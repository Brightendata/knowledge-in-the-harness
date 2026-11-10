"""One GRPO update of a LoRA adapter on one step's rollouts. Runs under torchrun, one process per GPU, in the model
server's environment (torch, transformers, peft) with src/ on PYTHONPATH; rsi.pipeline.grpo starts it.

    torchrun --standalone --nproc_per_node <n> -m rsi.grpo.train <settings.json>

settings.json: base (the model weights), adapter (the adapter to update, null to start one), out, rollouts (the step's
tau2 Results, run with vLLM's token ids and logprobs) and the [grpo] values the update uses. Writes the new adapter,
optimizer.pt and train.json to out.
"""
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface, Gemma4UnifiedForConditionalGeneration

# Tokens of one pack at most: on an A800 (80 GB) the trainer peaks at about 26 GiB plus 1.06 GiB per 1,000 pack tokens.
PACK_LIMIT = 40_000


def calls(sim):
    """(prompt ids, completion ids, completion logprobs) of each model call of a conversation, in order; the load_skill
    calls the agent answered itself come before the reply that followed them."""
    out = []
    for m in sim["messages"]:
        raw = m.get("raw_data") if m["role"] == "assistant" else None
        if not raw:
            continue
        for r in [load["response"] for load in raw.get("skill_loads", [])] + [raw]:
            choice = r["choices"][0]
            ids, logprobs = choice["provider_specific_fields"]["token_ids"], choice["logprobs"]["content"]
            assert len(ids) == len(logprobs), (len(ids), len(logprobs))
            out.append((r["prompt_token_ids"], ids, [t["logprob"] for t in logprobs]))
    return out


def pack(conversation):
    """A conversation's calls as one pack. The agent never sees its earlier thinking, so each call is a sample of its
    own, but its prompt is mostly the start of the last call's prompt: the pack holds that last prompt once, the trunk,
    and each call branches off where its prompt leaves the trunk. A branch is (where it leaves, the rest of its prompt,
    completion ids, logprobs); the rest is usually the few tokens opening the thinking channel at the end of a prompt,
    which the next prompt, without the thinking, does not repeat."""
    trunk = conversation[-1][0]
    branches = []
    for prompt, completion, logprobs in conversation:
        n = next((i for i, (a, b) in enumerate(zip(prompt, trunk)) if a != b), min(len(prompt), len(trunk)))
        branches.append((n, prompt[n:], completion, logprobs))
    return trunk, branches


def packs(conversation, limit=PACK_LIMIT):
    """A conversation's calls as packs of at most `limit` tokens: one pack, or its halves packed apart when longer, so
    that each completion still sees its whole prompt. A single call longer than that is left out."""
    trunk, branches = pack(conversation)
    if len(trunk) + sum(len(rest) + len(c) for _, rest, c, _ in branches) <= limit:
        return [(trunk, branches)]
    if len(conversation) == 1:
        return []
    half = len(conversation) // 2
    return packs(conversation[:half], limit) + packs(conversation[half:], limit)


def flatten(trunk, branches):
    """A pack as one sequence, the trunk then each branch: token ids; positions, a branch's continuing the trunk from
    where it left it; the branches' spans, (start, end, where it left the trunk); and, for each completion token, the
    index of the token whose logits predict it, the last token of its prompt for the first one."""
    ids, pos, spans, pred = list(trunk), list(range(len(trunk))), [], []
    for n, rest, completion, _ in branches:
        start, m = len(ids), len(rest) + len(completion)
        ids += rest + completion
        pos += range(n, n + m)
        spans.append((start, start + m, n))
        first = start + len(rest)
        pred += [first - 1 if rest else n - 1, *range(first, start + m - 1)]
    return ids, pos, spans, pred


class Blocks:
    """Who attends to whom in a flattened pack, in blocks for pack_attention: the trunk causally, each branch the trunk
    up to where it left it and itself causally; with a window, only the last `window` positions of each path, the
    trunk then in chunks of `window` queries. A block is (query start, query end, key ranges, mask), with no mask for
    plain causal attention. Dense masks over the whole pack would make SDPA attend to every pair and fall back to its
    slow masked kernel; these blocks keep the masked part small and skip the pairs a window leaves out."""

    def __init__(self, size, spans, window, device):
        self.blocks = []
        if window is None:
            self.blocks.append((0, size, [(0, size)], None))
        else:
            for s in range(0, size, window):
                e, a = min(s + window, size), max(0, s - window + 1)
                q, k = torch.arange(s, e, device=device)[:, None], torch.arange(a, e, device=device)[None, :]
                self.blocks.append((s, e, [(a, e)], (k <= q) & (k > q - window)))
        for start, end, n in spans:
            a = 0 if window is None else max(0, n - window + 1)
            q = torch.arange(n, n + end - start, device=device)[:, None]
            k = torch.arange(a, n + end - start, device=device)[None, :]
            self.blocks.append((start, end, [(a, n), (start, end)], (k <= q) if window is None else
                                (k <= q) & (k > q - window)))


def pack_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    """Attention over a pack for HF models, registered as "rsi_pack": attention_mask is the Blocks of the layer's
    attention type."""
    rep = query.shape[1] // key.shape[1]
    out = []
    for qs, qe, ranges, mask in attention_mask.blocks:
        k = torch.cat([key[:, :, a:b] for a, b in ranges], 2).repeat_interleave(rep, 1)
        v = torch.cat([value[:, :, a:b] for a, b in ranges], 2).repeat_interleave(rep, 1)
        out.append(F.scaled_dot_product_attention(query[:, :, qs:qe], k, v, attn_mask=mask, is_causal=mask is None,
                                                  scale=scaling))
    return torch.cat(out, 2).transpose(1, 2).contiguous(), None


AttentionInterface.register("rsi_pack", pack_attention)


def token_logprobs(lm, ids, pos, spans, pred, targets, chunk=1024):
    """Log-probabilities of a pack's completion tokens under lm, a Gemma4UnifiedForConditionalGeneration with the
    "rsi_pack" attention. The head runs in chunks, recomputed in the backward pass: over a 262k vocabulary the logits
    of a whole pack would not fit."""
    text = lm.config.get_text_config()
    size = len(ids) - sum(e - s for s, e, _ in spans)
    blocks = {"full_attention": Blocks(size, spans, None, ids.device),
              "sliding_attention": Blocks(size, spans, text.sliding_window, ids.device)}
    hidden = lm.model(input_ids=ids[None], position_ids=pos[None], use_cache=False,
                      attention_mask=blocks).last_hidden_state[0]

    def head(h, t):
        logits = lm.lm_head(h).float()
        logits = torch.tanh(logits / text.final_logit_softcapping) * text.final_logit_softcapping
        return logits.gather(1, t[:, None])[:, 0] - torch.logsumexp(logits, -1)

    return torch.cat([checkpoint(head, hidden[pred[i:i + chunk]], targets[i:i + chunk], use_reentrant=False)
                      for i in range(0, len(pred), chunk)])


def samples(paths):
    """The packs of the conversations of the step's informative groups, each with its advantage and weight. A task's
    conversations form a group; its advantages are the rewards standardized within the group, and a group whose
    conversations all score the same carries no signal. The weight averages a conversation's tokens, then the group's
    conversations, then the groups, as GRPO's loss does (Shao et al. 2024)."""
    groups = {}
    for path in paths:
        for sim in json.loads(Path(path).read_text())["simulations"]:
            if sim.get("reward_info"):
                groups.setdefault(sim["task_id"], []).append(sim)
    informative = [g for g in groups.values() if len({s["reward_info"]["reward"] for s in g}) > 1]
    out, dropped = [], 0
    for g in informative:
        rewards = torch.tensor([float(s["reward_info"]["reward"]) for s in g])
        for sim, adv in zip(g, ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).tolist()):
            conversation = calls(sim)
            parts = packs(conversation)
            dropped += len(conversation) - sum(len(branches) for _, branches in parts)
            tokens = sum(len(c) for _, branches in parts for _, _, c, _ in branches)
            out += [(trunk, branches, adv, 1 / (tokens * len(g) * len(informative))) for trunk, branches in parts]
    return out, {"groups": len(groups), "informative": len(informative),
                 "conversations": sum(len(g) for g in informative), "dropped_calls": dropped}


def share(items, rank, world):
    """This rank's packs: longest first, each to the least loaded rank, a pack's cost its length plus its attention's
    quadratic part (about equal at 24k tokens for this model)."""
    def cost(item):
        n = len(item[0]) + sum(len(r) + len(c) for _, r, c, _ in item[1])
        return n + n * n / 24000
    load, mine = [0.0] * world, []
    for item in sorted(items, key=cost, reverse=True):
        r = load.index(min(load))
        load[r] += cost(item)
        if r == rank:
            mine.append(item)
    return mine


def objective(ratio, adv, clip, dual_clip):
    """PPO's clipped objective of a run's tokens, from their ratios to the sampler's probabilities and the run's
    advantage, with the dual clip for a negative advantage (Ye et al. 2020; verl's clip_ratio_c): a token of a failed
    run that the policy now favours more than dual_clip times as much as the sampler did adds dual_clip x the
    advantage and no gradient, so that a few such tokens cannot swamp the step's gradient."""
    out = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    return out.clamp(min=dual_clip * adv) if adv < 0 else out


def accumulate(lm, params, acc, mine, settings, parts, label):
    """Add each pack's gradient to acc, as plain accumulation would, except a pack whose gradient per token is out of
    all proportion: its norm per unit of advantage, weight and completion token (the norm of the mean of ratio x the
    gradient of the log-probability over its tokens) above [grpo] max_token_grad. Normal conversations sit around 1; a
    context that blows up the gradient of every later token puts one at hundreds to tens of thousands, where alone it
    would take the update over, so it is left out. Returns the packs' stats, summed as main() records them, and each
    pack's gradient per token."""
    device = acc[0].device
    stats, per_token = torch.zeros(5, device=device), []
    for i, (trunk, branches, adv, weight) in enumerate(mine, 1):
        started = time.time()
        for p in params:
            p.grad = None
        ids, pos, spans, pred = flatten(trunk, branches)
        ids, pos, pred = (torch.tensor(x, device=device) for x in (ids, pos, pred))
        targets = torch.tensor([t for _, _, c, _ in branches for t in c], device=device)
        sampled = torch.tensor([x for _, _, _, lp in branches for x in lp], device=device)
        logprobs = token_logprobs(lm, ids, pos, spans, pred, targets)
        ratio = torch.exp(logprobs - sampled)
        loss = -weight * parts * objective(ratio, adv, settings["clip"], settings["dual_clip"]).sum()
        loss.backward()
        norm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)).item()
        per_token.append(norm / (parts * abs(adv) * weight * len(targets)))
        # a NaN gradient fails every comparison; kept, it would turn the whole adapter and the optimizer state into NaN
        left_out = not per_token[-1] <= settings["max_token_grad"]
        if not left_out:
            for a, p in zip(acc, params):
                if p.grad is not None:
                    a.add_(p.grad)
        stats += torch.tensor([len(ids), len(targets), (logprobs - sampled).abs().sum().item(),
                               ((ratio - 1).abs() > settings["clip"]).sum().item(), loss.item()], device=device)
        print(f"{label}, pack {i}/{len(mine)}, {len(ids)} tokens ({len(targets)} completion), grad norm {norm:.3f}, "
              f"{per_token[-1]:.1f} per token{' (left out)' if left_out else ''}, {time.time() - started:.1f} s, "
              f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return stats, per_token


def copy_k_to_v(path, text_config):
    """Write the k_proj LoRA of each global attention layer also as its v_proj LoRA. Those layers have no v_proj: V is
    the output of k_proj, so here the k_proj LoRA moves V too; vLLM loads k_proj's weights into both the K and the V
    slot of its fused projection but applies a k_proj LoRA to K only, and applies the copy to V."""
    if not text_config.attention_k_eq_v:
        return
    global_layers = {i for i, t in enumerate(text_config.layer_types) if t == "full_attention"}
    weights = load_file(path / "adapter_model.safetensors")
    for key in list(weights):
        m = re.search(r"\.layers\.(\d+)\.self_attn\.k_proj\.", key)
        if m and int(m[1]) in global_layers:
            weights[key.replace(".k_proj.", ".v_proj.")] = weights[key].clone()
    save_file(weights, path / "adapter_model.safetensors")


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    items, counts = samples(settings["rollouts"])

    lm = Gemma4UnifiedForConditionalGeneration.from_pretrained(settings["base"], dtype=torch.bfloat16,
                                                                attn_implementation="rsi_pack", device_map={"": device})
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if settings["adapter"]:
        model = PeftModel.from_pretrained(lm, settings["adapter"], is_trainable=True)
    else:
        model = get_peft_model(lm, LoraConfig(r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
                                              target_modules=settings["lora_targets"], lora_dropout=0.0))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    # the ranks sum their gradients, so they must hold the same adapter; a new one's lora_A is random
    for p in params:
        dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=0.0)
    # an earlier round's checkpoint comes without its optimizer state, which a round starts afresh
    if settings["adapter"] and (Path(settings["adapter"]) / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(Path(settings["adapter"]) / "optimizer.pt", map_location=device))

    # The step's packs in [grpo] mini_batches random parts, an optimizer update after each: on-policy for the first,
    # the later ones corrected by the ratio to the sampler's logprobs and clipped, as in PPO (objective). Each part's
    # weights are scaled to the whole step's.
    parts = settings["mini_batches"]
    order = random.Random(0).sample(items, len(items))
    # summed over ranks and parts: tokens of the packs, completion tokens, |logprob - sampler's|, ratios clipped, loss
    stats, norms, per_token = torch.zeros(5, device=device), [], []
    acc = [torch.zeros_like(p) for p in params]
    for b in range(parts):
        for a in acc:
            a.zero_()
        part_stats, part_per_token = accumulate(lm, params, acc, share(order[b::parts], rank, world), settings, parts,
                                                f"rank {rank}: part {b + 1}/{parts}")
        stats += part_stats
        per_token += part_per_token
        for p, a in zip(params, acc):
            p.grad = a
            dist.all_reduce(p.grad)
        norms.append(torch.nn.utils.clip_grad_norm_(params, settings["max_grad_norm"]).item())
        if rank == 0:
            print(f"update {b + 1}/{parts}: grad norm {norms[-1]:.3f}", flush=True)
        if order[b::parts]:
            opt.step()
        opt.zero_grad()
    dist.all_reduce(stats)
    left_out = torch.tensor(float(sum(not u <= settings["max_token_grad"] for u in per_token)), device=device)
    largest = torch.tensor(max(per_token, default=0.0), device=device)
    dist.all_reduce(left_out)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)

    if rank == 0:
        tokens, targets, gap, clipped, loss = stats.tolist()
        metrics = {**counts, "updates": sum(1 for b in range(parts) if order[b::parts]), "tokens": int(tokens),
                   "completion_tokens": int(targets), "loss": loss / parts, "grad_norms": norms,
                   "packs_left_out": int(left_out.item()), "max_token_grad": largest.item(),
                   "logprob_gap": gap / max(targets, 1), "clipped": clipped / max(targets, 1)}
        tmp = Path(settings["out"] + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        copy_k_to_v(tmp, lm.config.get_text_config())
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
        (tmp / "train.json").write_text(json.dumps(metrics, indent=1) + "\n")
        os.replace(tmp, settings["out"])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
