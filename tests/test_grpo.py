"""GRPO without the GPU server: a pack's logprobs on a tiny random Gemma 4 against each call run alone, the samples
taken from tau2 Results, rollouts per task, the steps of an epoch and the model gate's choice. A minute or two on a
CPU."""
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

import rsi.benchmarks.run as run_module
from rsi.benchmarks.run import served, start_system
from rsi.grpo.train import (accumulate, calls, copy_k_to_v, flatten, objective, pack, packs, samples, share,
                            token_logprobs)
from rsi.pipeline import grpo as grpo_step
from rsi.pipeline import model_gate as gate_step
from rsi.pipeline.grpo import batches, rollouts_for
from rsi.pipeline.model_gate import choose


def raw(prompt, completion):
    """An agent call's raw_data as vLLM returns it through litellm with return_token_ids and logprobs."""
    return {"prompt_token_ids": prompt, "choices": [{"provider_specific_fields": {"token_ids": completion},
                                                     "logprobs": {"content": [{"logprob": -0.5} for _ in completion]}}]}


def conversation(task, reward, calls_):
    """A tau2 simulation: tau2's greeting, then one agent message per call, a load_skill call riding on the first."""
    (p0, c0), *rest = calls_
    messages = [{"role": "assistant", "raw_data": None}, {"role": "user", "content": "hi"}]
    if rest:
        (p1, c1), *rest = rest
        loads = [{"name": "x", "response": raw(p0, c0)}]
        messages.append({"role": "assistant", "raw_data": {**raw(p1, c1), "skill_loads": loads}})
    else:
        messages.append({"role": "assistant", "raw_data": {**raw(p0, c0), "skill_loads": []}})
    for p, c in rest:
        messages += [{"role": "tool", "content": "{}"},
                     {"role": "assistant", "raw_data": {**raw(p, c), "skill_loads": []}}]
    return {"task_id": task, "trial": 0, "reward_info": {"reward": reward}, "messages": messages}


def tiny_lm(attention="sdpa"):
    """A random Gemma 4 like the 12B in miniature, the same weights on every call: sliding layers and a global one
    whose K and V share k_proj, a window of 4 positions."""
    torch.manual_seed(0)
    cfg = Gemma4UnifiedConfig(text_config=dict(
        num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"], hidden_size=64,
        intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
        head_dim=16, global_head_dim=32, sliding_window=4, vocab_size=300, attention_k_eq_v=True,
        final_logit_softcapping=30.0))
    return Gemma4UnifiedForConditionalGeneration._from_config(cfg, attn_implementation=attention, dtype=torch.float32)


class PackTest(unittest.TestCase):
    def test_a_pack_gives_each_completion_its_own_calls_logprobs(self):
        reference, packed = tiny_lm().eval(), tiny_lm("rsi_pack").eval()
        rng = random.Random(0)
        tok = lambda n: [rng.randrange(1, 300) for _ in range(n)]  # noqa: E731
        # as the chat template renders them: each prompt ends with a few tokens opening the thinking channel, which
        # the next prompt does not repeat; it goes on with the reply without its thinking and what came back. One
        # prompt leaves the trunk early, one shares nothing with it; the window of 4 cuts into trunk and branches.
        opening = [7, 8, 9]
        h1 = tok(11)
        h2 = h1 + tok(6)
        h3 = h2 + tok(7)
        prompts = [h1 + opening, h1[:2] + tok(5) + opening, [0] + tok(3) + opening, h2 + opening, h3 + opening]
        conv = [(p, tok(n), [0.0] * n) for p, n in zip(prompts, (5, 9, 4, 4, 6))]
        trunk, branches = pack(conv)
        self.assertEqual([(n, len(rest)) for n, rest, _, _ in branches], [(11, 3), (2, 8), (0, 7), (17, 3), (27, 0)])
        ids, pos, spans, pred = flatten(trunk, branches)
        ids, pos, pred = (torch.tensor(x) for x in (ids, pos, pred))
        targets = torch.tensor([t for _, c, _ in conv for t in c])
        got = token_logprobs(packed, ids, pos, spans, pred, targets, chunk=4)
        want = []
        for p, c, _ in conv:
            logits = reference(input_ids=torch.tensor([p + c]), use_cache=False).logits[0, len(p) - 1:-1].float()
            want.append(logits.log_softmax(-1).gather(1, torch.tensor(c)[:, None])[:, 0])
        torch.testing.assert_close(got, torch.cat(want), atol=1e-5, rtol=1e-5)
        got.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in packed.parameters()))

    def test_a_global_layers_k_proj_lora_is_written_also_as_its_v_proj(self):
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        config = LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
        model = get_peft_model(tiny_lm(), config)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            saved = load_file(Path(tmp) / "adapter_model.safetensors")
            copy_k_to_v(Path(tmp), model.get_base_model().config.get_text_config())
            weights = load_file(Path(tmp) / "adapter_model.safetensors")
            added = sorted(set(weights) - set(saved))
            # layer 5, the global one, had no v_proj; the sliding layers keep their own
            self.assertEqual(added, [f"base_model.model.model.language_model.layers.5.self_attn.v_proj.lora_{x}.weight"
                                     for x in "AB"])
            for key in added:
                torch.testing.assert_close(weights[key], weights[key.replace("v_proj", "k_proj")])
            # the next step loads the adapter back for training, the copies left out
            again = PeftModel.from_pretrained(tiny_lm(), tmp, is_trainable=True)
            for name, p in model.named_parameters():
                if "lora_" in name:
                    torch.testing.assert_close(dict(again.named_parameters())[name], p)

    def test_calls_come_in_order_and_branch_off_the_last_prompt(self):
        made = [([1, 2, 3, 9], [7]), ([1, 2, 3, 4, 9], [6]), ([5, 6], [8])]
        sim = conversation("t", 1.0, made)
        self.assertEqual([(p, c) for p, c, _ in calls(sim)], made)
        # a prompt sharing nothing with the last one still packs, its whole prompt in its branch
        trunk, branches = pack(calls(sim))
        self.assertEqual((trunk, [(n, r, c) for n, r, c, _ in branches]),
                         ([5, 6], [(0, [1, 2, 3, 9], [7]), (0, [1, 2, 3, 4, 9], [6]), (2, [], [8])]))

    def test_a_long_conversation_packs_in_parts_and_an_overlong_call_is_left_out(self):
        made = [([1, 2, 3, 9], [7, 7]), ([1, 2, 3, 4, 9], [6]), ([1, 2, 3, 4, 5, 9], [8, 8]),
                ([1, 2, 3, 4, 5, 6, 9], [5])]
        conv = calls(conversation("t", 1.0, made))

        def completions(limit):
            out = []
            for trunk, branches in packs(conv, limit):
                self.assertLessEqual(len(flatten(trunk, branches)[0]), limit)
                out.append([c for _, _, c, _ in branches])
            return out

        # 16 tokens packed whole, 9 + 11 in halves; at 7 the first two calls go alone and the last two, 8 each, go
        self.assertEqual(completions(16), [[[7, 7], [6], [8, 8], [5]]])
        self.assertEqual(completions(15), [[[7, 7], [6]], [[8, 8], [5]]])
        self.assertEqual(completions(7), [[[7, 7]], [[6]]])


class SampleTest(unittest.TestCase):
    def test_groups_advantages_and_weights(self):
        sims = ([conversation("a", r, [([1, 2], [3] * (2 + i))]) for i, r in enumerate([1.0, 0.0, 0.0])]
                + [conversation("b", 1.0, [([1], [2])]) for _ in range(2)]
                + [conversation("c", r, [([1], [2, 3]), ([1, 5], [6])]) for r in [0.0, 1.0, 1.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            path.write_text(json.dumps({"simulations": sims}))
            items, counts = samples([path])
        self.assertEqual(counts, {"groups": 3, "informative": 2, "conversations": 7, "dropped_calls": 0})
        # every group counts the same, every conversation in it the same, whatever their lengths
        total = sum(w * sum(len(c) for _, _, c, _ in b) for _, b, _, w in items)
        self.assertAlmostEqual(total, 1.0)
        advantages = [a for _, _, a, _ in items]
        self.assertAlmostEqual(sum(advantages[:3]), 0.0, places=5)
        self.assertGreater(advantages[0], 0)
        self.assertLess(advantages[1], 0)

    def test_a_failed_runs_token_far_more_likely_now_adds_no_gradient(self):
        # ratios to the sampler: within the clip, beyond it, and beyond the dual clip of 3
        ratio = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
        objective(ratio, -1.0, 0.2, 3.0).sum().backward()
        self.assertEqual(ratio.grad.tolist(), [-1.0, -1.0, 0.0])
        # a positive advantage keeps PPO's clip alone: no gradient above 1 + 0.2
        ratio = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
        objective(ratio, 1.0, 0.2, 3.0).sum().backward()
        self.assertEqual(ratio.grad.tolist(), [1.0, 0.0, 0.0])

    def packs_on_policy(self):
        """A LoRA model with a random adapter, and packs of three conversations whose sampler logprobs are the
        model's own, so that every token of every pack has a gradient."""
        lm = tiny_lm("rsi_pack")
        model = get_peft_model(lm, LoraConfig(r=2, lora_alpha=2, init_lora_weights=False, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.train()
        made = [[([1, 2, 3], [4, 5, 6]), ([1, 2, 3, 6, 7], [8, 9])], [([1, 2], [3] * 5)],
                [([1, 2, 9], [10, 11]), ([1, 2, 9, 12], [13, 14, 15]), ([1, 2, 9, 12, 16], [17])]]
        items = []
        for conv, adv in zip(made, (1.2, -0.7, -1.5)):
            trunk, branches = pack(calls(conversation("t", 1.0, conv)))
            ids, pos, spans, pred = (torch.tensor(x) if i != 2 else x for i, x in enumerate(flatten(trunk, branches)))
            with torch.no_grad():
                own = token_logprobs(lm, ids, pos, spans, pred,
                                     torch.tensor([t for _, _, c, _ in branches for t in c])).tolist()
            sized, at = [], 0
            for n, rest, c, _ in branches:
                sized.append((n, rest, c, own[at:at + len(c)]))
                at += len(c)
            items.append((trunk, sized, adv, 0.05))
        return lm, [p for p in model.parameters() if p.requires_grad], items

    def plain(self, lm, params, items, parts):
        """What the trainer did before any pack was left out: backward pass after backward pass into .grad."""
        for p in params:
            p.grad = None
        for trunk, branches, adv, weight in items:
            ids, pos, spans, pred = flatten(trunk, branches)
            ids, pos, pred = (torch.tensor(x) for x in (ids, pos, pred))
            logprobs = token_logprobs(lm, ids, pos, spans, pred,
                                      torch.tensor([t for _, _, c, _ in branches for t in c]))
            ratio = torch.exp(logprobs - torch.tensor([x for _, _, _, lp in branches for x in lp]))
            (-weight * parts * objective(ratio, adv, 0.2, 3.0).sum()).backward()
        return [p.grad.clone() for p in params]

    def test_packs_under_the_threshold_add_up_exactly_as_plain_accumulation(self):
        lm, params, items = self.packs_on_policy()
        plain = self.plain(lm, params, items, 4)
        acc = [torch.zeros_like(p) for p in params]
        with mock.patch("sys.stdout", new=io.StringIO()):
            stats, per_token = accumulate(lm, params, acc, items,
                                          {"clip": 0.2, "dual_clip": 3.0, "max_token_grad": 1e9}, 4, "test")
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(acc, plain)))
        self.assertTrue(all(u > 0 for u in per_token))
        self.assertEqual(int(stats[1]), sum(len(c) for _, b, _, _ in items for _, _, c, _ in b))

    def test_a_pack_over_the_threshold_is_left_out_and_nothing_else_changes(self):
        lm, params, items = self.packs_on_policy()
        acc = [torch.zeros_like(p) for p in params]
        with mock.patch("sys.stdout", new=io.StringIO()):
            _, per_token = accumulate(lm, params, acc, items,
                                      {"clip": 0.2, "dual_clip": 3.0, "max_token_grad": 1e9}, 4, "test")
        # the gradient per token is the pack's gradient norm per unit of advantage, weight and completion token
        alone = [self.plain(lm, params, [item], 4) for item in items]
        for u, grads, (_, branches, adv, weight) in zip(per_token, alone, items):
            norm = torch.sqrt(sum((g ** 2).sum() for g in grads)).item()
            self.assertAlmostEqual(u * 4 * abs(adv) * weight * sum(len(c) for _, _, c, _ in branches) / norm, 1.0,
                                   places=5)
        top = per_token.index(max(per_token))
        threshold = (sorted(per_token)[-1] + sorted(per_token)[-2]) / 2
        acc = [torch.zeros_like(p) for p in params]
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            accumulate(lm, params, acc, items, {"clip": 0.2, "dual_clip": 3.0, "max_token_grad": threshold}, 4,
                       "test")
        kept = self.plain(lm, params, [item for i, item in enumerate(items) if i != top], 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(acc, kept)))
        self.assertEqual(out.getvalue().count("(left out)"), 1)

    def test_a_pack_with_a_nan_gradient_is_left_out(self):
        lm, params, items = self.packs_on_policy()
        # a NaN sampler logprob in the successful conversation makes its gradient NaN
        trunk, ((n, rest, c, lp), *others), adv, weight = items[0]
        items[0] = (trunk, [(n, rest, c, [float("nan")] + lp[1:]), *others], adv, weight)
        acc = [torch.zeros_like(p) for p in params]
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            _, per_token = accumulate(lm, params, acc, items,
                                      {"clip": 0.2, "dual_clip": 3.0, "max_token_grad": 1e9}, 4, "test")
        self.assertTrue(torch.isnan(torch.tensor(per_token[0])))
        self.assertEqual(out.getvalue().count("(left out)"), 1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(acc, self.plain(lm, params, items[1:], 4))))

    def test_every_pack_goes_to_one_rank(self):
        sizes = [(9, 2), (30, 5), (4, 1), (12, 3), (7, 7)]
        items = [([0] * n, [(n, [], [1] * m, [0.0] * m)], 1.0, 1.0) for n, m in sizes]
        mine = [share(items, r, 3) for r in range(3)]
        self.assertEqual(sorted(id(x) for m in mine for x in m), sorted(id(x) for x in items))


class StepTest(unittest.TestCase):
    def test_rollouts_follow_the_latest_runs(self):
        counts = [16, 8, 5]
        self.assertEqual([rollouts_for(s, n, counts) for s, n in [(0, 4), (1, 4), (3, 4), (4, 4), (5, 5), (0, 16)]],
                         [16, 8, 8, 5, 5, 16])

    def test_each_epoch_deals_every_task_once(self):
        g = {"shuffle_seed": 1, "steps_per_epoch": 6}
        tasks = [f"t{i}" for i in range(23)]
        for epoch in (1, 2):
            steps = batches(tasks, epoch, g)
            self.assertEqual(len(steps), 6)
            self.assertEqual(sorted(t for s in steps for t in s), sorted(tasks))
            self.assertLessEqual(max(map(len, steps)) - min(map(len, steps)), 1)
        self.assertNotEqual(batches(tasks, 1, g), batches(tasks, 2, g))

    def test_steps_checkpoints_and_resume(self):
        # 5 tasks, 2 epochs of 2 steps, a checkpoint after the first and the last two; a task's rollouts all succeed
        # when its id is even; the round starts from an earlier round's adapter
        start = "artifacts/checkpoints/r0/airline/epoch2"
        g = {"epochs": 2, "steps_per_epoch": 2, "checkpoints": [0.5, 1.5, 2], "rollouts": [3, 2, 1], "excluded": ["t9"],
             "shuffle_seed": 0, "base": "B", "python": "P", "gpus": "2,3", "server_gpus": "0,1", "concurrency": 8,
             "lora_rank": 4, "lora_alpha": 4, "lora_targets": ["q_proj"], "lr": 1e-5, "mini_batches": 2, "clip": 0.2,
             "dual_clip": 3.0, "max_token_grad": 100, "max_grad_norm": 1.0}
        cfg = {"system": {"model": "base"}, "agent": {}, "user": {"train": {}}}
        recs = [{"task": {"id": f"t{i}"}} for i in [0, 1, 2, 3, 4, 9]]
        profile = {"t0": [True] * 4, "t1": [False] * 4, "t2": [True, False] * 2, "t3": [False] * 4, "t4": [True] * 4}
        runs, trained, served, updated = [], [], [], []

        def run(domain, task_recs, plugins, model, user, trials, seed, save_path, concurrency=None, agent_extra=None):
            runs.append((model, trials, sorted(r["task"]["id"] for r in task_recs)))
            return {r["task"]["id"]: [int(r["task"]["id"][1:]) % 2 == 0] * trials for r in task_recs}

        def train(argv, **kwargs):
            settings = json.loads(Path(argv[-1]).read_text())
            out = Path(settings["out"])
            trained.append(out.parent.name)
            updated.append(Path(settings["adapter"]).relative_to(tmp).as_posix())
            out.mkdir(parents=True)
            (out / "adapter_model.safetensors").write_text("w")
            (out / "optimizer.pt").write_text("o")
            (out / "train.json").write_text(json.dumps({"informative": 1, "groups": 2, "loss": 0.0,
                                                        "grad_norms": [0.0, 0.0], "packs_left_out": 0,
                                                        "max_token_grad": 0.0, "logprob_gap": 0.0, "clipped": 0.0,
                                                        "dropped_calls": 0}))

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.multiple(grpo_step, ROOT=Path(tmp), config=lambda: cfg, grpo_config=lambda: g,
                                    records=lambda domain, split: recs, outcomes=lambda r: profile if r == "P" else r,
                                    Results=SimpleNamespace(load=lambda path: "P"), run=run,
                                    empty_replies=lambda results: 0, serve=lambda name, path: served.append(name),
                                    server=lambda gpus: served.append(f"server {gpus}"),
                                    start_system=lambda round_: {"airline": {"model": start}}), \
                mock.patch.object(grpo_step.subprocess, "run", side_effect=train), \
                mock.patch("sys.stdout", new=io.StringIO()) as term:
            (Path(tmp) / start).mkdir(parents=True)
            (Path(tmp) / start / "adapter_model.safetensors").write_text("M(k-1)")
            grpo_step.grpo("airline", 1, ["pa", "pr"])
            # the earlier round's adapter is only read
            self.assertEqual([p.name for p in (Path(tmp) / start).iterdir()], ["adapter_model.safetensors"])
            self.assertEqual((Path(tmp) / start / "adapter_model.safetensors").read_text(), "M(k-1)")
            out = Path(tmp) / "artifacts/logs/r1/grpo/airline"
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["start", "step", "step", "step", "step", "stop"])
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "artifacts/checkpoints/r1/airline").iterdir()),
                             ["epoch0.5", "epoch1.5", "epoch2"])
            self.assertFalse((Path(tmp) / "artifacts/checkpoints/r1/airline/epoch1.5/optimizer.pt").exists())
            # the model server on its GRPO GPUs first; a step's rollouts run while the step before it trains: from the
            # adapter two steps back
            self.assertEqual(served, ["server 0,1", "airline-r1-s01", "airline-r1-s02"])
            self.assertEqual(list(dict.fromkeys(m for m, _, _ in runs)), [start, "airline-r1-s01", "airline-r1-s02"])
            # the first update goes on from the earlier round's adapter, each later one from the step before
            self.assertEqual(updated, [start] + [f"artifacts/logs/r1/grpo/airline/steps/s0{s}/adapter"
                                                 for s in (1, 2, 3)])
            self.assertEqual(json.loads((out / "config.json").read_text())["model"], start)
            # epoch 1 by the profile: t0, t4 always solved, t1, t3 never, t2 sometimes; epoch 2 by epoch 1's rollouts,
            # where the even tasks always succeed and the odd ones never do
            per_task = {}
            for _, trials, ids in runs:
                for t in ids:
                    per_task.setdefault(t, []).append(trials)
            self.assertEqual(per_task, {"t0": [1, 1], "t1": [3, 3], "t2": [2, 1], "t3": [3, 3], "t4": [1, 1]})
            self.assertEqual(trained, ["s01", "s02", "s03", "s04"])
            self.assertEqual([p.parent.name for p in out.glob("steps/*/adapter")], ["s04"])
            self.assertEqual(list(out.glob("steps/*/train_settings.json")), [])
            # the round's log gets the start and the end, the summary a line per step
            self.assertEqual([line.split(":")[0] for line in term.getvalue().splitlines()], ["airline GRPO"] * 2)
            self.assertEqual([line.split()[3] for line in (out / "summary.log").read_text().splitlines()],
                             ["1", "2", "3", "4"])
            # stopped before step 4's event: its rollouts run again (they resume from their files), the update does not
            (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events[:4]))
            runs.clear()
            grpo_step.grpo("airline", 1, ["pa", "pr"])
            self.assertEqual({m for m, _, _ in runs}, {"airline-r1-s02"})
            self.assertEqual(trained, ["s01", "s02", "s03", "s04"])
            lines = (out / "events.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["event"] for line in lines][-2:], ["step", "stop"])

    def test_a_round_starts_from_what_earlier_gates_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            def gate(path, accepted, saved):
                (Path(tmp) / path).mkdir(parents=True)
                (Path(tmp) / path / "decision.json").write_text(json.dumps({"accepted": accepted, "saved": saved}))

            # r1: both plugin gates and airline's model gate accept; r2: retail's plugin only on the retry, airline's
            # model gate accepts none
            gate("r1/gate/plugin_retail", True, "R1")
            gate("r1/gate/plugin_airline", True, "A1")
            gate("r1/gate/model_airline", "epoch1.5", "artifacts/checkpoints/r1/airline/epoch1.5")
            gate("r1/gate/model_retail", None, None)
            gate("r2/gate/plugin_retail", False, "R2?")
            gate("r2/gate/plugin_retail_retry", True, "R2")
            gate("r2/gate/model_airline", None, None)
            systems = [start_system(k, Path(tmp)) for k in (1, 2, 3)]
        self.assertEqual(systems[0], {"retail": {"plugin": "plugins/retail", "model": "gemma-4-12b"},
                                      "airline": {"plugin": "plugins/airline", "model": "gemma-4-12b"}})
        m1 = "artifacts/checkpoints/r1/airline/epoch1.5"
        self.assertEqual(systems[1]["airline"], {"plugin": "A1", "model": m1})
        # airline's r2 model gate accepted none: r3 starts from r1's adapter
        self.assertEqual(systems[2], {"retail": {"plugin": "R2", "model": "gemma-4-12b"},
                                      "airline": {"plugin": "A1", "model": m1}})

    def test_an_adapter_is_served_by_its_domain_round_and_checkpoint(self):
        with mock.patch.object(run_module, "serve") as serve:
            self.assertEqual(served("gemma-4-12b"), "gemma-4-12b")
            self.assertEqual(served("artifacts/checkpoints/r1/airline/epoch1.5"), "airline-r1-epoch1.5")
        serve.assert_called_once_with("airline-r1-epoch1.5", "artifacts/checkpoints/r1/airline/epoch1.5")

    def test_the_gate_takes_the_best_finalist_that_passes(self):
        def scored(passes, s):
            return {"passes": passes, "score": {"candidate": s}}
        scores = {"epoch2": scored(True, 0.70), "epoch2.5": scored(False, 0.80), "epoch3": scored(True, 0.72),
                  "epoch3.5": scored(True, 0.72)}
        self.assertEqual(choose(scores), "epoch3")
        self.assertIsNone(choose({"epoch4": scored(False, 0.9)}))

    def test_finalists_are_the_best_screened_in_checkpoint_order(self):
        # screened pass^1 by the stand-in pass_k: epoch1 0.6, epoch2 0.8, epoch2.5 0.7, epoch3 0.8
        screened = {"epoch1": 0.6, "epoch2": 0.8, "epoch2.5": 0.7, "epoch3": 0.8}
        with mock.patch.object(gate_step, "pass_k", lambda r, k: {"t": r}):
            self.assertEqual(gate_step.finalists(screened, 2), ["epoch2", "epoch3"])
            self.assertEqual(gate_step.finalists(screened, 3), ["epoch2", "epoch2.5", "epoch3"])
            self.assertEqual(gate_step.finalists({"epoch1": 0.5}, 2), ["epoch1"])

    def test_the_gate_screens_then_runs_the_finalists_against_the_current_system(self):
        # r2: retail's plugin gate scored H(k) on M(k-1) under seed 7; airline's evolution kept no edit, so it had none
        # and the gate runs M(k-1) itself. Three checkpoints each; the screening ranks epoch2 and epoch3 first.
        cfg = {"user": {"gate": {}}, "gate": {"trials": 8, "bootstrap": 10, "pass4_weight": 0}}
        m1 = "artifacts/checkpoints/r1/airline/epoch2"
        screen_p1 = {"epoch1": 0.6, "epoch2": 0.8, "epoch3": 0.7}
        runs, compared = [], []

        def run(domain, recs, plugins, model, user, trials, seed, save_path, concurrency=None):
            runs.append((model, seed, trials, Path(save_path).relative_to(tmp).as_posix()))
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            return screen_p1.get(Path(model).name) if "/screen/" in str(save_path) else model

        def paired(current, candidate, k, boot):
            compared.append((candidate, current, f"pass^{k}" if isinstance(k, int) else f"score {k}"))
            return {"current": 0.5, "candidate": 0.6, "diff": 0.1, "ci95": [0.0, 0.2]}

        with tempfile.TemporaryDirectory() as tmp:
            gates = Path(tmp) / "artifacts/logs/r2/gate"
            (gates / "plugin_retail").mkdir(parents=True)
            (gates / "plugin_retail/decision.json").write_text(json.dumps({"seed": 7, "accepted": True}))
            (gates / "plugin_retail/candidate.json").write_text("{}")
            for d in ("airline", "retail"):
                for c in screen_p1:
                    (Path(tmp) / f"artifacts/checkpoints/r2/{d}/{c}").mkdir(parents=True)
            with mock.patch.multiple(gate_step, ROOT=Path(tmp), config=lambda: cfg, records=lambda d, split: [],
                                     grpo_config=lambda section: {"gpus": "0,1,2,3", "concurrency": 8,
                                                                  "screen_trials": 3, "finalists": 2,
                                                                  "pass4_weight": 0.2},
                                     server=lambda gpus: None, served=lambda model: model, run=run, paired=paired,
                                     pass_k=lambda r, k: {"t": r},
                                     Results=SimpleNamespace(load=lambda path: Path(path).name),
                                     start_system=lambda round_: {"airline": {"model": m1}}), \
                    mock.patch("sys.stdout", new=io.StringIO()):
                gate_step.model_gate(["airline", "retail"], 2, ["pa", "pr"])
            decisions = {d: json.loads((gates / f"model_{d}/decision.json").read_text()) for d in ("airline", "retail")}
        own, screen = 2_500_000, 2_600_000
        ckpt = "artifacts/checkpoints/r2/{}/{}".format
        self.assertEqual(sorted(runs), sorted(
            [(ckpt(d, c), screen, 3, f"artifacts/logs/r2/gate/model_{d}/screen/{c}.json")
             for d in ("airline", "retail") for c in screen_p1]
            + [(m1, own, 8, "artifacts/logs/r2/gate/model_airline/current.json")]
            + [(ckpt("airline", c), own, 8, f"artifacts/logs/r2/gate/model_airline/{c}.json") for c in ("epoch2", "epoch3")]
            + [(ckpt("retail", c), 7, 8, f"artifacts/logs/r2/gate/model_retail/{c}.json") for c in ("epoch2", "epoch3")]))
        # airline's finalists against its own run of M(k-1), retail's against the plugin gate's candidate, on pass^1,
        # pass^4 and the model gate's own score, not the plugin gate's
        keys = ("pass^1", "pass^4", "score {1: 0.8, 4: 0.2}")
        self.assertEqual(sorted(set(compared)), sorted(
            [(ckpt("airline", c), m1, k) for c in ("epoch2", "epoch3") for k in keys]
            + [(ckpt("retail", c), "candidate.json", k) for c in ("epoch2", "epoch3") for k in keys]))
        self.assertEqual((decisions["airline"]["seed"], decisions["airline"]["current"]),
                         (own, "artifacts/logs/r2/gate/model_airline/current.json"))
        self.assertEqual((decisions["retail"]["seed"], decisions["retail"]["current"]),
                         (7, "artifacts/logs/r2/gate/plugin_retail/candidate.json"))
        self.assertEqual(list(decisions["retail"]["finalists"]), ["epoch2", "epoch3"])
        self.assertEqual(decisions["airline"]["accepted"], "epoch2")

    def test_without_screening_every_checkpoint_runs_against_the_current_system(self):
        # screen_trials 0: retail's three checkpoints all run V 8 times under its plugin gate's seed, 7
        cfg = {"user": {"gate": {}}, "gate": {"trials": 8, "bootstrap": 10, "pass4_weight": 0}}
        names = ["epoch3", "epoch3.5", "epoch4"]
        runs = []

        def run(domain, recs, plugins, model, user, trials, seed, save_path, concurrency=None):
            runs.append((Path(model).name, seed, trials, Path(save_path).name))
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            return model

        with tempfile.TemporaryDirectory() as tmp:
            gates = Path(tmp) / "artifacts/logs/r1/gate"
            (gates / "plugin_retail").mkdir(parents=True)
            (gates / "plugin_retail/decision.json").write_text(json.dumps({"seed": 7, "accepted": True}))
            (gates / "plugin_retail/candidate.json").write_text("{}")
            for c in names:
                (Path(tmp) / f"artifacts/checkpoints/r1/retail/{c}").mkdir(parents=True)
            with mock.patch.multiple(gate_step, ROOT=Path(tmp), config=lambda: cfg, records=lambda d, split: [],
                                     grpo_config=lambda section: {"gpus": "0", "concurrency": 8,
                                                                  "screen_trials": 0, "finalists": 2,
                                                                  "pass4_weight": 0.2},
                                     server=lambda gpus: None, served=lambda model: model, run=run,
                                     paired=lambda current, candidate, k, boot: {"current": 0.5, "candidate": 0.6,
                                                                                 "diff": 0.1, "ci95": [0.0, 0.2]},
                                     Results=SimpleNamespace(load=lambda path: Path(path).name)), \
                    mock.patch("sys.stdout", new=io.StringIO()):
                gate_step.model_gate(["retail"], 1, ["pa", "pr"])
            decision = json.loads((gates / "model_retail/decision.json").read_text())
        self.assertEqual(sorted(runs), sorted((c, 7, 8, f"{c}.json") for c in names))
        self.assertEqual(list(decision["finalists"]), names)
        self.assertNotIn("screen", decision)
        self.assertEqual(decision["accepted"], "epoch3")


if __name__ == "__main__":
    unittest.main()
