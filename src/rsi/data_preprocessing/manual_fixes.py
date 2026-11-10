"""Wording fixes for the few tasks the review found ambiguous, applied to the generators' output.

    python -m rsi.data_preprocessing.manual_fixes

Run after the generators; rewrites data/processed/{airline,retail}_final.jsonl in place. Running it again changes nothing.
"""
import json
from pathlib import Path

from rsi.data_preprocessing.airline.change import REBOOK, SICK_CANCEL

ROOT = Path(__file__).resolve().parents[3]
# task id -> (old, new): one speaker is told apart by the color the other one is changed to
RETAIL = {
    "generated_exchange_mind_W3002300": ("just the bluetooth speaker with color green;",
                                         "just the bluetooth speaker that is green now;"),
    # told apart from its twin by size L, while the only navy half-zip jacket in stock is XL: the user would hold on to L
    "generated_exchange_twin_W6874763": ("say the fleece jacket with size L.",
                                         "say the fleece jacket with size L; the new one may be another size."),
    # two items of one pending order said in two sentences: told to bring up one request at a time, the user may hold the
    # second item back until the first is changed, after which the order cannot be modified again
    "generated_pair_status_modify_2+new_order_sophia_garcia_5795": (
        "Bring up each request only after the previous one has been dealt with.",
        "Make sure you mention your requests together, in the order given."),
    "generated_lost_pair_lost_item+modify_2_anya_lee_8315": (
        "Bring up each request only after the previous one has been dealt with.",
        "Make sure you mention your requests together, in the order given."),
}
# a refused basic economy change with a cancellation as the fallback: unsaid, the user may take the upgrade to economy
# that lets the change through, which the policy allows
NO_UPGRADE = "You do not want to upgrade the cabin of the reservation whose flights you want changed."
FALLBACKS = [s.split("{")[0] for s in SICK_CANCEL + REBOOK]
FIELD = {"retail": "reason_for_call", "airline": "task_instructions"}  # where the sentences fixed are


def fixed(domain, tid, text):
    if domain == "retail" and tid in RETAIL:
        old, new = RETAIL[tid]
        assert text.count(old) + text.count(new) == 1, tid
        return text.replace(old, new)
    hits = [f for f in FALLBACKS if f in text] if domain == "airline" else []
    if hits and NO_UPGRADE not in text:
        assert len(hits) == 1 and text.count(hits[0]) == 1, tid
        i = text.index(hits[0])  # a line of its own, or a sentence of the reservation's line in a task of several
        return text[:i] + NO_UPGRADE + (text[i - 1] if i else "\n") + text[i:]
    return text


for domain in ("airline", "retail"):
    path = ROOT / f"data/processed/{domain}_final.jsonl"
    lines, changed = [], []
    for line in open(path):
        x = json.loads(line)
        s, field = x["task"]["user_scenario"]["instructions"], FIELD[domain]
        new = fixed(domain, x["task"]["id"], s[field])
        if new != s[field]:
            s[field] = new
            line = json.dumps(x) + "\n"
            changed.append(x["task"]["id"])
        lines.append(line)
    path.write_text("".join(lines))
    print(domain, len(changed), "changed:", " ".join(changed))
