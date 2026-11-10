"""Every generated retail task through policy rules the tools do not enforce and a reading of each request's spec (in the
generation report), both written without the generator's helpers; the tasks chosen for V and the training batches also
through tau2's own evaluator. Run after rsi.data_preprocessing.retail_generator. With SHARD=k/n the slow tests check
every n-th task from the k-th, so that n processes share them; the default, 0/20, checks every 20th task in a few
minutes, and SHARD=0/1 checks all of them (about two CPU-hours for both domains)."""
import json
import os
import re
import unittest
from collections import Counter

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Action, Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from rsi.data_preprocessing.common import ROOT, WRITES, outcome
from rsi.data_preprocessing.retail.facts import YES_NO
from rsi.data_preprocessing.retail.items import option_name
from rsi.data_preprocessing.retail_generator import FINAL, GENERATED, REPORT, TAU2_RETAIL_DB

ORDER_WRITES = WRITES["retail"] - {"modify_user_address"}
WORDS = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5, "portable": 0, "small": 1, "medium": 2, "large": 3,
         "low": 1, "high": 3, "beginner": 1, "intermediate": 2, "expert": 3, "basic": 1, "professional": 2,
         "standard": 1, "high-back": 2, "A6": 1, "A5": 2, "A4": 3, "1080p": 1080, "2K": 1440, "4K": 2160, "5K": 2880}
# what a reply may write another way, so that no option value or product name to state may have it
RESPELLED = r"(?i)\d|[-+]|grey|crew neck|full size|hard cover|soft cover|auto shutoff|not resistant|smart watch"
UP = r"bigger|larger|longer|taller|higher|brighter|more|thicker|heavier|harder"
DOWN = r"smaller|shorter|lower|less|thinner|lighter|easier|fewer"
SHARD, SHARDS = map(int, os.environ.get("SHARD", "0/20").split("/"))


def order_of(value):
    """A number that orders an option's values, or None for a value without an order."""
    if value in WORDS:
        return WORDS[value]
    m = re.match(r"(\d+(?:\.\d+)?)\s*(TB)?", value)
    return None if m is None else float(m.group(1)) * (1000 if m.group(2) else 1)


def money(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def worded(text, key, value):
    """The option value is in the text, as it is or in the customer's words for it ("no backlight")."""
    return value in text or YES_NO.get((key, value), value) in text


def reward(task, dump, trajectory):
    def environment(solo_mode=False):
        # each call must build its own DB, or the evaluator's two environments would share one
        return get_environment(db=RetailDB.model_validate(dump), solo_mode=solo_mode)

    r = EnvironmentEvaluator.calculate_reward(environment_constructor=environment, task=task,
                                              full_trajectory=trajectory).reward
    return r * CommunicateEvaluator.calculate_reward(task=task, full_trajectory=trajectory).reward


def replay(calls, dump):
    env = get_environment(db=RetailDB.model_validate(dump))
    messages = []
    for k, c in enumerate(calls):
        call = ToolCall(id=f"call_{k}", name=c["name"], arguments=c["arguments"], requestor="assistant")
        messages += [AssistantMessage(role="assistant", tool_calls=[call]), env.get_response(call)]
    return messages


def policy_violations(task, db):
    """Retail policy rules the tools do not enforce, on the reference replayed step by step: one customer per
    conversation; an exchange is for another option, not the same item; an order whose items were modified can no
    longer be modified."""
    env = get_environment(db=RetailDB.model_validate(db.model_dump()))
    live, found, owner, modified = env.tools.db, [], None, set()
    for a in task.evaluation_criteria.actions or []:
        g = a.arguments
        who = live.orders[g["order_id"]].user_id if a.name in ORDER_WRITES else g.get("user_id")
        owner = owner or who
        if who != owner:
            found.append(f"{a.name} for another customer, {who}")
        if a.name == "exchange_delivered_order_items" and any(x == y for x, y in zip(g["item_ids"], g["new_item_ids"])):
            found.append("exchanges an item for the same item")
        if a.name in ("modify_pending_order_address", "modify_pending_order_payment") and g["order_id"] in modified:
            found.append(f"modifies {g['order_id']} after its items were modified")
        if a.name == "modify_pending_order_items":
            modified.add(g["order_id"])
        env.make_tool_call(tool_name=a.name, requestor=a.requestor, **g)
    return found


class RetailGenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = RetailDB.load(ROOT / TAU2_RETAIL_DB)
        cls.dump = cls.db.model_dump()
        cls.tasks = [Task.model_validate(json.loads(line)["task"]) for line in open(GENERATED)]
        cls.chosen = [Task.model_validate(json.loads(line)["task"]) for line in open(FINAL)]
        cls.specs = json.loads(REPORT.read_text())["specs"]
        cls.tags = {json.loads(line)["task"]["id"]: set(json.loads(line)["tags"]) for line in open(GENERATED)}

    def text(self, task):
        return str(task.user_scenario.instructions)

    def test_no_evaluation_customer(self):
        test = get_tasks("base")
        official = set()
        for t in test:
            s = json.dumps(t.model_dump(mode="json"))
            official |= {self.db.orders["#" + o].user_id for o in re.findall(r"W\d{7}", s) if "#" + o in self.db.orders}
            official |= {u.user_id for u in self.db.users.values() if u.user_id in s or u.email in s or (
                f"{u.name.first_name} {u.name.last_name}" in s and u.address.zip in s)}
        for t in self.tasks:
            for a in t.evaluation_criteria.actions:
                g = a.arguments
                who = self.db.orders[g["order_id"]].user_id if "order_id" in g else g["user_id"]
                self.assertNotIn(who, official, t.id)
            self.assertFalse(any(f"{self.db.users[u].name.first_name} {self.db.users[u].name.last_name}" in
                                 t.user_scenario.instructions.known_info and self.db.users[u].address.zip in
                                 t.user_scenario.instructions.known_info for u in official), t.id)

    def test_variant_groups_differ_only_in_their_condition(self):
        """The tasks of a variant group act on the same order and item."""
        groups = {}
        for line in open(GENERATED):
            r = json.loads(line)
            groups.setdefault(r["group"], []).append(r["task"]["id"])
        for g, ids in groups.items():
            if len(ids) > 1:
                parts = [self.specs[i][0] for i in ids]
                self.assertEqual(len({(p.get("order"), p.get("item")) for p in parts}), 1, g)

    def test_outcomes_are_new(self):
        taken = {outcome(t, "retail") for t in get_tasks("base")}
        mine = Counter(outcome(t, "retail") for t in self.tasks if outcome(t, "retail"))
        self.assertEqual(set(mine) & taken, set())
        self.assertEqual([o for o, n in mine.items() if n > 1], [])

    def test_requests_apart(self):
        """No request of a task changes what another one names by the database: the default address it ships to, the
        address of the order it takes an address from, or the item of the order it matches a variant to."""
        for t in self.tasks:
            parts = self.specs[t.id]
            if len(parts) < 2:
                continue
            user = next(p["user"] if "user" in p else self.db.orders[p.get("order") or p["orders"][0]].user_id
                        for p in parts if "user" in p or "order" in p or p.get("orders"))
            for i, p in enumerate(parts):
                reads = set()
                if p.get("source") == "default":
                    reads.add("default")
                if p.get("source") == "order":
                    reads |= {o for o in self.db.users[user].orders if (self.db.orders[o].address.address1,
                              self.db.orders[o].address.zip) == (p["address"]["address1"], p["address"]["zip"])}
                reads |= {s["other"] for s in p.get("items", []) if isinstance(s, dict) and "other" in s}
                fact = p.get("named") if isinstance(p.get("named"), dict) else {}
                if fact.get("by") == "price":  # the order it is compared with
                    reads.add(fact["value"]["rival"])
                for q in parts[:i] + parts[i + 1:]:
                    writes = {c["arguments"].get("order_id", "default") for c in q["calls"]}
                    self.assertEqual(reads & writes, set(), t.id)
                    if fact.get("by") == "city":  # no other order is shipped to the city that names this one
                        self.assertNotIn(fact["value"], {c["arguments"]["city"] for c in q["calls"]
                                                         if c["name"] == "modify_pending_order_address"}, t.id)
                    # nor does another request change items of a product, which the unsaid options name alone
                    mine = {self.item(c["arguments"]["order_id"], x).name for c in p["calls"]
                            if c["name"] in ("modify_pending_order_items", "exchange_delivered_order_items")
                            for x in c["arguments"]["item_ids"]}
                    theirs = {self.item(c["arguments"]["order_id"], x).name for c in q["calls"]
                              if c["name"] in ("modify_pending_order_items", "exchange_delivered_order_items")
                              for x in c["arguments"]["item_ids"]}
                    self.assertEqual(mine & theirs, set(), t.id)

    def test_info_unsaid(self):
        """No string the agent must state is in the scenario, which the user would say for it."""
        for t in self.tasks:
            text = self.text(t).lower()
            self.assertEqual([x for x in t.evaluation_criteria.communicate_info if x.lower() in text], [], t.id)

    def test_policy(self):
        for t in self.tasks[SHARD::SHARDS]:
            parts = self.specs[t.id]
            self.assertEqual([a.model_dump(include={"name", "arguments"}) for a in t.evaluation_criteria.actions],
                             [c for p in parts for c in p["calls"]], t.id)
            self.assertEqual(policy_violations(t, self.db), [], t.id)
            for k, p in enumerate(parts):
                if p["forbidden"] and p["by_policy"]:
                    bad = t.model_copy(deep=True)
                    calls = [c for i, q in enumerate(parts) for c in ([q["forbidden"]] if i == k else q["calls"])]
                    bad.evaluation_criteria.actions = [Action(action_id=f"x{i}", **c) for i, c in enumerate(calls)]
                    self.assertNotEqual(policy_violations(bad, self.db), [], t.id)

    def test_scores(self):
        for t in self.chosen[SHARD::SHARDS]:
            parts = self.specs[t.id]
            calls = [c for p in parts for c in p["calls"]]
            info = t.evaluation_criteria.communicate_info
            said = [AssistantMessage(role="assistant", content=f"That is {x}.") for x in info]
            self.assertEqual(reward(t, self.dump, replay(calls, self.dump) + said), 1.0, t.id)
            self.assertEqual(reward(t, self.dump, []), 0.0 if calls or info else 1.0, t.id)
            if info:
                self.assertEqual(reward(t, self.dump, replay(calls, self.dump)), 0.0, t.id)
                # nor does an agent state it by reading out the details of a write it confirms
                args = json.dumps([c["arguments"] for p in parts for c in p["calls"] + [p["forbidden"] or {}]
                                   if c])
                self.assertEqual([x for x in info if x in args], [], t.id)
            for k, p in enumerate(parts):
                if p["forbidden"]:
                    bad = [c for i, q in enumerate(parts) for c in ([p["forbidden"]] if i == k else q["calls"])]
                    self.assertEqual(reward(t, self.dump, replay(bad, self.dump) + said), 0.0, t.id)
            ends = [k for k, p in enumerate(parts) if p.get("transfer")]
            self.assertTrue(all(k == len(parts) - 1 for k in ends), t.id)
            writing = [p["calls"] for p in parts if p["calls"]]
            if len(writing) > 1:  # the requests do not depend on their order, and each is needed
                back = [c for w in reversed(writing) for c in w]
                self.assertEqual(reward(t, self.dump, replay(back, self.dump) + said), 1.0, t.id)
                for i in range(len(writing)):
                    rest = [c for j, w in enumerate(writing) if j != i for c in w]
                    self.assertEqual(reward(t, self.dump, replay(rest, self.dump) + said), 0.0, t.id)

    def test_specs(self):
        for t in self.tasks[SHARD::SHARDS]:
            text = self.text(t)
            for p in self.specs[t.id]:
                if p.get("dropped"):  # the user drops it when asked to confirm: the agent must not make its call
                    self.assertIn("change", text.lower())
                    self.assertFalse(p.get("held"), t.id)  # nor holds back part of it until that moment
                    p = {**p, "calls": [p["forbidden"]], "forbidden": None}
                with self.subTest(task=t.id, family=p["family"]):
                    self.named(text, p)
                    check = getattr(self, "spec_" + p["family"], None)
                    if check:
                        check(t, text, p)

    def test_total_ops(self):
        """A total asked over several requests is what the reference calls refund, or charge, all in one direction."""
        for t in self.tasks:
            if "total_ops" not in self.tags[t.id]:
                continue
            moved = [m for a in t.evaluation_criteria.actions or [] if (m := self.moved(a))]
            self.assertGreater(len(moved), 1, t.id)
            self.assertEqual(len({m > 0 for m in moved}), 1, t.id)
            self.assertIn(money(abs(sum(moved))), t.evaluation_criteria.communicate_info, t.id)

    def moved(self, a):
        """What a reference call refunds to the customer (positive) or charges (negative)."""
        g = a.arguments
        if a.name == "cancel_pending_order":
            return sum(i.price for i in self.db.orders[g["order_id"]].items)
        if a.name == "return_delivered_order_items":
            return sum(self.item(g["order_id"], x).price for x in g["item_ids"])
        if a.name in ("modify_pending_order_items", "exchange_delivered_order_items"):
            return -sum(self.db.products[self.item(g["order_id"], x).product_id].variants[n].price
                        - self.item(g["order_id"], x).price for x, n in zip(g["item_ids"], g["new_item_ids"]))
        return 0

    def test_fast_refund(self):
        """A guaranteed refund time is asked about a refund that takes days (not to a gift card, which is refunded at
        once), from a request that comes before it."""
        for t in self.tasks[SHARD::SHARDS]:
            parts = self.specs[t.id]
            k = next((k for k, p in enumerate(parts) if p["family"] == "deny_fast_refund"), None)
            if k is not None:
                calls = [c for p in parts[:k] for c in p["calls"] + ([p["forbidden"]] if p.get("dropped") else [])]
                self.assertTrue(any(self.slow_refund(c) for c in calls), t.id)

    # ---------------------------------------------------------------- helpers

    def slow_refund(self, c):
        a = c["arguments"]
        if "order_id" not in a:
            return False
        order = self.db.orders[a["order_id"]]
        if c["name"] == "cancel_pending_order":
            pid = order.payment_history[0].payment_method_id
        elif c["name"] == "return_delivered_order_items":
            pid = a["payment_method_id"]
        elif c["name"] in ("modify_pending_order_items", "exchange_delivered_order_items"):
            old = {i.item_id: i for i in order.items}
            if sum(self.db.products[old[x].product_id].variants[n].price - old[x].price
                   for x, n in zip(a["item_ids"], a["new_item_ids"])) >= 0:
                return False
            pid = a["payment_method_id"]
        else:
            return False
        return self.db.users[order.user_id].payment_methods[pid].source != "gift_card"

    def named(self, text, p):
        """The order is told apart from all the customer's other orders, cancelled ones included: by its id, by a fact
        of it that no other order shares (the city it went to, its number of items, or the higher or lower price paid
        for a product two orders have), or by products no other order has all of; or, with a word for its status, by
        products no other order of that status has all of (a cancelled order never arrives either)."""
        names = p.get("named")
        if not p.get("order") or names in (None, "id"):
            if names == "id":
                self.assertIn(p["order"][1:], text)
            return
        order = self.db.orders[p["order"]]
        others = [self.db.orders[o] for o in self.db.users[order.user_id].orders if o != order.order_id]
        if isinstance(names, dict):
            by, value, said = names["by"], names["value"], names["text"]
            self.assertIn(said.lower(), text.lower())
            if by == "city":
                self.assertIn(value, said)
                self.assertEqual(order.address.city, value)
                self.assertTrue(all(o.address.city != value for o in others))
            elif by == "count":
                words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                         10: "ten"}
                self.assertIn(f"{words[value]} items", said)
                self.assertEqual(len(order.items), value)
                self.assertTrue(all(len(o.items) != value for o in others))
            elif by == "recent":  # the last of the customer's orders
                self.assertIn("most recent", said)
                self.assertTrue(others)
                self.assertEqual(self.db.users[order.user_id].orders[-1], order.order_id)
            else:
                self.assertIn(" ".join(w if w.isupper() else w.lower() for w in value["product"].split()), said)
                self.assertRegex(said, "pricier|cost more" if value["more"] else "cheaper|cost less")
                holders = [o for o in others if any(i.name == value["product"] for i in o.items)]
                self.assertEqual([o.order_id for o in holders], [value["rival"]])
                mine = [i.price for i in order.items if i.name == value["product"]]
                theirs = [i.price for i in holders[0].items if i.name == value["product"]]
                self.assertEqual((len(mine), len(theirs)), (1, 1))
                self.assertGreaterEqual(abs(mine[0] - theirs[0]), 5)
                self.assertEqual(mine[0] > theirs[0], value["more"])
            return
        for n in names:
            self.assertIn(" ".join(w if w.isupper() else w.lower() for w in n.split()), text)
        if not any(set(names) <= {i.name for i in r.items} for r in others):
            return
        self.assertFalse(p["family"] in ("exchange_pending", "change_delivered", "return_pending", "cancel_processed",
                                         "cancel_delivered") or p.get("said"), names)  # named without the status
        received = {"delivered", "exchange requested", "return requested"}
        kind = received if order.status in received else {"pending", "processed", "pending (item modified)", "cancelled"}
        # the status is said in the sentence that names the order by these products
        product = " ".join(w if w.isupper() else w.lower() for w in names[0].split())
        self.assertTrue(any(re.search("received|delivered" if order.status in received else
                                      "pending|hasn't shipped|still waiting|hasn't arrived", s)
                            for s in re.split(r"(?<=[.;])\s", text) if product in s), names)
        rivals = [o for o in others if o.status in kind]
        self.assertFalse(any(set(names) <= {i.name for i in r.items} for r in rivals), names)

    def item(self, order_id, item_id):
        return next(i for i in self.db.orders[order_id].items if i.item_id == item_id)

    def payment_named(self, text, order, pid):
        user = self.db.users[order.user_id]
        m = user.payment_methods[pid]
        if len(user.payment_methods) == 1:  # the only method, left unsaid as in the official tasks
            return
        if pid == order.payment_history[0].payment_method_id and "original payment method" in text:
            return
        if m.source == "credit_card":
            self.assertIn(f"ending in {m.last_four}", text)
        else:
            self.assertIn("PayPal" if m.source == "paypal" else "gift card", text)

    def variant(self, text, order, s):
        """The description of the new variant fits it alone among the variants in stock."""
        old = self.item(order.order_id, s["item"])
        variants = self.db.products[old.product_id].variants
        new = variants[s["new"]]
        self.assertTrue(new.available)
        self.assertNotEqual(s["new"], old.item_id)
        changed = {k: v for k, v in new.options.items() if old.options[k] != v}
        kept = {k: v for k, v in old.options.items() if k not in changed}

        def stock(fixed):
            return [k for k, v in variants.items() if v.available and all(v.options[o] == x for o, x in fixed.items())]

        # the options a description leaves unsaid stay as they are (the official tasks' default, unsaid); told from its
        # twin only when asked, the item's distinguishing words are in that answer
        name = " ".join(w if w.isupper() else w.lower() for w in old.name.split())
        if s.get("told"):
            self.assertIn(f"which {name}", text)
            self.assertIn(s["told"], text)
        if s["kind"] == "relaxed":  # the options said fit it alone in stock; none in stock keeps the others as well
            for k, v in s["stated"].items():
                self.assertTrue(worded(text, k, v), v)
                self.assertEqual(new.options[k], v)
            self.assertEqual(stock({**old.options, **s["stated"]}), [])
            self.assertEqual([x for x in stock(s["stated"]) if x != old.item_id], [s["new"]])
        elif s["kind"] == "exact":
            for k, v in changed.items():
                self.assertTrue(worded(text, k, v), v)
            self.assertEqual(stock({**kept, **changed}), [s["new"]])
        elif s["kind"] == "pref":
            fits = [x for x in stock({**kept, **s["stated"]}) if x != old.item_id]
            first = next(v for v in s["order"] if any(variants[x].options[s["open"]] == v for x in fits))
            self.assertEqual([x for x in fits if variants[x].options[s["open"]] == first], [s["new"]])
            for v in s["order"]:
                self.assertTrue(worded(text, s["open"], v), v)
            if first != s["order"][0] and kept:  # out of stock only with the others kept, which is then said
                self.assertIn(f"everything else the same apart from the {s['open']}", text)
        elif s["kind"] == "relative_pref":  # one option a step up or down, one open with a preference, the others kept
            key, open_ = s["step"], s["open"]
            values = sorted({v.options[key] for v in variants.values()}, key=order_of)
            self.assertEqual(values.index(new.options[key]) - values.index(old.options[key]), 1 if s["up"] else -1)
            self.assertEqual({k: v for k, v in changed.items() if k not in (key, open_)}, s["stated"])
            self.assertRegex(text, UP if s["up"] else DOWN)
            # "one size bigger" says how far; "a brighter one" is any value on that side
            fits = [k for k in stock({**kept, **s["stated"]}) if k != old.item_id and (
                variants[k].options[key] == new.options[key] if key in ("size", "pieces") else
                variants[k].options[key] != old.options[key]
                and (order_of(variants[k].options[key]) > order_of(old.options[key])) == s["up"])]
            first = next(v for v in s["order"] if any(variants[x].options[open_] == v for x in fits))
            self.assertEqual([x for x in fits if variants[x].options[open_] == first], [s["new"]])
            self.assertGreater(len({variants[x].options[open_] for x in fits}), 1)  # the preference decides
            for v in s["order"]:
                self.assertTrue(worded(text, open_, v), v)
            if first != s["order"][0] and kept:
                self.assertIn(f"everything else the same apart from the {open_}", text)
        elif s["kind"] == "relative":
            key = s["step"]
            values = sorted({v.options[key] for v in variants.values()}, key=order_of)
            self.assertEqual(values.index(new.options[key]) - values.index(old.options[key]), 1 if s["up"] else -1)
            self.assertEqual({k: v for k, v in changed.items() if k != key}, s["stated"])
            self.assertRegex(text, UP if s["up"] else DOWN)
            # "a longer one" fits no other variant in stock; "one size bigger" says how far
            if key not in ("size", "pieces"):
                side = [k for k in stock({**kept, **s["stated"]}) if k != old.item_id
                        and variants[k].options[key] != old.options[key]
                        and (order_of(variants[k].options[key]) > order_of(old.options[key])) == s["up"]]
                self.assertEqual(side, [s["new"]])
        elif s["kind"] == "extreme":
            fits = stock(s["fixed"])
            key = (lambda x: variants[x].price) if s["by"] == "price" else (
                lambda x: order_of(variants[x].options[s["by"]]))
            best = (max if s["most"] else min)(key(x) for x in fits)
            self.assertEqual([x for x in fits if key(x) == best], [s["new"]])
            if s["by"] == "price":
                self.assertIn("most expensive" if s["most"] else "cheapest", text)
            elif not s["fixed"]:  # "the longest zoom" alone would keep the other options, as the official tasks read it
                self.assertIn("whatever its options", text)
        elif s["kind"] == "bound":
            fits = [x for x in stock(s["stated"]) if x != old.item_id]
            self.assertEqual([x for x in fits if variants[x].price <= old.price], [s["new"]])
            self.assertIn("no more than", text)
        else:  # the same item as in the customer's other order that has this product
            other = self.db.orders[s["other"]]
            self.assertEqual(other.user_id, order.user_id)
            self.assertEqual([i.item_id for i in other.items if i.name == old.name], [s["new"]])
            holders = [o for o in self.db.users[order.user_id].orders if o != order.order_id
                       and any(i.name == old.name for i in self.db.orders[o].items)]
            self.assertEqual(holders, [s["other"]])

    def reason(self, text, reason):
        self.assertRegex(text, r"by mistake|wrong button|without meaning to" if reason == "ordered by mistake" else
                         r"no longer need|don't need|somewhere else|won't be using")

    # ---------------------------------------------------------------- one check per family

    def spec_cancel(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "pending")
        self.reason(text, p["reason"])
        self.assertEqual(p["calls"], [{"name": "cancel_pending_order",
                                       "arguments": {"order_id": order.order_id, "reason": p["reason"]}}])

    def spec_cancel_partial(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "pending")
        whole = {"name": "cancel_pending_order", "arguments": {"order_id": order.order_id}}
        if p["fallback"] == "whole":
            self.assertEqual(p["calls"][0]["arguments"]["reason"], p["reason"])
            self.reason(text, p["reason"])
        elif p["fallback"] == "keep":
            self.assertEqual(p["calls"], [])
            self.assertEqual({**p["forbidden"], "arguments": {"order_id": p["forbidden"]["arguments"]["order_id"]}},
                             whole)
        else:  # the item changed to the cheapest variant in stock
            old = self.item(order.order_id, p["item"])
            stock = sorted((v.price, k) for k, v in self.db.products[old.product_id].variants.items() if v.available)
            self.assertEqual(stock[0][1], p["new"])
            self.assertLess(stock[0][0], stock[1][0])
            self.assertIn("cheapest", text)

    def spec_modify(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "pending")
        self.change(t, text, p, order, "modify_pending_order_items")

    def spec_exchange(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "delivered")
        self.change(t, text, p, order, "exchange_delivered_order_items")

    def change(self, t, text, p, order, tool):
        if p.get("said"):  # the customer's word is the other action than the order's status allows
            self.assertNotEqual(p["said"], "exchange" if tool.startswith("exchange") else "modify")
            self.assertRegex(text, "exchange" if p["said"] == "exchange" else "change|modify")
        for s in p["items"]:
            self.variant(text, order, s)
        for pid in re.findall(r"a number noted for the [^,]+, (\d{10})", text):  # a product ID taken for an item's
            self.assertIn(pid, {i.product_id for i in order.items if i.item_id in {s["item"] for s in p["items"]}}
                          | {i.product_id for o in self.db.users[order.user_id].orders for i in self.db.orders[o].items})
        first = self.item(order.order_id, p["items"][0]["item"])
        twins = [i for i in order.items if i.name == first.name and i is not first]
        # the sentence keeping or holding back the first item tells it apart
        if twins and (p["mind"] in ("keep", "keep_payment") or p["held"]):
            name = " ".join(w if w.isupper() else w.lower() for w in first.name.split())
            said = next(s for s in re.split(r"(?<=[.;])\s", text) if name in s
                        and re.search(r"mind: \w+ only|decide to \w+ just|mention only|Start with", s))
            self.assertTrue(any(v in said and all(i.options[k] != v for i in twins) for k, v in first.options.items())
                            or "expensive" in said, said)
        kept = p["items"][:1] if p["mind"] in ("keep", "keep_payment") else p["items"]
        call = {"name": tool, "arguments": {"order_id": order.order_id, "item_ids": [s["item"] for s in kept],
                                            "new_item_ids": [s["new"] for s in kept], "payment_method_id": p["payment"]}}
        if p["mind"] == "return":  # at confirmation the customer returns the items instead, to the original method
            self.assertEqual(p["forbidden"], call)
            self.assertEqual(p["calls"], [{"name": "return_delivered_order_items", "arguments": {
                "order_id": order.order_id, "item_ids": [s["item"] for s in p["items"]],
                "payment_method_id": order.payment_history[0].payment_method_id}}])
            self.assertIn("return", text)
        else:
            self.assertEqual(p["calls"], [call])
        if p["mind"] in ("payment", "keep_payment"):  # another method is named first and replaced at confirmation
            before = p["forbidden"]["arguments"]["payment_method_id"]
            self.assertNotEqual(before, p["payment"])
            asked = p["items"] if p["mind"] == "keep_payment" else kept  # all the items, before one is dropped
            self.assertEqual(p["forbidden"], {"name": tool, "arguments": {
                "order_id": order.order_id, "item_ids": [s["item"] for s in asked],
                "new_item_ids": [s["new"] for s in asked], "payment_method_id": before}})
            self.payment_named(text, order, before)
        # in the order's own order, as tau2's modify tool needs
        index = [i.item_id for i in order.items]
        self.assertEqual([s["item"] for s in kept], sorted((s["item"] for s in kept), key=index.index))
        self.payment_named(text, order, p["payment"])

    def spec_out_of_stock(self, t, text, p):
        order = self.db.orders[p["order"]]
        old = self.item(order.order_id, p["item"])
        want = self.db.products[old.product_id].variants[p["want"]]
        self.assertFalse(want.available)
        for k, v in want.options.items():
            if old.options[k] != v:
                self.assertTrue(worded(text, k, v), v)
        if p["fallback"] == "other":
            self.variant(text, order, p["then"])
            new = self.db.products[old.product_id].variants[p["new"]].options
            if p["then"]["kind"] == "exact" and any(old.options[k] == v for k, v in new.items()):
                # the options kept are the current item's, not those of the variant asked for first
                self.assertIn("other options as on the one in your order", text)
            if p["then"]["kind"] == "extreme" and not p["then"]["fixed"]:  # not among the options asked for first
                self.assertIn("whatever its options", text)
        elif p["fallback"] == "cancel":
            self.reason(text, p["reason"])

    def spec_return(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "delivered")
        user = self.db.users[order.user_id]
        names = Counter(i.name for i in order.items)
        items = [self.item(order.order_id, x) for x in p["items"]]
        if p["scope"] == "all":
            self.assertEqual(sorted(p["items"]), sorted(i.item_id for i in order.items))
        elif p["scope"] == "except":
            left = [i for i in order.items if i.item_id not in p["items"]]
            self.assertEqual(len(left), 1)
            self.assertEqual(names[left[0].name], 1)
            self.assertIn("except" if "except" in text else "but", text)
        elif p["scope"] == "twin":  # one of several items of a product, told apart in the scenario
            self.assertEqual(len(items), 1)
            self.assertGreater(names[items[0].name], 1)
        else:
            self.assertTrue(all(names[i.name] == 1 for i in items))
        if p["scope"] == "held":  # one more item, named at confirmation, goes into the same return
            self.assertIn(p["late"], p["items"])
            self.assertGreater(len(p["items"]), 1)
            late = " ".join(w if w.isupper() else w.lower() for w in self.item(order.order_id, p["late"]).name.split())
            self.assertRegex(text, rf"confirm[^.]*the {re.escape(late)}")
        if p.get("transfer"):
            self.assertEqual(p["calls"], [])
            self.assertIn("human", text)
            return
        if p.get("remaining"):  # what the customer paid for the items left in the order
            self.assertEqual(p["remaining"], money(sum(i.price for i in order.items if i.item_id not in p["items"])))
        if p.get("tracking"):  # the tracking number of the order returned from, asked along with the return
            self.assertEqual(p["tracking"], order.fulfillments[0].tracking_id[0])
            self.assertIn(p["tracking"], t.evaluation_criteria.communicate_info)
            self.assertIn("tracking number of that order", text)
        original = order.payment_history[0].payment_method_id
        self.assertIn(p["payment"], (original, next((k for k, m in user.payment_methods.items()
                                                     if m.source == "gift_card"), None)))
        if "asked" in p:
            self.assertNotIn(p["asked"], (original,))
            self.payment_named(text, order, p["asked"])
        self.payment_named(text, order, p["payment"])

    def spec_address(self, t, text, p):
        order = self.db.orders[p["order"]]
        user = self.db.users[order.user_id]
        a = p["address"]
        if p["source"] == "given":
            self.assertIn(a["address1"], text)
            self.assertIn(a["zip"], text)
        elif p["source"] == "default":
            self.assertEqual(a, user.address.model_dump())
            self.assertRegex(text, "default|profile")
        else:
            sources = [o for o in user.orders if self.db.orders[o].address.model_dump() == a]
            self.assertTrue(sources)
            self.assertTrue(any(o in text for o in sources) or a["city"] in text)
        self.assertNotEqual(a, order.address.model_dump())

    def spec_address_items(self, t, text, p):
        """The address first, then the items of the same order: once its items are modified, an order can no longer be
        changed."""
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "pending")
        home = [{"name": "modify_user_address", "arguments": {"user_id": order.user_id, **p["address"]}}] \
            if p["home"] else []  # the customer moved there and wants it as their default address too
        self.assertEqual([c["name"] for c in p["calls"]],
                         ["modify_pending_order_address"] + [c["name"] for c in home] + ["modify_pending_order_items"])
        self.assertEqual(p["calls"][1:-1], home)
        if p["home"]:
            self.assertRegex(text, "default address")
            self.assertNotEqual(p["address"], self.db.users[order.user_id].address.model_dump())
        self.spec_address(t, text, {**p, "calls": p["calls"][:1]})
        self.change(t, text, {**p, "calls": p["calls"][-1:]}, order, "modify_pending_order_items")

    def spec_user_address(self, t, text, p):
        user = self.db.users[p["user"]]
        a = p["address"]
        self.assertNotEqual(a, user.address.model_dump())
        if p["source"] == "given":
            self.assertIn(a["address1"], text)
        else:
            sources = [o for o in user.orders if self.db.orders[o].address.model_dump() == a]
            self.assertTrue(any(o in text for o in sources) or a["city"] in text)

    def spec_payment(self, t, text, p):
        order = self.db.orders[p["order"]]
        user = self.db.users[order.user_id]
        self.assertNotEqual(p["payment"], order.payment_history[0].payment_method_id)
        m = user.payment_methods[p["payment"]]
        if m.source == "gift_card":
            self.assertGreaterEqual(m.balance, order.payment_history[0].amount)
        if p.get("gift_first"):
            gift = next(x for x in user.payment_methods.values() if x.source == "gift_card")
            self.assertLess(gift.balance, order.payment_history[0].amount)

    def spec_exchange_pending(self, t, text, p):
        self.assertEqual(self.db.orders[p["order"]].status, "pending")
        self.assertIn("exchange", text)

    def spec_change_delivered(self, t, text, p):
        self.assertEqual(self.db.orders[p["order"]].status, "delivered")

    def spec_return_pending(self, t, text, p):
        self.assertEqual(self.db.orders[p["order"]].status, "pending")
        self.reason(text, p["reason"])

    def spec_return_across(self, t, text, p):
        """One item from each of two delivered orders, each product the only one of its name among the customer's
        orders, each returned to its own order's payment method."""
        user = self.db.orders[p["orders"][0]].user_id
        names = Counter(i.name for o in self.db.users[user].orders for i in self.db.orders[o].items)
        calls = []
        for o, x in zip(p["orders"], p["items"]):
            order = self.db.orders[o]
            self.assertEqual(order.status, "delivered")
            self.assertEqual(names[self.item(o, x).name], 1)
            self.assertIn(" ".join(w if w.isupper() else w.lower() for w in self.item(o, x).name.split()), text)
            calls.append({"name": "return_delivered_order_items", "arguments": {
                "order_id": o, "item_ids": [x], "payment_method_id": order.payment_history[0].payment_method_id}})
        self.assertEqual(len(set(p["orders"])), 2)
        self.assertEqual(p["calls"], calls)

    def spec_cancel_across(self, t, text, p):
        """One item from each of two pending orders of several items, each product the only one of its name among the
        customer's orders; single items cannot be removed, so both whole orders are cancelled for the reason given."""
        user = self.db.orders[p["orders"][0]].user_id
        names = Counter(i.name for o in self.db.users[user].orders for i in self.db.orders[o].items)
        for o, x in zip(p["orders"], p["items"]):
            order = self.db.orders[o]
            self.assertEqual(order.status, "pending")
            self.assertGreater(len(order.items), 1)
            self.assertEqual(names[self.item(o, x).name], 1)
            self.assertIn(" ".join(w if w.isupper() else w.lower() for w in self.item(o, x).name.split()), text)
        self.assertEqual(len(set(p["orders"])), 2)
        self.reason(text, p["reason"])
        self.assertEqual(p["calls"], [{"name": "cancel_pending_order",
                                       "arguments": {"order_id": o, "reason": p["reason"]}} for o in p["orders"]])

    def spec_deny_lost(self, t, text, p):
        """A delivered item the customer lost: no write, and returning it is what the agent must not do."""
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "delivered")
        self.assertEqual(p["calls"], [])
        self.assertEqual(p["forbidden"], {"name": "return_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [p["item"]],
            "payment_method_id": order.payment_history[0].payment_method_id}})

    def spec_cancel_delivered(self, t, text, p):
        """Items of a delivered order the customer calls cancelling, returned to the original payment method."""
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "delivered")
        self.assertIn("cancel", text)
        self.assertEqual(p["calls"], [{"name": "return_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": p["items"],
            "payment_method_id": order.payment_history[0].payment_method_id}}])

    def spec_cancel_processed(self, t, text, p):
        self.assertEqual(self.db.orders[p["order"]].status, "processed")
        self.assertEqual(p["calls"], [])

    def spec_same_item(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertEqual(p["forbidden"]["arguments"]["new_item_ids"], [p["item"]])
        if p["then"] == "variant":
            self.variant(text, order, p["then_spec"])

    def spec_one_action(self, t, text, p):
        self.variant(text, self.db.orders[p["order"]], p["exchange"])
        self.assertIn(f"prefer the {p['prefer']}", text.replace("rather do the", "prefer the"))
        self.assertEqual(p["calls"][0]["name"], "exchange_delivered_order_items" if p["prefer"] == "exchange" else
                         "return_delivered_order_items")

    def spec_price_condition(self, t, text, p):
        order = self.db.orders[p["order"]]
        old = self.item(order.order_id, p["item"])
        stock = sorted((v.price, k) for k, v in self.db.products[old.product_id].variants.items() if v.available)
        self.assertEqual(stock[0][1], p["new"])
        self.assertIn(f"${p['limit']}", text)
        self.assertEqual(p["below"], stock[0][0] < p["limit"])
        self.assertIn(money(stock[0][0]), t.evaluation_criteria.communicate_info)

    def spec_count_condition(self, t, text, p):
        """How many variants are in stock decides: at least the customer's number, an exchange for the cheapest one in
        stock, else a return, both with the original payment method; the number is the count or one more, of a product
        with ten or more variants."""
        order = self.db.orders[p["order"]]
        old = self.item(order.order_id, p["item"])
        self.assertGreaterEqual(len(self.db.products[old.product_id].variants), 10)
        stock = sorted((v.price, k) for k, v in self.db.products[old.product_id].variants.items() if v.available)
        self.assertEqual(stock[0][1], p["new"])
        self.assertLess(stock[0][0], stock[1][0])
        self.assertEqual(p["threshold"], len(stock) if p["enough"] else len(stock) + 1)
        self.assertRegex(text, rf"at least {p['threshold']}\b|\b{p['threshold']} or more")
        original = order.payment_history[0].payment_method_id
        exchange = {"name": "exchange_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [old.item_id], "new_item_ids": [p["new"]],
            "payment_method_id": original}}
        ret = {"name": "return_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [old.item_id], "payment_method_id": original}}
        self.assertEqual((p["calls"], p["forbidden"]), ([exchange], ret) if p["enough"] else ([ret], exchange))

    def spec_paid_condition(self, t, text, p):
        """What the customer paid for the item, read from the order, decides: over the limit, a return (a pending order
        is cancelled, as ordered by mistake); otherwise an exchange (a modification) for the cheapest variant in stock,
        with the original payment method."""
        order = self.db.orders[p["order"]]
        old = self.item(order.order_id, p["item"])
        stock = sorted((v.price, k) for k, v in self.db.products[old.product_id].variants.items() if v.available)
        self.assertEqual(stock[0][1], p["new"])
        self.assertLess(stock[0][0], stock[1][0])
        self.assertIn(f"${p['limit']}", text)
        self.assertEqual(p["over"], old.price > p["limit"])
        self.assertIn(money(old.price), t.evaluation_criteria.communicate_info)
        original = order.payment_history[0].payment_method_id
        if order.status == "pending":
            self.reason(text, "ordered by mistake")
            keep = {"name": "modify_pending_order_items", "arguments": {
                "order_id": order.order_id, "item_ids": [old.item_id], "new_item_ids": [p["new"]],
                "payment_method_id": original}}
            drop = {"name": "cancel_pending_order", "arguments": {"order_id": order.order_id,
                                                                  "reason": "ordered by mistake"}}
        else:
            keep = {"name": "exchange_delivered_order_items", "arguments": {
                "order_id": order.order_id, "item_ids": [old.item_id], "new_item_ids": [p["new"]],
                "payment_method_id": original}}
            drop = {"name": "return_delivered_order_items", "arguments": {
                "order_id": order.order_id, "item_ids": [old.item_id], "payment_method_id": original}}
        self.assertEqual((p["calls"], p["forbidden"]), ([drop], keep) if p["over"] else ([keep], drop))

    def spec_cancel_refund(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertNotEqual(self.db.users[order.user_id].payment_methods[order.payment_history[0].payment_method_id]
                            .source, "gift_card")
        self.assertEqual(bool(p["calls"]), p["then"] == "cancel")

    def spec_all_pending_cancel(self, t, text, p):
        user = self.db.users[p["user"]]
        self.assertEqual(sorted(p["orders"]), sorted(o for o in user.orders if "pending" in self.db.orders[o].status))
        self.assertEqual([c["arguments"]["order_id"] for c in p["calls"]], p["orders"])
        self.reason(text, p["reason"])

    def spec_all_pending_address(self, t, text, p):
        user = self.db.users[p["user"]]
        self.assertEqual(sorted(p["orders"]), sorted(o for o in user.orders if "pending" in self.db.orders[o].status))
        # every pending order, then the profile
        self.assertEqual([c["arguments"].get("order_id") for c in p["calls"]], p["orders"] + [None])
        self.assertEqual(p["calls"][-1]["arguments"]["user_id"], p["user"])
        for c in p["calls"]:
            self.assertEqual({k: v for k, v in c["arguments"].items() if k not in ("order_id", "user_id")}, p["address"])
        if p.get("source") == "order":  # the address another order of theirs went to, which the agent looks up
            sources = [o for o in user.orders if self.db.orders[o].address.model_dump() == p["address"]]
            self.assertTrue(sources)
            self.assertFalse(set(sources) & set(p["orders"]))  # no pending order is there already
            self.assertTrue(any(o in text for o in sources) or p["address"]["city"] in text)
            self.assertNotEqual(p["address"], user.address.model_dump())
        else:
            self.assertIn(p["address"]["address1"], text)

    def spec_all_delivered(self, t, text, p):
        user = self.db.users[p["user"]]
        delivered = [o for o in user.orders if self.db.orders[o].status == "delivered"]
        self.assertEqual(sorted(p["orders"]), sorted(delivered))
        expected = sorted(i.item_id for o in delivered for i in self.db.orders[o].items if i.name != p["keep"])
        gift = next((k for k, m in user.payment_methods.items() if m.source == "gift_card"), None)
        if p.get("fallback"):  # every refund first asked on a card that paid for none of the orders
            asked = user.payment_methods[p["asked"]]
            self.assertNotEqual(asked.source, "gift_card")
            self.assertFalse(any(self.db.orders[o].payment_history[0].payment_method_id == p["asked"]
                                 for o in delivered))
            self.payment_named(text, self.db.orders[delivered[0]], p["asked"])
        if p.get("fallback") == "human":
            self.assertEqual(p["calls"], [])
            self.assertIn("human", text)
            calls = [p["forbidden"]]
            self.assertEqual(calls[0]["arguments"]["payment_method_id"],
                             self.db.orders[calls[0]["arguments"]["order_id"]].payment_history[0].payment_method_id)
            return
        returned = sorted(x for c in p["calls"] for x in c["arguments"]["item_ids"])
        self.assertEqual(returned, expected)
        for c in p["calls"]:
            o = self.db.orders[c["arguments"]["order_id"]]
            self.assertEqual(c["arguments"]["payment_method_id"],
                             gift if p.get("fallback") == "gift" else o.payment_history[0].payment_method_id)

    def spec_tracking(self, t, text, p):
        order = self.db.orders[p["order"]]
        self.assertEqual(t.evaluation_criteria.communicate_info.count(order.fulfillments[0].tracking_id[0]), 1)

    def spec_total(self, t, text, p):
        order = self.db.orders[p["order"]]
        paid = money(sum(x.amount for x in order.payment_history if x.transaction_type == "payment"))
        self.assertIn(paid, t.evaluation_criteria.communicate_info)

    def spec_attribute(self, t, text, p):
        order = self.db.orders[p["order"]]
        value = self.item(order.order_id, p["item"]).options[p["key"]]
        self.assertIn(value, t.evaluation_criteria.communicate_info)
        self.assertNotRegex(value, RESPELLED)
        self.assertIn(p["key"], text)

    def spec_priciest(self, t, text, p):
        items = sorted(self.db.orders[p["order"]].items, key=lambda i: i.price)
        self.assertLess(items[-2].price, items[-1].price)
        self.assertIn(money(items[-1].price), t.evaluation_criteria.communicate_info)
        self.assertIn(items[-1].name, t.evaluation_criteria.communicate_info)
        self.assertNotRegex(items[-1].name, RESPELLED)

    def spec_scope_items(self, t, text, p):
        """Every order of that status with the product, one item of it each, gets the same change, one call per order
        and the difference to its own payment method; orders of another status with it stay as they are."""
        user = self.db.users[p["user"]]
        mine = [self.db.orders[o] for o in user.orders if self.db.orders[o].status != "cancelled"]
        holders = [o for o in mine if any(i.name == p["product"] for i in o.items)]
        scope = [o for o in holders if o.status == p["status"]]
        self.assertEqual(p["orders"], [o.order_id for o in scope])
        self.assertGreater(len(holders), 1)
        self.assertIn(p["status"] if p["status"] == "pending" else "received", text.replace("delivered", "received"))
        self.assertIn(" ".join(w if w.isupper() else w.lower() for w in p["product"].split()), text)
        calls, diffs = [], []
        kind = p["kind"]
        for o in scope:
            [item] = [i for i in o.items if i.name == p["product"]]
            variants = self.db.products[item.product_id].variants
            stock = [k for k, v in variants.items() if v.available]
            if kind == "exact":
                self.assertTrue(worded(text, p["key"], p["value"]), p["value"])
                fits = [k for k in stock if variants[k].options == {**item.options, p["key"]: p["value"]}]
            elif kind == "pref":  # one option set, one open with a preferred value, the others kept
                (key, value), (open_, prefer) = p["key"], p["value"]
                self.assertNotEqual(item.options[key], value)
                same = [k for k in stock if all(variants[k].options[o_] == (value if o_ == key else x)
                                                for o_, x in item.options.items() if o_ != open_)]
                fits = [k for k in same if variants[k].options[open_] == prefer]
                for k, v in ((key, value), (open_, prefer)):
                    self.assertTrue(worded(text, k, v), v)
                self.assertIn(f"several {open_} options", text)
            else:  # the most expensive or the cheapest in stock, maybe with an option of the item's own
                if kind.endswith("_kept"):
                    stock = [k for k in stock if variants[k].options[p["key"]] == item.options[p["key"]]]
                    self.assertIn(f"same {option_name(p['key'], p['product'])}", text)
                best = (max if kind.startswith("most") else min)(variants[k].price for k in stock)
                fits = [k for k in stock if variants[k].price == best]
                self.assertIn("most expensive" if kind.startswith("most") else "cheapest", text)
            self.assertEqual(len(fits), 1)
            self.assertNotEqual(fits[0], item.item_id)
            diffs.append(variants[fits[0]].price - item.price)
            calls.append({"name": "modify_pending_order_items" if p["status"] == "pending" else
                          "exchange_delivered_order_items", "arguments": {
                              "order_id": o.order_id, "item_ids": [item.item_id], "new_item_ids": fits,
                              "payment_method_id": o.payment_history[0].payment_method_id}})
        if p.get("paying"):  # the gift card if it covers every difference, the other method if it covers none
            user = self.db.users[p["user"]]
            gift = next(k for k, m in user.payment_methods.items() if m.source == "gift_card")
            balance = user.payment_methods[gift].balance
            self.assertGreater(min(diffs), 0)
            self.assertTrue(balance >= sum(diffs) or balance < min(diffs))
            pay = gift if balance >= sum(diffs) else p["paying"]
            calls = [{**c, "arguments": {**c["arguments"], "payment_method_id": pay}} for c in calls]
            self.assertIn("gift card", text)
            self.payment_named(text, scope[0], p["paying"])
        self.assertEqual(p["calls"], calls)

    def spec_shipped_to(self, t, text, p):
        """All of the address, which is not where the customer lives (whose city and zip code may be in the scenario)."""
        order = self.db.orders[p["order"]]
        a, home = order.address, self.db.users[order.user_id].address
        for x in (a.address1, a.address2, a.city, a.zip):
            if x:
                self.assertIn(x, t.evaluation_criteria.communicate_info)
        self.assertNotEqual(a.city, home.city)
        self.assertNotEqual(a.zip, home.zip)

    def spec_budget_chain(self, t, text, p):
        """Every item of the pending order to its cheapest variant in stock, in one call in the order's own order of
        items, if that brings the order within the budget, which it is over now; else cancel it or keep it."""
        order = self.db.orders[p["order"]]
        self.assertEqual(order.status, "pending")
        self.assertEqual(p["items"], [i.item_id for i in order.items])
        self.assertEqual(len({i.name for i in order.items}), len(order.items))
        new = []
        for i in order.items:
            stock = sorted((v.price, k) for k, v in self.db.products[i.product_id].variants.items() if v.available)
            self.assertLess(stock[0][0], stock[1][0])
            self.assertLess(stock[0][0], i.price)
            new.append(stock[0][1])
        self.assertEqual(p["new"], new)
        total = sum(i.price for i in order.items)
        after = sum(self.db.products[i.product_id].variants[n].price for i, n in zip(order.items, new))
        self.assertIn(f"${p['budget']} ", text.replace(".", " ").replace(",", " ").replace(";", " "))
        self.assertLessEqual(p["budget"], total - 10)
        self.assertGreaterEqual(abs(p["budget"] - after), 10)
        self.assertEqual(p["within"], p["budget"] >= after)
        self.assertIn("cheapest", text)
        original = order.payment_history[0].payment_method_id
        modify = {"name": "modify_pending_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": p["items"], "new_item_ids": new, "payment_method_id": original}}
        cancel = {"name": "cancel_pending_order", "arguments": {"order_id": order.order_id, "reason": p["reason"]}}
        if p["then"] == "cancel":
            self.reason(text, p["reason"])
        if p["within"]:
            self.assertEqual(p["calls"], [modify])
        else:
            self.assertEqual(p["calls"], [cancel] if p["then"] == "cancel" else [])
            self.assertEqual(p["forbidden"], modify)
        info = t.evaluation_criteria.communicate_info
        if p["first"] in ("remove", "both"):
            self.assertIn(money(max(i.price for i in order.items)), info)
            self.assertRegex(text, "most expensive|costs the most")
        if p["first"] in ("split", "both"):
            self.assertRegex(text, "split")
        if any(x == money(after) for x in info):
            self.assertRegex(text, "new total|cost after")

    def spec_option_chain(self, t, text, p):
        """The variant with the option the item lacks, everything else the same: changed to if in stock and within the
        limit; out of stock, one fallback; over the limit, the other."""
        order = self.db.orders[p["order"]]
        item = self.item(order.order_id, p["item"])
        variants = self.db.products[item.product_id].variants
        self.assertNotEqual(item.options[p["key"]], p["value"])
        self.assertEqual([k for k, v in variants.items() if v.options == {**item.options, p["key"]: p["value"]}],
                         [p["new"]])
        new = variants[p["new"]]
        case = "gone" if not new.available else "under" if new.price <= p["limit"] else "over"
        self.assertEqual(case, p["case"])
        if case != "gone":
            self.assertGreaterEqual(abs(new.price - p["limit"]), 10)
        self.assertIn(f"${p['limit']}", text)
        self.assertTrue(worded(text, p["key"], p["value"]), p["value"])
        pending = order.status == "pending"
        original = order.payment_history[0].payment_method_id
        change = {"name": "modify_pending_order_items" if pending else "exchange_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [item.item_id], "new_item_ids": [p["new"]],
            "payment_method_id": original}}
        back = {"name": "return_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [item.item_id], "payment_method_id": original}}
        self.assertEqual(p["gone_then"], "cancel" if pending else "return")
        gone = {"name": "cancel_pending_order", "arguments": {"order_id": order.order_id,
                                                              "reason": p["reasons"][0]}} if pending else back
        over = {"cancel": {"name": "cancel_pending_order", "arguments": {"order_id": order.order_id,
                                                                         "reason": p["reasons"][1]}},
                "return": back, "keep": None}[p["price_then"]]
        self.assertNotEqual(p["reasons"][0], p["reasons"][1])
        expected = {"under": [change], "over": [over] if over else [], "gone": [gone]}[case]
        self.assertEqual(p["calls"], expected)
        if p["forbidden"]:
            self.assertIn(p["forbidden"], [x for x in (change, gone, over) if x and [x] != expected])
        if pending:
            self.reason(text, p["reasons"][0])
            if p["price_then"] == "cancel":
                self.reason(text, p["reasons"][1])

    def spec_linked(self, t, text, p):
        """An exchange (another variant in stock, out of stock, or the very same item, which the policy does not allow);
        if it becomes a return, another order of the customer's goes too, else it stays."""
        order = self.db.orders[p["order"]]
        other = self.db.orders[p["orders"][0]]
        self.assertEqual(order.status, "delivered")
        self.assertEqual(other.user_id, order.user_id)
        self.assertNotEqual(other.order_id, order.order_id)
        item = self.item(order.order_id, p["item"])
        self.assertNotIn(item.name, {i.name for i in other.items})
        variants = self.db.products[item.product_id].variants
        if p["how"] == "same":
            self.assertEqual(p["new"], item.item_id)
            self.assertRegex(text, "exactly like it|identical")
        else:
            new = variants[p["new"]]
            changed = {k: v for k, v in new.options.items() if item.options[k] != v}
            self.assertEqual(len(changed), 1)
            self.assertEqual(new.available, p["how"] == "exchange")
            [(k, v)] = changed.items()
            self.assertTrue(worded(text, k, v), v)
        original = order.payment_history[0].payment_method_id
        exchange = {"name": "exchange_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [item.item_id], "new_item_ids": [p["new"]],
            "payment_method_id": original}}
        back = {"name": "return_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [item.item_id], "payment_method_id": original}}
        if p["then"] == "cancel":
            self.assertEqual(other.status, "pending")
            follow = {"name": "cancel_pending_order", "arguments": {"order_id": other.order_id, "reason": p["reason"]}}
            self.reason(text, p["reason"])
        else:
            self.assertEqual(other.status, "delivered")
            thing = self.item(other.order_id, p["thing"])
            self.assertEqual(sum(i.name == thing.name for i in other.items), 1)
            follow = {"name": "return_delivered_order_items", "arguments": {
                "order_id": other.order_id, "item_ids": [thing.item_id],
                "payment_method_id": other.payment_history[0].payment_method_id}}
        self.assertEqual(p["calls"], [exchange] if p["how"] == "exchange" else [back, follow])
        self.assertIn(p["forbidden"], (exchange, back, follow))
        self.assertNotIn([p["forbidden"]], [p["calls"]])

    def spec_bound_chain(self, t, text, p):
        """The variant with the named option changed and the others kept, in stock: changed to if it costs less than what
        was paid; if it costs more, the fallback (return, cancel, or keep). No price within $1 of the one paid."""
        order = self.db.orders[p["order"]]
        item = self.item(order.order_id, p["item"])
        variants = self.db.products[item.product_id].variants
        fits = [k for k, v in variants.items() if v.available and v.options == {**item.options, **p["options"]}]
        self.assertEqual(fits, [p["new"]])
        self.assertGreaterEqual(abs(variants[p["new"]].price - item.price), 1)
        self.assertEqual(p["found"], variants[p["new"]].price < item.price)
        for o, x in p["options"].items():
            self.assertNotEqual(item.options[o], x)
            self.assertTrue(worded(text, o, x), x)
        self.assertIn("no more than what you paid", text)
        self.assertRegex(text, "everything else the same|other options unchanged")
        pending = order.status == "pending"
        original = order.payment_history[0].payment_method_id
        drop = {"cancel": {"name": "cancel_pending_order", "arguments": {"order_id": order.order_id,
                                                                         "reason": p["reason"]}},
                "return": {"name": "return_delivered_order_items", "arguments": {
                    "order_id": order.order_id, "item_ids": [item.item_id], "payment_method_id": original}},
                "keep": None}[p["then"]]
        self.assertEqual(p["then"] == "cancel", pending and p["then"] != "keep")
        if p["found"]:
            self.assertEqual(p["calls"], [{"name": "modify_pending_order_items" if pending else
                                           "exchange_delivered_order_items", "arguments": {
                                               "order_id": order.order_id, "item_ids": [item.item_id],
                                               "new_item_ids": [p["new"]], "payment_method_id": original}}])
        else:
            self.assertEqual(p["calls"], [drop] if drop else [])
            if p["then"] == "cancel":
                self.reason(text, p["reason"])

    def spec_card(self, t, text, p):
        """The card that paid for the order, one of the customer's payment methods, by last digits nothing else of theirs
        shows."""
        order = self.db.orders[p["order"]]
        user = self.db.users[order.user_id]
        m = user.payment_methods[order.payment_history[0].payment_method_id]
        self.assertEqual(m.source, "credit_card")
        self.assertGreater(len(user.payment_methods), 1)
        self.assertIn(m.last_four, t.evaluation_criteria.communicate_info)
        rest = json.dumps([self.db.orders[o].model_dump(mode="json") for o in user.orders])
        self.assertNotIn(m.last_four, rest)

    def spec_cheapest(self, t, text, p):
        stock = sorted(v.price for v in self.db.products[p["product"]].variants.values() if v.available)
        self.assertLess(stock[0], stock[1])
        self.assertIn(money(stock[0]), t.evaluation_criteria.communicate_info)



if __name__ == "__main__":
    unittest.main()
