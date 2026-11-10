"""Harness RSI without a model: H0 against the policy, the plugin agent's load_skill loop on a scripted model, rule
edits, the editor's code checks, step grouping, the write comparison and the pipeline's log. A few seconds."""
import io
import json
import random
import re
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tau2.agent.base_agent import AgentError
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.registry import registry

from rsi.agent import h0, plugin
from rsi.agent.plugin import PluginAgent, apply, rules, skill_text
from rsi.benchmarks import evaluate
from rsi.benchmarks import run as run_module
from rsi.benchmarks.run import paired
from rsi.harness_rsi import editor, evidence
from rsi.harness_rsi.evolve import START, Attempt, controls, group, harmful, support_ids, undo
from rsi.pipeline import harness_rsi, plugin_gate, run_pipeline

PLUGINS = ["plugins/retail", "plugins/airline"]


def squeeze(s):
    return " ".join(s.split())


class H0Test(unittest.TestCase):
    def test_quotes_add_up_to_the_policy_and_match_the_committed_plugins(self):
        with tempfile.TemporaryDirectory() as tmp:
            for domain, spec in h0.DOMAINS.items():
                out = Path(tmp) / domain
                h0.write(domain, out)
                entry = (out / "agents" / f"{domain}.md").read_text()
                parts = [entry.split("## Policy\n\n", 1)[1].split("\n\n## Routing\n", 1)[0]]
                for skill, (section, *_) in spec["skills"].items():
                    if section:
                        body = (out / "skills" / skill / "SKILL.md").read_text().split("## Policy\n\n", 1)[1]
                        parts.append(f"## {section}\n{body}")
                self.assertEqual(squeeze("\n".join(parts)), squeeze(spec["policy"].read_text()))
                for path in out.rglob("*"):
                    if path.is_file():
                        self.assertEqual(path.read_bytes(), (plugin.ROOT / "plugins" / domain /
                                                             path.relative_to(out)).read_bytes(), path)


def reply(content=None, calls=()):
    return AssistantMessage(role="assistant", content=content, cost=0.0, raw_data={"choices": []},
                            usage={"completion_tokens": 1, "prompt_tokens": 2}, generation_time_seconds=0.1,
                            tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)]
                            or None)


class AgentTest(unittest.TestCase):
    def agent(self):
        return PluginAgent(tools=[], domain_policy="POLICY", llm="openai/m", llm_args={}, plugins=PLUGINS, max_loads=4)

    def test_system_prompt_holds_the_catalogs_and_no_policy(self):
        prompt = self.agent().system_prompt
        self.assertNotIn("POLICY", prompt)
        self.assertTrue(prompt.startswith(f"<instructions>\n{plugin.INSTRUCTION}\n</instructions>\n<skills>\n"))
        self.assertEqual(prompt.count("\n| `"), 11)

    def test_load_skill_is_answered_inside_the_agent(self):
        agent = self.agent()
        state = agent.get_init_state()
        script = iter([reply(calls=[("load_skill", {"skill_name": "retail"})]),
                       reply(calls=[("load_skill", {"skill_name": "retail:exchange"}), ("get_user_details", {})]),
                       reply("How can I help?")])
        with mock.patch.object(plugin, "generate", lambda **_: next(script)):
            out, state = agent.generate_next_message(UserMessage(role="user", content="exchange"), state)
        self.assertEqual(out.content, "How can I help?")
        self.assertEqual([x["name"] for x in out.raw_data["skill_loads"]], ["retail", "retail:exchange"])
        self.assertEqual(out.usage, {"completion_tokens": 3, "prompt_tokens": 6})
        tools = [m for m in state.messages if isinstance(m, ToolMessage)]
        self.assertEqual([t.content.split("\n")[0] for t in tools], ["[Skill: retail]", "[Skill: retail:exchange]"])
        self.assertEqual([c.name for m in state.messages if isinstance(m, AssistantMessage) and m.tool_calls
                          for c in m.tool_calls], ["load_skill", "load_skill"])
        self.assertTrue(skill_text(agent.docs, "nope").startswith("Error: Skill 'nope' not found."))

    def test_too_many_loads_end_the_conversation(self):
        agent = self.agent()
        with mock.patch.object(plugin, "generate", lambda **_: reply(calls=[("load_skill", {"skill_name": "retail"})])):
            with self.assertRaises(AgentError):
                agent.generate_next_message(UserMessage(role="user", content="hi"), agent.get_init_state())

    def test_an_empty_reply_ends_the_conversation_as_an_agent_error_only_when_asked(self):
        def answer(agent):
            with mock.patch.object(plugin, "generate", lambda **_: reply()):
                return agent.generate_next_message(UserMessage(role="user", content="hi"), agent.get_init_state())[0]

        # by default tau2 rejects it, and runs the conversation again
        with self.assertRaises(ValueError):
            answer(self.agent()).validate()
        out = answer(PluginAgent(tools=[], domain_policy="POLICY", llm="openai/m", llm_args={}, plugins=PLUGINS,
                                 max_loads=4, end_on_empty=True))
        out.validate()
        self.assertTrue(PluginAgent.is_stop(out))
        self.assertEqual(out.raw_data, {"choices": [], "skill_loads": [], "empty_reply": True})
        # scored as an agent error: 0 before any check of the task
        sim = SimpleNamespace(messages=[out], termination_reason=TerminationReason.AGENT_STOP)
        info = plugin.evaluate(simulation=sim, task=None, evaluation_type=None, solo_mode=False, domain="retail")
        self.assertEqual((sim.termination_reason, info.reward), (TerminationReason.AGENT_ERROR, 0.0))


class EditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(plugin.ROOT / "plugins/retail", self.tmp / "retail")
        self.dir = self.tmp / "retail"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_rule_numbers_are_never_reused(self):
        path = self.dir / "skills/exchange/SKILL.md"
        add = {"op": "ADD", "file": "retail:exchange", "text": "Ask first.\nThen act."}
        self.assertEqual(apply(self.dir, [add, add]), ["retail:exchange-1", "retail:exchange-2"])
        apply(self.dir, [{"op": "DELETE", "rule_id": "retail:exchange-2"}, add,
                         {"op": "REPLACE", "rule_id": "retail:exchange-1", "text": "Check the status."}])
        self.assertEqual(rules(path), [("retail:exchange-1", "Check the status."), ("retail:exchange-3", "Ask first. Then act.")])
        self.assertIn(plugin.RULES_HEADER, path.read_text())
        apply(self.dir, [{"op": "DELETE", "rule_id": i} for i in ("retail:exchange-1", "retail:exchange-3")])
        self.assertNotIn(plugin.RULES, path.read_text())
        sibling = self.tmp / "sibling"  # a version made from the same parent after a rejected one
        shutil.copytree(plugin.ROOT / "plugins/retail", sibling)
        self.assertEqual(apply(sibling, [add], plugin.manifest(self.dir)["last_rule"]), ["retail:exchange-4"])

    def test_reply_schema_allows_only_this_plugins_targets(self):
        t = editor.targets(self.dir)
        self.assertEqual((t["op"]["enum"], t["rule_id"]["enum"]), (["ADD"], [""]))
        self.assertIn("retail:exchange", t["file"]["enum"])
        apply(self.dir, [{"op": "ADD", "file": "retail", "text": "Authenticate first."}])
        t = editor.targets(self.dir)
        self.assertEqual((t["op"]["enum"], t["rule_id"]["enum"]), (editor.OPS, ["retail-1", ""]))
        self.assertEqual(editor.analysis_schema(self.dir)["properties"]["op"]["enum"], editor.OPS + ["NONE"])

    def test_code_checks(self):
        apply(self.dir, [{"op": "ADD", "file": "retail", "text": "Authenticate first."}])

        def problem(text="Confirm all items before the call.", op="ADD", file="retail:modify", rule_id="",
                    support=("t1",), min_support=1, unclear=("t1",)):
            e = {"op": op, "file": file, "rule_id": rule_id, "text": text, "support": list(support), "reason": ""}
            return editor.edit_problem(e, self.dir, {"t1", "t2"}, set(unclear), min_support,
                                       {"#W1234567", "credit_card_1"}, {50.0}, set())

        self.assertIsNone(problem())
        self.assertIsNone(problem(text="Each extra bag costs $50."))
        self.assertIsNone(problem(op="REPLACE", rule_id="retail-1", file=""))
        self.assertIsNone(problem(unclear=()))
        self.assertIn("wrong or unclear", problem(op="REPLACE", rule_id="retail-1", file="", unclear=()))
        self.assertIn("wrong or unclear", problem(op="DELETE", rule_id="retail-1", file="", text="", unclear=("t2",)))
        self.assertIn("database IDs", problem(text="Never touch #W1234567."))
        self.assertIn("database IDs", problem(text="Refund to credit_card_1."))
        self.assertIn("amounts", problem(text="Offer $75 at most."))
        self.assertIn("English", problem(text="先确认商品。"))
        self.assertIn("not a document", problem(file="airline:modify"))
        self.assertIn("not a rule", problem(op="DELETE", rule_id="retail:modify-9", file=""))
        self.assertIn("not analysed", problem(support=("t9",)))
        self.assertIn("fewer than 2", problem(min_support=2))
        def quoted(*quotes):
            return {"op": "ADD", "evidence": [{"turn": 3, "quote": q} for q in quotes]}

        failed = '[2] agent: Please confirm all items.\n[4] agent calls f({"item_ids": ["123"]}) for `#W1`'
        pack = failed + "\n#### A successful run of the same task\n[3] user: I also want the tablet changed."
        self.assertIsNone(editor.quote_problem(quoted("all  ITEMS", 'item_ids: ["123"]) for #W1'), pack, failed))
        self.assertIsNone(editor.quote_problem(quoted("Please confirm... f({\"item_ids\""), pack, failed))
        self.assertIsNone(editor.quote_problem(quoted("all items", "also want the tablet"), pack, failed))
        self.assertIn("from the failed runs", editor.quote_problem(quoted("also want the tablet"), pack, failed))
        self.assertIn("not in the evidence", editor.quote_problem(quoted("all items", "item_ids 124"), pack, failed))
        self.assertIn("not in the evidence", editor.quote_problem(quoted("Please confirm… nothing"), pack, failed))
        self.assertIn("not in the evidence", editor.quote_problem(quoted("``"), pack, failed))
        self.assertIn("no evidence", editor.quote_problem(quoted(), pack, failed))
        def v(problem, old="none", verdict="block"):
            return {"verdict": verdict, "problem": problem, "old_problem": old}

        self.assertTrue(editor.blocked([v("A"), v("E")], "ADD"))
        self.assertTrue(editor.blocked([v("A"), v("D")], "REPLACE"))
        self.assertFalse(editor.blocked([v("A"), v("none", verdict="pass")], "ADD"))
        self.assertFalse(editor.blocked([v("A"), v("none")], "ADD"))
        self.assertFalse(editor.blocked([v("A"), None], "ADD"))
        self.assertFalse(editor.blocked([v("A", old="A"), v("A")], "REPLACE"))
        self.assertTrue(editor.blocked([v("A", old="A"), v("A")], "ADD"))
        self.assertFalse(editor.blocked([], "ADD"))
        proposals = [{"id": "a01", "task_id": "x+y"}, {"id": "a02", "task_id": "z"}]
        self.assertEqual(support_ids(["y", "x", "a02", "w"], proposals), ["w", "x+y", "z"])

    def test_a_rewrite_is_dropped_when_two_of_three_reviews_find_something_right_lost(self):
        # ask, the, user, call kept of ask, the, user, then, call, x
        self.assertAlmostEqual(editor.kept_share("Ask the user, then call `x`.", "ask the user and call y"), 4 / 6)

        def r(right="new"):
            return {"cases": [{"part": "p", "case": "c", "right": "both"}, {"part": "q", "case": "d", "right": right}]}

        self.assertTrue(editor.kept_by_review([r(), r(right="neither"), {"cases": []}]))
        # one review of three finding a case only the old text gets right, or giving no reply, does not decide; two do
        self.assertTrue(editor.kept_by_review([r(), r(right="old"), r()]))
        self.assertTrue(editor.kept_by_review([r(), None, r()]))
        self.assertFalse(editor.kept_by_review([r(right="old"), r(), r(right="old")]))
        self.assertFalse(editor.kept_by_review([r(right="old"), None, r()]))
        self.assertFalse(editor.kept_by_review([]))

    def test_rewrites_of_most_of_a_rule_are_reviewed_with_where_the_rule_came_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            out = logs / "r2/harness_rsi/airline"
            # round 1 wrote airline:modify-1 for task t1; round 2's plugin holds it and one more rule
            old, other = "Before changing flights, ask the user for the new dates and the cabin.", "Ask about bags."
            shutil.copytree(plugin.ROOT / "plugins/airline", out / "plugins" / START)
            apply(out / "plugins" / START, [{"op": "ADD", "file": "airline:modify", "rule_id": "", "text": old},
                                            {"op": "ADD", "file": "airline:modify", "rule_id": "", "text": other}])
            first = logs / "r1/harness_rsi/airline/steps/s01"
            first.mkdir(parents=True)
            written = {"op": "ADD", "file": "airline:modify", "rule_id": "", "text": old, "support": ["t1"],
                       "reason": "r", "written": "airline:modify-1"}
            (first / "merge.json").write_text(json.dumps({"edits": [written]}))
            # t1's second analysis gave no valid reply
            (first / "analysis.jsonl").write_text(
                "".join(json.dumps(a) + "\n" for a in [{"id": "a01", "task_id": "t1", "output": {"op": "ADD"}},
                                                         {"id": "a02", "task_id": "t2", "output": {"op": "ADD"}},
                                                         {"id": "a03", "task_id": "t1", "output": None}]))
            attempt = Attempt.__new__(Attempt)
            attempt.out, attempt.domain, attempt.cfg, attempt.policy, attempt.tools = out, "airline", {}, "P", "T"
            attempt.ev = {"review_below": 0.8}
            rewrite = {"op": "REPLACE", "file": "airline:modify", "rule_id": "airline:modify-1", "support": ["t2"],
                       "text": "Change the flights in one call."}
            tweak = {"op": "REPLACE", "file": "airline:modify", "rule_id": "airline:modify-2",
                     "text": "Ask about bags first.", "support": ["t2"]}
            add = {"op": "ADD", "file": "airline:book", "rule_id": "", "text": "Ask first.", "support": ["t2"]}
            proposals = [{"task_id": "t2", "unit": {"packs": [{}]}}]
            seen = []

            def review(cfg, policy, tools, plugin_dir, edit, props, origin, runs, checks, seed, call_id, raw):
                seen.append((edit["rule_id"], origin, runs, call_id))
                return {"cases": [{"part": "p", "case": "c", "right": "new" if call_id.endswith(".1") else right}]}

            # reviews 2 and 3 of the rewrite find a case only the old text gets right, or do not
            for right, dropped in (("both", []), ("old", [rewrite])):
                seen.clear()
                with mock.patch.object(editor, "review", review), \
                        mock.patch.object(evidence, "text", lambda pack, contrast, level: "run of t2"):
                    out_ = attempt.review([(rewrite, ["c"]), (tweak, ["c"]), (add, ["c"])], proposals,
                                          out / "plugins" / START, 7, out)
                # only the rewrite keeping under 0.8 of its rule's words is reviewed, three times, with where its old
                # text came from: the edit that wrote it and the analysis of the task behind it
                self.assertEqual(sorted(s[3] for s in seen), ["review1.1", "review1.2", "review1.3"])
                self.assertEqual(seen[0][1], (written, [{"id": "a01", "task_id": "t1", "output": {"op": "ADD"}}]))
                self.assertEqual(seen[0][2], "run of t2")
                self.assertEqual([d["edit"] for d in out_], dropped)
                # a rewrite the reviews drop is not listed to later steps as blocked
                self.assertTrue(all("problems" not in d for d in out_))

    def test_a_reply_without_thinking_is_tried_again_and_never_used(self):
        cfg = {"editor": {"retries": 2, "model": "m", "temperature": 1.0, "top_p": 0.95, "max_tokens": 9, "top_k": 64},
               "server": {"api_base": "http://x"}}

        def answers(*replies):  # (thinking, content) per try
            client = mock.MagicMock()
            client.chat.completions.create.side_effect = [SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=c, model_extra={"reasoning_content": t}),
                                         finish_reason="stop")], usage=None) for t, c in replies]
            return mock.patch.object(editor, "OpenAI", return_value=client)

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            with answers((None, '{"a": 1}'), ("hm", '{"a": 2}')):
                self.assertEqual(editor.ask(cfg, "s", "u", {}, 7, "a01", raw), {"a": 2})
            with answers((None, '{"a": 1}'), ("hm", "not json"), (None, '{"a": 3}')):
                self.assertIsNone(editor.ask(cfg, "s", "u", {}, 7, "a01", raw))
            with answers((None, '{"a": 1}'), ("hm", "not json"), (None, "")):
                self.assertIsNone(editor.ask(cfg, "s", "u", {}, 7, "a01", raw))
            with answers(("hm", ""), (None, ""), (None, "{")):
                self.assertIsNone(editor.ask(cfg, "s", "u", {}, 7, "a01", raw))
            self.assertEqual([json.loads(line)["seed"] for line in open(raw)], [7, 8, 7, 8, 9, 7, 8, 9, 7, 8, 9])

    def test_only_blocked_edits_are_listed_to_later_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Attempt.__new__(Attempt)
            attempt.out = Path(tmp)
            add = {"op": "ADD", "file": "airline:book", "rule_id": "", "text": "Pick the cheapest."}
            kept = {"op": "ADD", "file": "airline:book", "rule_id": "", "text": "Ask first.", "written": "airline:book-1"}
            bad = {"op": "ADD", "file": "airline:book", "rule_id": "", "text": "Add bags.", "written": "airline:book-2"}
            records = {1: {"edits": [kept], "dropped_by_checks": [
                           {"edit": add, "reason": "check A: x | B: y", "problems": ["A", "B"]},
                           {"edit": kept, "reason": "quote not in the evidence"}]},
                       2: {"edits": [], "dropped_by_checks": []},
                       3: {"edits": [kept, bad], "dropped_by_checks": [],
                           "removed_after_retest": [{"edit": bad, "problems": ["H"], "tasks": []}]}}
            for s, record in records.items():
                (attempt.out / f"steps/s{s:02d}").mkdir(parents=True)
                (attempt.out / f"steps/s{s:02d}/merge.json").write_text(json.dumps(record))
            events = [{"event": "step", "step": 1}, {"event": "merge", "step": 1, "edits": 1, "candidate": "v01"},
                      {"event": "retest", "step": 1, "candidate": "v01", "runs": 8, "kept": False,
                       "successes": {"current": 5, "candidate": 4}},
                      {"event": "step", "step": 2}, {"event": "merge", "step": 2, "edits": 0},
                      {"event": "step", "step": 3}, {"event": "merge", "step": 3, "edits": 2, "candidate": "v02"},
                      {"event": "harm", "step": 3, "candidate": "v02", "removed": ["airline:book-2"], "tasks": {},
                       "version": "v03"},
                      {"event": "retest", "step": 3, "candidate": "v03", "runs": 8, "kept": True, "rerun": True,
                       "successes": {"current": 4, "candidate": 8},
                       "pruned": {"candidate": "v02", "successes": {"current": 4, "candidate": 6}}}]
            (attempt.out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            _, _, current, versions, buffer = attempt.state()
        self.assertEqual((current, versions), ("v03", 3))
        self.assertEqual(buffer, [{"step": 1, "blocked": [{"edit": add, "problems": ["A", "B"]}]},
                                  {"step": 3, "blocked": [{"edit": bad, "problems": ["H"]}]}])
        self.assertEqual(editor.blocked_text(buffer).splitlines(), [
            "Step 1:",
            "- ADD airline:book: Pick the cheapest. (blocked before testing: it makes the agent do for every user what "
            "one user asked for or claimed and decides for the user)",
            "Step 3:",
            "- ADD airline:book: Add bags. (taken out after testing: it made the agent fail: a task whose failed runs "
            "named it failed far more often than before)"])
        self.assertEqual(editor.blocked_text([]), "None.")

    def test_a_rule_named_in_a_tasks_failed_runs_is_harmful_only_after_a_large_fall(self):
        def sim(task, trial, ok, thinking="", load_thinking=""):
            raw = {"choices": [{"message": {"reasoning_content": thinking}}],
                   "skill_loads": [{"name": "airline:modify",
                                    "response": {"choices": [{"message": {"reasoning_content": load_thinking}}]}}]}
            msg = AssistantMessage(role="assistant", content="ok", raw_data=raw)
            return SimpleNamespace(task_id=task, trial=trial, reward_info=SimpleNamespace(reward=1.0 if ok else 0.0),
                                   messages=[msg])

        def results(*tasks):  # (task, successes, thinking of each failed run)
            out = []
            for task, ok, failed in tasks:
                out += [sim(task, i, True) for i in range(ok)]
                out += [sim(task, ok + i, False, *t) for i, t in enumerate(failed)]
            return SimpleNamespace(simulations=out)

        rule = "By [airline:modify-3], add the bags."
        current = results(("t1", 6, []), ("t2", 6, []), ("t3", 5, [("",)]), ("t4", 6, []), ("t5", 6, []))
        candidate = results(
            ("t1", 1, [(rule,), ("", "modify-3 says so"), (rule,), ("modify-30",), (rule,)]),  # falls 5, named in 4 of 5
            ("t2", 3, [(rule,)] * 3),                                                         # falls 3 only
            ("t3", 0, [(rule,)] * 2 + [("",)] * 4),                                           # named in 2 of 6
            ("t4", 0, [(rule,)] * 6),                                                         # excluded
            ("t5", 0, [(rule,)] + [("",)] * 5))                                               # named in 1 of 6
        self.assertEqual(harmful(current, candidate, ["airline:modify-3", "airline:book-1"], 4, {"t4"}),
                         {"airline:modify-3": [{"task": "t1", "current": 6, "candidate": 1, "failed": 5, "naming": 4},
                                               {"task": "t3", "current": 5, "candidate": 0, "failed": 6, "naming": 2}]})

    def test_a_harmful_rule_comes_out_of_its_step(self):
        def sims(task, ok, failed, thinking=""):
            raw = {"choices": [{"message": {"reasoning_content": thinking}}]}
            return [SimpleNamespace(task_id=task, trial=i, reward_info=SimpleNamespace(reward=float(i < ok)),
                                    messages=[AssistantMessage(role="assistant", content="ok", raw_data=raw)])
                    for i in range(ok + failed)]

        def step(candidate, pruned=None):
            out = self.tmp / f"run{len(list(self.tmp.iterdir()))}"
            shutil.copytree(plugin.ROOT / "plugins/airline", out / "plugins" / START)
            attempt = Attempt.__new__(Attempt)
            attempt.out, attempt.domain, attempt.round, attempt.attempt = out, "airline", 9, 0
            attempt.recs, attempt.tasks, attempt.other = {"t1": {}, "t2": {}, "t3": {}}, {}, "plugins/retail"
            attempt.start, attempt.model = "plugins/airline", "m"
            attempt.cfg = {"user": {"train": {}}}
            attempt.ev = {"trials": 6, "keep_margin": 4, "harm_drop": 4, "excluded": []}
            attempt.system = lambda version: [version]
            attempt.analyse = lambda *args: []
            edits = [{"op": "ADD", "file": "airline:modify", "rule_id": "", "text": "Add bags."},
                     {"op": "ADD", "file": "airline:book", "rule_id": "", "text": "Ask first."}]
            attempt.merge = lambda *args: ({"edits": edits, "dropped_by_checks": [], "summary": "s"}, [])
            current = SimpleNamespace(simulations=sims("t1", 6, 0) + sims("t2", 2, 4) + sims("t3", 0, 6))  # 8
            ran = {"current.json": current, "candidate.json": candidate, "pruned.json": pruned}
            buffer = []
            (out / "events.jsonl").write_text("")
            with mock.patch("rsi.harness_rsi.evolve.run", lambda *args: ran[args[-1].name]):
                result = attempt.step(1, [], ["t1", "t2", "t3"], START, 0, buffer)
            attempt.index()
            return result, json.loads((out / "plugins/versions.json").read_text()), buffer, out

        named = "Rule [airline:modify-1] says to add them."
        # t1 falls 6 -> 2 and its failed runs name the first rule, which comes out; the step gained 6 with it, so the
        # rest is kept as it is
        passed = SimpleNamespace(simulations=sims("t1", 2, 4, named) + sims("t2", 6, 0) + sims("t3", 6, 0))
        (version, n), index, buffer, out = step(passed)
        self.assertEqual((version, n, index["v02"]["without"], index["v02"]["rerun"], index["v01"]["kept"]),
                         ("v02", 2, ["airline:modify-1"], False, False))
        self.assertEqual([r for path in plugin.documents(out / "plugins/v02").values() for r in rules(path)],
                         [("airline:book-1", "Ask first.")])
        self.assertEqual(buffer[-1]["blocked"][0]["problems"], ["H"])
        # the same fall in a step that gained nothing with the rule: the rest is run again on its own and decides
        failed = SimpleNamespace(simulations=sims("t1", 2, 4, named) + sims("t2", 3, 3) + sims("t3", 3, 3))
        good = SimpleNamespace(simulations=sims("t1", 6, 0) + sims("t2", 6, 0) + sims("t3", 6, 0))
        (version, n), index, _, _ = step(failed, good)
        self.assertEqual((version, index["v02"]["rerun"], index["v02"]["successes"]["candidate"]), ("v02", True, 18))
        same = SimpleNamespace(simulations=sims("t1", 6, 0) + sims("t2", 2, 4) + sims("t3", 0, 6))
        (version, n), index, _, _ = step(failed, same)
        self.assertEqual((version, index["v02"]["kept"]), (START, False))

    def test_rules_of_a_step_are_taken_back_out(self):
        start, cand = self.tmp / "start", self.tmp / "cand"
        apply(self.dir, [{"op": "ADD", "file": "retail:modify", "text": "A."}])
        shutil.copytree(self.dir, start)
        shutil.copytree(self.dir, cand)
        edits = [{"op": "ADD", "file": "retail:modify", "text": "B."},
                 {"op": "REPLACE", "rule_id": "retail:modify-1", "text": "A2."}]
        for e, written in zip(edits, apply(cand, edits)):
            e["written"] = written
        apply(cand, undo(edits, start))
        self.assertEqual(rules(cand / "skills/modify/SKILL.md"), [("retail:modify-1", "A.")])


class VersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_versions_count_accepted_gates_per_domain(self):
        def gate(r, d, accepted, plugin_dir):
            self.write(self.tmp / f"r{r}/gate/plugin_{d}/decision.json", {"accepted": accepted, "saved": plugin_dir})
        gate(1, "retail", True, "R1")
        gate(1, "airline", False, "A?")
        gate(2, "airline_retry", True, "A1")
        self.assertEqual(evaluate.plugins("retail", 0, self.tmp), ["plugins/retail", "plugins/airline"])
        self.assertEqual(evaluate.plugins("retail", 1, self.tmp), ["R1", "plugins/airline"])
        self.assertEqual(evaluate.plugins("airline", 1, self.tmp), ["R1", "A1"])
        with self.assertRaises(SystemExit):
            evaluate.plugins("retail", 2, self.tmp)
        # M<j>: the adapter of the domain's j-th accepted model gate
        self.write(self.tmp / "r1/gate/model_airline/decision.json", {"accepted": None, "saved": None})
        self.write(self.tmp / "r2/gate/model_airline/decision.json", {"accepted": "epoch2", "saved": "C2"})
        self.assertEqual(evaluate.model_of("airline", 0, self.tmp), "gemma-4-12b")
        self.assertEqual(evaluate.model_of("airline", 1, self.tmp), "C2")
        with self.assertRaises(SystemExit):
            evaluate.model_of("retail", 1, self.tmp)


class GroupingTest(unittest.TestCase):
    def test_steps_are_capped_per_category(self):
        # failure counts of the size H0 leaves in a training batch
        for counts, n in [({"modify": 32, "exchange": 16, "cancel": 9, "user-address": 9, "return": 9}, 5),
                          ({"modify": 77, "book": 42, "compensation": 15, "cancel": 9}, 10)]:
            failing = {c: [f"{c}/{i}" for i in range(k)] for c, k in counts.items()}
            for seed in range(20):
                steps = group(failing, 16, 8, random.Random(seed))
                self.assertEqual(len(steps), n)
                self.assertEqual(sorted(t for s in steps for t in s), sorted(t for v in failing.values() for t in v))
                self.assertLessEqual(max(map(len, steps)) - min(map(len, steps)), 8)
                for s in steps:
                    self.assertLessEqual(max(sum(t.split("/")[0] == c for t in s) for c in counts), 8)

    def test_controls_restart_once_used_up(self):
        good = [f"g{i}" for i in range(127)]  # airline B1: about 127 all-correct tasks for 10 steps
        steps = controls(good, 10, 16)
        self.assertTrue(all(len(s) == len(set(s)) == 16 for s in steps))
        self.assertEqual(set(t for s in steps[:7] for t in s), set(good[:112]))
        self.assertEqual(controls(good[:5], 2, 16), [good[:5], good[:5]])
        self.assertEqual(controls([], 3, 16), [[], [], []])


class GateTest(unittest.TestCase):
    def test_paired_pass_k(self):
        def results(wins):
            return SimpleNamespace(simulations=[SimpleNamespace(task_id=t, trial=i, reward_info=SimpleNamespace(
                reward=float(w)) if w is not None else None) for t, ws in wins.items() for i, w in enumerate(ws)])
        current = results({"a": [1, 1, 1, 0], "b": [1, 1, 1, 1], "c": [0, 0, 0, None]})
        candidate = results({"a": [1, 1, 1, 1], "b": [1, 1, 1, 1], "c": [1, 0, 0, 0]})
        p1, p4 = paired(current, candidate, 1, 100), paired(current, candidate, 4, 100)
        self.assertAlmostEqual(p1["diff"], (0.25 + 0.25) / 3)
        self.assertAlmostEqual(p4["current"], 1 / 3)
        self.assertAlmostEqual(p4["diff"], 1 / 3)
        self.assertLessEqual(p4["ci95"][0], p4["diff"])
        mix = paired(current, candidate, {1: 0.8, 4: 0.2}, 100)
        self.assertAlmostEqual(mix["diff"], 0.8 * p1["diff"] + 0.2 * p4["diff"])
        self.assertAlmostEqual(mix["current"], 0.8 * p1["current"] + 0.2 * p4["current"])

    def test_the_gate_takes_a_higher_score(self):
        # V: 200 tasks x 8 trials, so pass^1 moves in steps of 1/1600; a tie off by float error is no rise
        cases = {1 / 1600: True, 8 / 1600: True, 0.0: False, 1e-12: False, -1 / 1600: False}
        for d, expected in cases.items():
            self.assertEqual(plugin_gate.accepts(d), expected, d)
        self.assertEqual(plugin_gate.score(0.2), {1: 0.8, 4: 0.2})

    def test_h_k_is_the_plugin_a_gate_accepted_else_the_start(self):
        accepted, rejected = {"accepted": True, "saved": "S"}, {"accepted": False, "saved": "X"}
        cases = [(None, None, "plugins/retail"), (accepted, None, "S"), (rejected, accepted, "S"),
                 (rejected, rejected, "plugins/retail"), (rejected, None, "plugins/retail")]
        for first, retry, expected in cases:
            with mock.patch.object(harness_rsi, "evolve"), mock.patch.object(
                    harness_rsi, "gate", side_effect=lambda domain, round_, again: retry if again else first):
                self.assertEqual(harness_rsi.harness_rsi("retail", 1), expected, (first, retry))

    def test_pipeline_log_lines_start_with_the_time(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stdout", new=io.StringIO()) as term:
            log = Path(tmp) / "pipeline.log"
            tee = run_pipeline.Tee(log)
            print("first", file=tee)
            tee.write("second, written ")
            tee.write("in two parts\nthird\n")
            tee.log.close()
            lines = log.read_text().splitlines()
        self.assertEqual([line[20:] for line in lines], ["first", "second, written in two parts", "third"])
        self.assertTrue(all(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ", line[:20]) for line in lines))
        self.assertEqual(term.getvalue(), "first\nsecond, written in two parts\nthird\n")


class RegistryTest(unittest.TestCase):
    def test_runs_side_by_side_register_a_db_snapshot_once(self):
        # GRPO's rollouts and the model gate start runs in threads; with a DB load as slow as a real one, each of them
        # finds the snapshot unregistered, and tau2 refuses to register a name twice
        name = "retail-snapshot_for_test"
        self.addCleanup(registry._domains.pop, name, None)

        def load(path):
            time.sleep(0.2)
            return SimpleNamespace(model_copy=lambda deep: None)

        with mock.patch.dict(run_module.DB, {"retail": (SimpleNamespace(load=load), mock.Mock())}), \
                ThreadPoolExecutor(4) as pool:
            names = list(pool.map(lambda _: run_module.env_domain("retail", "data/snapshot_for_test.json"), range(4)))
        self.assertEqual(names, [name] * 4)


class DiffTest(unittest.TestCase):
    def test_labels(self):
        gold = [SimpleNamespace(name="cancel_pending_order", arguments={"order_id": "#A", "reason": "x"}),
                SimpleNamespace(name="modify_user_address", arguments={"user_id": "u", "city": "B"}),
                SimpleNamespace(name="return_delivered_order_items", arguments={"order_id": "#C"})]
        task = SimpleNamespace(evaluation_criteria=SimpleNamespace(actions=gold))
        calls = [("cancel_pending_order", {"order_id": "#A", "reason": "y"}),
                 ("return_delivered_order_items", {"order_id": "#D"}),
                 ("exchange_delivered_order_items", {"order_id": "#E"})]
        messages = []
        for i, (name, args) in enumerate(calls):
            messages += [AssistantMessage(role="assistant", turn_idx=2 * i,
                                          tool_calls=[ToolCall(id=str(i), name=name, arguments=args)]),
                         ToolMessage(id=str(i), role="tool", content="{}", turn_idx=2 * i + 1)]
        d = evidence.diff(task, SimpleNamespace(messages=messages))
        self.assertEqual([(x["turn"], x["label"], x["tool"]) for x in d],
                         [(0, "wrong arguments", "cancel_pending_order"), (2, "wrong target", "return_delivered_order_items"),
                          (4, "extra", "exchange_delivered_order_items"), (None, "missing", "modify_user_address")])
        self.assertEqual(evidence.category(task, SimpleNamespace(messages=messages)), "cancel")

    def test_arguments_compare_as_the_tool_reads_them(self):
        flight = {"flight_number": "HAT001", "date": "2024-05-20"}
        gold = [SimpleNamespace(name="book_reservation", arguments={
            "user_id": "u", "origin": "A", "destination": "B", "flight_type": "one_way", "cabin": "economy",
            "flights": [flight], "passengers": [{"first_name": "a", "last_name": "b", "dob": "1990-01-01"}],
            "payment_methods": [{"payment_id": "credit_card_1", "amount": 560}], "total_baggages": 1,
            "nonfree_baggages": 0, "insurance": "no"})]
        task = SimpleNamespace(evaluation_criteria=SimpleNamespace(actions=gold))

        def call(**change):
            args = {**gold[0].arguments, "flights": [{**flight, "origin": "A", "price": "116"}],
                    "payment_methods": [{"amount": "560", "payment_id": "credit_card_1"}], **change}
            return SimpleNamespace(messages=[
                AssistantMessage(role="assistant", turn_idx=0, tool_calls=[ToolCall(id="0", name="book_reservation",
                                                                                     arguments=args)]),
                ToolMessage(id="0", role="tool", content="{}", turn_idx=1)])

        self.assertEqual(evidence.diff(task, call()), [])
        d = evidence.diff(task, call(total_baggages=2))
        self.assertEqual([(x["label"], x["agent"]["total_baggages"], x["agent"]["flights"]) for x in d],
                         [("wrong arguments", 2, [flight])])


if __name__ == "__main__":
    unittest.main()
