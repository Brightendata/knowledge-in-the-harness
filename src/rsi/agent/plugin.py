"""Plugins in Knora-Claw's format, and tau2's LLMAgent with plugins installed.

A plugin directory:
    plugin.json              name, description; rule counters once rules exist
    agents/<name>.md         entry document, registered as <name>
    skills/<skill>/SKILL.md  one per category, registered as <name>:<skill>
A document may end with a "## Rules" section of numbered rules, the only part plugin evolution edits.
"""
import json
import re
from pathlib import Path

import tau2.runner.simulation
from loguru import logger
from tau2.agent.base_agent import AgentError
from tau2.agent.llm_agent import AGENT_INSTRUCTION, LLMAgent
from tau2.data_model.message import MultiToolMessage, ToolMessage
from tau2.data_model.simulation import TerminationReason
from tau2.environment.tool import as_tool
from tau2.evaluator.evaluator import evaluate_simulation
from tau2.registry import registry
from tau2.utils.llm_utils import generate

ROOT = Path(__file__).resolve().parents[3]
RULES = "## Rules"
RULES_HEADER = "These rules add detail to the policy. If a rule ever conflicts with the policy, follow the policy."
RULE_LINE = re.compile(r"^- \*\*\[(?P<id>[^\]]+)\]\*\* (?P<text>.+)$", re.M)
# tau2's stop token for agents (LLMSoloAgent); tau2's LLMAgent itself never stops a conversation
STOP = "###STOP###"
REMINDER = ("[IMPORTANT] You MUST strictly follow the workflow described above. Do not improvise or use a different "
            "approach unless this skill explicitly allows it.")
# tau2's agent instruction, pointing to the plugins instead of a policy in the prompt: the plugins hold the policy
POLICY_IN_PROMPT = "according to the <policy> provided below."
assert POLICY_IN_PROMPT in AGENT_INSTRUCTION
INSTRUCTION = AGENT_INSTRUCTION.replace(
    POLICY_IN_PROMPT, "according to the policy of the domain you serve. The policy is in the skills below: load the "
    "domain's entry skill first, and load the skill for each kind of request as the entry says.")
# Knora-Claw's skill listing in English, without its rule on reloading skills after context compression, plus the
# rule that a plugin's entry is loaded first.
CATALOG = """## Installed skills

Skills are collections of instructions, scripts and resources organized by domain. Use `load_skill` to get the full instructions.

**Available skills**:

| Name | Description |
| --- | --- |
{rows}

**Usage rules**:
- Scan the available skills on every turn. Call `load_skill(skill_name='...')` only when a skill clearly applies; if none matches, load none.
- If several skills apply, pick the most specific one, and load one skill at a time.
- Load a plugin's entry (the name without a colon) first; the entry says which of its skills to load next.
- After loading a skill, strictly follow its instructions; its workflow takes precedence over your default behavior.

Every skill has a SKILL.md with its full instructions; read it with `load_skill`."""


def frontmatter(text):
    """(fields, body) of a markdown document with a '---' frontmatter."""
    if not text.startswith("---\n"):
        return {}, text
    head, body = text[4:].split("\n---\n", 1)
    return dict(line.split(": ", 1) for line in head.splitlines() if ": " in line), body.lstrip("\n")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def manifest(plugin_dir):
    return json.loads((resolve(plugin_dir) / "plugin.json").read_text())


def documents(plugin_dir):
    """Registered name -> path of every document of a plugin, the entry first."""
    plugin_dir = resolve(plugin_dir)
    name = manifest(plugin_dir)["name"]
    docs = {name: plugin_dir / "agents" / f"{name}.md"}
    for path in sorted(plugin_dir.glob("skills/*/SKILL.md")):
        docs[f"{name}:{frontmatter(path.read_text())[0]['name']}"] = path
    return docs


def description(plugin_dir, name):
    """Catalog description: the manifest's for the entry, the frontmatter's for a skill."""
    if ":" not in name:
        return manifest(plugin_dir)["description"]
    return frontmatter(documents(plugin_dir)[name].read_text())[0].get("description", "")


def catalog(plugin_dirs):
    """The skill catalog of the system prompt, plugins in the order given."""
    rows = []
    for plugin_dir in plugin_dirs:
        for name in documents(plugin_dir):
            text = description(plugin_dir, name).replace("|", "\\|").replace("\n", " ").strip()
            rows.append(f"| `{name}` | {text[:107] + '...' if len(text) > 110 else text or '_no description_'} |")
    return CATALOG.format(rows="\n".join(rows))


def skill_text(docs, name):
    """What load_skill returns: knora-claw's format, without the skill directory hint."""
    if name not in docs:
        return f"Error: Skill '{name}' not found."
    return f"[Skill: {name}]\n{frontmatter(docs[name].read_text())[1].strip()}\n\n{REMINDER}"


def rules(path):
    """(id, text) of the numbered rules of a document, in order."""
    text = Path(path).read_text()
    at = text.find(f"\n{RULES}\n")
    return [] if at < 0 else [(m["id"], m["text"]) for m in RULE_LINE.finditer(text[at:])]


def write_rules(path, numbered):
    """Replace a document's Rules section; no section when there are no rules."""
    path = Path(path)
    text = path.read_text()
    at = text.find(f"\n{RULES}\n")
    text = (text if at < 0 else text[:at]).rstrip("\n") + "\n"
    if numbered:
        text += f"\n{RULES}\n\n{RULES_HEADER}\n\n" + "".join(f"- **[{i}]** {t}\n" for i, t in numbered)
    path.write_text(text)


def owner(rule_id):
    """Registered name of the document a rule id belongs to: 'retail:exchange-3' -> 'retail:exchange'."""
    return rule_id.rsplit("-", 1)[0]


def apply(plugin_dir, edits, numbered=None):
    """Apply ADD / REPLACE / DELETE edits to a plugin's rules in place; returns the id each edit wrote or removed.
    Rule numbers are never reused: the last number of each document is kept in plugin.json. numbered: last numbers
    already used elsewhere, such as by a sibling version, that new rules count on from."""
    plugin_dir = resolve(plugin_dir)
    docs = documents(plugin_dir)
    meta = manifest(plugin_dir)
    last = meta.setdefault("last_rule", {})
    for name, n in (numbered or {}).items():
        last[name] = max(last.get(name, 0), n)
    ids = []
    for e in edits:
        text = " ".join(e.get("text", "").split())
        if e["op"] == "ADD":
            last[e["file"]] = last.get(e["file"], 0) + 1
            ids.append(f"{e['file']}-{last[e['file']]}")
            write_rules(docs[e["file"]], rules(docs[e["file"]]) + [(ids[-1], text)])
        else:
            ids.append(e["rule_id"])
            path = docs[owner(e["rule_id"])]
            write_rules(path, [(i, text if i == e["rule_id"] else t) for i, t in rules(path)
                               if not (e["op"] == "DELETE" and i == e["rule_id"])])
    (plugin_dir / "plugin.json").write_text(json.dumps(meta, indent=2) + "\n")
    return ids


def load_skill(skill_name: str) -> str:
    """Load the full content of a skill by name.

    Use this when you determine that a user's request matches a skill's description and you need the full
    instructions to follow.

    IMPORTANT: After loading, you MUST strictly follow the skill's SKILL.md instructions. Do not improvise or use a
    different approach unless the skill explicitly allows it. The skill's workflow takes precedence over your default
    behavior.

    Only call this after checking Available Skills and deciding a skill is relevant based on its name and description.

    Args:
        skill_name: Name of the skill, as listed in Available Skills.
    """
    raise NotImplementedError("PluginAgent answers load_skill itself")


LOAD_SKILL = as_tool(load_skill)


class PluginAgent(LLMAgent):
    """tau2's LLMAgent with plugins: the system prompt holds the agent instruction and the skill catalog but no policy,
    which the agent reads by loading the plugins; the agent answers load_skill itself, calling the model again until it
    replies or calls an environment tool. The orchestrator, the user simulator and the DB replay never see these calls;
    they are kept, in order, in raw_data["skill_loads"] of the message returned, whose cost and usage add up all its
    model calls.

    A reply with neither text nor a tool call is one tau2 rejects, running the whole conversation again. With
    end_on_empty (GRPO's rollouts) it ends the conversation instead, as the agent error it is, scored 0 like every
    conversation that ends early: the agent stops with tau2's stop token in its place, the reply's raw_data marked
    "empty_reply", so that the trajectory keeps the call."""

    @classmethod
    def is_stop(cls, message):
        return bool((message.raw_data or {}).get("empty_reply"))

    def __init__(self, tools, domain_policy, llm, llm_args, plugins, max_loads, end_on_empty=False):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        # sorted by plugin name, so the system prompt does not depend on the order or the version paths given
        self.plugins = sorted((resolve(p) for p in plugins), key=lambda p: manifest(p)["name"])
        self.docs = {name: path for p in self.plugins for name, path in documents(p).items()}
        self.max_loads = max_loads
        self.end_on_empty = end_on_empty

    @property
    def system_prompt(self):
        return f"<instructions>\n{INSTRUCTION}\n</instructions>\n<skills>\n{catalog(self.plugins)}\n</skills>"

    def _generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        calls, loads = [], []
        while True:
            reply = generate(model=self.llm, tools=self.tools + [LOAD_SKILL], messages=state.system_messages + state.messages,
                             call_name="agent_response", **self.llm_args)
            calls.append(reply)
            load = next((c for c in reply.tool_calls or [] if c.name == "load_skill"), None)
            if load is None:
                break
            if len(loads) == self.max_loads:
                raise AgentError(f"more than {self.max_loads} load_skill calls before one reply")
            if len(reply.tool_calls) > 1:
                logger.warning(f"dropping the tool calls made together with load_skill: {reply.tool_calls}")
            name = str(load.arguments.get("skill_name", ""))
            loads.append({"name": name, "response": reply.raw_data})
            state.messages.append(reply.model_copy(update={"tool_calls": [load]}))
            state.messages.append(ToolMessage(id=load.id, role="tool", content=skill_text(self.docs, name)))
        reply.raw_data = {**(reply.raw_data or {}), "skill_loads": loads}
        reply.cost = sum(c.cost or 0.0 for c in calls)
        reply.usage = {k: sum((c.usage or {}).get(k, 0) for c in calls) for k in ("completion_tokens", "prompt_tokens")}
        reply.generation_time_seconds = sum(c.generation_time_seconds or 0.0 for c in calls)
        if self.end_on_empty and not (reply.has_content() or reply.is_tool_call()):
            reply = reply.model_copy(update={"content": STOP, "raw_data": {**reply.raw_data, "empty_reply": True}})
        return reply


def create_plugin_agent(tools, domain_policy, **kwargs):
    """tau2 agent factory. The plugin directories ride in llm_args, so that tau2's Results record them."""
    args = dict(kwargs.get("llm_args") or {})
    plugins, max_loads, end_on_empty = args.pop("plugins"), args.pop("max_skill_loads"), args.pop("end_on_empty", False)
    return PluginAgent(tools, domain_policy, llm=kwargs.get("llm"), llm_args=args, plugins=plugins, max_loads=max_loads,
                       end_on_empty=end_on_empty)


def evaluate(simulation, **kwargs):
    """tau2's evaluation, where a conversation the agent ended on an empty reply (end_on_empty) is an agent error."""
    if simulation.messages and (getattr(simulation.messages[-1], "raw_data", None) or {}).get("empty_reply"):
        simulation.termination_reason = TerminationReason.AGENT_ERROR
    return evaluate_simulation(simulation=simulation, **kwargs)


registry.register_agent_factory(create_plugin_agent, "plugin_agent")
tau2.runner.simulation.evaluate_simulation = evaluate
