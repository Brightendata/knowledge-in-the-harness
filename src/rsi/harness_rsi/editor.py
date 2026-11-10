"""The editor, M(0): per-task failure analysis, the merge of proposals into rule edits, and the code checks on what it
returns. Prompts follow SkillOpt's (Yang et al., 2026, arXiv:2605.23904) failure analyst, failure merge and ranking
prompts and its skill-aware reflection (SKILL_DEFECT / EXECUTION_LAPSE), adapted to a plugin of numbered rules under a
fixed policy. Replies are constrained
to a JSON schema by the model server.
"""
import json
import re
import threading
from collections import Counter

from openai import OpenAI

from rsi.agent.plugin import RULES, description, documents, frontmatter, rules

ANALYST = """You are an expert failure analyst for a customer-service agent. The agent serves the users of one business under a written policy, with tools that read and change the business's database. The policy is not in the agent's prompt: it comes in a plugin, which the agent reads with the tool load_skill. The plugin has an entry document, which quotes the general part of the policy and holds a routing table, and one skill for each kind of database change, which quotes that kind's part of the policy. The agent sees a part of the policy only after loading the document that quotes it. Every document may end with a Rules section of numbered rules. The rules are the only part of the plugin that can change, and your analysis is used to edit them.

You receive:
1. The policy. It is the ground truth.
2. The tools the agent calls, as the agent sees them.
3. The current plugin: each document's name, which part of the policy it quotes, and its numbered rules.
4. Edits blocked earlier in this run, each with the problem it was blocked for: by a check before testing, or taken out after testing because the agent failed with it.
5. The evidence for ONE task the agent failed, prepared by a script: the outcome; the reference database writes, in order, and where the agent's writes differ from them; the differences between the agent's database and the reference database; which documents the agent loaded and when, and whether it had loaded the skill of each write before making it; the turns where it went wrong; and the conversation, numbered by turn, with the agent's thinking near those turns. There may be a second failed run of the same task, or a successful run of it for contrast. A successful run is a different conversation, with its own turn numbers: use it only to see what went differently, and take your diagnosis and quotes from the failed runs.

Your task: find why the agent failed, and propose at most one edit to the rules that would prevent this kind of failure in other conversations.

## How to analyze
1. Start from the write comparison and the database differences. They show WHAT went wrong, not WHY: a missing write can mean the agent wrongly refused or transferred, misread the request, or never got that far; an extra write, one the reference does not have, means the agent made a change that the policy rules out in this case or that the user did not ask for. For an extra write, check each condition the policy sets for that kind of change against the tool results for that reservation, order or item alone, such as a status, a date, the user's account, or a claim of the user that the tools do not confirm: the condition that fails there is what the agent should have checked before the write. Compare also the order of the agent's writes with the order of the reference writes: one write can change what a later one may do.
2. Find the turn where the agent went wrong, and use its thinking there to see what it believed. When the user made several requests, find the one that turn belongs to and judge that request against the policy on its own. Compare with the successful run when there is one.
3. Decide whether the failure is the agent's. The reference is the right outcome for everything this user wanted, and users often state part of it only when the agent asks. So when the reference has a write, or an item in a write, that the user had not asked for when the agent made its write, or never asked for, do not conclude that the reference is wrong: list what the policy tells the agent to ask, remind or confirm before writes of this kind, and check in the conversation whether the agent did each before its write. If it did not, the failure is the agent's: asked as the policy says, the user would have stated the rest. Only when even an agent that did all of this could not have reached the reference (the user changed or contradicted the request, gave wrong information, or did not state the rest even when asked) is the failure not the agent's: then propose nothing.
4. Check what the agent had loaded by the turn where it went wrong. A document it had not loaded by then could not guide it.
5. When an amount the agent stated or wrote is wrong, recompute the right amount from the numbers in the evidence (tool results, reference writes, information the agent had to state) and write the arithmetic in the diagnosis, with the reference amount it must give. Propose a rule about a calculation only if your arithmetic reproduces the reference amount.
6. Relate the failure to the existing guidance, which is the policy's sentences and the plugin's numbered rules. Test: is there a sentence or rule that, if the agent had followed it, would have prevented this failure?
   - "missing": no: neither the policy nor the plugin covers the situation;
   - "not_loaded": yes, but it is in a document the agent had not loaded by that turn;
   - "unclear": yes, the agent had loaded it, but it is wrong, too vague, or easy to misread in this situation;
   - "not_followed": yes, the agent had loaded it, it is clear, and the agent did not follow it.
7. Decide where the edit goes:
   - "not_loaded": the entry, as a rule on when to load which skill. A rule in the document the agent did not load could not help.
   - A failure that cuts across kinds of request, such as identifying the user, handling several requests in one conversation, or transferring to a human agent: the entry.
   - Otherwise, the skill for this kind of request: what to check, ask or tell the user, and in which order.
8. Choose the edit. The agent had the policy and the rules it loaded and still failed, so a rule that says the same thing again in other words, even with "must", fixes nothing. An edit must name an action the agent did not take at the turn where it went wrong (a question to ask the user, a field or tool result to check, a value to compute, or the order of two tool calls) and say when it applies, as in "Before calling <tool>, ..." or "When the user asks for ..., ...".
   - The policy is never wrong and cannot be edited: REPLACE and DELETE apply only to the plugin's numbered rules. When a policy sentence was unclear or not followed, ADD a rule that turns it into such a step at the point of action.
   - The plugin's rules were kept because the version that added them raised successes. REPLACE a rule only if an agent doing exactly what it says would still go wrong here: fix what leads it wrong, and keep every question, check and step of it that is still right. DELETE a rule only if it contradicts the policy or the reference.
   - When a plugin rule is right but was not followed, or does not cover this failure, keep it: ADD a separate rule for the action it does not name, or choose NONE if it already names the action, as rewording it rarely changes what the agent does. Never ADD a rule that repeats an existing one.
   - When there is a successful run, check that an agent following your rule would still do what it did right there; if not, revise the rule or choose NONE.
   - Choose NONE when the evidence is insufficient, the failure is not the agent's, or you cannot name such an action.

## Requirements for a rule
- One or two sentences of English, phrased as a general instruction that applies to other users, orders and reservations.
- One point: what the agent does in one situation. If the failure needs more than one change, propose the one for the first turn where the agent went wrong.
- No IDs, names, amounts, dates, products or flights of this task. Amounts stated in the policy itself may be used.
- A rule about an amount uses the policy's terms and the quantities the tools return, and never takes one quantity for another, such as an amount paid for a price.
- Never violate or relax the policy: a rule may only be stricter or more specific.
- The policy, its sentences on refusing and on transferring included, applies to each of the user's requests, and to each reservation, order or item a request covers, on its own: a part it allows is carried out even when another part cannot be handled, which is refused or transferred by itself. A rule never has the agent refuse, skip or transfer a part that the policy allows because another part is ruled out, unless the user asked for them only together.
- A preference this user stated, such as wanting the cheapest option or a direct flight, or a claim this user made, is not a rule for every user: keep it as the rule's condition ("When the user asks for ..., ..."), never as what the agent does by default.
- Never tell the agent to skip or shorten listing the action details and getting the user's explicit confirmation before a database change.
- Never tell the agent to decide for the user anything the policy says the user provides or chooses (such as the payment method, the flights, the items or variants, or the reason), and never tell it to guess instead of asking.
- REPLACE only a rule about the same situation that leads the agent wrong, and keep every question, check and step of it that is still right; for another situation, or for what a right rule does not cover, ADD.
- DELETE only a plugin rule that contradicts the policy or the reference, or that made the agent fail. A rule that only looks redundant stays.
- Do not propose a blocked edit again, nor any edit with the problem a blocked edit was blocked for.

## Evidence
Quote exactly: copy a short span, at most 15 words, from the turn of the failed run you cite. Prefer words to IDs and numbers, which are easy to miscopy.

## Output
A JSON object with these fields:
- "diagnosis": one to three sentences on why the agent failed; with "NONE", why you propose nothing.
- "evidence": the turns that show it, each a turn number and an exact quote from that turn.
- "relation": "missing", "not_loaded", "unclear" or "not_followed".
- "existing": the policy sentence the failure relates to, quoted exactly, or the id of the numbered rule; "" when the relation is "missing".
- "op": "ADD", "REPLACE", "DELETE" or "NONE".
- "file": the plugin document, as listed in the plugin: for ADD the one the rule goes to, for REPLACE and DELETE the one that holds the rule.
- "rule_id": for REPLACE and DELETE, the rule's id; otherwise "".
- "text": for ADD and REPLACE, the rule; otherwise "".
- "confidence": "high", "medium" or "low"."""

ANALYST_INPUT = """## Policy
{policy}

## Tools
{tools}

## Current plugin
{plugin}

## Blocked edits
{blocked}

## Evidence
{evidence}

Before you answer, think through the steps under How to analyze, one by one."""

MERGE = """You are a skill-edit coordinator. A customer-service agent reads its business's policy from a plugin, with the tool load_skill: an entry document, which quotes the general part of the policy and holds a routing table, and one skill for each kind of database change, which quotes that kind's part of the policy. The agent sees a part of the policy only after loading the document that quotes it. Analysts independently proposed edits to the plugin's numbered rules, one analyst per failed task. Merge the proposals into one coherent, non-redundant set of at most {budget} edits.

You receive:
1. The policy. It is the ground truth.
2. The tools the agent calls, as the agent sees them.
3. The current plugin, with its numbered rules.
4. Edits blocked earlier in this run, each with the problem it was blocked for: by a check before testing, or taken out after testing because the agent failed with it.
5. The proposals: for each, its task, the analyst's diagnosis, the quoted evidence, the existing rule it relates to, and the proposed edit.

## Merge rules
1. Deduplicate: proposals that make the same point become one edit, worded as the one that names the most concrete action, supported by all their tasks.
2. Resolve conflicts: when proposals contradict each other, keep the better-justified one or combine them. Settle a contradiction by the policy first, then by the evidence of the tasks; if neither settles it, drop both. Proposals that compute the same amount in different ways contradict each other: keep the way that, by the arithmetic in the diagnoses, gives the reference amount in every one of their tasks; if none does, drop them all; if several do, keep the one built from the quantities the policy names, such as prices and fees per passenger or per item.
3. Prevalent patterns first: edits supported by several tasks address systematic failures and take priority. {support}
4. Rank by expected impact: an edit that fixes a failure recurring across tasks beats one that fixes a single case. Return between 0 and {budget} edits, best first: {budget} is a ceiling, not a target. The edits are kept or rejected together, so a weak edit can sink good ones: adopt an edit only if its diagnosis fits its evidence and it names an action the agent did not take (a question to ask the user, a field or tool result to check, a value to compute, or the order of two tool calls) and when it applies. A rule that says again what the policy or an existing rule already says, even with "must", is not an edit.
5. Independence: no two edits may touch the same rule or make the same point.
6. Consistency: a new rule must not contradict an existing plugin rule. To overturn a rule, REPLACE it, with evidence that an agent doing exactly what it says goes wrong. A rule that is right but was not followed stays as it is: adopt a more concrete step as a separate ADD; never add a duplicate.
7. Never violate or relax the policy: rules may only be stricter or more specific. In particular, never skip or shorten listing the action details and getting the user's explicit confirmation before a database change, and never decide for the user anything the policy says the user provides or chooses. The policy, its sentences on refusing and on transferring included, applies to each of the user's requests, and to each reservation, order or item a request covers, on its own: a part it allows is carried out even when another part cannot be handled, which is refused or transferred by itself. Never adopt a rule that has the agent refuse, skip or transfer a part that the policy allows because another part is ruled out, unless the user asked for them only together. Keep each rule's condition: a preference or a claim of the users behind a proposal becomes "when the user asks for ..." or "when the user says ...", never a default for every user; when you combine proposals, keep the condition under which their tasks failed.
8. Edit locally: each edit ADDs, REPLACEs or DELETEs one rule; never rewrite a document.
9. Do not adopt a blocked edit again, nor any edit with the problem a blocked edit was blocked for.
10. Each ADD goes to one document of this plugin: rules on when to load which skill (proposals whose relation is "not_loaded") and rules that cut across kinds of request to the entry, rules for one kind of request to that skill. Rule text names no IDs, names, amounts, dates, products or flights of any task; amounts stated in the policy are fine. A rule about an amount uses the policy's terms and the quantities the tools return, and never takes one quantity for another, such as an amount paid for a price.
11. DELETE only a rule that contradicts the policy or a reference, or that made the agent fail. REPLACE only a rule about the same situation whose own instruction led the agent wrong, and keep every question, check and step of it that is still right. A proposal to REPLACE a rule about another situation, or a right rule that was not followed, becomes an ADD, and so does one whose new text would drop a question, check or step of the old rule that is still right. Rules that only look redundant stay.
12. If no proposal deserves adoption, return no edits.
13. One point per edit: an edit says what the agent does in one situation. Never join points about different situations or different requests into one rule, even short ones, and even when one proposal makes them together: each point is its own edit, supported by the tasks behind it. When the points worth adopting outnumber {budget}, adopt the best-supported ones.

## Output
A JSON object with:
- "edits": at most {budget}, best first. Each has "op" ("ADD", "REPLACE" or "DELETE"), "file" (the plugin document's name: for ADD the one the rule goes to, for REPLACE and DELETE the one that holds the rule), "rule_id" (for REPLACE and DELETE; otherwise ""), "text" (for ADD and REPLACE, one or two sentences; otherwise ""), "support" (the task ids of the proposals behind it) and "reason" (one sentence).
- "not_adopted": each proposal you did not adopt, by its id, with the reason.
- "summary": one sentence describing the change as a whole, for the version history."""

SUPPORT_ANY = ("Adopt an edit supported by a single task when the policy and the rules have no step for the situation "
               "it names: a calculation, a check, a question or an order of calls that the agent lacked fills such a "
               "gap, however narrow the situation; a rule that only says again what a policy sentence says fills none.")
SUPPORT_MIN = "Every edit needs the support of at least {n} different tasks."

MERGE_INPUT = """## Policy
{policy}

## Tools
{tools}

## Current plugin
{plugin}

## Blocked edits
{blocked}

## Proposals ({n})
{proposals}"""

CHECK = """You check one rule edit before it is tested. A customer-service agent serves the users of one business under a written policy, which it reads from a plugin: an entry document and one skill for each kind of database change, each ending with numbered rules. Analysts studied conversations the agent failed and proposed rule edits, and a coordinator merged them. Your job is to stop an edit that would make the agent act wrongly in conversations with OTHER users, and to let every other edit through.

You receive:
1. The policy. It is the ground truth.
2. The tools the agent calls, as the agent sees them.
3. The current plugin, with its numbered rules.
4. The edit: ADD a new rule to a document, or REPLACE an existing rule's text (the old text is shown).
5. The proposals behind the edit: for each supporting task, the analyst's diagnosis and the quoted evidence.

## Problems
A rule has a problem only if you can describe a concrete conversation with another user in which an agent that follows the rule acts wrongly, in one of these ways:
A. Condition dropped: the rule makes the agent do for every user what one user asked for or preferred (such as the cheapest option or a certain card), or treat as true for every user what one user claimed. A rule must keep the condition it depends on: "When the user asks for the cheapest ..., ...", not "Choose the cheapest ...".
B. Deciding for the user: the rule makes the agent pick on its own something the user chooses or provides (which flight, item or variant, payment method, number of bags, passengers, the reason). The agent may choose by a criterion only when the user asked for that criterion, and the rule must say so in its condition.
C. Skipping the confirmation: the rule lets the agent change the database without first listing the action details and getting the user's explicit confirmation (yes).
D. Forbidding what the policy allows: the rule's trigger is wider than the situation it was written for, so the agent would refuse, block or transfer in cases where the policy allows or requires the action.
E. Wrong amount or count: the rule computes or states an amount or count against what the policy or a tool says: it leaves out a charge the policy names for it (such as a fee per passenger or per item), adds a charge the policy does not name, counts one person where all count, prices something otherwise than the policy says, or fills a tool argument with a quantity other than the one the argument asks for.
F. Wrong transfer: the rule makes the agent transfer to a human agent where the policy says to refuse, or refuse where the policy says to transfer.
G. Allowing what the policy forbids: the rule says the agent may or can do something that a policy sentence says the agent cannot, may not or can no longer do in that situation. Read that sentence for exactly what it limits: a limit on one tool, or on how many times a tool may be called, says nothing about other tools.

## How to check
- For a rule that forbids something or makes the agent transfer: list the cases it forbids or sends to a human agent, reading "and", "or", "unless" and "only" literally, and compare each case with the policy (D, F). The policy, its sentences on refusing and on transferring included, applies to each of the user's requests, and to each reservation, order or item a request covers, on its own: a part it allows is carried out even when another part cannot be handled, which is refused or transferred by itself. So a rule that makes the agent refuse, skip or transfer a part that the policy allows because another part is ruled out forbids what the policy allows (D), however strict it is, unless the user asked for them only together.
- For a rule that adds a check or a condition before an action (verify, ensure, only if): say what the agent does when the check fails or the condition does not hold. If it then refuses the action, or rejects something the user chose, in a case where the policy and the tools allow it, that is D.
- For a rule that lets the agent do something, or says it still can: find the policy sentences about that action in the situation the rule names, and compare (G). Judge each action by the reservation, order or item as it is when that action is taken: after a change the policy allows (such as a new class), the sentences about what it has become apply, unless a later action of the rule undoes or refunds that change, as a cancellation does.
- For a rule about a database change: check that the agent still lists the action details and gets the user's explicit yes before the change (C).
- For a rule that computes or states an amount or a count: write, term by term, what the policy and the tools say it is made of and what the rule makes of it, and compare (E). Reaching the amount through other quantities the tools return, such as what was already paid, is not a problem by itself: check the terms.
- For a rule that makes the agent pick something: check that the pick follows the user's own request or answer (B), and that the rule's condition says when (A). A step the policy requires in the case the rule names, before what the user asked for can be done, is the policy's handling of that case, not a pick, and naming its least form (such as the next class up) is not one either: the agent proposes it, and the user takes it or asks for more.
- For a REPLACE, judge the old text the same way, then the new text against the policy, not against the old text: changing the old rule's trigger, scope or content is what a REPLACE is for.

## Not a problem
- The rule repeats the policy, is long, vague or redundant, or seems unlikely to help.
- The rule adds a check, a question or a confirmation, even one the policy does not ask for, as long as it does not forbid anything the policy allows.
- The rule has the agent ask the user something before an action, such as which payment method to use or whether the request is complete, even where the policy or the tools do not need the answer: asking does not forbid the action.
- The rule is stricter or more specific than the policy in the situation it names, and the policy allows what it asks.
- The rule does not repeat a requirement the policy states elsewhere, such as a check before an action: the agent still has the policy, so leaving the requirement out of the rule does not skip it.
- A policy sentence can be read in more than one way, and one reading allows what the rule asks.
- You cannot tell whether the policy allows what the rule asks.
- The rule names, for a step the policy requires before what the user asked for can be done, the least such change (such as the next class up): the agent lists it for the user's confirmation, and a user who asks for another gets that instead.
- The rule first makes a change the policy allows, which stays, and then does what the policy allows for the reservation, order or item as that change has left it. A later action that undoes or refunds the first change, such as a cancellation, does not count as leaving it.

## Output
A JSON object with:
- "situations": one sentence on when the rule applies, as written.
- "old_problem": for a REPLACE, the old text's problem: "A" to "G", or "none"; for an ADD, "none".
- "problem": the new rule's problem: "A" to "G", or "none".
- "example": for a problem of the new rule, the conversation in which an agent following it would act wrongly, and what the policy requires there; otherwise "".
- "verdict": "block" if the new rule has a problem, otherwise "pass"."""

CHECK_INPUT = """## Policy
{policy}

## Tools
{tools}

## Current plugin
{plugin}

## Edit
{edit}

## Proposals behind it
{proposals}

Before you answer, go through "How to check" for this rule (for a REPLACE, the old text first), then the problems A to G one by one, and what is not a problem."""

REVIEW = """You review one rewrite of a rule before it is tested. A customer-service agent serves the users of one business under a written policy, which it reads from a plugin: an entry document and one skill for each kind of database change, each ending with numbered rules. Analysts studied conversations the agent failed and proposed rule edits, and a coordinator merged them. This edit REPLACEs an existing rule and rewrites most of its words. The existing rule was kept because the plugin with it succeeded more often in earlier tests. A rewrite that drops or changes a part of it that is still right loses what that part did for other users, and the test after this step rarely shows the loss; a rewrite that fixes a part that is wrong is what a REPLACE is for. Your job: decide whether the new text keeps every part of the old text that is right and changes only what is wrong.

You receive:
1. The policy. It is the ground truth.
2. The tools the agent calls, as the agent sees them.
3. The current plugin, with its numbered rules.
4. The edit: the rule's old text, its new text, and the reason the coordinator gave for it.
5. Why the old text was written: the reason given when it was added or last rewritten, and the analysts' diagnoses of the failed tasks it was written for.
6. The proposals behind the rewrite: for each supporting task, the analyst's diagnosis and quotes.
7. The failed runs of the supporting tasks, near the turns where the agent went wrong: the reference writes, where the agent's writes differ from them, and the conversation.
8. Two earlier checks of the edit. They judged each text on its own against the policy and did not compare the two: their passing the new text does not mean the rewrite keeps what was right.

## How to review
1. Split the old text into its parts: each condition (when it applies), and each thing it has the agent do (a check, a question, a tool call, the order of calls, an amount or count). Split as finely as the words allow: "all items, including those named in earlier turns" is two parts.
2. For each part, look for the cases in which an agent following the old text and an agent following the new text act differently because of how the new text treats that part. Look first at the cases the old text was written for, then at the cases of the rewrite's supporting tasks, then at cases like them with more items, requests or turns, other cabins or reservation types, other order statuses, more passengers. Cases in which both texts lead to the same action do not matter, even when they are the cases the old text was written for. The agent also follows the policy: where a part of the old text only repeats what a policy sentence requires in that case, such as listing the action details and getting the user's explicit yes before a database change, an agent following the new text still does it, unless the new text says otherwise. A part for which there is no such case is kept.
3. For each such case, decide which action is right by the policy, and by the reference for the tasks shown: the old text's, the new text's, both or neither. A policy sentence decides it, or a failed run in which the agent went wrong by doing what that very part of the old text says. That the agent failed in a task the rule applies to is not enough: a failure that came from something else, such as a request the user withdrew or changed, a part of the request the agent missed, or a step the rule does not mention, says nothing about the part. An action is wrong when it does for every user what one user asked for or claimed, decides for the user something the user chooses or provides, changes the database without listing the action details and getting the user's explicit yes, refuses or transfers where the policy allows the action, computes or states an amount or count against the policy or the tools, or does what the policy says the agent cannot do in that situation.

## Output
A JSON object with "cases": each case of step 2, with "part" (the words of the old text it is about), "case" (one sentence: the situation, what the old text has the agent do and what the new text has it do) and "right" ("old", "new", "both" or "neither"). No cases when the new text keeps every part."""

REVIEW_INPUT = """## Policy
{policy}

## Tools
{tools}

## Current plugin
{plugin}

## Edit
REPLACE {rule_id}
Old text: {old}
New text: {new}
Reason given for the rewrite: {reason}

## Why the old text was written
{origin}

## Proposals behind the rewrite
{proposals}

## Failed runs of the rewrite's supporting tasks
{runs}

## Earlier checks
{checks}

Before you answer, go through "How to review" for this rewrite: the parts of the old text, then for each part the cases in which the two texts make the agent act differently, and which action is right in each."""

OPS = ["ADD", "REPLACE", "DELETE"]


def schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def targets(plugin_dir):
    """Allowed values of the edit fields, enforced by constrained decoding: REPLACE and DELETE only once the plugin has
    rules, and only this plugin's document names and rule ids. Unconstrained, the editor often tries to REPLACE
    sentences of the policy; allowed an empty file, the merge leaves it empty on ADDs."""
    docs = documents(plugin_dir)
    ids = [i for path in docs.values() for i, _ in rules(path)]
    return {"op": {"type": "string", "enum": OPS if ids else ["ADD"]},
            "file": {"type": "string", "enum": list(docs)},
            "rule_id": {"type": "string", "enum": ids + [""]}}


def analysis_schema(plugin_dir):
    t = targets(plugin_dir)
    return schema({
        "diagnosis": {"type": "string"},
        "evidence": {"type": "array", "items": schema({"turn": {"type": "integer"}, "quote": {"type": "string"}})},
        "relation": {"type": "string", "enum": ["missing", "not_loaded", "unclear", "not_followed"]},
        "existing": {"type": "string"},
        "op": {"type": "string", "enum": t["op"]["enum"] + ["NONE"]},
        "file": t["file"],
        "rule_id": t["rule_id"],
        "text": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    })


def merge_schema(plugin_dir):
    return schema({
        "edits": {"type": "array", "items": schema({
            **targets(plugin_dir), "text": {"type": "string"}, "support": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"}})},
        "not_adopted": {"type": "array", "items": schema({"proposal": {"type": "string"}, "reason": {"type": "string"}})},
        "summary": {"type": "string"},
    })


PROBLEMS = ["A", "B", "C", "D", "E", "F", "G", "none"]
CHECK_SCHEMA = schema({
    "situations": {"type": "string"},
    "old_problem": {"type": "string", "enum": PROBLEMS},
    "problem": {"type": "string", "enum": PROBLEMS},
    "example": {"type": "string"},
    "verdict": {"type": "string", "enum": ["block", "pass"]},
})
CHECKS = 2  # each edit is checked this many times, with different seeds, and dropped only when every check blocks it
REVIEWS = 3  # a rewrite of most of a rule is reviewed this many times, with different seeds (kept_by_review)
REVIEW_SCHEMA = schema({
    "cases": {"type": "array", "items": schema({
        "part": {"type": "string"}, "case": {"type": "string"},
        "right": {"type": "string", "enum": ["old", "new", "both", "neither"]}})},
})
CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*dollars?\b", re.I)
_lock = threading.Lock()


def ask(cfg, system, user, reply_schema, seed, call_id, raw_path):
    """The editor's parsed reply, or None when no try gives valid JSON after thinking. Gemma often answers without
    thinking, and analyses without thinking were mostly wrong, so such a reply is tried again and never used. Every
    try, with the full messages, the thinking and the token counts, is appended to raw_path."""
    e = cfg["editor"]
    client = OpenAI(base_url=cfg["server"]["api_base"], api_key="EMPTY", timeout=3600)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for i in range(e["retries"] + 1):
        r = client.chat.completions.create(
            model=e["model"], messages=messages, temperature=e["temperature"], top_p=e["top_p"],
            max_tokens=e["max_tokens"], seed=seed + i,
            response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": reply_schema}},
            extra_body={"top_k": e["top_k"]})
        m = r.choices[0].message
        extra = m.model_extra or {}
        thinking = extra.get("reasoning_content") or extra.get("reasoning")
        with _lock, open(raw_path, "a") as f:
            f.write(json.dumps({"id": call_id, "try": i, "seed": seed + i, "messages": messages, "thinking": thinking,
                                "content": m.content, "finish_reason": r.choices[0].finish_reason,
                                "usage": r.usage.model_dump() if r.usage else None}) + "\n")
        try:
            reply = json.loads(m.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if thinking:
            return reply
    return None


def plugin_view(plugin_dir):
    """The plugin as the editor reads it: each document's name, what of the policy it quotes, and its rules. The
    quoted policy itself is not repeated: the policy comes in full before it."""
    out = []
    for name, path in documents(plugin_dir).items():
        body = frontmatter(path.read_text())[1]
        entry = ":" not in name
        out += [f"### {name} ({'entry, loaded first' if entry else 'skill'})",
                f"Catalog description: {description(plugin_dir, name)}"]
        if entry:
            routing = body.split("## Routing\n", 1)[1].split(f"\n{RULES}\n")[0].strip()
            out += ["Quotes the policy from its beginning up to its first action section.",
                    f"Routing (fixed):\n{routing}"]
        elif "## Policy" in body:
            out.append(f'Quotes the policy section "{body.splitlines()[0].lstrip("#").strip()}".')
        else:
            out.append("The policy has no section of its own for it; the general rules apply.")
        numbered = rules(path)
        out += (["Rules:"] + [f"- [{i}] {t}" for i, t in numbered]) if numbered else ["Rules: none yet."]
        out.append("")
    return "\n".join(out).strip()


def edit_line(e):
    where = e["file"] if e["op"] == "ADD" else e["rule_id"]
    return f"{e['op']} {where}" + ("" if e["op"] == "DELETE" else f": {e['text']}")


CHECK_PROBLEMS = {"A": "makes the agent do for every user what one user asked for or claimed",
                  "B": "decides for the user",
                  "C": "lets the agent change the database without the user's confirmation",
                  "D": "forbids what the policy allows",
                  "E": "computes or states an amount or count against the policy or the tools",
                  "F": "transfers where the policy says to refuse, or refuses where it says to transfer",
                  "G": "lets the agent do what the policy says it cannot",
                  "H": "made the agent fail: a task whose failed runs named it failed far more often than before"}


def blocked_text(buffer):
    """The edits the check blocked earlier in this run, with the problems it found, and the rules the rerun of their
    step took out (evolve.harmful). The other edits of a step the retest did not keep are not listed: a step's edits are
    judged together on a noisy rerun, so its rejection says little about any one of them, and the editor takes a listed
    edit for a harmful one and never proposes it again."""
    if not buffer:
        return "None."
    lines = []
    for b in buffer:
        lines.append(f"Step {b['step']}:")
        for x in b["blocked"]:
            when = "taken out after testing" if "H" in x["problems"] else "blocked before testing"
            problems = " and ".join(CHECK_PROBLEMS[p] for p in sorted(set(x["problems"])))
            lines.append(f"- {edit_line(x['edit'])} ({when}: it {problems})")
    return "\n".join(lines)


def proposals_text(analyses):
    out = []
    for a in analyses:
        o = a["output"]
        quotes = "; ".join(f'turn {q["turn"]}: "{q["quote"]}"' for q in o["evidence"])
        existing = f" (existing: {o['existing']})" if o["existing"] else ""
        out += [f"### {a['id']} (task {a['task_id']})", f"Diagnosis: {o['diagnosis']}", f"Evidence: {quotes}",
                f"Relation: {o['relation']}{existing}", f"Proposed: {edit_line(o)} (confidence {o['confidence']})", ""]
    return "\n".join(out).strip()


def merge_system(budget, min_support):
    support = SUPPORT_MIN.format(n=min_support) if min_support > 1 else SUPPORT_ANY
    return MERGE.format(budget=budget, support=support)


def normalize(s):
    """Words and numbers only: a quote may drop the backticks around an ID or the quotes around a JSON key."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def quote_problem(output, evidence, failed):
    """Why an analysis may not be merged, or None: a proposal must quote evidence, every quote must be text of the
    evidence it was given (each piece of a quote cut with an ellipsis), and at least one must come from the failed runs,
    not only from the successful run shown for contrast. The cited turn is not checked: the editor often cites a turn or
    two off, or quotes a fact the pack states, such as an amount the agent never said."""
    if output["op"] == "NONE":
        return None
    if not output["evidence"]:
        return "no evidence quoted"
    text, failed_text, from_failed = normalize(evidence), normalize(failed), False
    for q in output["evidence"]:
        pieces = [p for p in map(normalize, re.split(r"\.\.\.|…", q["quote"])) if p]
        if not pieces or any(p not in text for p in pieces):
            return f"quote not in the evidence: {q['quote']!r}"
        from_failed |= all(p in failed_text for p in pieces)
    return None if from_failed else "no quote from the failed runs"


def tools_text(tools):
    """The tools as the agent sees them, one JSON schema per line, without the schemas' titles."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "title"}
        return [strip(x) for x in o] if isinstance(o, list) else o
    return "\n".join(json.dumps(strip(t.openai_schema["function"]), ensure_ascii=False) for t in tools)


def check(cfg, policy, tools, plugin_dir, edit, proposals, seed, call_id, raw_path):
    """The editor's verdict on one merged edit (CHECK), or None without a valid reply. A DELETE is not checked."""
    if edit["op"] == "DELETE":
        return None
    if edit["op"] == "ADD":
        text = f"ADD to {edit['file']}: {edit['text']}"
    else:
        old = next(t for path in documents(plugin_dir).values() for i, t in rules(path) if i == edit["rule_id"])
        text = f"REPLACE {edit['rule_id']}\nOld text: {old}\nNew text: {edit['text']}"
    behind = [a for a in proposals if a["task_id"] in set(edit["support"])]
    user = CHECK_INPUT.format(policy=policy, tools=tools, plugin=plugin_view(plugin_dir), edit=text,
                              proposals=proposals_text(behind) or "None.")
    return ask(cfg, CHECK, user, CHECK_SCHEMA, seed, call_id, raw_path)


def kept_share(old, new):
    """Share of the old text's words, counted with repeats, that the new text still has."""
    a, b = Counter(re.findall(r"\w+", old.lower())), Counter(re.findall(r"\w+", new.lower()))
    return sum((a & b).values()) / max(1, sum(a.values()))


def review(cfg, policy, tools, plugin_dir, edit, proposals, origin, runs, checks, seed, call_id, raw_path):
    """The editor's review of a REPLACE against the rule's old text (REVIEW), or None without a valid reply. origin:
    the edit that wrote the old text and the analyses behind it, or None; runs: the supporting tasks' failed runs as
    text; checks: the edit's CHECK verdicts."""
    old = next(t for path in documents(plugin_dir).values() for i, t in rules(path) if i == edit["rule_id"])
    behind = [a for a in proposals if a["task_id"] in set(edit["support"])]
    why = "Not recorded."
    if origin:
        written, analyses = origin
        why = f"{written['op']}, for: {written.get('reason') or 'no reason given'}\n\n{proposals_text(analyses)}"
    lines = [f"- Check {j}: no valid reply." if v is None else
             f"- Check {j}: old text {v['old_problem']}, new text {v['problem']}, verdict {v['verdict']}"
             + (f"; example: {v['example']}" if v["example"] else "") for j, v in enumerate(checks, 1)]
    user = REVIEW_INPUT.format(policy=policy, tools=tools, plugin=plugin_view(plugin_dir), rule_id=edit["rule_id"],
                               old=old, new=edit["text"], reason=edit.get("reason") or "none given", origin=why,
                               proposals=proposals_text(behind) or "None.", runs=runs or "None.",
                               checks="\n".join(lines) or "None.")
    return ask(cfg, REVIEW, user, REVIEW_SCHEMA, seed, call_id, raw_path)


def kept_by_review(verdicts):
    """Whether a reviewed REPLACE stays: fewer than two of its reviews find a case in which only the old text's action
    is right, a review without a valid reply counting as one that does. A single review's slip, either way, does not
    decide. Problems of the new text alone are the checks' to find (CHECK), which block only when both agree: one
    review's reading of an amount or a condition is too noisy to drop an edit on."""
    return bool(verdicts) and sum(v is None or any(c["right"] == "old" for c in v["cases"]) for v in verdicts) < 2


def blocked(verdicts, op):
    """Whether an edit's checks block it: every check found a problem in the new rule and, for a REPLACE, none in the
    old text, which a REPLACE with a problem of its own still improves on. A check without a valid reply never blocks."""
    return bool(verdicts) and all(
        v is not None and v["verdict"] == "block" and v["problem"] != "none"
        and (op != "REPLACE" or v["old_problem"] == "none") for v in verdicts)


def db_keys(db_paths):
    """Every ID of the databases: table keys, and the keys of payment methods and of product variants."""
    keys = set()
    for path in db_paths:
        for table in json.loads(path.read_text()).values():
            if isinstance(table, dict):
                keys |= set(table)
                for rec in table.values():
                    if isinstance(rec, dict):
                        for field in ("payment_methods", "variants"):
                            if isinstance(rec.get(field), dict):
                                keys |= set(rec[field])
    return keys


def amounts(text):
    return {float(x.replace(",", "")) for m in AMOUNT.finditer(text) for x in m.groups() if x}


def edit_problem(e, plugin_dir, analyzed, unclear, min_support, keys, policy_amounts, touched):
    """Why an edit may not be applied, or None. unclear: the tasks whose analysis found an existing rule wrong or
    unclear. A REPLACE or DELETE needs one of them among its support: a rule that is right but was not followed stays,
    as the prompts say, and the merge does not always keep to it."""
    docs = documents(plugin_dir)
    ids = {i for path in docs.values() for i, _ in rules(path)}
    if e["op"] == "ADD":
        if e["file"] not in docs:
            return f"{e['file']!r} is not a document of this plugin"
    elif e["rule_id"] not in ids:
        return f"{e['rule_id']!r} is not a rule of this plugin"
    elif e["rule_id"] in touched:
        return f"{e['rule_id']} is already edited in this step"
    if not set(e["support"]) <= analyzed:
        return "supported by tasks not analysed in this step"
    if len(set(e["support"])) < min_support:
        return f"supported by fewer than {min_support} tasks"
    if e["op"] != "ADD" and not set(e["support"]) & unclear:
        return "REPLACE or DELETE without an analysis that found the rule wrong or unclear"
    if e["op"] != "DELETE":
        if not e["text"].strip():
            return "empty rule"
        named = sorted(w for w in re.findall(r"#?\w+", e["text"]) if w in keys)
        if named:
            return f"names database IDs {named}"
        extra = sorted(amounts(e["text"]) - policy_amounts)
        if extra:
            return f"names amounts not in the policy {extra}"
        if CJK.search(e["text"]):
            return "not in English"
    return None
