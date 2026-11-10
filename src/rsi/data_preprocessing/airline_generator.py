"""Generates the airline validation set V and the three training batches from tau2's airline database and policy, aimed
at the knowledge points of the official tasks.

Usage, from the repository root:
    python -m rsi.data_preprocessing.airline_generator
About 3 minutes on all cores; every run writes the same bytes.

Reads tau2-bench/data/tau2/domains/airline/db.json, the official base tasks (their customers and reference writes are
left out) and data/test_annotation.json (the target share of each knowledge point). Writes
- data/processed/airline_generated.jsonl: every candidate, {"task", "source": "generated", "source_id", "db", "tags",
  "customer", "group"};
- data/processed/airline_final.jsonl and data/splits/airline.json: the tasks chosen for V (200) and the training batches
  round1-round3 (300 each), which rsi.benchmarks.run and the pipeline read;
- data/reports/airline_generate.json: candidates per family and tag, each split's counts against its targets, and per
  task what each request states (its spec) and, for a request the agent must refuse, the call that would grant it.

Then check every task with the tests, 8 processes in about 5 minutes; all 8 must print OK:
    for k in 0 1 2 3 4 5 6 7; do SHARD=$k/8 python -m unittest tests.test_airline_generate & done; wait

The requests are in airline/ (cancel.py, change.py, modify.py, book.py, compensation.py, linked.py); this file
registers them as families with quotas, combines them into tasks of several requests and writes each task's scenario.
"""
import json
import random
import re
from collections import Counter

from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_tasks

from rsi.data_preprocessing.airline.book import book, book_again
from rsi.data_preprocessing.airline.cancel import (CANCEL_POOL, cancel, cancel_negotiate, cancel_or_change, chain,
                                                   scope_cancel, upgrade_cancel)
from rsi.data_preprocessing.airline.change import (change, change_both, change_or_cancel, change_or_rebook, reroute,
                                                   reroute_rebook)
from rsi.data_preprocessing.airline.compensation import balances, compensation
from rsi.data_preprocessing.airline.facts import CITY, day, facts, status
from rsi.data_preprocessing.airline.linked import (ALLOWED, basic_upgrade_budget, book_pay_by_total, book_same_cap,
                                                   cabin_bags, cancel_total, change_upgrade_budget)
from rsi.data_preprocessing.airline.modify import (bags, cabin_change, fewer_passengers, insurance, passengers,
                                                   scope_downgrade, segment_upgrade, several)
from rsi.data_preprocessing.common import (CHUNK, ROOT, TRANSFER_LAST, Made, behaviours, leaks, line, make_task,
                                           outcome, outputs, owned, traits, write, writes_tag)

TAU2_AIRLINE_DB = "tau2-bench/data/tau2/domains/airline/db.json"
GENERATED, FINAL, SPLITS, REPORT = outputs("airline")
PAIR_CAP = 120  # requests per family across tasks of several requests, so that the easiest to combine do not crowd out the rest

# Variant groups of linked.py, the user's limit on either side: (pool, builder taking the side) on a reservation, or
# the builder for a customer.
LINKED = {
    "cabin_bags": (lambda r, f: f["upcoming"] and r.cabin != "business" and r.nonfree_baggages == 0, cabin_bags),
    "change_upgrade_budget": (lambda r, f: f["upcoming"] and r.cabin == "economy" and r.flight_type == "round_trip",
                              change_upgrade_budget),
    "basic_upgrade_budget": (lambda r, f: f["upcoming"] and r.cabin == "basic_economy" and f["age"] > 25
                             and not f["airline"], basic_upgrade_budget),
}
LINKED_USERS = {"book_same_cap": book_same_cap, "book_pay_by_total": book_pay_by_total}
RESERVATION_FAMILIES = {  # family: (pool, builder)
    **{f"cancel_{b}": (CANCEL_POOL[b], lambda db, r, rng, b=b: cancel(db, r, rng, b)) for b in CANCEL_POOL},
    "change": (lambda r, f: f["upcoming"] and r.cabin != "basic_economy",
               lambda db, r, rng: change(db, r, rng, "ok")),
    "change_same_day": (lambda r, f: f["upcoming"] and r.cabin != "basic_economy",
                        lambda db, r, rng: change(db, r, rng, "same_day")),
    "change_cabin": (lambda r, f: f["upcoming"] and r.cabin != "basic_economy",
                     lambda db, r, rng: change(db, r, rng, "cabin")),
    "change_both": (lambda r, f: f["upcoming"] and r.cabin != "basic_economy" and r.flight_type == "round_trip",
                    change_both),
    "upgrade_cancel": (lambda r, f: CANCEL_POOL["none"](r, f) and f["upcoming"], upgrade_cancel),
    "change_basic": (lambda r, f: f["upcoming"] and r.cabin == "basic_economy",
                     lambda db, r, rng: change(db, r, rng, "basic")),
    "change_upgrade": (lambda r, f: f["upcoming"] and r.cabin == "basic_economy" and f["age"] > 25
                       and not f["airline"],
                       lambda db, r, rng: change(db, r, rng, "upgrade")),
    "reroute": (lambda r, f: f["upcoming"] and r.cabin != "basic_economy", reroute),
    "reroute_rebook": (lambda r, f: f["upcoming"] and r.cabin == "business" and r.flight_type == "one_way",
                       reroute_rebook),
    "cabin": (lambda r, f: f["upcoming"], cabin_change),
    "cabin_budget": (lambda r, f: f["upcoming"], lambda db, r, rng: cabin_change(db, r, rng, budget=True)),
    "bags": (lambda r, f: not f["flown"], bags),
    "bags_remove": (lambda r, f: not f["flown"] and r.total_baggages > 1,
                    lambda db, r, rng: bags(db, r, rng, remove=True)),
    "passengers": (lambda r, f: not f["flown"], passengers),
    # A reservation with both a cancelled and a delayed flight would grant two kinds of certificate; none is used.
    "compensation_cancelled": (lambda r, f: f["airline"] and not f["delayed"] and f["eligible"],
                               lambda db, r, rng: compensation(db, r, rng, "cancelled")),
    "compensation_ineligible": (lambda r, f: f["airline"] and not f["delayed"] and not f["eligible"],
                                lambda db, r, rng: compensation(db, r, rng, "ineligible")),
    "compensation_delayed": (lambda r, f: f["delayed"] and not f["airline"],
                             lambda db, r, rng: compensation(db, r, rng, "delayed")),
    "compensation_delayed_change": (lambda r, f: f["delayed"] and not f["airline"],
                                    lambda db, r, rng: compensation(db, r, rng, "delayed_change")),
    "compensation_false": (lambda r, f: not f["airline"] and not f["delayed"],
                           lambda db, r, rng: compensation(db, r, rng, "false")),
    "cancel_or_change": (lambda r, f: f["upcoming"] and r.cabin == "economy" and f["age"] > 25 and r.insurance == "no",
                         cancel_or_change),
    "fewer_downgrade": (lambda r, f: f["upcoming"] and len(r.passengers) > 1 and r.cabin != "basic_economy",
                        lambda db, r, rng: fewer_passengers(db, r, rng, "downgrade")),
    "fewer_cancel": (lambda r, f: f["upcoming"] and len(r.passengers) > 1 and not 23 <= f["age"] <= 25,
                     lambda db, r, rng: fewer_passengers(db, r, rng, "cancel")),
    "insurance": (lambda r, f: f["upcoming"], insurance),
    "cancel_negotiate": (CANCEL_POOL["none"], cancel_negotiate),
    "several": (lambda r, f: f["upcoming"], several),
    "chain": (lambda r, f: f["upcoming"] and r.cabin == "basic_economy" and r.insurance == "no" and f["age"] > 25
              and not f["airline"], chain),
    "change_or_cancel": (lambda r, f: f["upcoming"] and r.cabin == "basic_economy" and r.insurance == "yes",
                         change_or_cancel),
    "change_or_rebook": (lambda r, f: f["upcoming"] and r.cabin == "basic_economy" and r.insurance == "yes",
                         change_or_rebook),
    "segment": (lambda r, f: f["upcoming"] and r.cabin != "business",
                lambda db, r, rng: segment_upgrade(db, r, rng, rng.random() < 0.5)),
    "cancel_total": (lambda r, f: any(CANCEL_POOL[b](r, f) for b in ALLOWED), cancel_total),
    # a variant group's request on its own, either side of the limit, for tasks of several requests
    **{f"{name}_any": (pool, lambda db, r, rng, build=build: build(db, r, rng, rng.random() < 0.5))
       for name, (pool, build) in LINKED.items()},
}
DELAYED = RESERVATION_FAMILIES["compensation_delayed"][0]  # a reservation with a delay to complain about
USER_FAMILIES = {
    "book": book,
    "book_certificates": lambda db, u, rng: book(db, u, rng, "certificates"),
    "book_cards": lambda db, u, rng: book(db, u, rng, "cards"),
    "book_gift_cards": lambda db, u, rng: book(db, u, rng, "gift_cards"),
    "book_certificate": lambda db, u, rng: book(db, u, rng, "certificate"),
    "book_certificate_gift": lambda db, u, rng: book(db, u, rng, "certificate_gift"),
    "book_gift_only": lambda db, u, rng: book(db, u, rng, "gift_only"),
    # the balances asked first are those the booking then spends; alone, since another request could spend them too
    **{f"book_sums_{s}": lambda db, u, rng, s=s: book(db, u, rng, s, sums=True)
       for s in ("gift_cards", "certificate_gift", "gift_only")},
    "book_again": book_again,
    "book_friend": lambda db, u, rng: book_again(db, u, rng, friend=True),
    "balances": balances,
    "scope_cancel_all": lambda db, u, rng: scope_cancel(db, u, rng, "all"),
    "scope_cancel_alone": lambda db, u, rng: scope_cancel(db, u, rng, "alone"),
    "scope_cancel_long": lambda db, u, rng: scope_cancel(db, u, rng, "long"),
    "scope_downgrade": scope_downgrade,
    **{f"{name}_any": lambda db, u, rng, build=build: build(db, u, rng, rng.random() < 0.5)
       for name, build in LINKED_USERS.items()},
}
# Variant groups: the same customer and story with one condition changed, so that the reference changes with it.
RESERVATION_VARIANTS = {
    "segment_upgrade": (lambda r, f: f["upcoming"] and r.cabin != "business",
                        [lambda db, r, rng: segment_upgrade(db, r, rng, True),
                         lambda db, r, rng: segment_upgrade(db, r, rng, False)]),
    "basic_change": (lambda r, f: f["upcoming"] and r.cabin == "basic_economy" and f["age"] > 25 and not f["airline"],
                     [lambda db, r, rng: change(db, r, rng, "basic"), lambda db, r, rng: change(db, r, rng, "upgrade")]),
    "cancel_fallback": (lambda r, f: f["upcoming"] and r.cabin == "economy" and f["age"] > 25 and r.insurance == "no",
                        [lambda db, r, rng: cancel(db, r, rng, "none", cabin_claim=False), cancel_or_change]),
    "change_budget": (lambda r, f: f["upcoming"] and r.cabin != "basic_economy",
                      [lambda db, r, rng: change(db, r, rng, "budget_within"),
                       lambda db, r, rng: change(db, r, rng, "budget_over")]),
    "cancel_upgrade": (lambda r, f: CANCEL_POOL["none"](r, f) and f["upcoming"],
                       [lambda db, r, rng: cancel(db, r, rng, "none", cabin_claim=False), upgrade_cancel]),
    **{name: (pool, [lambda db, r, rng, build=build: build(db, r, rng, True),
                     lambda db, r, rng, build=build: build(db, r, rng, False)]) for name, (pool, build) in LINKED.items()},
}
USER_VARIANTS = {
    "book_cap": [lambda db, u, rng: book(db, u, rng, cap="within"), lambda db, u, rng: book(db, u, rng, cap="over")],
    **{name: [lambda db, u, rng, build=build: build(db, u, rng, True),
              lambda db, u, rng, build=build: build(db, u, rng, False)] for name, build in LINKED_USERS.items()},
}
# Families a task of several requests draws from, one request each on its own reservation. A cancellation after a
# flown segment is not among them: the policy has the agent transfer it, which ends the call.
PAIRED = ("cancel_recent", "cancel_business", "cancel_insured", "cancel_none", "cancel_yesterday", "change",
          "change_basic", "change_upgrade", "reroute", "cabin", "cabin_budget", "bags", "bags_remove", "passengers",
          "compensation_cancelled", "compensation_delayed", "compensation_false", "fewer_downgrade", "insurance",
          "cancel_negotiate", "segment", "book", "book_certificates", "book_gift_cards", "book_again", "balances",
          "fewer_cancel", "chain", "change_same_day", "change_both", "upgrade_cancel", "book_friend", "change_or_rebook",
          "book_certificate", "book_certificate_gift", "book_gift_only", "change_cabin",
          *(f"{name}_any" for name in [*LINKED, *LINKED_USERS]))
# Tasks per family, or as many as the database allows; families with few instances come first, so that the others do
# not use them up. "pair" and "triple" draw two or three requests from PAIRED; a variant family counts groups.
QUOTA = {
    **{f: 40 for f in ("cancel_recent", "compensation_delayed_change", "compensation_ineligible", "compensation_delayed",
                       "cancel_airline", "compensation_cancelled", "compensation_false", "cancel_yesterday",
                       "cancel_flown", "cancel_business", "cancel_insured", "cancel_insured_other", "cancel_none",
                       "change_or_cancel", "change_or_rebook", "chain", "scope_cancel_long", "scope_cancel_alone",
                       "scope_cancel_all",
                       "scope_downgrade", "balances", "bags_remove", "fewer_cancel", "cancel_negotiate")},
    **{f: 30 for f in ("change", "change_basic", "change_upgrade", "cabin", "cabin_budget", "fewer_downgrade",
                       "cancel_or_change", "book")},
    **{f: 50 for f in ("bags", "passengers", "reroute")},
    **{f: 70 for f in ("book_again", "book_certificates", "book_cards", "book_gift_cards", "book_certificate",
                       "book_certificate_gift", "book_gift_only")},
    **{f"book_sums_{s}": 30 for s in ("gift_cards", "certificate_gift", "gift_only")},
    **{f: 60 for f in ("change_same_day", "change_both", "upgrade_cancel", "book_friend")},
    "change_cabin": 80, "reroute_rebook": 40,
    "several": 90, "segment_upgrade": 40, "basic_change": 30, "cancel_fallback": 30, "book_cap": 35,
    "change_budget": 35, "cancel_upgrade": 30,
    **{f: 50 for f in ("change_upgrade_budget", "book_same_cap", "book_pay_by_total", "basic_upgrade_budget",
                       "cancel_total", "cabin_bags")},
    # triple_write: three requests that all write, for the official tasks' share of three or more writes
    "pair": 700, "triple": 400, "triple_write": 300,
    # only as a request of a task of several: on their own the choice of V and the batches takes none of them
    **{f: 0 for f in ("book_certificate", "book_certificate_gift", "change", "change_both", "change_upgrade",
                      "fewer_downgrade", "upgrade_cancel")},
}
# Per customer and reservation across the candidates; the tasks of several requests come last and need room left
PER_USER = 30
PER_RESERVATION = 8
# Rates at which tasks show a behaviour; above the official shares, so that the choice of V and the batches can match
# those shares and still find enough tasks.
RATES = {"reactive": 0.22, "emotion": 0.15, "no_transfer": 0.15, "offtopic": 0.1}

# Each slot has a few wordings; a task picks one per slot with its own random.Random(task id).
CONFIRM = ("Confirm when the agent asks for confirmation.", "Say yes when the agent asks you to confirm.",
           "Agree once the agent summarizes what it will do.")
INSIST = ("If the agent refuses, push back {n} times before you accept it.",
          "If the agent says no, ask again {n} more times, then accept the decision.",
          "Do not take the first no for an answer: insist {n} times, then give up.")
ACCEPT_END = ("If the agent says it is not possible, accept that and end the conversation.",
              "If it cannot be done, thank the agent and end the call.",
              "Should the agent refuse, accept it and say goodbye.")
ACCEPT_ON = ("If the agent says it is not possible, accept that and go on with your other request.",
             "If it cannot be done, accept it and move on to your other request.",
             "Should the agent refuse, accept it and continue with the rest of your call.")
KNOWN = ("You are {name}. Your user id is {user_id}.", "Your name is {name} and your user id is {user_id}.",
         "You are {name}, with user id {user_id}.")
UNKNOWN = ("You do not remember your reservation number.", "You do not have your reservation id with you.",
           "You don't have your reservation number at hand.")
ALSO = ("Besides that, {}", "In addition, {}", "On the same call, {}")
LOWERED = {"You", "You'd", "On", "For", "Book", "Please", "The", "Ask"}
ORDER = ("Mention all of your requests at the start of the conversation.",
         "Bring up each request only after the previous one has been dealt with.")
LEAVE_IT = ("You are not good with schedules: let the agent work out which flights meet your conditions.",
            "You want the agent to figure out the flights that fit; you will not compare them yourself.",
            "You prefer not to compare the options yourself; ask the agent to find the right flights for you.")
# the request named by what it is about, as its line is headed: an ordinal depends on what the simulator counts
CHANGE_MIND = ("When the agent asks you to confirm your request for {what}, change your mind: you do not want that done "
               "after all.",
               "Once the agent is about to carry out your request for {what}, tell it you have changed your mind and "
               "want it left as it is.",
               "You change your mind about your request for {what} when asked to confirm it, and drop it.")
PAYING = {"book_reservation", "cancel_reservation", "update_reservation_flights", "update_reservation_baggages"}
INFO_TAGS = {"price_calc", "info_amount", "info_count"}


def heading(db, part):
    """What a request's lines are about, named as the request names it, to head them in a task of several requests."""
    if part.spec["family"].startswith("book"):
        return "For the new booking"
    rid = part.spec.get("reservation")
    if rid is None:
        return None
    if rid in part.request:
        return f"For reservation {rid}"
    r = db.reservations[rid]  # trip() makes origin, destination and first date unique
    return f"For your {day(r.flights[0].date)} trip from {CITY[r.origin]} to {CITY[r.destination]}"


def assemble(db, task_id, user_id, parts, rng, mind=None):
    """The task and its knowledge-point tags. A request asks for its amount only if no other request of the task moves
    money, which the question could be taken to be about. With several requests, each request's lines become one line
    headed by what they are about, since their payment methods, claims, budgets and bags would otherwise read as the
    whole call's; and a refused request never ends the call while another may still be pending. mind: the index of a
    request the user drops when asked to confirm it (parts already as drop() leaves them)."""
    user = db.users[user_id]
    requests = [parts[0].request]
    for p in parts[1:]:  # a request starting with a name keeps its capital
        first = p.request.split()[0]
        requests.append(rng.choice(ALSO).format(p.request[0].lower() + p.request[1:] if first in LOWERED else p.request))
    lines, info, tags = [], [], set()
    if len(parts) > 1 and rng.random() < 0.3:
        lines.append(rng.choice(ORDER[1:] if any(p.transfer for p in parts) else ORDER))
        tags.add("order")
    for k, p in enumerate(parts):
        own = list(p.lines)
        if p.transfer and len(parts) > 1:
            own.append(rng.choice(TRANSFER_LAST))
        if p.ask and not any(c["name"] in PAYING for q in parts if q is not p for c in q.calls):
            own.append(p.ask)
            info += p.info
            tags |= p.tags
        elif p.info and not p.ask:  # the request itself asks for the information
            info += p.info
            tags |= p.tags
        else:
            tags |= p.tags - INFO_TAGS
        if (p.forbidden or p.needs_company) and not p.calls and p.by_policy and not p.transfer:
            if k < len(parts) - 1:
                own.append(rng.choice(ACCEPT_ON))
            elif rng.random() < 0.65:
                own.append(rng.choice(INSIST).format(n=rng.choice((2, 3))))
                tags.add("pressure")
            else:  # with other requests, the agent may answer this one first: the user does not hang up on them
                own.append(rng.choice(ACCEPT_ON if len(parts) > 1 else ACCEPT_END))
        if "reservation" in p.spec and p.spec["reservation"] not in p.request:
            tags.add("find_res")
        head = heading(db, p) if len(parts) > 1 else None
        lines += [f"{head}: {' '.join(own)}"] if head and own else own
    if mind is not None:
        lines.append(rng.choice(CHANGE_MIND).format(what=(heading(db, parts[mind]) or "For that request")[4:]))
    calls = [c for p in parts for c in p.calls]
    if calls:
        lines.append(rng.choice(CONFIRM))
    # a user who asks for a human, or whom the policy has the agent transfer, does not also refuse a transfer
    refused = any((p.forbidden or p.needs_company) and not p.calls for p in parts) and not any(p.transfer for p in parts)
    if any(t.startswith("sel_") for p in parts for t in p.tags) and rng.random() < 0.15:
        lines.append(rng.choice(LEAVE_IT))
        tags.add("agent_decides")
    for b in behaviours(rng, RATES):
        if b != "no_transfer" or refused:
            lines.insert(rng.randrange(len(lines) + 1), line(rng, b))
            tags.add(b)
    persona = traits(rng)
    if persona:
        lines.insert(rng.randrange(len(lines) + 1), persona)
    tags |= {"out_write" if calls else "out_nowrite", writes_tag(len(calls))}
    if len(parts) > 1:
        tags.add("multi")
        # some request is refused; one that only asks for information, or that the user drops, is not
        if calls and any((p.forbidden or p.needs_company) and not p.calls and not p.spec.get("dropped") for p in parts):
            tags.add("out_partial")
    if any(p.transfer for p in parts):
        tags.add("out_transfer")
    known = rng.choice(KNOWN).format(name=f"{user.name.first_name} {user.name.last_name}", user_id=user_id)
    known = " ".join([known] + [x for p in parts for x in p.known])
    said = " ".join(requests + lines)  # no "you do not remember the reservation number" next to one given by its id
    unknown = rng.choice(UNKNOWN) if "find_res" in tags and not any(r in said for r in user.reservations) and \
        rng.random() < 0.3 else None
    return make_task("airline", task_id, parts, " ".join(requests), known, unknown, lines, info), \
        sorted(tags | branches(db, parts))


def branches(db, parts):
    """Pseudo-tags for the branch of a policy rule a task takes, so that V and each batch have every branch (BRANCHES):
    in a task of one request, a basic economy reservation upgraded first and changed in a second call, or any other
    changed along with its cabin in one call (with one request the agent gets a branch right often enough that V shows
    the harm of a rule written for its sibling); in any task, a cancellation that only the 24-hour rule allows and the
    user does not bring up, as in the official tasks."""
    out = set()
    if len(parts) == 1:
        flights = [c["arguments"] for c in parts[0].calls if c["name"] == "update_reservation_flights"]
        if flights:
            r = db.reservations[flights[0]["reservation_id"]]
            old = {(f.flight_number, f.date) for f in r.flights}
            new = [{(f["flight_number"], f["date"]) for f in a["flights"]} for a in flights]
            if r.cabin == "basic_economy" and len(new) == 2 and new[0] == old != new[1]:
                out.add("br_basic_two_calls")
            if r.cabin != "basic_economy" and len(new) == 1 and new[0] != old and flights[0]["cabin"] != r.cabin:
                out.add("br_one_call")
    for p in parts:
        moved = {c["arguments"]["reservation_id"] for c in p.calls if c["name"] == "update_reservation_flights"}
        for c in p.calls:
            rid = c["arguments"].get("reservation_id")
            if c["name"] == "cancel_reservation" and rid not in moved:
                r = db.reservations[rid]
                f = facts(db, r)
                covered = r.insurance == "yes" and p.spec.get("reason") in ("health", "weather")
                if f["age"] <= 24 and r.cabin != "business" and not covered and not f["airline"] and \
                        "recent" not in p.spec.get("claims", []):
                    out.add("br_cancel_24h")
    return out


# least tasks with each branch in V and in each training batch; the official tasks change a reservation's flights and
# cabin in one call more often than they upgrade a basic economy one first, and V keeps that order. The database has
# about twenty reservations a cancellation within 24 hours fits.
BRANCHES = {"br_one_call": (6, 6), "br_basic_two_calls": (0, 6), "br_cancel_24h": (6, 12)}


def ambiguous(db, parts):
    """Whether one request gives an illness or the weather as its reason while another expects the cancellation of an
    insured reservation to be refused: the agent could take the reason for both."""
    if not any(p.spec.get("reason") in ("health", "weather") or p.spec["family"] == "change_or_cancel" for p in parts):
        return False
    refused = [p.forbidden["arguments"]["reservation_id"] for p in parts
               if p.forbidden and p.forbidden["name"] == "cancel_reservation"]
    refused += [x for p in parts for x in p.spec.get("keep", [])]
    return any(db.reservations[x].insurance == "yes" for x in refused)


def wallet(db, part):
    """The gift cards and certificates the part's calls pay with or refund to, or whose balances it asks for: the user
    names them by balance, which another request's call would change."""
    ids = set(part.spec.get("methods", []))
    for c in part.calls + ([part.forbidden] if part.forbidden else []):
        g = c["arguments"]
        ids |= {g.get("payment_id")} | {m["payment_id"] for m in g.get("payment_methods", [])}
        if c["name"] == "cancel_reservation":
            ids |= {x.payment_id for x in db.reservations[g["reservation_id"]].payment_history}
    return {x for x in ids if x and not x.startswith("credit_card")}


def people(part):
    """(the passengers the request takes off a trip, the ones it books): someone who "cannot travel anymore" is not
    booked on another trip in the same call."""
    s = part.spec
    off = {tuple(s["drop"])} if s["family"] == "fewer_passengers" else {tuple(s["old"])} if s["family"] == "passengers" \
        else set()
    on = {(x["first_name"], x["last_name"]) for c in part.calls + ([part.forbidden] if part.forbidden else [])
          if c["name"] == "book_reservation" for x in c["arguments"]["passengers"]}
    return off, on


def reservations_of(part):
    s = part.spec
    return ([s["reservation"]] if "reservation" in s else []) + s.get("cancel", []) + s.get("keep", []) + s.get(
        "reservations", [])


def generate(db, official, taken):
    """Tasks per QUOTA family on customers of no evaluation task, each customer in at most PER_USER tasks and each
    reservation in at most PER_RESERVATION; a task whose reference writes are those of an evaluation task, or of a task
    made before, is skipped. Returns (family, task, parts, tags, customer, group) per task."""
    usable = [r for r in db.reservations.values()
              if r.user_id not in official and r.status != "cancelled" and r.origin != r.destination]
    fact = {r.reservation_id: {**facts(db, r), "statuses": [status(db, f) for f in r.flights]} for r in usable}
    users = sorted({r.user_id for r in usable})
    made = Made("airline", db, taken, assemble, reservations_of, (PER_USER, PER_RESERVATION), lambda p: True, 0.35,
                INFO_TAGS)
    paired = Counter()

    def combine(family, n, u, suffix):
        """n requests of customer u for a task of several, with their families, or None."""
        pick = random.Random(f"{family}_{u}{suffix}")
        mine = [r for r in usable if r.user_id == u and made.uses[r.reservation_id] < PER_RESERVATION]
        order = pick.sample(PAIRED, len(PAIRED))
        if any(DELAYED(r, fact[r.reservation_id]) for r in mine) and pick.random() < 0.5:
            order.insert(0, order.pop(order.index("compensation_delayed")))  # few customers have one to report
        if any(CANCEL_POOL["recent"](r, fact[r.reservation_id]) for r in mine):
            order.insert(0, order.pop(order.index("cancel_recent")))  # nor a reservation booked within 24 hours
        parts, names = [], []
        for name in order:
            if len(parts) == n:
                break
            kinds = {p.spec["family"] for p in parts}
            if paired[name] >= PAIR_CAP or name.split("_")[0] in {k.split("_")[0] for k in kinds}:
                continue
            busy = {x for p in parts for x in reservations_of(p)}
            if name in USER_FAMILIES:
                part = USER_FAMILIES[name](db, u, pick)
            else:
                pool, build = RESERVATION_FAMILIES[name]
                fits = [r for r in mine if r.reservation_id not in busy and pool(r, fact[r.reservation_id])]
                part = build(db, pick.choice(fits), pick) if fits else None
            # Requests never share a reservation, a gift card or a certificate: one that names it ("the same flight as
            # on X", "all my upcoming trips", "my $58 gift card") would read differently once another has changed it.
            if part is None or busy & set(reservations_of(part)) or (part.transfer and len(parts) < n - 1) \
                    or any(wallet(db, part) & wallet(db, p) or leaks(part, p) or leaks(p, part) for p in parts) \
                    or any(people(part)[0] & people(p)[1] or people(p)[0] & people(part)[1] for p in parts):
                continue
            if family == "triple_write" and not part.calls:
                continue
            parts.append(part)
            names.append(name)
        if len(parts) == n and not any(p.transfer for p in parts[:-1]) and not ambiguous(db, parts):
            return parts, names
        return None

    for family, quota in QUOTA.items():
        if not quota:
            continue
        rng = random.Random(family)
        if family in RESERVATION_FAMILIES or family in RESERVATION_VARIANTS:
            pool, builds = RESERVATION_VARIANTS.get(family) or (RESERVATION_FAMILIES[family][0],
                                                                [RESERVATION_FAMILIES[family][1]])
            candidates = [r for r in usable if pool(r, fact[r.reservation_id])]
            rng.shuffle(candidates)
            made.singles(family, quota, candidates, lambda r: r.reservation_id, lambda r: r.user_id, builds)
            continue
        candidates = list(users)
        rng.shuffle(candidates)
        if family in USER_FAMILIES or family in USER_VARIANTS:
            made.singles(family, quota, candidates, str, str, USER_VARIANTS.get(family) or [USER_FAMILIES[family]])
            continue
        n = {"pair": 2, "triple": 3, "triple_write": 3}[family]
        count = 0
        for attempt in range(3):  # several tasks of a kind per customer, each with other requests
            suffix = f"_{attempt}" if attempt else ""
            for start in range(0, len(candidates), CHUNK):
                if count == quota:
                    break
                # A customer's requests depend on their own reservations and on which families are used up: built
                # ahead for a chunk, and built again if a family fills up before the customer's turn.
                full = {f for f in PAIRED if paired[f] >= PAIR_CAP}
                chunk = [u for u in candidates[start:start + CHUNK] if made.free(u)]
                built = {u: combine(family, n, u, suffix) for u in chunk}
                made.prefetch([b[0] for b in built.values() if b])
                for u in chunk:
                    if count == quota:
                        break
                    if {f for f in PAIRED if paired[f] >= PAIR_CAP} != full:
                        built[u] = combine(family, n, u, suffix)
                    if built[u] is None:
                        continue
                    parts, names = built[u]
                    if made.record(family, [f"generated_{family}_" + "+".join(names) + f"_{u}{suffix}"], u, [parts]):
                        count += 1
                        paired.update(names)
    made.pool.shutdown()
    return made.tasks


def customer_of(db, task):
    """The customer an evaluation task is about, by the user id in it."""
    text = json.dumps(task.model_dump(mode="json"))
    return next(u for u in db.users if re.search(rf"\b{re.escape(u)}\b", text))


if __name__ == "__main__":
    db = FlightDB.load(ROOT / TAU2_AIRLINE_DB)
    test = get_tasks("base")
    text = json.dumps([t.model_dump(mode="json") for t in test])
    official = {u for u in db.users if u in text}
    # the database has a delay to complain about for only about twenty customers
    delayed = {r.user_id for r in db.reservations.values() if DELAYED(r, facts(db, r))}

    def owned_of(u):
        return owned(len(db.users[u].reservations), "airline")

    write("airline", TAU2_AIRLINE_DB, generate(db, official, {outcome(t, "airline") for t in test}), delayed, owned_of,
          {t.id: owned_of(customer_of(db, t)) for t in test}, BRANCHES)
