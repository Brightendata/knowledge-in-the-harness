"""Every generated airline task through policy rules the tools do not enforce and a reading of each request's spec (in
the generation report), both written without the generator's helpers; the tasks chosen for V and the training batches
also through tau2's own evaluator. Run after rsi.data_preprocessing.airline_generator. With SHARD=k/n the slow tests
check every n-th task from the k-th, so that n processes share them; the default, 0/20, checks every 20th task in a few
minutes, and SHARD=0/1 checks all of them (about two CPU-hours for both domains)."""
import json
import os
import re
import unittest
from collections import Counter

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Action, Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.domains.airline.tools import AirlineTools
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from rsi.data_preprocessing.airline_generator import FINAL, GENERATED, REPORT, TAU2_AIRLINE_DB
from rsi.data_preprocessing.common import ROOT, WRITES, outcome

CITY = {a.iata: a.city for a in AirlineTools(None).list_all_airports()}
ALLOWED = {"recent", "business", "airline", "insured"}
UP = {"basic_economy": "economy", "economy": "business"}
DOWN = {"business": "economy", "economy": "basic_economy"}
FREE = {"regular": {"basic_economy": 0, "economy": 1, "business": 2},
        "silver": {"basic_economy": 1, "economy": 2, "business": 3},
        "gold": {"basic_economy": 2, "economy": 3, "business": 4}}
DAY_BEFORE = "2024-05-14T15:00:00"  # 24 hours before the policy's current time
SHARD, SHARDS = map(int, os.environ.get("SHARD", "0/20").split("/"))
COVERED_REASON = re.compile(r"\b(?:sick\w*|ill|illness|unwell|health|medical\w*|hospital\w*|injur\w*|surgery|flu|"
                            r"fever|weather|storms?|hurricane|snow\w*|blizzard)\b", re.I)


def flight_status(db, flight):
    return db.flights[flight.flight_number].dates[flight.date].status


def flown(db, reservation):
    return any(flight_status(db, f) in ("landed", "flying") for f in reservation.flights)


def certificate_basis(db, user_id, amount, changed):
    """A reservation of the user whose flight status grants this certificate, judged on the database before the
    conversation: $100 per passenger for a cancelled flight, $50 per passenger for a delayed one once the reservation
    is changed or cancelled, and only for a silver or gold member, an insured reservation or a business one."""
    user = db.users[user_id]
    for rid in user.reservations:
        r = db.reservations[rid]
        if not (user.membership in ("silver", "gold") or r.insurance == "yes" or r.cabin == "business"):
            continue
        status = {flight_status(db, f) for f in r.flights}
        if "cancelled" in status and amount == 100 * len(r.passengers):
            return rid
        if "delayed" in status and amount == 50 * len(r.passengers) and rid in changed:
            return rid
    return None


def policy_violations(tasks, db):
    """Airline policy rules the tools do not enforce, checked on the reference replayed step by step. Compensation is
    judged on the database before the conversation, since references often change the delayed flight first."""
    reasons, dump = {}, db.model_dump()
    for t in tasks:
        env = get_environment(db=FlightDB.model_validate(dump))
        live = env.tools.db
        writes = [a for a in t.evaluation_criteria.actions or [] if a.name in WRITES["airline"]]
        changed = {a.arguments["reservation_id"] for a in writes
                   if a.name in ("update_reservation_flights", "cancel_reservation")}
        found = []
        for a in t.evaluation_criteria.actions or []:
            g = a.arguments
            r = live.reservations.get(g.get("reservation_id", ""))
            if a.name == "update_reservation_flights" and r is not None:
                old = [(f.flight_number, f.date) for f in r.flights]
                new = [(f["flight_number"], f["date"]) for f in g["flights"]]
                legs = [live.flights[n] for n, _ in new]
                if old != new and r.cabin == "basic_economy":
                    found.append("changes the flights of a basic economy reservation")
                end = r.destination if r.flight_type == "one_way" else r.origin
                if old != new and (legs[0].origin != r.origin or legs[-1].destination != end
                                   or r.destination not in {leg.destination for leg in legs}):
                    found.append("changes the origin, destination or trip type")
                if any(flight_status(live, f) in ("landed", "flying") for f in r.flights
                       if (f.flight_number, f.date) not in new):
                    found.append("replaces a flown segment")
                if g["cabin"] != r.cabin and flown(live, r):
                    found.append("changes the cabin after a segment was flown")
            if a.name == "update_reservation_baggages" and r is not None:
                free = FREE[live.users[r.user_id].membership][r.cabin] * len(r.passengers)
                if g["total_baggages"] < r.total_baggages or g["nonfree_baggages"] < r.nonfree_baggages:
                    found.append(f"removes bags ({r.total_baggages} -> {g['total_baggages']} in total, "
                                 f"{r.nonfree_baggages} -> {g['nonfree_baggages']} paid)")
                elif g["nonfree_baggages"] != max(0, g["total_baggages"] - free):
                    found.append(f"charges {g['nonfree_baggages']} bags where {free} are free")
            if a.name == "book_reservation":
                user = live.users[g["user_id"]]
                kinds = Counter(user.payment_methods[p["payment_id"]].source for p in g["payment_methods"])
                if kinds["certificate"] > 1 or kinds["credit_card"] > 1 or kinds["gift_card"] > 3:
                    found.append(f"pays with {dict(kinds)}")
                if len(g["passengers"]) > 5:
                    found.append("books more than five passengers")
                free = FREE[user.membership][g["cabin"]] * len(g["passengers"])
                if g["nonfree_baggages"] != max(0, g["total_baggages"] - free):
                    found.append(f"charges {g['nonfree_baggages']} bags where {free} are free")
            if a.name == "cancel_reservation" and r is not None:
                if flown(live, r):
                    found.append(f"cancels {r.reservation_id}, which has a flown segment")
                elif not (r.created_at >= DAY_BEFORE or r.cabin == "business"
                          or any(flight_status(live, f) == "cancelled" for f in r.flights)):
                    if r.insurance != "yes":
                        found.append(f"cancels {r.reservation_id}, which no cancellation rule allows")
                    elif not COVERED_REASON.search(str(t.user_scenario.instructions)):
                        found.append(f"cancels {r.reservation_id} on insurance without a health or weather reason")
            if a.name == "send_certificate" and certificate_basis(db, g["user_id"], g["amount"], changed) is None:
                found.append(f"sends a ${g['amount']} certificate that no cancelled or delayed flight grants")
            env.make_tool_call(tool_name=a.name, requestor=a.requestor, **g)
        if found:
            reasons[t.id] = "; ".join(found)
    return reasons


def reward(task, dump, trajectory):
    def environment(solo_mode=False):
        # each call must build its own DB, or the evaluator's two environments would share one
        return get_environment(db=FlightDB.model_validate(dump), solo_mode=solo_mode)

    r = EnvironmentEvaluator.calculate_reward(environment_constructor=environment, task=task,
                                              full_trajectory=trajectory).reward
    return r * CommunicateEvaluator.calculate_reward(task=task, full_trajectory=trajectory).reward


def replay(calls, dump):
    env = get_environment(db=FlightDB.model_validate(dump))
    messages = []
    for k, c in enumerate(calls):
        call = ToolCall(id=f"call_{k}", name=c["name"], arguments=c["arguments"], requestor="assistant")
        messages += [AssistantMessage(role="assistant", tool_calls=[call]), env.get_response(call)]
    return messages


def charged(dump, call):
    """What the call adds to its reservation's payment history (negative for a refund, 0 for nothing)."""
    env = get_environment(db=FlightDB.model_validate(dump))
    history = env.tools.db.reservations[call["arguments"]["reservation_id"]].payment_history
    n = len(history)
    env.make_tool_call(tool_name=call["name"], **call["arguments"])
    return sum(x.amount for x in history[n:])


def legs(r):
    """The outbound flights and, for a round trip, the return flights."""
    if r.flight_type == "one_way":
        return [r.flights]
    k = [f.origin for f in r.flights].index(r.destination)
    return [r.flights[:k], r.flights[k:]]


def at(clock, days=0):
    """Minutes after midnight of the departure day, for a schedule time like '03:00:00+1'."""
    hm, _, plus = clock.partition("+")
    h, m = map(int, hm.split(":")[:2])
    return (days + int(plus or 0)) * 1440 + 60 * h + m


def search(db, origin, destination, date, cabin, seats):
    """Every itinerary the search tools list with seats for everyone: legs, price per seat, departure, arrival."""
    tools = AirlineTools(db)
    found = [{"legs": [(f.flight_number, date)], "price": f.prices[cabin], "dep": at(f.scheduled_departure_time_est),
              "arr": at(f.scheduled_arrival_time_est)}
             for f in tools.search_direct_flight(origin, destination, date) if f.available_seats[cabin] >= seats]
    for a, b in tools.search_onestop_flight(origin, destination, date):
        if a.available_seats[cabin] >= seats and b.available_seats[cabin] >= seats:
            found.append({"legs": [(a.flight_number, a.date), (b.flight_number, b.date)],
                          "price": a.prices[cabin] + b.prices[cabin], "dep": at(a.scheduled_departure_time_est),
                          "arr": at(b.scheduled_arrival_time_est, int(b.date[-2:]) - int(date[-2:]))})
    return found


def pick(found, preference, current=None, hour=None, after=None):
    """The itineraries the user's preference names; it singles one out when this returns one. after: for "later", the
    departure (minutes) of the user's current flight."""
    if preference == "same flight":
        return [o["legs"] for o in found if [n for n, _ in o["legs"]] == current]
    if preference == "later":
        later = [o for o in found if o["dep"] > after]
        return [o["legs"] for o in later if o["dep"] == min(x["dep"] for x in later)] if later else []
    if preference == "second cheapest":
        prices = sorted({o["price"] for o in found})
        if len(prices) < 3 or [o["price"] for o in found].count(prices[0]) != 1:
            return []
        return [o["legs"] for o in found if o["price"] == prices[1]]
    nonstop = [o for o in found if len(o["legs"]) == 1]
    pool, key = {
        "cheapest": (found, lambda o: o["price"]),
        "cheapest nonstop": (nonstop, lambda o: o["price"]),
        "earliest nonstop": (nonstop, lambda o: o["dep"]),
        "fastest": (found, lambda o: o["arr"] - o["dep"]),
        "after": ([o for o in found if o["dep"] >= 60 * (hour or 0)], lambda o: o["price"]),
        "arrive by": ([o for o in found if o["arr"] <= 60 * (hour or 0)], lambda o: o["price"]),
    }[preference]
    if len(pool) < 2:
        return []
    best = min(key(o) for o in pool)
    return [o["legs"] for o in pool if key(o) == best]


def clock(hour):
    return f"{hour % 12 or 12}{'am' if hour < 12 else 'pm'}"


def dated(date):
    return f"May {int(date[-2:])}"


def born(dob):
    y, m, d = dob.split("-")
    month = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December")[int(m) - 1]
    return f"{month} {int(d)}, {y}"


class AirlineGenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = FlightDB.load(ROOT / TAU2_AIRLINE_DB)
        cls.dump = cls.db.model_dump()
        cls.tasks = [Task.model_validate(json.loads(line)["task"]) for line in open(GENERATED)]
        cls.chosen = [Task.model_validate(json.loads(line)["task"]) for line in open(FINAL)]
        cls.specs = json.loads(REPORT.read_text())["specs"]

    def text(self, task):
        return str(task.user_scenario.instructions)

    def names_payment(self, text, user, payment_id):
        """The scenario names the method in a way that fits no other method of the user."""
        m = user.payment_methods[payment_id]
        if m.source == "credit_card":
            self.assertIn(f"ending in {m.last_four}", text)
            self.assertEqual([x.last_four for x in user.payment_methods.values() if x.source == "credit_card"]
                             .count(m.last_four), 1)
            return
        kind = "gift card" if m.source == "gift_card" else "travel certificate"
        same = [x for x in user.payment_methods.values() if x.source == m.source]
        self.assertIn(kind, text)
        if len(same) > 1:
            self.assertIn(f"${m.amount:g} {kind}", text)
            self.assertEqual([x.amount for x in same].count(m.amount), 1)

    def names_trip(self, text, r):
        """The scenario names the reservation by its id, or by its cities and dates, which fit no other reservation of
        the user."""
        if r.reservation_id in text:
            return
        js = legs(r)
        for code in (r.origin, r.destination):
            self.assertIn(CITY[code], text)
        for j in js:
            self.assertIn(dated(j[0].date), text)
        key = (r.origin, r.destination, r.flights[0].date)
        others = [o for o in self.db.users[r.user_id].reservations if o != r.reservation_id]
        self.assertNotIn(key, [(self.db.reservations[o].origin, self.db.reservations[o].destination,
                                self.db.reservations[o].flights[0].date) for o in others])

    def test_no_evaluation_customer(self):
        official = json.dumps([t.model_dump(mode="json") for t in get_tasks("base")])
        for t in self.tasks:
            for s in self.specs[t.id]:
                user = s.get("user") or self.db.reservations[s["reservation"]].user_id
                self.assertNotIn(user, official, t.id)

    def test_outcomes_are_new(self):
        taken = {outcome(t, "airline") for t in get_tasks("base")}
        mine = Counter(outcome(t, "airline") for t in self.tasks if outcome(t, "airline"))
        self.assertEqual(set(mine) & taken, set())
        self.assertEqual([o for o, n in mine.items() if n > 1], [])

    def test_requests_apart(self):
        """No two requests of a task are about the same reservation, gift card or certificate: one naming it ("the same
        flight as on X", "all my upcoming trips", "my $58 gift card") would read differently once another had changed
        it."""
        def stored(p):  # gift cards and certificates its calls pay with or refund to, or whose balances it asks for
            ids = set(p.get("methods", []))
            for c in p["calls"] + ([p["forbidden"]] if p["forbidden"] else []):
                g = c["arguments"]
                ids |= {g.get("payment_id")} | {m["payment_id"] for m in g.get("payment_methods", [])}
                if c["name"] == "cancel_reservation":
                    ids |= {x.payment_id for x in self.db.reservations[g["reservation_id"]].payment_history}
            return {x for x in ids if x and self.db.users[t_user].payment_methods.get(x) is not None
                    and self.db.users[t_user].payment_methods[x].source != "credit_card"}

        for t in self.tasks:
            seen = Counter(x for p in self.specs[t.id] for x in {p.get("reservation"), *p.get("cancel", []),
                                                                 *p.get("keep", []), *p.get("reservations", [])} if x)
            self.assertEqual([x for x, n in seen.items() if n > 1], [], t.id)
            t_user = re.search(r"user id is (\w+)|with user id (\w+)", t.user_scenario.instructions.known_info)
            t_user = next(g for g in t_user.groups() if g)
            cards = Counter(x for p in self.specs[t.id] for x in stored(p))
            self.assertEqual([x for x, n in cards.items() if n > 1], [], t.id)
            # someone the user takes off a trip as no longer travelling is not booked on another in the same call
            off = {tuple(p["drop"] if p["family"] == "fewer_passengers" else p["old"]) for p in self.specs[t.id]
                   if p["family"] in ("fewer_passengers", "passengers")}
            on = {(x["first_name"], x["last_name"]) for p in self.specs[t.id]
                  for c in p["calls"] + ([p["forbidden"]] if p["forbidden"] else [])
                  if c["name"] == "book_reservation" for x in c["arguments"]["passengers"]}
            self.assertEqual(off & on, set(), t.id)

    def test_info_unsaid(self):
        """No amount the agent must state is in the scenario, which the user would say for it."""
        for t in self.tasks:
            text = self.text(t)
            self.assertEqual([x for x in t.evaluation_criteria.communicate_info if x in text], [], t.id)

    def test_no_transfer_conflict(self):
        """A user who asks for a human, or whom the policy has the agent transfer, does not also refuse a transfer."""
        for t in self.tasks:
            if any(p.get("transfer") or p.get("basis") == "flown" for p in self.specs[t.id]):
                self.assertNotRegex(self.text(t), r"no circumstances do you want to be transferred|refuse to be "
                                                  r"transferred|offers to transfer you, decline", t.id)

    def test_unknown_reservation(self):
        """A user who does not have the reservation number at hand names no reservation by its id."""
        for t in self.tasks:
            ins = t.user_scenario.instructions
            if ins.unknown_info:
                said = " ".join([ins.known_info, ins.reason_for_call, ins.task_instructions])
                self.assertEqual([w for w in re.findall(r"\b[A-Z0-9]{6}\b", said) if w in self.db.reservations], [],
                                 t.id)

    def test_policy(self):
        tasks = self.tasks[SHARD::SHARDS]
        self.assertEqual(policy_violations(tasks, self.db), {})
        for t in tasks:
            parts = self.specs[t.id]
            self.assertEqual([a.model_dump(include={"name", "arguments"}) for a in t.evaluation_criteria.actions],
                             [c for p in parts for c in p["calls"]], t.id)
            for k, p in enumerate(parts):
                if not p["forbidden"]:
                    continue
                bad = t.model_copy(deep=True)
                calls = [c for i, q in enumerate(parts) for c in ([q["forbidden"]] if i == k else q["calls"])]
                bad.evaluation_criteria.actions = [Action(action_id=f"x{i}", **c) for i, c in enumerate(calls)]
                if p["by_policy"]:
                    self.assertIn(t.id, policy_violations([bad], self.db), t.id)
                else:  # the user declines an upgrade over their budget, which the policy would allow
                    self.assertEqual(policy_violations([bad], self.db), {}, t.id)

    def test_scores(self):
        for t in self.chosen[SHARD::SHARDS]:
            parts = self.specs[t.id]
            calls = [c for p in parts for c in p["calls"]]
            info = t.evaluation_criteria.communicate_info
            said = [AssistantMessage(role="assistant", content=f"That comes to ${int(x):,}.") for x in info]
            self.assertEqual(reward(t, self.dump, replay(calls, self.dump) + said), 1.0, t.id)
            self.assertEqual(reward(t, self.dump, []), 0.0 if calls or info else 1.0, t.id)
            if info:
                self.assertEqual(reward(t, self.dump, replay(calls, self.dump)), 0.0, t.id)
            for k, p in enumerate(parts):
                if p["forbidden"]:
                    bad = [c for i, q in enumerate(parts) for c in ([p["forbidden"]] if i == k else q["calls"])]
                    self.assertEqual(reward(t, self.dump, replay(bad, self.dump) + said), 0.0, t.id)
            # a request the agent ends by transferring the user comes last; one no tool could grant comes with a write
            ends = [k for k, p in enumerate(parts) if p.get("transfer") or p.get("basis") == "flown"]
            self.assertTrue(all(k == len(parts) - 1 for k in ends), t.id)
            if any(p["family"] == "insurance" for p in parts):
                self.assertTrue(calls, t.id)
            writing = [p["calls"] for p in parts if p["calls"]]
            if len(writing) > 1:  # the requests do not depend on their order, and each is needed
                back = [c for w in reversed(writing) for c in w]
                self.assertEqual(reward(t, self.dump, replay(back, self.dump) + said), 1.0, t.id)
                for i in range(len(writing)):
                    rest = [c for j, w in enumerate(writing) if j != i for c in w]
                    self.assertEqual(reward(t, self.dump, replay(rest, self.dump) + said), 0.0, t.id)

    def test_specs(self):
        seen = Counter(p["family"] for t in self.tasks for p in self.specs[t.id])
        for t in self.tasks[SHARD::SHARDS]:
            text = self.text(t)
            for p in self.specs[t.id]:
                if p.get("dropped"):  # the user drops it when asked to confirm: the agent must not make its call
                    self.assertIn("change your mind", text.lower().replace("changed your mind", "change your mind"))
                    p = {**p, "calls": [p["forbidden"]], "forbidden": None}
                with self.subTest(task=t.id, family=p["family"]):
                    getattr(self, "spec_" + p["family"])(t, text, p)
        self.assertEqual(set(seen), {"cancel", "change", "reroute", "cabin", "bags", "passengers", "book", "compensation",
                                     "segment_upgrade", "fewer_passengers", "insurance", "balances", "scope_cancel",
                                     "scope_downgrade", "several", "chain", "change_or_cancel", "book_again",
                                     "change_both", "upgrade_cancel", "change_or_rebook", "cabin_bags",
                                     "change_upgrade_budget", "book_same_cap", "book_pay_by_total",
                                     "basic_upgrade_budget", "cancel_total", "reroute_rebook"})

    def claims(self, p, r, granted):
        """What the user claims about the reservation or their membership is true exactly when the request is granted."""
        statuses = [self.db.flights[f.flight_number].dates[f.date].status for f in r.flights]
        truth = {"recent": r.created_at >= "2024-05-14T15:00:00", "insured": r.insurance == "yes",
                 "business": r.cabin == "business", "airline": "cancelled" in statuses,
                 "gold": self.db.users[r.user_id].membership == "gold"}
        for claim in p.get("claims", []):
            self.assertEqual(truth[claim], granted, claim)

    def spec_cancel(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        call = {"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}}
        allowed = p["basis"] in ALLOWED
        self.assertEqual(p["calls"], [call] if allowed else [])
        self.assertEqual(p["forbidden"], None if allowed else call)
        self.claims(p, r, allowed)
        if allowed:
            self.states(t, sum(x.amount for x in r.payment_history))

    def moved(self, text, r, journey, date, preference, cabin, payment, hour=None):
        """The flight change that moves the journey to the date by the preference, as the agent would find it."""
        if date == legs(r)[journey][0].date:  # other flights on the day the reservation already has
            self.assertRegex(text, "same (day|date)")
        else:
            self.assertIn(dated(date), text)
        if hour is not None:
            self.assertIn(clock(hour), text)
        self.names_payment(text, self.db.users[r.user_id], payment)
        js = legs(r)
        j = js[journey]
        found = search(self.db, j[0].origin, j[-1].destination, date, cabin, len(r.passengers))
        if date == j[0].date:  # other flights on the same day: the user's own are no change
            found = [o for o in found if [n for n, _ in o["legs"]] != [f.flight_number for f in j]]
        new = pick(found, preference, [f.flight_number for f in j], hour,
                   at(self.db.flights[j[0].flight_number].scheduled_departure_time_est))
        self.assertEqual(len(new), 1)
        flights = [[(f.flight_number, f.date) for f in jj] for jj in js]
        flights[journey] = new[0]
        return {"name": "update_reservation_flights",
                "arguments": {"reservation_id": r.reservation_id, "cabin": cabin, "payment_id": payment,
                              "flights": [{"flight_number": n, "date": d} for jj in flights for n, d in jj]}}

    def states(self, t, amount):
        """A one-request task that asks for an amount asks for this one."""
        info = t.evaluation_criteria.communicate_info
        if info and len(self.specs[t.id]) == 1:
            self.assertEqual(info, [str(abs(amount))])

    def spec_change(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        if "fallback_for" in p:  # the cancellation the user asks for first, which is refused
            self.claims(p["fallback_for"], r, granted=False)
        expected = self.moved(text, r, p["journey"], p["date"], p["preference"], p["cabin"], p["payment"], p.get("hour"))
        if p.get("upgrade_first"):
            self.assertEqual(r.cabin, "basic_economy")
            # the card pays or receives the difference of both calls, so the sentence naming it must say so (another
            # request of the task may upgrade first too, with a sentence of its own)
            self.assertTrue(any("both" in s for s in text.split("\n") if "upgrad" in s and "to economy" in s
                                and ("first" in s or "so that the change" in s)), text)
            first = {**expected, "arguments": {**expected["arguments"], "flights": [
                {"flight_number": f.flight_number, "date": f.date} for f in r.flights]}}
            self.assertEqual(p["calls"], [first, expected])
        elif r.cabin == "basic_economy":
            self.assertEqual((p["calls"], p["forbidden"]), ([], expected))
        elif "budget" in p:  # the user takes the change only up to the limit
            self.assertIn(f"${p['budget']}", text)
            self.assertEqual(p["calls"] or [p["forbidden"]], [expected])
            self.assertEqual(bool(p["calls"]), 0 < charged(self.dump, expected) <= p["budget"])
        elif p["cabin"] != r.cabin:  # the cabin switched with the flights, in the same call
            self.assertEqual({r.cabin, p["cabin"]}, {"economy", "business"})
            for cabin in (r.cabin, p["cabin"]):
                self.assertIn(cabin, t.user_scenario.instructions.reason_for_call)
            self.assertEqual(p["calls"], [expected])
            self.states(t, charged(self.dump, expected))
            if len(self.specs[t.id]) == 1:  # done as two calls, the payment history differs
                said = [AssistantMessage(role="assistant", content=f"That comes to ${int(x):,}.")
                        for x in t.evaluation_criteria.communicate_info]
                first = {**expected, "arguments": {**expected["arguments"], "flights": [
                    {"flight_number": f.flight_number, "date": f.date} for f in r.flights]}}
                self.assertEqual(reward(t, self.dump, replay([first, expected], self.dump) + said), 0.0)
        else:
            self.assertEqual(p["calls"], [expected])
            self.states(t, charged(self.dump, expected))

    def spec_reroute(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.assertIn(CITY[p["destination"]], text)
        self.assertEqual(p["calls"], [])
        self.assertEqual(self.db.flights[p["forbidden"]["arguments"]["flights"][-1]["flight_number"]].destination,
                         p["destination"])

    def spec_cabin(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.names_payment(text, self.db.users[r.user_id], p["payment"])
        call = {"name": "update_reservation_flights",
                "arguments": {"reservation_id": r.reservation_id, "cabin": p["cabin"], "payment_id": p["payment"],
                              "flights": [{"flight_number": f.flight_number, "date": f.date} for f in r.flights]}}
        self.assertEqual(p["calls"] or [p["forbidden"]], [call])
        difference = charged(self.dump, call)
        self.states(t, difference)
        if "budget" in p:
            self.assertIn(f"${p['budget']}", text)
            self.assertEqual(bool(p["calls"]), 0 < difference <= p["budget"])
            if len(self.specs[t.id]) == 1:  # with other requests that move money, the question is left out
                self.assertIn(str(difference), t.evaluation_criteria.communicate_info)

    def spec_bags(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        call = p["calls"][0] if p["calls"] else p["forbidden"]
        if "add" in p:
            self.names_payment(text, self.db.users[r.user_id], p["payment"])
            self.assertEqual(call["arguments"]["total_baggages"], r.total_baggages + p["add"])
            self.states(t, charged(self.dump, call))
        else:
            self.assertEqual(p["calls"], [])
            self.assertEqual(call["arguments"]["total_baggages"], r.total_baggages - p["remove"])
            self.assertIn(f"of the {r.total_baggages} checked bags", text)

    def spec_passengers(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        user = self.db.users[r.user_id]
        self.assertNotEqual(p["old"], [user.name.first_name, user.name.last_name])
        self.assertIn(" ".join(p["old"]), text)
        self.assertIn(f"{p['new']['first_name']} {p['new']['last_name']}", text)
        if [p["new"]["first_name"], p["new"]["last_name"]] == [user.name.first_name, user.name.last_name]:
            self.assertEqual(p["new"]["dob"], user.dob)  # the user, whose date of birth the agent reads in the profile
            self.assertIn("profile", text)
        else:
            self.assertIn(born(p["new"]["dob"]), text)
        expected = [p["new"] if [x.first_name, x.last_name] == p["old"] else x.model_dump() for x in r.passengers]
        self.assertEqual(p["calls"][0]["arguments"]["passengers"], expected)

    def spec_book(self, t, text, p):
        user = self.db.users[p["user"]]
        call = p["calls"][0]["arguments"]
        for code in (p["origin"], p["destination"]):
            self.assertIn(CITY[code], text)
        for date in p["dates"]:
            self.assertIn(dated(date), text)
        if "hour" in p:
            self.assertIn(clock(p["hour"]), text)
        self.assertEqual(p["passengers"][0], {"first_name": user.name.first_name, "last_name": user.name.last_name,
                                              "dob": user.dob})
        saved = [x.model_dump() for x in user.saved_passengers]
        for x in p["passengers"]:
            self.assertTrue(x == p["passengers"][0] or x in saved)
            if "profile" not in text:  # otherwise the agent reads dates of birth from the profile
                self.assertIn(born(x["dob"]), text)
        routes = [(p["origin"], p["destination"])] + [(p["destination"], p["origin"])] * (len(p["dates"]) - 1)
        flights = []
        for (o, d), date in zip(routes, p["dates"]):
            found = search(self.db, o, d, date, p["cabin"], len(p["passengers"]))
            new = pick(found, p["preference"], None, p.get("hour"))
            self.assertEqual(len(new), 1)
            flights += new[0]
        self.assertEqual(call["flights"], [{"flight_number": n, "date": d} for n, d in flights])
        self.assertEqual(call["passengers"], p["passengers"])
        self.assertEqual([x["payment_id"] for x in call["payment_methods"]], p["payment"])
        paid = {x["payment_id"]: x["amount"] for x in call["payment_methods"]}
        kinds = [user.payment_methods[k].source for k in p["payment"]]
        # a certificate first, then gift cards, a credit card last; every stored balance but the last method's is used
        # up, and the last one takes only what is still owed (the rest of a certificate is lost)
        self.assertEqual(kinds, sorted(kinds, key=["certificate", "gift_card", "credit_card"].index))
        for k in p["payment"][:-1]:
            if user.payment_methods[k].source != "credit_card":
                self.assertEqual(paid[k], user.payment_methods[k].amount)
        gifts = [k for k in p["payment"] if user.payment_methods[k].source == "gift_card"]
        if len(gifts) > 1:  # smallest balance first, every gift card unless the ones used cover the price
            self.assertIn("smallest", text)
            mine = sorted((k for k, m in user.payment_methods.items() if m.source == "gift_card"),
                          key=lambda k: user.payment_methods[k].amount)
            self.assertEqual(gifts, mine[:len(gifts)])
            if len(gifts) < len(mine):
                self.assertEqual(p["payment"][-1], gifts[-1])
        for k in p["payment"]:
            if user.payment_methods[k].source != "gift_card" or len(gifts) == 1:
                self.names_payment(text, user, k)
        price = sum(x["amount"] for x in call["payment_methods"])
        if "cap" in p:  # the companions come along only within the user's limit (no bags, no insurance)
            self.assertIn(f"${p['limit']}", text)
            everyone = p["forbidden"]["arguments"]["passengers"] if p["cap"] == "over" else p["passengers"]
            self.assertEqual(len(p["passengers"]) > 1, p["cap"] == "within")
            fare = price // len(p["passengers"])
            self.assertEqual(fare * len(everyone) <= p["limit"], p["cap"] == "within")
        info = t.evaluation_criteria.communicate_info
        if "sums" in p:  # asked before the booking, which spends some of these balances
            self.assertRegex(text, r"Before anything is booked|only then book")
            self.assertTrue(set(p["sums"]) & {user.payment_methods[k].source for k in p["payment"]})
            for source, said in p["sums"].items():
                found = [m.amount for m in user.payment_methods.values() if m.source == source]
                self.assertGreater(len(found), 1)
                self.assertEqual(said, str(int(sum(found))))
                self.assertIn(said, info)
            info = [x for x in info if x not in p["sums"].values()]
            self.assertIn(info, ([], [str(price)]))
        else:
            self.states(t, price)

    def spec_compensation(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.assertIn(f"{p['flight']} on {dated(p['date'])}", text)
        self.assertIn((p["flight"], p["date"]), [(f.flight_number, f.date) for f in r.flights])
        amount = (100 if p["claim"] == "cancelled" else 50) * len(r.passengers)
        if "miscount" in p:  # the user's count of passengers is wrong; the reservation's is what counts
            self.assertNotEqual(p["miscount"], len(r.passengers))
            self.assertIn(f" {p['miscount']}", text)
        certificate = {"name": "send_certificate", "arguments": {"user_id": r.user_id, "amount": amount}}
        self.claims(p, r, granted=bool(p["calls"]))
        if self.db.flights[p["flight"]].dates[p["date"]].status != p["claim"]:  # a false claim earns nothing
            self.assertEqual((p["calls"], p["forbidden"]), ([], certificate))
        elif "date_new" in p:  # a delayed flight moved to another day
            journey = [k for k, j in enumerate(legs(r)) if p["flight"] in [f.flight_number for f in j]]
            self.assertEqual(journey, [p["journey"]])
            change = self.moved(text, r, p["journey"], p["date_new"], p["preference"], r.cabin, p["payment"])
            self.assertEqual(p["calls"], [change, certificate])
        else:
            self.assertEqual(p["calls"] or [p["forbidden"]], [certificate])


    # ---------------------------------------------------------------- the requests added for the test-aligned pool

    def upcoming(self, r):
        return all(self.db.flights[f.flight_number].dates[f.date].status == "available" for f in r.flights)

    def unclear(self, user):
        """Reservations of the user that are partly still to come but not all bookable (a flight later today, or after
        one flown or cancelled), which the user might count among their upcoming trips."""
        found = []
        for x in user.reservations:
            r = self.db.reservations[x]
            statuses = [self.db.flights[f.flight_number].dates[f.date].status for f in r.flights]
            if r.status != "cancelled" and not self.upcoming(r) and \
                    any(s in ("available", "on time", "delayed") for s in statuses):
                found.append(r)
        return found

    def cancellable(self, r):
        """A cancellation rule allows it for a change of plans: booked within 24 hours, business class, or a flight the
        airline cancelled."""
        statuses = [self.db.flights[f.flight_number].dates[f.date].status for f in r.flights]
        return r.created_at >= "2024-05-14T15:00:00" or r.cabin == "business" or "cancelled" in statuses

    def whole(self, r, cabin, payment):
        return {"name": "update_reservation_flights",
                "arguments": {"reservation_id": r.reservation_id, "cabin": cabin, "payment_id": payment,
                              "flights": [{"flight_number": f.flight_number, "date": f.date} for f in r.flights]}}

    def spec_segment_upgrade(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.names_payment(text, self.db.users[r.user_id], p["payment"])
        self.assertEqual(p["cabin"], UP[r.cabin])
        self.assertIn(f"${p['budget']}", text)
        call = self.whole(r, p["cabin"], p["payment"])
        self.assertEqual(p["calls"] or [p["forbidden"]], [call])
        self.assertEqual(bool(p["calls"]), charged(self.dump, call) <= p["budget"])

    def spec_fewer_passengers(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        user = self.db.users[r.user_id]
        self.names_trip(text, r)
        self.assertIn(p["drop"], [[x.first_name, x.last_name] for x in r.passengers])
        self.assertNotEqual(p["drop"], [user.name.first_name, user.name.last_name])
        self.assertIn(" ".join(p["drop"]), text)
        self.assertGreater(len(r.passengers), 1)
        if p["fallback"] == "downgrade":
            self.names_payment(text, user, p["payment"])
            self.assertEqual(p["cabin"], DOWN[r.cabin])
            call = self.whole(r, p["cabin"], p["payment"])
            self.assertEqual(p["calls"], [call])
            self.assertLess(charged(self.dump, call), 0)
        else:
            call = {"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}}
            self.assertEqual(p["calls"] or [p["forbidden"]], [call])
            self.assertEqual(bool(p["calls"]), self.cancellable(r))

    def spec_insurance(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.assertEqual((p["calls"], p["forbidden"]), ([], None))
        self.assertEqual(p["add"], r.insurance == "no")

    def spec_balances(self, t, text, p):
        user = self.db.users[p["user"]]
        for source, said in p["sums"].items():
            found = [m.amount for m in user.payment_methods.values() if m.source == source]
            self.assertGreater(len(found), 1)
            self.assertEqual(said, str(int(sum(found))))
            self.assertIn(said, t.evaluation_criteria.communicate_info)
        self.assertEqual((p["calls"], p["forbidden"]), ([], None))

    def spec_scope_cancel(self, t, text, p):
        user = self.db.users[p["user"]]
        self.assertEqual(self.unclear(user), [])
        mine = [self.db.reservations[x] for x in user.reservations if self.db.reservations[x].status != "cancelled"]
        hit = [r for r in mine if self.upcoming(r)]
        if p["which"] == "alone":
            hit = [r for r in hit if len(r.passengers) == 1]
        elif p["which"] == "long":
            self.assertIn(f"{p['hours']} hours", text)
            hit = [r for r in hit if any(at(self.db.flights[f.flight_number].scheduled_arrival_time_est)
                                         - at(self.db.flights[f.flight_number].scheduled_departure_time_est)
                                         > 60 * p["hours"] for f in r.flights)]
        self.assertEqual(sorted(p["cancel"] + p["keep"]), sorted(r.reservation_id for r in hit))
        self.assertEqual(sorted(p["cancel"]), sorted(r.reservation_id for r in hit if self.cancellable(r)))
        self.assertEqual(p["calls"], [{"name": "cancel_reservation", "arguments": {"reservation_id": x}}
                                      for x in p["cancel"]])
        if p["keep"]:
            self.assertEqual(p["forbidden"], {"name": "cancel_reservation",
                                              "arguments": {"reservation_id": p["keep"][0]}})

    def spec_scope_downgrade(self, t, text, p):
        user = self.db.users[p["user"]]
        self.assertEqual([r for r in self.unclear(user) if r.cabin == "business"], [])
        hit = [self.db.reservations[x] for x in user.reservations if self.db.reservations[x].status != "cancelled"
               and self.db.reservations[x].cabin == "business" and self.upcoming(self.db.reservations[x])]
        self.assertEqual(p["reservations"], [r.reservation_id for r in hit])
        self.assertEqual(p["calls"], [self.whole(r, "economy", r.payment_history[0].payment_id) for r in hit])
        if t.evaluation_criteria.communicate_info and len(self.specs[t.id]) == 1:
            refund = -sum(charged(self.dump, c) for c in p["calls"])
            self.assertEqual(t.evaluation_criteria.communicate_info, [str(refund)])

    def spec_several(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        user = self.db.users[r.user_id]
        self.names_trip(text, r)
        self.names_payment(text, user, p["payment"])
        self.assertIn("order", text)
        cabin, total, calls = r.cabin, r.total_baggages, []
        passengers = [x.model_dump() for x in r.passengers]
        for op in p["ops"]:
            if op == "cabin":
                cabin = UP[cabin]
                calls.append(self.whole(r, cabin, p["payment"]))
            elif op == "passenger":
                old, new = p["replace"][:2], p["replace"][2]
                self.assertIn(new, [x.model_dump() for x in user.saved_passengers])
                self.assertIn(f"{' '.join(old)} with {new['first_name']} {new['last_name']}", text)
                self.assertIn(born(new["dob"]), text)  # or the user could not tell the agent
                passengers = [new if [x["first_name"], x["last_name"]] == old else x for x in passengers]
                calls.append({"name": "update_reservation_passengers",
                              "arguments": {"reservation_id": r.reservation_id, "passengers": passengers}})
            else:  # bags free by the cabin the reservation has at that point
                total += p["add"]
                nonfree = max(0, total - FREE[user.membership][cabin] * len(r.passengers))
                if "cabin" in p["ops"][p["ops"].index(op):]:  # an upgrade after them would make paid bags free
                    self.assertEqual(nonfree, 0)
                calls.append({"name": "update_reservation_baggages",
                              "arguments": {"reservation_id": r.reservation_id, "total_baggages": total,
                                            "nonfree_baggages": nonfree, "payment_id": p["payment"]}})
        self.assertEqual(p["calls"], calls)

    def spec_chain(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.assertEqual((r.cabin, r.insurance), ("basic_economy", "no"))
        self.assertFalse(self.cancellable(r))
        self.assertIn(dated(p["date"]), text)
        self.assertIn("insurance", text)
        self.assertEqual((p["calls"], p["forbidden"]),
                         ([], {"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}}))

    def spec_change_or_cancel(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        user = self.db.users[r.user_id]
        self.names_trip(text, r)
        self.assertEqual((r.cabin, r.insurance), ("basic_economy", "yes"))
        self.assertIn([user.name.first_name, user.name.last_name], [[x.first_name, x.last_name] for x in r.passengers])
        self.assertEqual(p["calls"], [{"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}}])
        expected = self.moved(text, r, p["journey"], p["date"], p["preference"], p["cabin"], p["payment"], p.get("hour"))
        self.assertEqual(p["forbidden"], expected)

    def spec_change_or_rebook(self, t, text, p):
        """The change is refused for basic economy; the trip is cancelled under insurance for the user's illness and the
        flights the change would have given are booked anew, for the same passengers and cabin."""
        r = self.db.reservations[p["reservation"]]
        user = self.db.users[r.user_id]
        self.names_trip(text, r)
        self.assertEqual((r.cabin, r.insurance, r.flight_type), ("basic_economy", "yes", "one_way"))
        self.assertIn([user.name.first_name, user.name.last_name], [[x.first_name, x.last_name] for x in r.passengers])
        self.assertRegex(text, COVERED_REASON)
        self.assertGreater(p["date"], r.flights[0].date)  # the illness keeps the user from the original date
        expected = self.moved(text, r, p["journey"], p["date"], p["preference"], p["cabin"], p["payment"], p.get("hour"))
        self.assertEqual(p["forbidden"], expected)
        flights = expected["arguments"]["flights"]
        price = sum(self.db.flights[f["flight_number"]].dates[f["date"]].prices[r.cabin] for f in flights)
        self.assertEqual(p["calls"], [
            {"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}},
            {"name": "book_reservation", "arguments": {
                "user_id": r.user_id, "origin": r.origin, "destination": r.destination, "flight_type": "one_way",
                "cabin": r.cabin, "flights": flights, "passengers": [x.model_dump() for x in r.passengers],
                "payment_methods": [{"payment_id": p["payment"], "amount": price * len(r.passengers)}],
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no"}}])

    def spec_reroute_rebook(self, t, text, p):
        """A new destination is refused; the business class trip is cancelled (any reason will do) and a one-way trip to
        the new destination on the same day is booked, on the flights the preference singles out, for the same
        passengers and cabin, paid with one card."""
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.assertEqual((r.cabin, r.flight_type), ("business", "one_way"))
        self.assertNotEqual(p["destination"], r.destination)
        self.assertEqual(p["date"], r.flights[0].date)
        self.assertIn(CITY[p["destination"]], text)
        found = search(self.db, r.origin, p["destination"], p["date"], r.cabin, len(r.passengers))
        hit = pick(found, p["preference"], hour=p.get("hour"))
        self.assertEqual(len(hit), 1)
        flights = [{"flight_number": n, "date": d} for n, d in hit[0]]
        price = sum(self.db.flights[f["flight_number"]].dates[f["date"]].prices[r.cabin] for f in flights)
        self.names_payment(text, self.db.users[r.user_id], p["payment"])
        self.assertEqual(p["calls"], [
            {"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}},
            {"name": "book_reservation", "arguments": {
                "user_id": r.user_id, "origin": r.origin, "destination": p["destination"], "flight_type": "one_way",
                "cabin": r.cabin, "flights": flights, "passengers": [x.model_dump() for x in r.passengers],
                "payment_methods": [{"payment_id": p["payment"], "amount": price * len(r.passengers)}],
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no"}}])
        self.assertEqual(p["forbidden"], {"name": "update_reservation_flights", "arguments": {
            "reservation_id": r.reservation_id, "cabin": r.cabin, "flights": flights, "payment_id": p["payment"]}})

    def spec_book_again(self, t, text, p):
        user = self.db.users[p["user"]]
        before = self.db.reservations[p["reservation"]]
        self.assertEqual(before.user_id, p["user"])
        self.assertEqual((before.flight_type, len(before.flights)), ("one_way", 1))
        self.names_trip(text, before)
        call = p["calls"][0]["arguments"]
        self.assertEqual(call["flights"], [{"flight_number": before.flights[0].flight_number, "date": p["date"]}])
        self.assertEqual((call["cabin"], call["origin"], call["destination"]),
                         (before.cabin, before.origin, before.destination))
        me = {"first_name": user.name.first_name, "last_name": user.name.last_name, "dob": user.dob}
        if call["passengers"] == [me]:  # the user, on another day
            self.assertIn(dated(p["date"]), text)
        else:  # someone saved in the user's profile, on the same flight and day
            self.assertEqual(p["date"], before.flights[0].date)
            self.assertIn("same", text)
            friend = call["passengers"][0]
            self.assertIn(friend, [x.model_dump() for x in user.saved_passengers])
            self.assertIn(f"{friend['first_name']} {friend['last_name']}", text)
            self.assertEqual(len(call["passengers"]), 1)
        self.names_payment(text, user, p["payment"][0])

    def spec_change_both(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.names_payment(text, self.db.users[r.user_id], p["payment"])
        js = legs(r)
        self.assertEqual(len(js), 2)
        shifts = {int(d[-2:]) - int(j[0].date[-2:]) for j, d in zip(js, p["dates"])}
        self.assertEqual(len(shifts), 1)  # both journeys move by the same number of days
        shift = {1: "one day later", 2: "two days later", -1: "one day earlier", -2: "two days earlier"}[shifts.pop()]
        self.assertIn(shift, text)
        flights = []
        for j, date in zip(js, p["dates"]):
            new = pick(search(self.db, j[0].origin, j[-1].destination, date, r.cabin, len(r.passengers)),
                       p["preference"], None, p.get("hour"))
            self.assertEqual(len(new), 1)
            flights += new[0]
        self.assertEqual(p["calls"], [{"name": "update_reservation_flights",
                                       "arguments": {"reservation_id": r.reservation_id, "cabin": r.cabin,
                                                     "payment_id": p["payment"],
                                                     "flights": [{"flight_number": n, "date": d} for n, d in flights]}}])

    def spec_upgrade_cancel(self, t, text, p):
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.names_payment(text, self.db.users[r.user_id], p["payment"])
        self.assertFalse(self.cancellable(r))
        self.assertNotEqual(r.insurance, "yes")
        self.assertTrue(self.upcoming(r))
        self.assertIn("business", text)
        cancel = {"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}}
        self.assertEqual(p["calls"], [self.whole(r, "business", p["payment"]), cancel])
        self.assertEqual(p["forbidden"], cancel)
        self.claims(p, r, granted=False)

    # ---------------------------------------------------------------- a value the agent works out decides the next step

    def spec_cabin_bags(self, t, text, p):
        """The whole reservation goes up one cabin only within the user's limit; the bags added after that are free by
        the cabin it has then, and so many that the upgrade changes how many are paid for."""
        r = self.db.reservations[p["reservation"]]
        user = self.db.users[r.user_id]
        self.names_trip(text, r)
        self.names_payment(text, user, p["payment"])
        self.assertTrue(self.upcoming(r))
        self.assertEqual((p["cabin"], r.nonfree_baggages), (UP[r.cabin], 0))
        self.assertIn(f"${p['budget']}", text)
        self.assertIn(f"{('one', 'two', 'three')[p['add'] - 1]} checked bag", text)
        if "partial" in p:  # the journey or passenger the user first asks to upgrade alone, which the policy refuses
            self.assertIn(p["partial"], text)
        upgrade = self.whole(r, p["cabin"], p["payment"])
        cost = charged(self.dump, upgrade)
        self.assertGreaterEqual(abs(cost - p["budget"]), 10)
        within = cost <= p["budget"]
        total = r.total_baggages + p["add"]
        paid = {c: max(0, total - FREE[user.membership][c] * len(r.passengers)) for c in (r.cabin, p["cabin"])}
        self.assertNotEqual(paid[r.cabin], paid[p["cabin"]])
        self.assertEqual(within, cost + 50 * paid[p["cabin"]] <= p["budget"])  # with the bags counted in, the same
        # and with the bag fees the upgrade saves taken off it
        self.assertEqual(within, cost - 50 * (paid[r.cabin] - paid[p["cabin"]]) <= p["budget"])
        bags = {"name": "update_reservation_baggages",
                "arguments": {"reservation_id": r.reservation_id, "total_baggages": total,
                              "nonfree_baggages": paid[p["cabin"] if within else r.cabin], "payment_id": p["payment"]}}
        self.assertEqual(p["calls"], [upgrade, bags] if within else [bags])
        if not p.get("dropped"):
            self.assertEqual(p["forbidden"], None if within else upgrade)
        self.states(t, cost)

    def spec_change_upgrade_budget(self, t, text, p):
        """A journey moved and wanted in business class, which only the whole reservation can be: all of it in business
        class with the change, in one call, within the user's limit; else the change alone in economy or nothing, as
        the user says. The preference picks the same flights in either cabin."""
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.assertEqual((r.cabin, r.flight_type), ("economy", "round_trip"))
        self.assertTrue(self.upcoming(r))
        self.assertIn("business", t.user_scenario.instructions.reason_for_call)
        self.assertIn(f"${p['budget']}", text)
        self.assertRegex(text, r"stay in economy|keep economy|changed, in economy" if p["otherwise"] == "change"
                         else r"as it is|any change at all")
        business = self.moved(text, r, p["journey"], p["date"], p["preference"], "business", p["payment"], p.get("hour"))
        economy = self.moved(text, r, p["journey"], p["date"], p["preference"], "economy", p["payment"], p.get("hour"))
        self.assertEqual(business["arguments"]["flights"], economy["arguments"]["flights"])
        cost = charged(self.dump, business)
        self.assertGreaterEqual(abs(cost - p["budget"]), 10)
        within = cost <= p["budget"]
        self.assertEqual(p["calls"], [business] if within else [economy] if p["otherwise"] == "change" else [])
        if not p.get("dropped"):
            self.assertEqual(p["forbidden"], None if within else business)
        self.states(t, cost)
        if within and len(self.specs[t.id]) == 1:  # done as two calls, either one first, the payment history differs
            said = [AssistantMessage(role="assistant", content=f"That comes to ${int(x):,}.")
                    for x in t.evaluation_criteria.communicate_info]
            for two in ([self.whole(r, "business", p["payment"]), business], [economy, business]):
                self.assertEqual(reward(t, self.dump, replay(two, self.dump) + said), 0.0)

    def spec_book_same_cap(self, t, text, p):
        """The same nonstop flight as on one of the user's one-way reservations, for the user and a saved passenger if
        the fare for both is within the user's limit, else for the user alone, who is within it; no bags, no
        insurance."""
        user = self.db.users[p["user"]]
        r = self.db.reservations[p["reservation"]]
        self.assertEqual((r.user_id, r.flight_type, len(r.flights), r.cabin), (p["user"], "one_way", 1, p["cabin"]))
        self.names_trip(text, r)
        me = {"first_name": user.name.first_name, "last_name": user.name.last_name, "dob": user.dob}
        friend = p["passengers"][1]
        self.assertEqual((p["passengers"][0], len(p["passengers"])), (me, 2))
        self.assertIn(friend, [x.model_dump() for x in user.saved_passengers])
        self.assertIn(f"{friend['first_name']} {friend['last_name']}", text)
        on = [[x.first_name, x.last_name] for x in r.passengers]
        self.assertNotIn([friend["first_name"], friend["last_name"]], on + [[me["first_name"], me["last_name"]]])
        if "profile" not in text:  # otherwise the agent reads dates of birth from the profile
            for x in p["passengers"]:
                self.assertIn(born(x["dob"]), text)
        if p["date"] == r.flights[0].date:  # the same day: neither of them is on that reservation
            self.assertRegex(text, "same (day|date)")
            self.assertNotIn([me["first_name"], me["last_name"]], on)
        else:
            self.assertIn(dated(p["date"]), text)
            self.assertIn({"basic_economy": "basic economy", "economy": "economy", "business": "business"}[r.cabin], text)
        self.assertRegex(text, "no checked bags|not check any bags|not need any checked bags|No checked bags")
        self.assertRegex(text, "not want travel insurance|decline travel insurance|not buy travel insurance")
        self.names_payment(text, user, p["payment"][0])
        number = r.flights[0].flight_number
        fare = self.db.flights[number].dates[p["date"]].prices[r.cabin]
        self.assertIn(f"${p['limit']}", text)
        self.assertGreaterEqual(abs(2 * fare - p["limit"]), 10)
        self.assertLessEqual(fare, p["limit"])
        within = 2 * fare <= p["limit"]

        def booking(people):
            return {"name": "book_reservation", "arguments": {
                "user_id": p["user"], "origin": r.origin, "destination": r.destination, "flight_type": "one_way",
                "cabin": r.cabin, "flights": [{"flight_number": number, "date": p["date"]}], "passengers": people,
                "payment_methods": [{"payment_id": p["payment"][0], "amount": fare * len(people)}],
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no"}}

        self.assertEqual(p["calls"], [booking(p["passengers"] if within else [me])])
        if not p.get("dropped"):
            self.assertEqual(p["forbidden"], None if within else booking(p["passengers"]))
        info = t.evaluation_criteria.communicate_info
        if info and len(self.specs[t.id]) == 1:
            self.assertEqual(info, [str(2 * fare)])

    def spec_book_pay_by_total(self, t, text, p):
        """A booking, checked as the kind of booking it is, paid by the plan its total picks: the first if the total is
        more than the user's threshold, the other if not. A plan uses up a travel certificate or a gift card worth less
        than the total, or neither, and puts the rest on a credit card."""
        user = self.db.users[p["user"]]
        (self.spec_book_again if "reservation" in p else self.spec_book)(t, text, p)
        call = p["calls"][0]["arguments"]
        total = sum(x["amount"] for x in call["payment_methods"])
        self.assertIn(f"${p['threshold']}", text)
        self.assertGreaterEqual(abs(total - p["threshold"]), 10)
        self.assertNotEqual(p["above"], p["at_most"])

        def paid(plan):
            *stored, card = plan
            self.assertEqual(user.payment_methods[card].source, "credit_card")
            self.assertLessEqual(len(stored), 1)
            for k in stored:
                self.assertIn(user.payment_methods[k].source, ("certificate", "gift_card"))
                self.assertLess(user.payment_methods[k].amount, total)
            for k in plan:
                self.names_payment(text, user, k)
            used = [{"payment_id": k, "amount": user.payment_methods[k].amount} for k in stored]
            return used + [{"payment_id": card, "amount": total - sum(x["amount"] for x in used)}]

        chosen, other = (p["above"], p["at_most"]) if total > p["threshold"] else (p["at_most"], p["above"])
        self.assertEqual(p["payment"], chosen)
        self.assertEqual(call["payment_methods"], paid(chosen))
        forbidden = {"name": "book_reservation", "arguments": {**call, "payment_methods": paid(other)}}
        if not p.get("dropped"):
            self.assertEqual(p["forbidden"], forbidden)

    def spec_basic_upgrade_budget(self, t, text, p):
        """A basic economy change, refused, with the user's fallback: an upgrade to economy with the current flights,
        then the change, both paid with the same method, only if the two together cost at most the user's limit."""
        r = self.db.reservations[p["reservation"]]
        self.names_trip(text, r)
        self.assertEqual(r.cabin, "basic_economy")
        self.assertTrue(self.upcoming(r))
        self.assertIn(f"${p['budget']}", text)
        change = self.moved(text, r, p["journey"], p["date"], p["preference"], "economy", p["payment"], p.get("hour"))
        upgrade = self.whole(r, "economy", p["payment"])
        env = get_environment(db=FlightDB.model_validate(self.dump))
        env.make_tool_call(tool_name=upgrade["name"], **upgrade["arguments"])
        cost = charged(self.dump, upgrade) + charged(env.tools.db.model_dump(), change)
        self.assertGreaterEqual(abs(cost - p["budget"]), 10)
        within = cost <= p["budget"]
        self.assertEqual(p["calls"], [upgrade, change] if within else [])
        self.assertEqual(p["forbidden"], None if within else upgrade)
        self.states(t, cost)

    def spec_cancel_total(self, t, text, p):
        """A cancellation one of the rules allows, and the sum of the payments recorded on the user's other upcoming
        reservations; some of the user's other reservations are not upcoming, and none could be counted either way."""
        r = self.db.reservations[p["reservation"]]
        user = self.db.users[r.user_id]
        self.names_trip(text, r)
        self.assertIn(p["basis"], ALLOWED)
        self.assertFalse(flown(self.db, r))
        self.assertTrue(self.cancellable(r) or r.insurance == "yes" and COVERED_REASON.search(text))
        call = {"name": "cancel_reservation", "arguments": {"reservation_id": r.reservation_id}}
        self.assertEqual((p["calls"], p["forbidden"]), ([call], None))
        self.claims(p, r, granted=True)
        self.assertEqual(self.unclear(user), [])
        mine = [self.db.reservations[x] for x in user.reservations
                if x != r.reservation_id and self.db.reservations[x].status != "cancelled"]
        others = [x for x in mine if self.upcoming(x)]
        self.assertLess(len(others), len(mine))  # the agent has to leave some out
        self.assertEqual(p["reservations"], sorted(x.reservation_id for x in others))
        self.assertIn("upcoming", t.user_scenario.instructions.reason_for_call)
        total = sum(x.amount for o in others for x in o.payment_history)
        self.assertEqual(t.evaluation_criteria.communicate_info, [str(int(total))])


if __name__ == "__main__":
    unittest.main()
