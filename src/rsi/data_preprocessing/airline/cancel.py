"""Cancel flight: a cancellation the policy allows (booked within 24 hours, business class, a flight the airline
cancelled, insurance with a health or weather reason) or refuses, and the requests that start with one: falling back on
a flight change or on an upgrade to business class, bargaining, a chain of refusals, and every upcoming reservation that
fits a condition at once."""
from rsi.data_preprocessing.airline.change import WHICH, change
from rsi.data_preprocessing.airline.facts import (CITY, CLAIM, FIRST_DAY, LAST_DAY, PLANS, SICK, WHY, _call,
                                                  _difference, _flights, borderline, day, facts, minutes, pay, payer,
                                                  trip)
from rsi.data_preprocessing.common import Part

CANCEL = ("You want to cancel {trip}.", "You are calling to cancel {trip}.", "You need to cancel {trip}.")
CANCEL_REST = ("You have already flown part of {trip}, and you want to cancel the rest of it.",
               "Part of {trip} is already behind you; you want the remaining flights cancelled.",
               "You want to cancel what is left of {trip}, part of which you have already flown.")
WHY_ALL = ("If the agent asks why, you say that {why}.", "Your reason for cancelling is that {why}.",
           "When asked for the reason, you explain that {why}.")
WEATHER = ("a hurricane is forecast to hit {city} on your travel dates",
           "a severe snowstorm is forecast for {city} on your travel dates")
AIRLINE_WHY = ("the airline cancelled one of your flights and the trip no longer works for you",
               "one of your flights was cancelled by the airline, so you no longer want the trip")
# The policy has the agent offer compensation only when the user asks for it.
TAKE_OFFER = ("You do not ask for compensation, but if the agent offers some, you accept it.",
              "If the agent offers you compensation, you take it, though you do not ask for any.",
              "You are not calling for compensation, but you would accept it if offered.")
YESTERDAY = ("You booked this trip yesterday, so you expect to be able to cancel it for free.",
             "Mention that you only booked it yesterday.", "Point out that you made the booking just yesterday.")
NO_ALTERNATIVE = ("You do not want to change the flights or the cabin of that trip instead.",
                  "For that trip you are only interested in a cancellation, not in a change.",
                  "If the agent suggests changing that trip instead of cancelling it, you decline.")
CANCEL_PURPOSE = {
    "recent": "cancellation the policy allows: booked within the last 24 hours",
    "business": "cancellation the policy allows: a business class reservation",
    "airline": "cancellation the policy allows: the airline cancelled a flight; no compensation unless asked for",
    "insured": "cancellation the policy allows: travel insurance and a {reason} reason",
    "none": "cancellation no rule allows",
    "insured_other": "cancellation the policy does not allow: travel insurance, but not a health or weather reason",
    "yesterday": "cancellation the policy does not allow: booked the day before, but more than 24 hours ago",
    "flown": "cancellation the policy does not allow: part of the trip has been flown",
}
CANCEL_POOL = {
    "recent": lambda r, f: not f["flown"] and f["age"] < 23 and r.cabin != "business",
    "business": lambda r, f: not f["flown"] and f["age"] > 25 and r.cabin == "business" and not f["airline"],
    "airline": lambda r, f: not f["flown"] and f["airline"],
    "insured": lambda r, f: (not f["flown"] and f["age"] > 25 and r.cabin != "business" and not f["airline"]
                             and r.insurance == "yes"),
    "none": lambda r, f: (not f["flown"] and f["age"] > 25 and r.cabin != "business" and not f["airline"]
                          and r.insurance == "no"),
    "insured_other": lambda r, f: (not f["flown"] and f["age"] > 25 and r.cabin != "business" and not f["airline"]
                                   and r.insurance == "yes"),
    "yesterday": lambda r, f: not f["flown"] and 25 < f["age"] < 39 and r.cabin != "business" and not f["airline"],
    "flown": lambda r, f: (f["flown"] and not f["airline"] and not f["delayed"]
                           and any(s == "available" for s in f["statuses"])),
}


def cancel(db, r, rng, basis, cabin_claim=True):
    """Cancel flight: allowed if booked within 24 hours, in business class, after the airline cancelled a flight, or
    with travel insurance for a health or weather reason, and never once part of the trip has been flown. Without
    cabin_claim, a refused request never claims the business cabin."""
    text = trip(db, r, rng)
    if text is None:
        return None
    f = facts(db, r)
    spec = {"family": "cancel", "reservation": r.reservation_id, "basis": basis, "reason": "plans"}
    why = rng.choice(PLANS)
    if basis == "insured":
        spec["reason"] = "health" if f["passenger"] and rng.random() < 0.6 else "weather"
        why = rng.choice(SICK) if spec["reason"] == "health" else rng.choice(WEATHER).format(city=CITY[r.destination])
    elif basis == "airline":
        spec["reason"], why = "airline", rng.choice(AIRLINE_WHY)
    lines = [rng.choice(WHY).format(why=why)]
    purpose = CANCEL_PURPOSE[basis].format(reason=spec["reason"])
    call = _call("cancel_reservation", reservation_id=r.reservation_id)
    if basis in ("recent", "business", "airline", "insured"):
        if basis == "airline" and f["eligible"] and rng.random() < 0.5:
            lines.append(rng.choice(TAKE_OFFER))
        # the airline's cancellation is already the user's reason; the official tasks leave the 24-hour rule for the
        # agent to find
        if basis != "airline" and rng.random() < (0.2 if basis == "recent" else 0.5):
            spec["claims"] = [basis]
            lines.append(rng.choice(CLAIM[basis]))
        return Part(spec, purpose, rng.choice(CANCEL).format(trip=text), lines, [call],
                    tags={"cancel_ok"} | ({"verify_claim"} if spec.get("claims") else set()))
    if basis == "flown":
        return Part(spec, purpose, rng.choice(CANCEL_REST).format(trip=text), lines, forbidden=call,
                    tags={"cancel_flown", "transfer"}, transfer=True)
    false = [k for k, holds in (("recent", f["age"] < 24), ("insured", r.insurance == "yes"),
                                ("business", r.cabin == "business"), ("airline", f["airline"]))
             if not holds and (cabin_claim or k != "business")]
    if basis == "yesterday":
        lines.append(rng.choice(YESTERDAY))
        false.remove("recent")
    spec["claims"] = rng.sample(false, rng.choice((0, 1, 1, 2)))
    lines += [rng.choice(CLAIM[k]) for k in spec["claims"]] + [rng.choice(NO_ALTERNATIVE)]
    if spec["claims"]:
        purpose += ", despite the user's false claims"
    # the agent checks what the user claims (and when a booking "from yesterday" was made) against the reservation
    return Part(spec, purpose, rng.choice(CANCEL).format(trip=text), lines, forbidden=call,
                tags={"cancel_refuse"} | ({"false_claim", "verify_claim"} if spec["claims"] else set())
                | ({"verify_claim"} if basis == "yesterday" else set()))


FALLBACK = ("If the agent says the reservation cannot be cancelled, you would rather move {which} to {date}.",
            "Should cancelling not be possible, you instead want to change {which} to {date}.",
            "If cancelling is not allowed, you would like to fly on {date} instead, for {which}.")


def cancel_or_change(db, r, rng):
    """A cancellation no rule allows, with a flight change, which the policy allows, as the user's fallback."""
    refused = cancel(db, r, rng, "none", cabin_claim=False)  # a claimed business cabin would carry into the change
    moved = change(db, r, rng, "ok")
    if refused is None or moved is None:
        return None
    j = moved.spec["journey"]
    which = "that trip" if r.flight_type == "one_way" else rng.choice(WHICH[j]).format(trip="that trip")
    lines = [x for x in refused.lines if x not in NO_ALTERNATIVE]
    lines.append(rng.choice(FALLBACK).format(which=which, date=day(moved.spec["date"])))
    spec = {**moved.spec, "fallback_for": refused.spec}
    return Part(spec, "cancellation no rule allows, then the flight change the user falls back on", refused.request,
                lines + moved.lines, moved.calls, refused.forbidden, moved.info, moved.ask,
                tags=refused.tags | moved.tags | {"fallback"})


UPGRADE_CANCEL = ("If the agent says the reservation cannot be cancelled, ask it to upgrade the reservation to business "
                  "class first, paying with {pay}, and then cancel it.",
                  "Should cancelling be refused, you want the reservation moved up to business class with {pay} and "
                  "cancelled right after that.",
                  "If it cannot be cancelled as it is, have it upgraded to business class (charge {pay}) and then "
                  "cancelled.")


def upgrade_cancel(db, r, rng):
    """A cancellation no rule allows, with the user's fallback: an upgrade to business class, which any reservation whose
    flights are all to come may take, after which the cancellation is allowed."""
    refused = cancel(db, r, rng, "none", cabin_claim=False)
    if refused is None:
        return None
    flights = [(f.flight_number, f.date) for f in r.flights]
    if any(db.flights[n].dates[d].available_seats["business"] < len(r.passengers) for n, d in flights):
        return None
    user = db.users[r.user_id]
    pid = payer(user, _difference(db, r, "business", flights), rng)
    if pid is None:
        return None
    lines = [x for x in refused.lines if x not in NO_ALTERNATIVE]
    lines.append(rng.choice(UPGRADE_CANCEL).format(pay=pay(user, pid)))
    calls = [_call("update_reservation_flights", reservation_id=r.reservation_id, cabin="business",
                   flights=_flights(flights), payment_id=pid), refused.forbidden]
    spec = {**refused.spec, "family": "upgrade_cancel", "payment": pid}
    return Part(spec, "a cancellation no rule allows, then an upgrade to business class and the cancellation",
                refused.request, lines, calls, refused.forbidden,
                tags=refused.tags | {"cabin_change", "cabin_diff", "mod_pay", "cancel_ok", "fallback", "order"})


NEGOTIATE = ("If the agent will not refund you, ask for a travel voucher or at least half of the fare back.",
             "When refused, try to negotiate: a gift card, a voucher, or even a 10% refund would do.",
             "Do not take no for an answer: ask for a partial refund or a credit for future travel instead.")


def cancel_negotiate(db, r, rng):
    """Cancel flight, which no rule allows, with the user bargaining for a partial refund or a voucher, which the
    policy does not offer either."""
    part = cancel(db, r, rng, "none")
    if part is None:
        return None
    part.lines.append(rng.choice(NEGOTIATE))
    part.spec["negotiate"] = True
    part.tags |= {"deny_other", "pressure"}
    part.purpose += "; no partial refund or voucher either"
    return part


SCOPE_CANCEL = {
    "all": ("You'd like every upcoming trip of yours cancelled.", "You need every one of your upcoming trips cancelled.",
            "Please cancel all the trips you still have coming up."),
    "alone": ("You want to cancel all of your upcoming reservations that have only one passenger.",
              "Cancel every upcoming trip of yours on which only one person flies.",
              "You'd like the agent to cancel your upcoming reservations with a single passenger."),
    "long": ("You want to cancel all of your upcoming reservations that include a flight longer than {h} hours.",
             "Cancel every upcoming trip of yours with any flight over {h} hours long.",
             "You no longer want any upcoming reservation with a flight of more than {h} hours."),
}
DURATIONS = ("You want the agent to tell you how long each of those flights is before cancelling.",
             "Ask the agent for the duration of the flights so that you can check its choice.",
             "Have the agent tell you the flight times it found.")
PROCEED = ("Even if the agent says some of them will not be refunded, you want to go ahead.",
           "You understand some may not be refundable; you still want them cancelled.", "")
FIGURE_OUT = ("You do not know your reservation ids; let the agent work out which ones these are.",
              "You don't remember the details of each trip, so the agent has to find the right ones.",
              "Have the agent look up which reservations fit; you will not list them yourself.")


def flight_hours(db, f):
    """A flight's scheduled duration in hours."""
    x = db.flights[f.flight_number]
    return (minutes(x.scheduled_arrival_time_est) - minutes(x.scheduled_departure_time_est)) / 60


def scope_cancel(db, user_id, rng, which):
    """Cancel flight, for every upcoming reservation that fits a condition: each is cancelled only if a cancellation
    rule allows it for the user's reason (a change of plans), whatever the user says about refunds."""
    user = db.users[user_id]
    mine = [db.reservations[x] for x in user.reservations if db.reservations[x].status != "cancelled"]
    upcoming = [r for r in mine if facts(db, r)["upcoming"]]
    # a past trip the airline cancelled and the user never flew could still be cancelled for a refund, which a user
    # cancelling "everything" might accept
    if any(borderline(db, r) or (facts(db, r)["airline"] and not facts(db, r)["flown"]) for r in mine):
        return None
    if which == "alone":
        hit = [r for r in upcoming if len(r.passengers) == 1]
    elif which == "long":
        lengths = sorted({flight_hours(db, f) for r in upcoming for f in r.flights})
        cut = [h for h in range(2, 7) if h not in lengths]
        if not cut:
            return None
        h = rng.choice(cut)
        hit = [r for r in upcoming if any(flight_hours(db, f) > h for f in r.flights)]
    else:
        hit = upcoming
    if len(hit) < 2 or (which != "all" and len(hit) == len(upcoming)):  # the condition has to rule some out
        return None
    ok = [r for r in hit if facts(db, r)["age"] < 23 or r.cabin == "business" or facts(db, r)["airline"]]
    if any(23 <= facts(db, r)["age"] <= 25 for r in hit):  # too close to the 24-hour line
        return None
    refused = [r for r in hit if r not in ok]
    spec = {"family": "scope_cancel", "user": user_id, "which": which, "cancel": [r.reservation_id for r in ok],
            "keep": [r.reservation_id for r in refused], **({"hours": h} if which == "long" else {})}
    lines = [rng.choice(FIGURE_OUT), rng.choice(WHY_ALL).format(why=rng.choice(PLANS)), rng.choice(PROCEED)]
    tags = {"scope_all", "filter_res", "find_res", "agent_decides"} | ({"cancel_ok"} if ok else set())
    tags |= ({"cancel_refuse"} if refused else set()) | ({"duration"} if which == "long" else set())
    if which == "long":
        lines.append(rng.choice(DURATIONS))
        tags.add("info_attr")
    calls = [_call("cancel_reservation", reservation_id=r.reservation_id) for r in ok]
    forbidden = _call("cancel_reservation", reservation_id=refused[0].reservation_id) if refused else None
    return Part(spec, "cancelling every reservation that fits, each only as the cancellation rules allow",
                rng.choice(SCOPE_CANCEL[which]).format(h=h if which == "long" else ""), [x for x in lines if x], calls,
                forbidden, tags=tags)


CHAIN = ("If you cannot get a refund, you want to move the trip to {date} instead.",
         "Should cancelling be refused, ask to change the flight to {date}.",
         "If the agent will not cancel it, try to have the trip moved to {date}.")
CHAIN_INSURE = ("If that does not work either, ask to add travel insurance to the booking.",
                "If the date cannot be changed, you at least want travel insurance added.",
                "Failing that, try to get travel insurance put on the reservation.")
NO_CABIN = ("Under no circumstances will you upgrade the cabin of that trip.",
            "You will not pay for a cabin upgrade on that trip.", "Upgrading the cabin of that trip is out of the question.")


def chain(db, r, rng):
    """Three requests in turn, each refused: a cancellation no rule allows, then a date change of basic economy, then
    travel insurance after booking."""
    if r.cabin != "basic_economy" or r.insurance != "no" or r.flight_type != "one_way":
        return None
    part = cancel(db, r, rng, "none", cabin_claim=False)
    if part is None:
        return None
    d = int(r.flights[0].date[-2:]) + rng.choice((-2, -1, 1, 2))
    if not FIRST_DAY <= d <= LAST_DAY:
        return None
    part.lines = [x for x in part.lines if x not in NO_ALTERNATIVE]
    part.lines += [rng.choice(CHAIN).format(date=day(f"2024-05-{d:02d}")), rng.choice(CHAIN_INSURE), rng.choice(NO_CABIN)]
    part.spec.update({"family": "chain", "date": f"2024-05-{d:02d}"})
    part.tags |= {"fallback", "basic_no_change", "mod_flights", "ins_no_add"}
    part.purpose = "a cancellation, a basic economy date change and insurance after booking, each refused in turn"
    return part
