"""Refunds and compensation: a certificate for a flight the airline cancelled, or for a delayed one once the reservation
is changed, only for an eligible user who asks; none for a delay alone, an ineligible user or a flight that was not
cancelled. Also the sums of the user's gift card and travel certificate balances."""
from rsi.data_preprocessing.airline.change import MOVE, PREFER, WHICH, _new_journey
from rsi.data_preprocessing.airline.facts import (CABIN, CLAIM, PAY_CHANGE, _call, _difference, _flights, clock, day,
                                                  disrupted, facts, journeys, pay, payer, pref_tags, sayable, status, trip)
from rsi.data_preprocessing.common import Part

COMPLAIN = {
    "cancelled": ("You are calling because the airline cancelled flight {flight} on {date}, part of {trip}, and you "
                  "want compensation.",
                  "You want compensation because flight {flight} on {date}, part of {trip}, was cancelled by the "
                  "airline.",
                  "You are upset that flight {flight} on {date} on {trip} was cancelled, and you ask what compensation "
                  "you will get."),
    "delayed": ("You are calling because flight {flight} on {date}, part of {trip}, is delayed, and you want "
                "compensation.",
                "You want compensation because flight {flight} on {date}, part of {trip}, has been delayed.",
                "You are upset that flight {flight} on {date} on {trip} is delayed, and you ask what compensation you "
                "will get."),
}
KEEP = ("You do not want to change or cancel that trip.", "You want to keep that trip exactly as it is.",
        "Changing or cancelling that trip is not something you want.")
MISCOUNT = ("If the agent asks how many passengers were on the reservation, say {n}; this is wrong, and you admit it if "
            "the agent corrects you.",
            "You believe {n} people are on that reservation and say so if asked, but you accept the agent's correction.",
            "When asked about the number of passengers, you insist once that there were {n}, then admit you may be "
            "wrong.")
WHY_DELAYED = ("You also want to know why the flight was delayed.", "Ask the agent what caused the delay.")
WAIT_FOR_OFFER = ("Do not ask for compensation right away: complain first and see whether the agent offers something; "
                  "only after a few messages ask for it yourself.",
                  "Start by complaining without asking for anything; try to get the agent to be the one who offers "
                  "compensation, and ask explicitly only if it does not.",
                  "First just describe how upset you are and wait for an offer; if none comes, ask for compensation.")


def compensation(db, r, rng, kind):
    """Refunds and compensation: when the user asks for it, a $100 certificate per passenger for a flight the airline
    cancelled, or $50 per passenger for a delayed flight once the reservation is changed or cancelled, and only for a
    silver or gold member, an insured reservation or business class. kind is "cancelled" (granted), "ineligible",
    "delayed" (kept as it is), "delayed_change", or "false" (the user claims the airline cancelled a flight that it did
    not)."""
    text = trip(db, r, rng)
    if text is None:
        return None
    f = facts(db, r)
    user = db.users[r.user_id]
    others = [x for x in disrupted(db, r.user_id) if x != r.reservation_id]
    if others or (kind == "false" and (f["airline"] or f["delayed"])):
        return None  # the reservation the user names is the only one that could grant a certificate
    claimed = "delayed" if kind in ("delayed", "delayed_change") else "cancelled"
    hit = [x for x in r.flights if (status(db, x) == claimed) != (kind == "false")]
    if not hit:
        return None
    flight = rng.choice(hit)
    spec = {"family": "compensation", "reservation": r.reservation_id, "flight": flight.flight_number,
            "date": flight.date, "claim": claimed}
    request = rng.choice(COMPLAIN[claimed]).format(flight=flight.flight_number, date=day(flight.date), trip=text)
    certificate = _call("send_certificate", user_id=r.user_id,
                        amount=(100 if claimed == "cancelled" else 50) * len(r.passengers))
    holds = {"gold": user.membership == "gold", "insured": r.insurance == "yes", "business": r.cabin == "business"}
    true = [k for k, h in holds.items() if h]
    claims = []
    if kind in ("cancelled", "delayed_change") and true and rng.random() < 0.5:
        spec["claims"] = [rng.choice(true)]
        claims = [rng.choice(CLAIM[spec["claims"][0]])]
    extra = set()
    if rng.random() < 0.5:  # the certificate goes by the reservation's passengers, not the user's count
        spec["miscount"] = len(r.passengers) + 1
        claims.append(rng.choice(MISCOUNT).format(n=spec["miscount"]))
        extra = {"wrong_info", "verify_claim"}
    if kind == "delayed_change":
        if not f["eligible"] or r.cabin == "basic_economy" or f["flown"]:
            return None
        delayed = next(k for k, jj in enumerate(journeys(r)) if flight in jj)
        moved = _new_journey(db, r, rng, r.cabin, ("cheapest nonstop", "earliest nonstop", "cheapest"), delayed)
        if moved is None:
            return None
        j, date, preference, hour, legs, flights = moved
        pid = payer(user, _difference(db, r, r.cabin, flights), rng)
        if pid is None:
            return None
        spec.update({"journey": j, "date_new": date, "preference": preference, "payment": pid,
                     **({"hour": hour} if hour is not None else {})})
        which = "that trip" if r.flight_type == "one_way" else rng.choice(WHICH[j]).format(trip="that trip")
        request += " " + rng.choice(MOVE).format(which=which, date=day(date))
        lines = [rng.choice(PREFER[preference]).format(cabin=CABIN[r.cabin], date=day(date),
                                                       hour=clock(hour) if hour is not None else ""),
                 rng.choice(PAY_CHANGE).format(pay=pay(user, pid))] + claims
        change_call = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=r.cabin,
                            flights=_flights(flights), payment_id=pid)
        return Part(spec, "compensation for a delayed flight after the reservation is changed", request, lines,
                    [change_call, certificate],
                    tags={"comp_ok", "comp_delay", "status_check", "mod_flights", "mod_pay"} | pref_tags(preference, legs)
                    | ({"verify_claim"} if claims else set()) | extra)
    lines = [rng.choice(KEEP)] + claims
    if kind == "cancelled":
        tags = {"comp_ok", "status_check"} | ({"verify_claim"} if claims else set()) | extra
        if rng.random() < 0.4:  # the agent offers nothing until the user asks
            lines.append(rng.choice(WAIT_FOR_OFFER))
            tags.add("comp_no_proactive")
        return Part(spec, "compensation the policy grants for a cancelled flight", request, lines, [certificate],
                    tags=tags)
    tags = {"comp_refuse", "status_check"} | extra
    if kind == "ineligible":
        false = [k for k, h in holds.items() if not h]
        spec["claims"] = rng.sample(false, rng.choice((0, 1, 1, 2)))
        lines += [rng.choice(CLAIM[k]) for k in spec["claims"]]
        tags |= {"false_claim", "verify_claim"} if spec["claims"] else set()
        purpose = "compensation the policy does not grant: regular member, no insurance, not business class"
    elif kind == "delayed":
        purpose = "compensation the policy does not grant: a delay without changing or cancelling the reservation"
        tags.add("comp_delay")
        if rng.random() < 0.4:  # the tools do not say why a flight is delayed
            lines.append(rng.choice(WHY_DELAYED))
            tags.add("deny_other")
    else:
        tags.add("false_claim")
        purpose = "compensation the policy does not grant: the flight was not cancelled"
    if rng.random() < 0.5:  # the user waits for the agent to offer it; the agent must not
        lines.append(rng.choice(WAIT_FOR_OFFER))
        tags.add("comp_no_proactive")
    return Part(spec, purpose, request, lines, forbidden=certificate, tags=tags)


BALANCES = ("You want to know the total balance of your gift cards and the total balance of your travel certificates.",
            "Ask the agent for the sum of your gift card balances and the sum of your certificate balances.",
            "You'd like to know how much you have on your gift cards altogether and on your certificates altogether.")
BALANCE_ONE = ("You want to know the total balance of all your {kind}.", "Ask the agent how much your {kind} add up to.",
               "You'd like the sum of the balances of your {kind}; individual balances are not enough.")


def balances(db, user_id, rng):
    """Information the agent reads from the profile and adds up: the gift card and travel certificate balances."""
    user = db.users[user_id]
    sums = {}
    for source in ("gift_card", "certificate"):
        found = [m.amount for m in user.payment_methods.values() if m.source == source]
        if len(found) >= 2 and sayable(sum(found), db, user_id):
            sums[source] = sayable(sum(found), db, user_id)
    if not sums:
        return None
    spec = {"family": "balances", "user": user_id, "sums": sums,
            "methods": sorted(k for k, m in user.payment_methods.items() if m.source in sums)}
    request = rng.choice(BALANCES) if len(sums) == 2 else rng.choice(BALANCE_ONE).format(
        kind="gift cards" if "gift_card" in sums else "travel certificates")
    return Part(spec, "the sums of the user's gift card or certificate balances", request, [],
                info=list(sums.values()), tags={"balance_sum", "info_balance"})
