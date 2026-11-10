"""Requests in which a value the agent works out decides what happens next on the same reservation or booking: an
upgrade within the user's limit, then checked bags free by the cabin the reservation ends up in; a flight change with
the whole reservation in business class within a limit, else the change alone or nothing; the same flight as on a
reservation, booked for a companion too within a limit; a booking paid by the plan its total picks; a basic economy
change, upgraded first within a limit; a cancellation and the total paid for the other upcoming reservations."""
from tau2.domains.airline.tools import AirlineTools

from rsi.data_preprocessing.airline.book import (AGAIN_UNKNOWN, CAP, INSURANCE, KNOWN_OTHER, KNOWN_SELF, NO_BAGS,
                                                 OTHERS_IN_PROFILE, PAY_ONE, book, book_again)
from rsi.data_preprocessing.airline.cancel import CANCEL_POOL, cancel
from rsi.data_preprocessing.airline.change import (MOVE, PREFER, SAME_DAY, UPGRADE_FIRST, WHICH, _new_journey, change,
                                                   next_one)
from rsi.data_preprocessing.airline.facts import (CABIN, DOB_IN_PROFILE, FIRST_DAY, FREE_BAGS, LAST_DAY, PAY_CHANGE,
                                                  _call, _count, _difference, _flights, birthday, borderline, choose,
                                                  clock, day, facts, journeys, listed, options, pay, payer, pref_tags,
                                                  sayable, trip)
from rsi.data_preprocessing.airline.modify import BUDGET, ONE_PERSON, SEGMENT, UPGRADE
from rsi.data_preprocessing.common import Part

UP = {"basic_economy": "economy", "economy": "business"}
ALLOWED = ("recent", "business", "airline", "insured")  # cancellation rules that allow it


def _limit(amount, rng, within):
    """A limit in tens on the side of the amount the variant needs and at least $10 from it, so that rounding cannot
    move it across; or None."""
    limit = int(amount * (rng.uniform(1.15, 1.5) if within else rng.uniform(0.5, 0.85)))
    limit -= limit % 10
    return limit if limit > 0 and abs(limit - amount) >= 10 and within == (limit > amount) else None


# ---------------------------------------------------------------- an upgrade within a limit, then bags

WHOLE_OR_KEEP = ("If that is not possible, you are willing to upgrade the whole reservation, as long as the extra cost "
                 "is at most ${budget} in total; otherwise you keep your current cabin.",
                 "Should only a part not be possible, upgrade everything instead, but only if that costs ${budget} or "
                 "less altogether; if more, you stay in your current cabin.",
                 "If the agent says it has to be the whole reservation, you accept that for at most ${budget} in total, "
                 "and otherwise keep your current cabin.")
BAGS_NEXT = ("Once the cabin is settled, upgraded or not, you also want {bags} added to that reservation.",
             "After the cabin question is decided either way, you want {bags} added to the same reservation.",
             "When the cabin has been dealt with, whether or not it changed, add {bags} to that reservation.")
PAY_CABIN_BAGS = ("Pay whatever you owe, for the cabin or the bags, with {pay}.",
                  "Charge any cost, for the cabin or for baggage, to {pay}.",
                  "Use {pay} for anything the cabin or the bags cost.")
ASK_WHOLE = ("Ask the agent what upgrading the whole reservation would cost.",
             "You want to know the exact extra cost of upgrading the whole reservation.",
             "Before deciding, ask how much the upgrade would cost for the whole reservation.")


def cabin_bags(db, r, rng, within):
    """Change cabin, then baggage, on one reservation whose flights are all to come: the whole reservation goes up one
    cabin if that costs at most the user's limit (within), else it keeps its cabin; about half of the users first ask
    to upgrade one journey or one passenger, which the policy refuses. Then checked bags are added, free by the cabin
    the reservation has by then, and so many that the cabin changes how many of them are paid for."""
    up = UP.get(r.cabin)
    text = trip(db, r, rng)
    if up is None or text is None or r.nonfree_baggages:
        return None
    user = db.users[r.user_id]
    flights = [(f.flight_number, f.date) for f in r.flights]
    free = {c: FREE_BAGS[user.membership][c] * len(r.passengers) for c in (r.cabin, up)}
    adds = [k for k in (1, 2, 3) if r.total_baggages + k > free[r.cabin]]  # some paid for in the current cabin
    difference = _difference(db, r, up, flights)
    if not adds or difference <= 0 or any(db.flights[n].dates[d].available_seats[up] < len(r.passengers)
                                          for n, d in flights):
        return None
    k = rng.choice(adds)
    total = r.total_baggages + k
    nonfree = {c: max(0, total - free[c]) for c in free}
    pid = payer(user, max(difference + 50 * nonfree[up], 50 * nonfree[r.cabin]), rng)  # a gift card covers either way
    # the bags paid for after an upgrade within the limit fit in it too, should the user count them in; over it, the
    # upgrade stays over once the bag fees it saves are taken off
    limit = _limit(difference + 50 * nonfree[up] if within else difference - 50 * (nonfree[r.cabin] - nonfree[up]),
                   rng, within)
    if pid is None or limit is None:
        return None
    partial = None
    # a part can be named on about half of the reservations: a round trip, or several passengers
    if (r.flight_type == "round_trip" or len(r.passengers) > 1) and rng.random() < 0.85:
        if r.flight_type == "round_trip" and (len(r.passengers) == 1 or rng.random() < 0.5):
            partial = rng.choice(("outbound flight", "return flight"))
            request = rng.choice(SEGMENT).format(which=partial, trip=text, cabin=CABIN[up])
        else:
            p = rng.choice(r.passengers)
            partial = f"{p.first_name} {p.last_name}"
            request = rng.choice(ONE_PERSON).format(name=partial, trip=text, cabin=CABIN[up])
    if partial is None:
        request = rng.choice(UPGRADE).format(trip=text, cabin=CABIN[up])
    request += " " + rng.choice(BAGS_NEXT).format(bags=_count(k, "checked bag"))
    lines = [rng.choice(WHOLE_OR_KEEP if partial else BUDGET).format(budget=limit),
             rng.choice(PAY_CABIN_BAGS).format(pay=pay(user, pid))]
    spec = {"family": "cabin_bags", "reservation": r.reservation_id, "cabin": up, "budget": limit, "add": k,
            "payment": pid, **({"partial": partial} if partial else {})}
    upgrade = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=up, flights=_flights(flights),
                    payment_id=pid)

    def add_bags(cabin):
        return _call("update_reservation_baggages", reservation_id=r.reservation_id, total_baggages=total,
                     nonfree_baggages=nonfree[cabin], payment_id=pid)

    tools = AirlineTools(db)
    shown = {x for f in r.flights for x in listed(tools, f.origin, f.destination, f.date)}
    shown |= {50 * n for n in nonfree.values()} | {difference + sum(f.price for f in r.flights) * len(r.passengers)}
    amount = sayable(difference, db, r.user_id, shown)
    info, ask = ([amount], rng.choice(ASK_WHOLE)) if amount and rng.random() < 0.6 else ([], "")
    tags = {"cabin_change", "cabin_diff", "budget", "cond_value", "price_calc", "bags_add", "bag_allowance", "mod_pay",
            "order"} | ({"cabin_whole", "fallback"} if partial else set()) | ({"info_amount"} if info else set())
    first = "a partial upgrade, refused, then " if partial else ""
    if within:
        return Part(spec, f"{first}an upgrade within the user's limit, then checked bags free by the new cabin", request,
                    lines, [upgrade, add_bags(up)], info=info, ask=ask, tags=tags)
    return Part(spec, f"{first}an upgrade over the user's limit, declined, then checked bags free by the current cabin",
                request, lines, [add_bags(r.cabin)], upgrade, info, ask, by_policy=False, tags=tags)


# ---------------------------------------------------------------- a change with business class within a limit

JOURNEY_UP = ("You'd also like to fly that journey in business class.",
              "On that journey you want to sit in business class.", "You also want that journey in business class.")
WHOLE_UP = ("If business class is not possible for that journey alone, you'd take business class for the whole "
            "reservation along with the change, as long as the two together cost you at most ${budget}.",
            "Should the agent say the cabin must be the same for the whole trip, upgrade the whole reservation to "
            "business class with the change, but only if the change and the upgrade together cost ${budget} or less.",
            "If business class is only possible for the whole reservation, you accept it together with the change, "
            "provided you pay no more than ${budget} for both.")
OTHERWISE = {"change": ("If it costs more than that, just make the flight change and stay in economy.",
                        "If it costs more, change the flights anyway but keep economy.",
                        "Above that, you only want the flights changed, in economy."),
             "nothing": ("If it costs more than that, leave the reservation as it is.",
                         "If it costs more, you do not want any change at all.",
                         "Above that, keep the reservation exactly as it is.")}
ASK_WITH_BUSINESS = ("Ask the agent how much the change with the whole reservation in business class would cost.",
                     "You want to know the exact extra cost of the change with business class for the whole reservation.",
                     "Before deciding, ask what the change plus business class for everything would cost in total.")


def _pick(db, r, j, date, preference, hour, cabin, same_day):
    """The flights of the itinerary the preference singles out for journey j on that date in the cabin, picked as
    _new_journey picks it, or None."""
    js = journeys(r)
    everything = options(AirlineTools(db), js[j][0].origin, js[j][-1].destination, date, cabin, len(r.passengers))
    current = [f.flight_number for f in js[j]]
    found = [o for o in everything if not same_day or [n for n, _ in o[0]] != current]
    if preference == "later":
        return list(next_one(db, js[j], found) or []) or None
    hit = choose(found, preference, current, hour)
    if hit is None or same_day and choose(everything, preference, current, hour) != hit:
        return None
    return list(hit[0])


def change_upgrade_budget(db, r, rng, within):
    """Change flights and cabin of a round trip in economy: one journey moves, and the user wants that journey in
    business class, which the policy refuses, the cabin being the same for the whole reservation. The user falls back
    on the whole reservation in business class with the change, in one call, if the extra cost is at most a limit
    (within); else, as the story has it, on the change alone in economy or on nothing. The preference singles out the
    same itinerary in either cabin, and the change done as two calls (cabin or flights first) leaves another payment
    history."""
    if r.cabin != "economy" or r.flight_type != "round_trip":
        return None
    text = trip(db, r, rng)
    if text is None:
        return None
    user = db.users[r.user_id]
    same = rng.random() < 0.5
    for _ in range(5):  # a preference that compares prices often picks another itinerary in economy
        moved = _new_journey(db, r, rng, "business", same_day=same)
        if moved is None or _pick(db, r, *moved[:4], "economy", same) == moved[4]:
            break
    else:
        return None
    if moved is None:
        return None
    j, date, preference, hour, legs, flights = moved
    current = [(f.flight_number, f.date) for f in r.flights]
    if any(db.flights[n].dates[d].available_seats["business"] < len(r.passengers) for n, d in flights + current):
        return None
    difference = _difference(db, r, "business", flights)
    first, alone = _difference(db, r, "business", current), _difference(db, r, "economy", flights)
    if difference <= 0 or 0 in (first, difference - first, alone, difference - alone):
        return None
    pid = payer(user, max(difference, first, alone), rng)
    otherwise = rng.choice(("change", "nothing"))
    limit = _limit(difference, rng, within)
    if pid is None or limit is None:
        return None
    request = rng.choice(SAME_DAY if same else MOVE).format(which=rng.choice(WHICH[j]).format(trip=text), date=day(date))
    request += " " + rng.choice(JOURNEY_UP)
    lines = [rng.choice(PREFER[preference]).format(cabin=CABIN["business"], date=day(date),
                                                   hour=clock(hour) if hour is not None else ""),
             rng.choice(WHOLE_UP).format(budget=limit) + " " + rng.choice(OTHERWISE[otherwise]),
             rng.choice(PAY_CHANGE).format(pay=pay(user, pid))]
    spec = {"family": "change_upgrade_budget", "reservation": r.reservation_id, "journey": j, "date": date,
            "preference": preference, "budget": limit, "payment": pid, "otherwise": otherwise,
            **({"hour": hour} if hour is not None else {})}
    tools, js = AirlineTools(db), journeys(r)
    shown = listed(tools, js[j][0].origin, js[j][-1].destination, date)
    shown |= {x for f in r.flights for x in listed(tools, f.origin, f.destination, f.date)}
    shown |= {abs(x) for x in (first, alone, difference - first, difference - alone)}
    amount = sayable(difference, db, r.user_id, shown)
    info, ask = ([amount], rng.choice(ASK_WITH_BUSINESS)) if amount and rng.random() < 0.5 else ([], "")
    tags = {"mod_flights", "cabin_change", "cabin_diff", "cabin_whole", "budget", "cond_value", "price_calc",
            "fallback"} | pref_tags(preference, legs) | ({"same_day"} if same else set()) | (
        {"info_amount"} if info else set())

    def update(cabin):
        return _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=cabin,
                     flights=_flights(flights), payment_id=pid)

    if within:
        return Part(spec, "a flight change with one journey in business class, refused, then with the whole reservation "
                    "in business class within the user's limit, in one call", request, lines, [update("business")],
                    info=info, ask=ask, tags=tags | {"cabin_flights", "mod_pay"})
    if otherwise == "change":
        return Part(spec, "a flight change with one journey in business class, refused, over the user's limit for the "
                    "whole reservation: the change alone", request, lines, [update("economy")], update("business"), info,
                    ask, by_policy=False, tags=tags | {"mod_pay"})
    return Part(spec, "a flight change with one journey in business class, refused, over the user's limit for the whole "
                "reservation: nothing changes", request, lines, forbidden=update("business"), info=info, ask=ask,
                by_policy=False, tags=tags)


# ---------------------------------------------------------------- bookings whose total decides

SAME_FLIGHT = ("You want to book the same flight as on {trip} again, on {date}, in {cabin}, for yourself and {friend}.",
               "You'd like to fly the same flight as on {trip} once more, on {date}, in {cabin}, with {friend}.",
               "Book yourself and {friend} on the same flight as on {trip}, this time on {date}, in {cabin}.")
SAME_FLIGHT_DAY = ("You want to book yourself and {friend} on exactly the same flight as {trip}, same day and cabin.",
                   "Put yourself and {friend} on the very same flight as {trip}, on the same date and in the same cabin.",
                   "You'd like seats for yourself and {friend} on the same flight as {trip}, same date, same cabin.")
ASK_FOR_TWO = ("Ask the agent how much it would cost for the two of you.",
               "You want to know the total price for both of you before deciding.",
               "Before booking, ask what the flight costs for two.")


def book_same_cap(db, user_id, rng, within):
    """Book flight, the same nonstop flight as on one of the user's one-way reservations, on another day or, for
    travellers not on that reservation, on the same day: for the user and a passenger saved in their profile if the
    total for both is at most the user's limit (within), else for the user alone, who is within it."""
    user = db.users[user_id]
    cards = sorted(k for k, m in user.payment_methods.items() if m.source == "credit_card" and pay(user, k))
    me = {"first_name": user.name.first_name, "last_name": user.name.last_name, "dob": user.dob}
    if not cards:
        return None
    for rid in rng.sample(user.reservations, len(user.reservations)):
        r = db.reservations[rid]
        if r.flight_type != "one_way" or len(r.flights) != 1 or r.status == "cancelled":
            continue
        text = trip(db, r, rng)
        on = {(p.first_name, p.last_name) for p in r.passengers}
        friends = [p.model_dump() for p in user.saved_passengers
                   if (p.first_name, p.last_name) not in on | {(me["first_name"], me["last_name"])}]
        number, old = r.flights[0].flight_number, r.flights[0].date
        dates = [d for d in (f"2024-05-{x:02d}" for x in range(FIRST_DAY, LAST_DAY + 1))
                 if (d != old or (me["first_name"], me["last_name"]) not in on) and d in db.flights[number].dates
                 and db.flights[number].dates[d].status == "available"
                 and db.flights[number].dates[d].available_seats[r.cabin] >= 2]
        if text is None or not friends or not dates:
            continue
        others = [d for d in dates if d != old]
        date = old if old in dates and (not others or rng.random() < 0.3) else rng.choice(others)
        friend, card = rng.choice(friends), rng.choice(cards)
        fare = db.flights[number].dates[date].prices[r.cabin]
        limit = _limit(2 * fare, rng, within)
        if limit is None or limit < fare:
            return None
        name = f"{friend['first_name']} {friend['last_name']}"
        request = (rng.choice(SAME_FLIGHT_DAY).format(trip=text, friend=name) if date == old else
                   rng.choice(SAME_FLIGHT).format(trip=text, date=day(date), cabin=CABIN[r.cabin], friend=name))
        lines = [x for x in (rng.choice(AGAIN_UNKNOWN), rng.choice(CAP).format(limit=limit), rng.choice(NO_BAGS),
                             rng.choice(INSURANCE["no"]), rng.choice(PAY_ONE).format(a=pay(user, card))) if x]
        known = [rng.choice(DOB_IN_PROFILE) if rng.random() < 0.35 else
                 rng.choice(KNOWN_SELF).format(dob=birthday(user.dob)),
                 rng.choice(OTHERS_IN_PROFILE) if rng.random() < 0.5 else
                 rng.choice(KNOWN_OTHER).format(name=name, dob=birthday(friend["dob"]))]
        amount = sayable(2 * fare, db, user_id, listed(AirlineTools(db), r.origin, r.destination, date))
        info, ask = ([amount], rng.choice(ASK_FOR_TWO)) if amount and rng.random() < 0.5 else ([], "")
        spec = {"family": "book_same_cap", "user": user_id, "reservation": rid, "date": date, "cabin": r.cabin,
                "passengers": [me, friend], "limit": limit, "payment": [card]}
        tags = {"book", "book_pax", "sel_same_as", "search_direct", "book_insurance", "budget", "cond_value",
                "price_calc", "fallback"} | ({"info_amount"} if info else set())

        def booking(people):
            return _call("book_reservation", user_id=user_id, origin=r.origin, destination=r.destination,
                         flight_type="one_way", cabin=r.cabin, flights=_flights([(number, date)]), passengers=people,
                         payment_methods=[{"payment_id": card, "amount": fare * len(people)}], total_baggages=0,
                         nonfree_baggages=0, insurance="no")

        if within:
            return Part(spec, "booking a companion too on the flight of one of the user's reservations, within the "
                        "user's limit", request, lines, [booking([me, friend])], info=info, ask=ask, known=known,
                        tags=tags)
        return Part(spec, "booking the flight of one of the user's reservations: over the user's limit for two, so for "
                    "the user alone", request, lines, [booking([me])], booking([me, friend]), info, ask, by_policy=False,
                    known=known, tags=tags)
    return None


PLAN = {"card": ("put the whole amount on {card}", "charge all of it to {card}"),
        "stored": ("use the full balance of {x} first and put the rest on {card}",
                   "use up {x} and charge the remainder to {card}")}
BY_TOTAL = ("If the total comes to more than ${t}, {a}; otherwise, {b}.",
            "How you pay depends on the total: if it is more than ${t}, {a}; if it is ${t} or less, {b}.",
            "Should the booking cost more than ${t} in total, {a}. If not, {b}.")


def book_pay_by_total(db, user_id, rng, above):
    """Book flight (the flights a preference singles out, or the same flight as on a reservation), paid by the one of
    two plans the total picks: one if it is more than the user's threshold (above), the other if not. Each plan is one
    the policy allows, with methods from the profile: a travel certificate or a gift card used up first, or neither,
    and the rest on a credit card."""
    user = db.users[user_id]
    cards = sorted(k for k, m in user.payment_methods.items() if m.source == "credit_card" and pay(user, k))
    stored = sorted(k for k, m in user.payment_methods.items() if m.source != "credit_card" and m.amount and pay(user, k))
    if not cards or not stored:
        return None
    part = book_again(db, user_id, rng) if rng.random() < 0.5 else book(db, user_id, rng)
    if part is None or "change_mind" in part.tags:  # a total that changes at the confirmation would pick anew
        return None
    booking = part.calls[0]["arguments"]
    total = sum(x["amount"] for x in booking["payment_methods"])
    card = rng.choice(cards)
    plans = [[k, card] for k in stored if user.payment_methods[k].amount < total] + [[card]]
    t = _limit(total, rng, not above)
    if len(plans) < 2 or t is None:
        return None
    high, low = rng.sample(plans, 2)

    def paid(plan):
        used = [{"payment_id": k, "amount": int(user.payment_methods[k].amount)} for k in plan[:-1]]
        return used + [{"payment_id": card, "amount": total - sum(x["amount"] for x in used)}]

    def said(plan):
        if len(plan) == 1:
            return rng.choice(PLAN["card"]).format(card=pay(user, card))
        return rng.choice(PLAN["stored"]).format(x=pay(user, plan[0]), card=pay(user, card))

    chosen, other = (high, low) if above else (low, high)
    part.lines[-1] = rng.choice(BY_TOTAL).format(t=t, a=said(high), b=said(low))  # in place of the payment line
    part.calls = [_call("book_reservation", **{**booking, "payment_methods": paid(chosen)})]
    part.forbidden, part.by_policy = _call("book_reservation", **{**booking, "payment_methods": paid(other)}), False
    part.spec.update({"family": "book_pay_by_total", "payment": chosen, "threshold": t, "above": high, "at_most": low})
    part.tags |= {"book_pay", "pay_alloc", "budget", "cond_value", "price_calc"}
    part.purpose += f"; paid by the plan the total picks, {'above' if above else 'at most'} the user's threshold"
    return part


# ---------------------------------------------------------------- a basic economy change upgraded first within a limit

UPGRADE_IF = ("If the agent says a basic economy reservation cannot be changed, you are willing to upgrade it to economy "
              "first and then change the flight, but only if the two steps together cost at most ${budget}; otherwise "
              "you keep the reservation as it is.",
              "Should basic economy rule out the change, you would upgrade the whole reservation to economy and then "
              "move the flight, as long as both steps together cost no more than ${budget}; if more, leave it unchanged.",
              "If the change is only possible after an upgrade to economy, you accept upgrading first, but only for "
              "${budget} or less for both steps together; otherwise keep the reservation as it is.")
PAY_STEPS = ("Pay or receive any difference for either step with {pay}.",
             "Use {pay} for whatever the upgrade and the change cost or refund.",
             "Any charge or refund for the two steps goes to {pay}.")
ASK_STEPS = ("Ask the agent how much the upgrade and the change would cost together.",
             "You want to know what the two steps cost in total before deciding.",
             "Before agreeing, ask what the upgrade and the change add up to.")


def basic_upgrade_budget(db, r, rng, within):
    """Change flights, which the policy refuses for basic economy unless the reservation is first upgraded to economy:
    the user takes the upgrade and then the change only if the two together cost at most a limit (within), else keeps
    the reservation as it is."""
    part = change(db, r, rng, "upgrade") if r.cabin == "basic_economy" else None
    if part is None:
        return None
    user, pid = db.users[r.user_id], part.spec["payment"]
    upgrade, move = part.calls
    flights = [(x["flight_number"], x["date"]) for x in move["arguments"]["flights"]]
    # after the upgrade every flight is priced at today's economy fare, so the two steps cost the economy fares of the
    # new flights less what was paid
    total = _difference(db, r, "economy", flights)
    first = _difference(db, r, "economy", [(f.flight_number, f.date) for f in r.flights])
    limit = _limit(total, rng, within) if total > 0 else None
    if limit is None:
        return None
    named = {s.format(pay=pay(user, pid)) for s in UPGRADE_FIRST}
    lines = [x for x in part.lines if x not in named]
    lines[1:1] = [rng.choice(UPGRADE_IF).format(budget=limit), rng.choice(PAY_STEPS).format(pay=pay(user, pid))]
    tools, journey = AirlineTools(db), journeys(r)[part.spec["journey"]]
    shown = listed(tools, journey[0].origin, journey[-1].destination, part.spec["date"])
    shown |= {x for f in r.flights for x in listed(tools, f.origin, f.destination, f.date)}
    amount = sayable(total, db, r.user_id, shown | {abs(first), abs(total - first)})
    info, ask = ([amount], rng.choice(ASK_STEPS)) if amount and rng.random() < 0.6 else ([], "")
    part.spec.update({"family": "basic_upgrade_budget", "budget": limit})
    tags = part.tags | {"budget", "cond_value", "price_calc", "order"} | ({"info_amount"} if info else set())
    if within:
        return Part(part.spec, "a basic economy flight change, refused, then an upgrade to economy and the change "
                    "within the user's limit", part.request, lines, part.calls, info=info, ask=ask, tags=tags)
    return Part(part.spec, "a basic economy flight change, refused; an upgrade and the change would cost more than the "
                "user's limit, so nothing changes", part.request, lines, forbidden=upgrade, info=info, ask=ask,
                by_policy=False, tags=tags - {"mod_pay"})


# ---------------------------------------------------------------- a cancellation and the total paid for the others

TOTAL_PAID = ("You also want to know how much you have paid in total for your other upcoming reservations.",
              "You also want the agent to add up what you paid for all of your other upcoming reservations.",
              "On top of that, ask how much you paid altogether for your other upcoming reservations.")


def cancel_total(db, r, rng):
    """Cancel flight, which one of the rules allows, and how much the user paid for their other upcoming reservations:
    the payments their histories record (a cancellation adds a refund there as a negative payment; none of these has
    one). Only for users none of whose reservations could be taken either way, and some of whose other reservations
    are not upcoming, so that the agent has to tell them apart."""
    user = db.users[r.user_id]
    mine = [db.reservations[x] for x in user.reservations if db.reservations[x].status != "cancelled"]
    others = [x for x in mine if x.reservation_id != r.reservation_id and facts(db, x)["upcoming"]]
    if any(borderline(db, x) for x in mine) or len(others) == len(mine) - 1:
        return None
    f = facts(db, r)
    bases = [b for b in ALLOWED if CANCEL_POOL[b](r, f)]
    amount = sayable(sum(p.amount for x in others for p in x.payment_history), db, r.user_id)
    part = cancel(db, r, rng, rng.choice(bases)) if bases and amount else None
    if part is None:
        return None
    part.request += " " + rng.choice(TOTAL_PAID)
    part.info = [amount]
    part.spec.update({"family": "cancel_total", "reservations": sorted(x.reservation_id for x in others)})
    part.tags |= {"filter_res", "price_calc", "info_amount"}
    part.purpose += "; and the total paid for the user's other upcoming reservations"
    return part
