"""Change flights: a journey moved to another day, or to other flights on its day, by a preference that singles out one
itinerary, the cabin switched with it or not; both journeys of a round trip at once; a new destination (refused); a
basic economy ticket (refused, or upgraded to economy first, or cancelled under travel insurance and booked anew)."""
from tau2.domains.airline.tools import AirlineTools

from rsi.data_preprocessing.airline.facts import (BACK, CABIN, COST, FIRST_DAY, FREE_BAGS, HOW_LONG, LAST_DAY, PAY_CHANGE,
                                                  SICK, _call, _difference, _flights, choose, clock, day, facts,
                                                  journeys, listed, minutes, options, pay, payer, place, pref_tags,
                                                  sayable, single_out, trip)
from rsi.data_preprocessing.common import Part

# The new date never directly follows {which}: a round trip's description ends with its return date.
MOVE = ("You want to move {which}; the new date is {date}.",
        "You need to change the date of {which}; it should be {date} instead.",
        "You would like to fly on {date} instead, for {which}.")
WHICH = (("the outbound flight of {trip}", "the outbound journey of {trip}"),
         ("the return flight of {trip}", "the return journey of {trip}"))
PREFER = {
    "same flight": ("You want the same flight number, just on the new date.",
                    "You'd like to keep the same flight, only on {date}.", "Ask for the same flight on {date}."),
    "cheapest": ("You want the cheapest {cabin} option that day, nonstop or with one stop.",
                 "Whichever {cabin} option is cheapest that day is fine, with or without a connection.",
                 "Price matters most: you want the cheapest {cabin} itinerary that day, even if it has a stop."),
    "cheapest nonstop": ("You only want a nonstop flight, the cheapest one in {cabin} that day.",
                         "It has to be nonstop; among the nonstop flights that day you want the cheapest {cabin} one.",
                         "You want the cheapest nonstop {cabin} flight that day."),
    # the cabin too: the earliest or fastest flight may be full in it and have seats in another
    "earliest nonstop": ("You want the nonstop {cabin} flight that leaves earliest that day.",
                         "It has to be nonstop in {cabin}, and you want the earliest departure that day.",
                         "You'd like the first nonstop {cabin} flight of the day."),
    "second cheapest": ("You want the second cheapest {cabin} option that day, nonstop or with one stop; the cheapest "
                        "ones are usually inconvenient.",
                        "Not the cheapest, but the next cheapest {cabin} itinerary that day, with or without a stop.",
                        "You'd like the second-lowest-priced {cabin} option that day."),
    "fastest": ("You want the fastest {cabin} itinerary that day, counting the time of any layover.",
                "Total travel time matters most: the quickest {cabin} option that day, connection included.",
                "You'd like whichever {cabin} option gets you there in the least total time that day."),
    "after": ("You cannot leave before {hour}; among the {cabin} options that day departing then or later, you want the "
              "cheapest.",
              "It has to depart at {hour} or later that day; of those, pick the cheapest {cabin} option.",
              "You can only fly from {hour} on that day, and you want the cheapest {cabin} option leaving then or later."),
    "arrive by": ("You need to arrive by {hour} that same day; among the {cabin} options that do, you want the cheapest.",
                  "It must get you there no later than {hour} that day; of those, the cheapest {cabin} option.",
                  "You want the cheapest {cabin} itinerary that day that arrives by {hour}."),
    "later": ("You want the next {cabin} flight after yours that same day; if there is none, the earliest one the next "
              "day.",
              "Move it to the first {cabin} flight that leaves later that day, or failing that to the earliest one the day "
              "after.",
              "You'd like the first {cabin} itinerary that departs after your current one that day; if none does, the "
              "first one the next morning."),
}
NO_UPGRADE = ("You do not want to upgrade the cabin of that trip or cancel it; you only want the flights changed.",
              "If the agent suggests upgrading or cancelling that trip, you decline both.",
              "For that trip, you will neither pay for a cabin upgrade nor cancel it.")
UPGRADE_FIRST = ("If the agent says a basic economy reservation cannot be changed, you agree to upgrade it to economy "
                 "first and then change the flight, paying or receiving any difference for both steps with {pay}.",
                 "If basic economy cannot be changed, you are willing to upgrade the whole reservation to economy so that "
                 "the change can be made, with {pay} covering any difference for both the upgrade and the change.",
                 "You accept upgrading to economy first if that is the only way to change the flight; use {pay} for "
                 "both steps.")
NO_CANCEL = ("You do not want to cancel that trip.", "Cancelling that trip is not an option for you.",
             "You want to keep that trip, not cancel it.")
REROUTE = ("You want to change {trip} so that it goes to {city} instead of {old}.",
           "You need {trip} to end in {city} rather than {old}.",
           "You'd like {trip} to take you to {city} instead of {old}.")
KEEP_IF_NOT = ("If that is not possible, you keep the reservation as it is.",
               "If it cannot be done, you leave the booking unchanged.",
               "If the agent cannot do it, you do not want any other change.")
ASK_HUMAN = ("If the agent says the change is not possible, ask to be transferred to a human agent.",
             "Should the agent refuse, you want to speak to a human agent.",
             "If it cannot be done, ask the agent to transfer you to a person.")


def _new_journey(db, r, rng, cabin, prefs=tuple(PREFER), journey=None, same_day=False, later=False):
    """A journey of the reservation (or the given one) moved to another day (with later, a later one), or with same_day
    to other flights on its day, by a preference that singles out one itinerary: (journey index, new date, preference,
    its hour or None, the new journey's flights, all flights of the reservation after the change), or None. On the same
    day, "later" is the first itinerary leaving after the current one."""
    tools, js = AirlineTools(db), journeys(r)
    prefs = tuple(p for p in prefs if p != "later")
    if same_day:
        picks = [(j, int(js[j][0].date[-2:])) for j in range(len(js)) if journey in (None, j)]
        prefs = tuple(p for p in prefs if p != "same flight") + ("later",)
    else:
        picks = [(j, d) for j in range(len(js)) for d in range(FIRST_DAY, LAST_DAY + 1)
                 if journey in (None, j) and 0 < abs(d - int(js[j][0].date[-2:])) <= 3
                 and (not later or d > int(js[j][0].date[-2:]))
                 and (j == 1 or len(js) == 1 or d < int(js[1][0].date[-2:]))
                 and (j == 0 or d > int(js[0][-1].date[-2:]))]
    rng.shuffle(picks)
    for j, d in picks:
        date = f"2024-05-{d:02d}"
        everything = options(tools, js[j][0].origin, js[j][-1].destination, date, cabin, len(r.passengers))
        current = [f.flight_number for f in js[j]]
        # the itinerary the user has is not a change
        found = [o for o in everything if not same_day or [n for n, _ in o[0]] != current]
        fits = [(p, h) for p in prefs if p != "same flight" or len(js[j]) == 1
                for h in [(None if next_one(db, js[j], found) else False) if p == "later"
                          else single_out([found], p, rng, current)] if h is not False]
        if same_day:  # nor may the preference pick the user's own flights over it
            fits = [(p, h) for p, h in fits
                    if p == "later" or choose(everything, p, current, h) == choose(found, p, current, h)]
        if not fits:
            continue
        preference, hour = rng.choice(fits + 2 * [x for x in fits if "nonstop" in x[0]])  # official tasks favour nonstop
        new = list(next_one(db, js[j], found) if preference == "later" else choose(found, preference, current, hour)[0])
        kept = [[(f.flight_number, f.date) for f in jj] for jj in js]
        kept[j] = new
        return j, date, preference, hour, new, [leg for jj in kept for leg in jj]
    return None


def next_one(db, journey, found):
    """The flights of the itinerary that leaves first after the journey's first flight, if one alone does, else None."""
    dep = minutes(db.flights[journey[0].flight_number].scheduled_departure_time_est)
    later = sorted((minutes(o[2]), o[0]) for o in found if minutes(o[2]) > dep)
    return later[0][1] if later and (len(later) == 1 or later[0][0] < later[1][0]) else None


SAME_DAY = ("You want to switch {which} to other flights on the same day.",
            "You'd like different flights for {which}, keeping the same date.",
            "You need {which} moved to another flight that same day.")
CHANGE_BUDGET = ("You will only make the change if it costs at most ${budget} in total; otherwise keep your current "
                 "flights.", "Your limit for the change is ${budget} altogether; if it costs more, leave the reservation "
                             "as it is.")
SWITCH = {"economy": "business", "business": "economy"}
CABIN_TOO = ("With the change, you also want the whole reservation in {cabin} instead of {old}.",
             "At the same time, you want to fly {cabin} rather than {old}.",
             "You'd also like to switch from {old} to {cabin} as part of this change.")
WAIVE = ("Mention that you bought travel insurance, so you expect any change fee to be waived.",
         "Insist that your travel insurance should cover any change fee.")


def change(db, r, rng, kind):
    """Change flights: a reservation other than basic economy can move a journey to another day, or to other flights on
    its day, without changing its origin, destination or trip type; a basic economy one cannot, unless it is first
    upgraded to another cabin. kind: "ok", "same_day", "basic", "upgrade", "budget_within" / "budget_over" (the user
    takes the change only up to a price), or "cabin": economy and business switch cabins with the change, in the same
    call (done as two, the payment history differs)."""
    text = trip(db, r, rng)
    if text is None:
        return None
    user = db.users[r.user_id]
    cabin = "economy" if kind == "upgrade" else SWITCH.get(r.cabin) if kind == "cabin" else r.cabin
    # as in cabin_change, a new cabin leaves the agent no bags to settle
    if cabin is None or kind == "cabin" and (
            r.nonfree_baggages if cabin == "business" else
            r.total_baggages > FREE_BAGS[user.membership][cabin] * len(r.passengers)):
        return None
    # a basic economy ticket, or a change on a budget, is often to be moved within its own day, as in the official tasks
    same = kind == "same_day" or (kind != "ok" and rng.random() < 0.5)
    moved = _new_journey(db, r, rng, cabin, same_day=same)
    if moved is None:
        return None
    j, date, preference, hour, legs, flights = moved
    # Every flight of the reservation takes the new cabin, the ones kept too; so do the current ones, so that the change
    # done as two calls (the cabin first) goes through as well and only its payment history tells it apart.
    current = [(f.flight_number, f.date) for f in r.flights]
    if kind == "cabin" and any(db.flights[n].dates[d].available_seats[cabin] < len(r.passengers)
                               for n, d in flights + current):
        return None
    which = text if r.flight_type == "one_way" else rng.choice(WHICH[j]).format(trip=text)
    spec = {"family": "change", "reservation": r.reservation_id, "journey": j, "date": date, "cabin": cabin,
            "preference": preference, **({"hour": hour} if hour is not None else {})}
    lines = [rng.choice(PREFER[preference]).format(cabin=CABIN[cabin], date=day(date),
                                                   hour=clock(hour) if hour is not None else "")]
    request = rng.choice(SAME_DAY if same else MOVE).format(which=which, date=day(date))
    chosen = pref_tags(preference, legs) | ({"same_day"} if same else set())
    if kind == "cabin":
        request += " " + rng.choice(CABIN_TOO).format(cabin=CABIN[cabin], old=CABIN[r.cabin])
        chosen |= {"cabin_change", "cabin_diff", "cabin_flights"}
    if kind == "upgrade":
        spec["upgrade_first"] = True
        upgrade = _difference(db, r, cabin, [(f.flight_number, f.date) for f in r.flights])
        upgraded = r.model_copy(deep=True, update={"cabin": cabin})
        for f in upgraded.flights:
            f.price = db.flights[f.flight_number].dates[f.date].prices[cabin]
        pid = payer(user, max(upgrade, upgrade + _difference(db, upgraded, cabin, flights)), rng)
        if pid is None:
            return None
        spec["payment"] = pid
        lines += [rng.choice(UPGRADE_FIRST).format(pay=pay(user, pid)), rng.choice(NO_CANCEL)]
        calls = [_call("update_reservation_flights", reservation_id=r.reservation_id, cabin=cabin,
                       flights=_flights((f.flight_number, f.date) for f in r.flights), payment_id=pid),
                 _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=cabin,
                       flights=_flights(flights), payment_id=pid)]
        return Part(spec, "basic economy flight change after an upgrade to economy", request, lines, calls,
                    tags={"mod_flights", "basic_no_change", "fallback", "cabin_change", "cabin_diff", "mod_pay"} | chosen)
    difference = _difference(db, r, cabin, flights)
    first = _difference(db, r, cabin, current)  # the cabin alone, the first of the change done as two calls
    # a step that costs nothing records no payment, and two calls then end as one does
    if kind == "cabin" and 0 in (first, difference - first):
        return None
    # a gift card covering the larger of the two amounts also pays for the change done as two calls
    pid = payer(user, max(difference, first) if kind == "cabin" else difference, rng)
    if pid is None:
        return None
    spec["payment"] = pid
    lines.append(rng.choice(PAY_CHANGE).format(pay=pay(user, pid)))
    call = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=cabin, flights=_flights(flights),
                 payment_id=pid)
    if kind == "basic":
        lines.append(rng.choice(NO_UPGRADE))
        return Part(spec, "flight change the policy does not allow: basic economy", request, lines, forbidden=call,
                    tags={"mod_flights", "basic_no_change"})
    tools, journey = AirlineTools(db), journeys(r)[j]
    shown = listed(tools, journey[0].origin, journey[-1].destination, date)
    if kind == "cabin":  # the kept flights are priced anew too
        shown |= {x for n, d in flights for x in db.flights[n].dates[d].prices.values()}
    amount = sayable(difference, db, r.user_id, shown)
    if kind.startswith("budget"):  # the user asks the price, then takes the change only within the limit
        within = kind == "budget_within"
        if difference <= 0 or amount is None:
            return None
        limit = int(difference * (rng.uniform(1.15, 1.5) if within else rng.uniform(0.5, 0.85)))
        limit -= limit % 10
        if limit <= 0 or within != (limit >= difference):
            return None
        spec["budget"] = limit
        lines.insert(1, rng.choice(CHANGE_BUDGET).format(budget=limit))
        tags = {"mod_flights", "budget", "cond_value", "price_calc", "info_amount"} | chosen
        if within:
            return Part(spec, "flight change within the user's limit", request, lines, [call], info=[amount],
                        ask=rng.choice(COST), tags=tags | {"mod_pay"})
        return Part(spec, "flight change over the user's limit: the user only asks for the price", request, lines,
                    forbidden=call, info=[amount], ask=rng.choice(COST), by_policy=False, tags=tags)
    part = Part(spec, "flight and cabin change in one call" if kind == "cabin" else "flight change the policy allows",
                request, lines, [call], tags={"mod_flights", "mod_pay"} | chosen)
    if rng.random() < (0.6 if preference == "fastest" else 0.2):  # asked, not checked
        part.lines.append(rng.choice(HOW_LONG))
        part.tags.add("info_attr")
    if rng.random() < 0.15:  # a belief the policy has nothing to answer to: there are no change fees to waive
        part.lines.append(rng.choice(WAIVE))
        part.tags |= {"false_claim", "pressure"}
    if amount and rng.random() < 0.6:
        part.ask, part.info = rng.choice(COST if difference > 0 else BACK), [amount]
        part.tags |= {"price_calc", "info_amount"}
    return part


BOTH_WAYS = ("You want to move both flights of {trip}, the outbound and the return, {shift}.",
             "Both flights of {trip} need to move {shift}, the outbound and the return.",
             "You'd like to shift {trip}, both directions, {shift}.")
SHIFT = {1: "one day later", 2: "two days later", -1: "one day earlier", -2: "two days earlier"}


def change_both(db, r, rng):
    """Change flights, both journeys of a round trip moved by the same number of days in one change, the same
    preference singling out each new itinerary."""
    text = trip(db, r, rng)
    js = journeys(r)
    if text is None or len(js) != 2 or r.cabin == "basic_economy":
        return None
    user = db.users[r.user_id]
    tools = AirlineTools(db)
    for shift in rng.sample(sorted(SHIFT), len(SHIFT)):
        out, back = int(js[0][0].date[-2:]) + shift, int(js[1][0].date[-2:]) + shift
        if not (FIRST_DAY <= out and back <= LAST_DAY and out < back):
            continue
        dates = [f"2024-05-{out:02d}", f"2024-05-{back:02d}"]
        found = [options(tools, j[0].origin, j[-1].destination, d, r.cabin, len(r.passengers)) for j, d in zip(js, dates)]
        fits = [(p, h) for p in PREFER if p not in ("same flight", "later") for h in [single_out(found, p, rng)]
                if h is not False]
        if not fits:
            continue
        preference, hour = rng.choice(fits)
        new = [list(choose(f, preference, hour=hour)[0]) for f in found]
        flights = [leg for n in new for leg in n]
        difference = _difference(db, r, r.cabin, flights)
        pid = payer(user, difference, rng)
        if pid is None:
            return None
        spec = {"family": "change_both", "reservation": r.reservation_id, "dates": dates, "preference": preference,
                "payment": pid, **({"hour": hour} if hour is not None else {})}
        lines = [rng.choice(PREFER[preference]).format(cabin=CABIN[r.cabin], date="", hour=clock(hour) if hour is not
                                                       None else "").replace("that day", "each day").replace(
                     "that same day", "each day"), rng.choice(PAY_CHANGE).format(pay=pay(user, pid))]
        call = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=r.cabin,
                     flights=_flights(flights), payment_id=pid)
        tags = {"mod_flights", "mod_pay"} | pref_tags(preference, new[0]) | pref_tags(preference, new[1])
        part = Part(spec, "both journeys of a round trip moved in one change", rng.choice(BOTH_WAYS).format(
            trip=text, shift=SHIFT[shift]), lines, [call], tags=tags)
        amount = sayable(difference, db, r.user_id, {x for f in found for o in f for x in (o[1],)})
        if amount and rng.random() < 0.5:
            part.ask, part.info = rng.choice(COST if difference > 0 else BACK), [amount]
            part.tags |= {"price_calc", "info_amount"}
        return part
    return None


def reroute(db, r, rng):
    """Change flights: the destination cannot be changed."""
    text = trip(db, r, rng)
    if text is None or r.flight_type != "one_way":
        return None
    user, date = db.users[r.user_id], r.flights[0].date
    tools = AirlineTools(db)
    other = sorted({f.destination for f in tools.search_direct_flight(r.origin, None, date)
                    if f.destination != r.destination and f.available_seats[r.cabin] >= len(r.passengers)})
    pid = payer(user, 0, rng)
    if not other or pid is None:
        return None
    city = rng.choice(other)
    flight = next(f for f in tools.search_direct_flight(r.origin, city, date)
                  if f.available_seats[r.cabin] >= len(r.passengers))
    spec = {"family": "reroute", "reservation": r.reservation_id, "destination": city}
    request = rng.choice(REROUTE).format(trip=text, city=place(city), old=place(r.destination))
    forbidden = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=r.cabin,
                      flights=_flights([(flight.flight_number, date)]), payment_id=pid)
    f = facts(db, r)
    # where a rule allows cancelling, the agent may offer to cancel and book anew, which this user does not want
    keep = [NO_CANCEL[0]] if r.cabin == "business" or f["age"] < 24 or f["airline"] else []
    if rng.random() < 0.4:  # the user, not the policy, asks for a human once the change is refused
        spec["transfer"] = True
        return Part(spec, "flight change the policy does not allow: a new destination; the user then asks for a human",
                    request, [rng.choice(ASK_HUMAN)] + keep, forbidden=forbidden,
                    tags={"same_od", "fallback", "transfer"}, transfer=True)
    return Part(spec, "flight change the policy does not allow: a new destination", request,
                [rng.choice(KEEP_IF_NOT)] + keep, forbidden=forbidden, tags={"same_od"})


SICK_CANCEL = ("If a basic economy reservation cannot be changed, you are willing to cancel it instead, using your "
               "travel insurance: {why}.",
               "Should the change be refused because the ticket is basic economy, cancel the trip under your travel "
               "insurance, since {why}.",
               "If the agent cannot change it, cancel the whole trip with your travel insurance; {why}.")


def change_or_cancel(db, r, rng):
    """Change flights, refused for basic economy, with a cancellation under travel insurance for the user's illness as
    the fallback, which the policy allows."""
    if r.cabin != "basic_economy" or r.insurance != "yes" or not facts(db, r)["passenger"]:
        return None
    part = change(db, r, rng, "basic")
    if part is None:
        return None
    part.lines = [x for x in part.lines if x not in NO_UPGRADE]
    part.lines.append(rng.choice(SICK_CANCEL).format(why=rng.choice(SICK)))
    part.calls = [_call("cancel_reservation", reservation_id=r.reservation_id)]
    part.spec.update({"family": "change_or_cancel", "reason": "health"})
    part.tags |= {"fallback", "cancel_ok"}
    part.purpose = "a basic economy flight change, refused, then a cancellation under travel insurance"
    return part


REBOOK = ("If the agent says a basic economy reservation cannot be changed, ask it to cancel the reservation under your "
          "travel insurance, since {why}, and to book you a new trip on the flights you want instead.",
          "Should the change be refused because the ticket is basic economy, you want the trip cancelled with your travel "
          "insurance, as {why}, and a new booking made for the new date.")
REBOOK_WHY = ("you are sick with the flu and cannot fly on the original date",
              "your doctor has told you not to fly before {date} after a minor surgery",
              "you injured your back and cannot travel on the original date")
NEW_BOOKING = ("For the new booking: the same passengers and cabin, no checked bags, no travel insurance, paid with {card}.",
               "Book the new trip for the same passengers in the same cabin, without checked bags or insurance, and pay "
               "with {card}.")


ELSEWHERE = ("If the agent says the destination cannot be changed, ask it to cancel that reservation and book you a "
             "one-way trip from {o} to {d} on {date} instead, for the same passengers in the same cabin.",
             "Should a new destination not be possible, you want that reservation cancelled and a new one-way trip "
             "booked from {o} to {d} on {date}, same passengers and cabin.")
ELSEWHERE_PAY = ("No checked bags or travel insurance for the new trip; pay with {card}.",
                 "You need no bags or insurance on the new booking, and it goes on {card}.")


def reroute_rebook(db, r, rng):
    """Change flights to another destination, which the policy refuses (origin, destination and trip type stay), with
    the user's fallback: cancelling the reservation, which business class allows whatever the reason, and booking a
    one-way trip to the new destination on the same day, on the flights a preference singles out."""
    user = db.users[r.user_id]
    f = facts(db, r)
    cards = sorted(k for k, m in user.payment_methods.items() if m.source == "credit_card" and pay(user, k))
    # within a day of booking any cancellation is allowed, which would blur the basis
    if r.flight_type != "one_way" or r.cabin != "business" or not cards or not f["upcoming"] or f["age"] < 25:
        return None
    text = trip(db, r, rng)
    if text is None:
        return None
    date, tools = r.flights[0].date, AirlineTools(db)
    cities = sorted({x.destination for x in tools.search_direct_flight(r.origin, None, date)}
                    - {r.destination, r.origin})
    for city in rng.sample(cities, len(cities)):
        found = options(tools, r.origin, city, date, r.cabin, len(r.passengers))
        fits = [(p, h) for p in PREFER if p not in ("same flight", "later") for h in [single_out([found], p, rng)]
                if h is not False]
        if not fits:
            continue
        preference, hour = rng.choice(fits + 2 * [x for x in fits if "nonstop" in x[0]])
        legs = list(choose(found, preference, hour=hour)[0])
        card = rng.choice(cards)
        price = sum(db.flights[n].dates[d].prices[r.cabin] for n, d in legs) * len(r.passengers)
        spec = {"family": "reroute_rebook", "reservation": r.reservation_id, "destination": city, "date": date,
                "preference": preference, "payment": card, **({"hour": hour} if hour is not None else {})}
        lines = [rng.choice(ELSEWHERE).format(o=place(r.origin), d=place(city), date=day(date)),
                 rng.choice(PREFER[preference]).format(cabin=CABIN[r.cabin], date=day(date),
                                                       hour=clock(hour) if hour is not None else ""),
                 rng.choice(ELSEWHERE_PAY).format(card=pay(user, card)),
                 rng.choice(WHY_ANY).format(why=rng.choice(("your plans have changed", "you have to be elsewhere")))]
        calls = [_call("cancel_reservation", reservation_id=r.reservation_id),
                 _call("book_reservation", user_id=r.user_id, origin=r.origin, destination=city,
                       flight_type="one_way", cabin=r.cabin, flights=_flights(legs),
                       passengers=[p.model_dump() for p in r.passengers],
                       payment_methods=[{"payment_id": card, "amount": price}], total_baggages=0, nonfree_baggages=0,
                       insurance="no")]
        forbidden = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=r.cabin,
                          flights=_flights(legs), payment_id=card)
        return Part(spec, "a new destination, refused, then a cancellation business class allows and a new booking",
                    rng.choice(REROUTE).format(trip=text, city=place(city), old=place(r.destination)), lines, calls,
                    forbidden, tags={"same_od", "fallback", "cancel_ok", "book", "book_insurance"}
                    | pref_tags(preference, legs) | ({"book_pax"} if len(r.passengers) > 1 else set()))
    return None


WHY_ANY = ("If the agent asks why you are cancelling that trip, you say that {why}.",
           "Your reason for cancelling that trip is that {why}.")


def change_or_rebook(db, r, rng):
    """Change flights, refused for basic economy, with the user's fallback: a cancellation under travel insurance for
    their illness, which the policy allows, and a new booking of the flights they wanted."""
    user = db.users[r.user_id]
    cards = sorted(k for k, m in user.payment_methods.items() if m.source == "credit_card" and pay(user, k))
    text = trip(db, r, rng)
    if text is None or not cards or r.flight_type != "one_way" or not facts(db, r)["passenger"]:
        return None
    # the illness keeps the user from the original date, so the new one is later
    moved = _new_journey(db, r, rng, r.cabin, prefs=tuple(p for p in PREFER if p != "same flight"), later=True)
    if moved is None:
        return None
    _, date, preference, hour, legs, flights = moved
    card = rng.choice(cards)
    price = sum(db.flights[n].dates[d].prices[r.cabin] for n, d in flights) * len(r.passengers)
    spec = {"family": "change_or_rebook", "reservation": r.reservation_id, "journey": 0, "date": date, "cabin": r.cabin,
            "preference": preference, "payment": card, "reason": "health", **({"hour": hour} if hour is not None else {})}
    lines = [rng.choice(PREFER[preference]).format(cabin=CABIN[r.cabin], date=day(date),
                                                   hour=clock(hour) if hour is not None else ""),
             rng.choice(REBOOK).format(why=rng.choice(REBOOK_WHY).format(date=day(date))),
             rng.choice(NEW_BOOKING).format(card=pay(user, card))]
    calls = [_call("cancel_reservation", reservation_id=r.reservation_id),
             _call("book_reservation", user_id=r.user_id, origin=r.origin, destination=r.destination,
                   flight_type="one_way", cabin=r.cabin, flights=_flights(flights),
                   passengers=[p.model_dump() for p in r.passengers],
                   payment_methods=[{"payment_id": card, "amount": price}], total_baggages=0, nonfree_baggages=0,
                   insurance="no")]
    forbidden = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=r.cabin,
                      flights=_flights(flights), payment_id=card)
    return Part(spec, "a basic economy flight change, refused, then a cancellation under travel insurance and a new "
                "booking", rng.choice(MOVE).format(which=text, date=day(date)), lines, calls, forbidden,
                tags={"mod_flights", "basic_no_change", "fallback", "cancel_ok", "book"} | pref_tags(preference, legs)
                | ({"book_pax"} if len(r.passengers) > 1 else set()))
