"""What the airline requests share: the database facts about a reservation, how its owner names a reservation or a
payment method, finding flights and the one a preference singles out, the amounts the agent states, and sentences
several requests use."""
from datetime import datetime

from tau2.domains.airline.tools import AirlineTools

CITY = {a.iata: a.city for a in AirlineTools(None).list_all_airports()}
CABIN = {"basic_economy": "basic economy", "economy": "economy", "business": "business class"}
FREE_BAGS = {  # free checked bags per passenger, by membership and cabin
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}
NOW = datetime(2024, 5, 15, 15)  # the policy's current time
FIRST_DAY, LAST_DAY = 16, 30  # flights can be booked from May 16 to May 30


# ---------------------------------------------------------------- facts

def status(db, flight):
    return db.flights[flight.flight_number].dates[flight.date].status


def journeys(r):
    """The flights of the outbound journey and, for a round trip, of the return journey."""
    if r.flight_type == "one_way":
        return [r.flights]
    k = next(i for i, f in enumerate(r.flights) if f.origin == r.destination)
    return [r.flights[:k], r.flights[k:]]


def facts(db, r):
    statuses = [status(db, f) for f in r.flights]
    user = db.users[r.user_id]
    return {
        "flown": any(s in ("landed", "flying") for s in statuses),
        "upcoming": all(s == "available" for s in statuses),
        "airline": "cancelled" in statuses,
        "delayed": "delayed" in statuses,
        "age": (NOW - datetime.fromisoformat(r.created_at)).total_seconds() / 3600,
        "eligible": user.membership in ("silver", "gold") or r.insurance == "yes" or r.cabin == "business",
        "passenger": any((p.first_name, p.last_name) == (user.name.first_name, user.name.last_name) for p in r.passengers),
    }


def borderline(db, r):
    """Whether the user may count the reservation as upcoming though not all of its flights can still be booked: some
    flight is still to come (later today, or after one already flown or cancelled)."""
    statuses = [status(db, f) for f in r.flights]
    return any(s in ("available", "on time", "delayed") for s in statuses) and any(s != "available" for s in statuses)


def disrupted(db, user_id):
    """The user's reservations with a flight the airline cancelled or delayed."""
    return [rid for rid in db.users[user_id].reservations
            if any(status(db, f) in ("cancelled", "delayed") for f in db.reservations[rid].flights)]


def place(code):
    return f"{CITY[code]} ({code})"


def day(date):
    return f"May {int(date[-2:])}"


def birthday(dob):
    months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December")
    y, m, d = dob.split("-")
    return f"{months[int(m) - 1]} {int(d)}, {y}"


TRIP = {
    "one_way": ("your one-way trip from {o} to {d} on {out}", "your one-way flight from {o} to {d} on {out}",
                "your {out} one-way trip from {o} to {d}"),
    "round_trip": ("your round trip from {o} to {d}, leaving {out} and coming back {back}",
                   "your round trip from {o} to {d} ({out} out, {back} back)",
                   "your {o} to {d} round trip that leaves {out} and returns {back}"),
}


BY_ID = ("reservation {rid}", "your reservation {rid}", "booking {rid}")
ID_SHARE = 0.5  # per request; the official tasks give every reservation they need by its id in about two of five


def trip(db, r, rng):
    """The reservation as its owner names it: by its id, as in about two of five official tasks, or by its trip, or
    None if that description would fit another of their reservations."""
    if rng.random() < ID_SHARE:
        return rng.choice(BY_ID).format(rid=r.reservation_id)
    js = journeys(r)
    key = (r.origin, r.destination, r.flights[0].date)
    for other in db.users[r.user_id].reservations:
        o = db.reservations[other]
        if other != r.reservation_id and (o.origin, o.destination, o.flights[0].date) == key:
            return None
    back = day(js[1][0].date) if len(js) > 1 else None
    return rng.choice(TRIP[r.flight_type]).format(o=place(r.origin), d=place(r.destination), out=day(js[0][0].date),
                                                  back=back)


def pay(user, payment_id):
    """A payment method as its owner names it, or None if the name would fit another of their methods."""
    m = user.payment_methods[payment_id]
    if m.source == "credit_card":
        same = [x for x in user.payment_methods.values() if x.source == "credit_card" and x.last_four == m.last_four]
        name = f"your {m.brand.title()} credit card ending in {m.last_four}"
    else:
        same = [x for x in user.payment_methods.values() if x.source == m.source]
        kind = "gift card" if m.source == "gift_card" else "travel certificate"
        name = f"your {kind}" if len(same) == 1 else f"your ${m.amount:g} {kind}"
        same = same if len(same) == 1 else [x for x in same if x.amount == m.amount]
    return name if len(same) == 1 else None


def payer(user, amount, rng):
    """A credit card, or a gift card that covers the amount, that the user can name; the tools take no travel
    certificate for a change."""
    fits = sorted(k for k, m in user.payment_methods.items()
                  if (m.source == "credit_card" or (m.source == "gift_card" and m.amount >= amount)) and pay(user, k))
    return rng.choice(fits) if fits else None


# ---------------------------------------------------------------- flights and the one a preference singles out

def minutes(clock, day=0):
    """Minutes from midnight of the first flight's day for a schedule time such as '03:00:00+1'."""
    t, _, plus = clock.partition("+")
    h, m, _ = map(int, t.split(":"))
    return (day + int(plus or 0)) * 1440 + 60 * h + m


def options(tools, origin, destination, date, cabin, passengers):
    """Every direct or one-stop itinerary the search tools return with seats for all passengers, as (flights as
    ((number, date), ...), price per passenger, departure time, arrival in minutes from midnight of the departure day).
    The second flight of a one-stop itinerary may leave the next day."""
    found = [(((f.flight_number, date),), f.prices[cabin], f.scheduled_departure_time_est,
              minutes(f.scheduled_arrival_time_est))
             for f in tools.search_direct_flight(origin, destination, date) if f.available_seats[cabin] >= passengers]
    for a, b in tools.search_onestop_flight(origin, destination, date):
        if min(a.available_seats[cabin], b.available_seats[cabin]) >= passengers:
            later = (int(b.date[-2:]) - int(date[-2:]))
            found.append((((a.flight_number, a.date), (b.flight_number, b.date)), a.prices[cabin] + b.prices[cabin],
                          a.scheduled_departure_time_est, minutes(b.scheduled_arrival_time_est, later)))
    return found


# preferences that take an hour: the earliest departure, or the latest arrival on the departure day
TIMED = ("after", "arrive by")


def choose(found, preference, current=None, hour=None):
    """The one itinerary the preference singles out among at least two candidates, or None."""
    if preference == "same flight":
        hit = [o for o in found if [n for n, _ in o[0]] == current]
        return hit[0] if len(hit) == 1 else None
    if preference == "fastest":
        pool, key = found, lambda o: o[3] - minutes(o[2])
    elif preference == "second cheapest":
        prices = sorted({o[1] for o in found})
        hit = [o for o in found if len(prices) > 2 and o[1] == prices[1]]
        return hit[0] if len(hit) == 1 and sum(o[1] == prices[0] for o in found) == 1 else None
    elif preference in TIMED:
        fits = (lambda o: minutes(o[2]) >= 60 * hour) if preference == "after" else (lambda o: o[3] <= 60 * hour)
        pool, key = [o for o in found if fits(o)], lambda o: o[1]
        if len(pool) == len(found):  # the hour has to rule something out
            return None
    else:
        pool = found if preference == "cheapest" else [o for o in found if len(o[0]) == 1]
        key = (lambda o: o[1]) if preference.startswith("cheapest") else (lambda o: o[2])
    best = [o for o in pool if key(o) == min((key(x) for x in pool), default=None)]
    return best[0] if len(pool) > 1 and len(best) == 1 else None


def hours(found, preference, rng):
    """An hour for a timed preference that singles out one itinerary of found, or None."""
    hs = list(range(7, 21)) if preference == "after" else list(range(10, 24))
    rng.shuffle(hs)
    return next((h for h in hs if choose(found, preference, hour=h)), None)


def single_out(found, preference, rng, current=None):
    """The hour a timed preference needs (None for the others) if the preference singles out one itinerary in each of
    found (a list, one per journey), else False."""
    if preference in TIMED:
        h = hours(found[0], preference, rng)
        return h if h is not None and all(choose(f, preference, hour=h) for f in found) else False
    return None if all(choose(f, preference, current) for f in found) else False


PREFER_TAGS = {"same flight": {"sel_same_as"}, "cheapest": {"sel_cheapest"},
               "cheapest nonstop": {"sel_cheapest"}, "earliest nonstop": {"sel_time"},
               "second cheapest": {"sel_cheapest"}, "fastest": {"sel_time", "duration"}, "after": {"sel_time", "sel_cheapest"},
               "arrive by": {"sel_time", "sel_cheapest"}, "later": {"sel_time"}}  # its next-day fallback never applies


def pref_tags(preference, legs):
    """Tags of a flight choice: what singles it out, and the searches: nonstop flights always, one-stop ones when the
    choice has a stop, as in the reference actions of the official tasks."""
    return PREFER_TAGS[preference] | {"search_direct"} | ({"search_onestop"} if len(legs) > 1 else set())


def clock(hour):
    return f"{hour % 12 or 12}{'am' if hour < 12 else 'pm'}"


# ---------------------------------------------------------------- amounts and calls

def listed(tools, origin, destination, date):
    """Every price and flight number the search tools list for the route on that day, which the agent may mention."""
    shown = set()
    for f in tools.search_direct_flight(origin, destination, date):
        shown |= set(f.prices.values()) | {f.flight_number}
    for a, b in tools.search_onestop_flight(origin, destination, date):
        shown |= set(a.prices.values()) | set(b.prices.values()) | {a.prices[c] + b.prices[c] for c in a.prices}
        shown |= {a.flight_number, b.flight_number}
    return shown


def sayable(amount, db, user_id, shown=()):
    """The amount as tau2's communicate check looks for it in the agent's replies, or None where a reply could contain it
    by chance: under three digits, or inside something the agent is likely to say anyway (a price, payment or balance of
    the customer's, a price or flight number the search tools list, a card's last digits, an ID)."""
    s = str(abs(int(amount)))
    user = db.users[user_id]
    seen = {x if isinstance(x, str) else str(int(x)) for x in shown} | set(user.payment_methods) | {user_id}
    seen |= {str(int(m.amount)) for m in user.payment_methods.values() if m.source != "credit_card"}
    seen |= {m.last_four for m in user.payment_methods.values() if m.source == "credit_card"}
    for rid in user.reservations:
        r = db.reservations[rid]
        seen |= {rid} | {f.flight_number for f in r.flights}
        seen |= {str(f.price) for f in r.flights} | {str(p.amount) for p in r.payment_history}
        seen.add(str(sum(p.amount for p in r.payment_history)))
    return s if len(s) >= 3 and not any(s in x for x in seen) else None


def _difference(db, r, cabin, flights):
    """What update_reservation_flights charges (or refunds, if negative): flights it keeps in the same cabin keep their
    price, the others cost today's price in the new cabin."""
    kept = {(f.flight_number, f.date): f.price for f in r.flights} if cabin == r.cabin else {}
    new = sum(kept[n, d] if (n, d) in kept else db.flights[n].dates[d].prices[cabin] for n, d in flights)
    return (new - sum(f.price for f in r.flights)) * len(r.passengers)


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


def _flights(pairs):
    return [{"flight_number": n, "date": d} for n, d in pairs]


WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _count(n, noun):
    return f"{WORDS[n - 1] if n <= len(WORDS) else n} {noun}{'s' if n > 1 else ''}"


# ---------------------------------------------------------------- sentences several requests use
# Each slot has a few wordings; a task picks one per slot with its own random.Random(task id).

# Facts the policy conditions on, as the user asserts them. The claims of a refused request are false; a granted request
# sometimes comes with a true one, so that a claim alone does not tell the agent to refuse.
CLAIM = {
    "recent": ("You insist that you made this booking less than a day ago.",
               "Claim that you booked the trip only a few hours ago.",
               "Say that you booked it within the last 24 hours, so it must be free to cancel."),
    "insured": ("Claim that you bought travel insurance with this booking.",
                "Insist that the reservation includes travel insurance.",
                "Say you are sure you added insurance when you booked."),
    "business": ("Say that the tickets are business class.", "Insist that you paid for business class on this trip.",
                 "Claim that this is a business class reservation."),
    "airline": ("Claim that the airline cancelled one of your flights.",
                "Say you were told that the airline cancelled one of your flights.",
                "Insist that the airline itself cancelled part of the trip."),
    "gold": ("Claim that you are a gold member.", "Insist that your membership is gold.",
             "Say you have gold status with the airline."),
}
PAY_CHANGE = ("Any fare difference should go on {pay}.", "Use {pay} to pay or receive any price difference.",
              "If there is a price difference, settle it with {pay}.")
COST = ("Ask the agent how much extra you will pay.", "You want to know exactly how much more it will cost.",
        "Before confirming, ask how much you will be charged.")
BACK = ("Ask the agent how much you will get back.", "You want to know exactly how much will be refunded.",
        "Before confirming, ask how much money you will get back.")
HOW_LONG = ("Ask the agent how long the trip will take in total, layover included.",
            "You want to know the total travel time of the flights the agent picks.",
            "Have the agent tell you how many hours the whole journey takes.")
# "that trip" names the reservation once a task has several requests (assemble), so that each reason goes with its trip
WHY = ("If the agent asks why you are cancelling that trip, you say that {why}.",
       "Your reason for cancelling that trip is that {why}.",
       "When asked why you want to cancel that trip, you explain that {why}.")
PLANS = ("your plans have changed", "the event you were travelling for has been called off",
         "you have to stay home for work", "you no longer need the trip")
SICK = ("you are sick with the flu and cannot travel", "you were injured in a fall and cannot travel",
        "you need surgery and your doctor told you not to travel")
FREE_COUNT = ("Ask the agent how many checked bags you get for free.", "You want to know how many of your bags are free.",
              "Before booking, ask how many bags are included for free.")
FREE_COUNT_ADDED = FREE_COUNT[:2] + ("Before confirming, ask how many of your bags are free.",)
DOB_IN_PROFILE = ("Your date of birth is in your profile, so you would rather not give it.",
                  "You prefer not to say your date of birth: it is in your user profile.",
                  "If asked for your date of birth, tell the agent it is in your profile.")
