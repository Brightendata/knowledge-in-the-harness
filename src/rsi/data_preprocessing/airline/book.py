"""Book flight: a trip on the flights a preference singles out, for the user and companions saved in the profile, with
free bags by membership and cabin, travel insurance, and payments within the policy's limits; the same flight as on an
earlier reservation."""
from tau2.domains.airline.tools import AirlineTools

from rsi.data_preprocessing.airline.facts import (CABIN, CITY, DOB_IN_PROFILE, FIRST_DAY, FREE_BAGS, FREE_COUNT,
                                                  HOW_LONG, LAST_DAY, _call, _count, _flights, birthday, choose, clock,
                                                  day, options, pay, place, pref_tags, sayable, single_out, trip)
from rsi.data_preprocessing.common import Part

BOOK = ("You want to book {kind} from {o} to {d}, {dates}, in {cabin}, for {who}.",
        "You are calling to book {kind} from {o} to {d} in {cabin}, {dates}, for {who}.",
        "You'd like to book {kind} in {cabin} from {o} to {d}, {dates}, for {who}.")
PREFER_BOOK = {
    "cheapest": ("You want the cheapest {cabin} option{each}, nonstop or with one stop.",
                 "Whichever {cabin} itinerary is cheapest{each} is fine, with or without a connection.",
                 "Price matters most: the cheapest {cabin} option{each}, even with a stop."),
    "cheapest nonstop": ("You only fly nonstop, and you want the cheapest nonstop {cabin} flight{each}.",
                         "It has to be nonstop; you want the cheapest nonstop {cabin} flight{each}.",
                         "You want the cheapest nonstop {cabin} flight{each}."),
    "earliest nonstop": ("You want the nonstop flight that leaves earliest in the day{each}.",
                         "It has to be nonstop, with the earliest departure of the day{each}.",
                         "You'd like the first nonstop flight of the day{each}."),
    "second cheapest": ("You want the second cheapest {cabin} option{each}, nonstop or with one stop; the cheapest one "
                        "is usually not great.",
                        "Not the cheapest, but the next cheapest {cabin} itinerary{each}, with or without a stop.",
                        "You'd like the second-lowest-priced {cabin} option{each}."),
    "fastest": ("You want the fastest itinerary{each}, counting the time of any layover.",
                "Total travel time matters most: the quickest option{each}, connection included.",
                "You'd like whichever option takes the least total time{each}."),
    "after": ("You cannot leave before {hour}; among the {cabin} options departing then or later{each}, you want the "
              "cheapest.",
              "It has to depart at {hour} or later{each}; of those, pick the cheapest {cabin} option.",
              "You can only fly from {hour} on{each}, and you want the cheapest {cabin} option leaving then or later."),
    "arrive by": ("You need to arrive by {hour} on the day you leave{each}; among the {cabin} options that do, you want "
                  "the cheapest.",
                  "It must get you there no later than {hour} the same day{each}; of those, the cheapest {cabin} option.",
                  "You want the cheapest {cabin} itinerary that arrives by {hour} on the day it leaves{each}."),
}
INSURANCE_MIND = {"yes": ("When the agent asks you to confirm the booking, change your mind: you do want travel insurance "
                          "after all.",
                          "Once the agent summarizes the booking, you reconsider and ask to add travel insurance.",
                          "At the confirmation step, decide that you want the insurance after all."),
                  "no": ("When the agent asks you to confirm the booking, change your mind: you do not want the travel "
                         "insurance after all.",
                         "Once the agent summarizes the booking, you reconsider and drop the travel insurance.",
                         "At the confirmation step, decide that you would rather not buy the insurance.")}
OTHERS_IN_PROFILE = ("The dates of birth of your companions are saved in your profile.",
                     "If asked for your companions' dates of birth, say they are in your profile.",
                     "You do not remember your companions' dates of birth; they are in your profile.")
BAGS = ("You will check {n} in total.", "You are bringing {n} to check in.", "You need to check {n} in total.")
NO_BAGS = ("You do not need any checked bags.", "You will not check any bags.", "No checked bags are needed.")
INSURANCE = {"yes": ("You want travel insurance.", "You'd like to add travel insurance.",
                     "You want the booking to include travel insurance."),
             "no": ("You do not want travel insurance.", "You decline travel insurance.",
                    "You will not buy travel insurance.")}
PAY_ONE = ("Pay with {a}.", "You want to pay with {a}.", "Use {a} for the whole booking.")
PAY_SPLIT = ("Use the full balance of {a} first and put the rest on {b}.",
             "Pay with all of {a}, and the remaining amount with {b}.", "First use up {a}, then charge the rest to {b}.")
ASK_TOTAL = ("Ask the agent for the total price.", "You want to know the total cost before confirming.",
             "Before confirming, ask how much the booking costs in total.")
TWO_CERTIFICATES = ("You want to pay with both of your travel certificates and put the rest on {card}.",
                    "You'd like to use your two travel certificates and pay the remainder with {card}.",
                    "Use {a} and {b}, and charge whatever is left to {card}.")
ONE_CERTIFICATE = ("If the agent says only one travel certificate can be used, use {a} and put the rest on {card}.",
                   "Should only one certificate be allowed, you use {a}, with the rest on {card}.",
                   "If two certificates are not allowed, pay with {a} and {card} instead.")
TWO_CARDS = ("You want to split the payment evenly between {a} and {b}.",
             "You'd like to pay half with {a} and half with {b}.",
             "Divide the cost equally between {a} and {b}.")
ONE_CARD = ("If the agent says you cannot pay with two credit cards, put everything on {a}.",
            "Should two cards not be allowed, pay the whole amount with {a}.",
            "If only one credit card can be used, use {a} for all of it.")
KNOWN_SELF = ("Your date of birth is {dob}.", "You were born on {dob}.", "Your date of birth: {dob}.")
KNOWN_OTHER = ("{name}'s date of birth is {dob}.", "{name} was born on {dob}.", "{name}: born {dob}.")


def _itinerary(db, user, rng):
    """A trip the user could book, with the preference that singles out its flights:
    (flight type, origin, destination, dates, cabin, passengers, preference, chosen itineraries, all options), or None."""
    tools = AirlineTools(db)
    origin, destination = rng.sample(sorted(CITY), 2)
    flight_type = rng.choice(("one_way", "round_trip"))
    out = rng.randint(FIRST_DAY, LAST_DAY - 2)
    dates = [f"2024-05-{out:02d}"]
    if flight_type == "round_trip":
        dates.append(f"2024-05-{min(LAST_DAY, out + rng.randint(2, 6)):02d}")
    cabin = rng.choice(list(CABIN))
    saved = [p for p in user.saved_passengers if (p.first_name, p.last_name) != (user.name.first_name, user.name.last_name)]
    people = [{"first_name": user.name.first_name, "last_name": user.name.last_name, "dob": user.dob}]
    people += [p.model_dump() for p in rng.sample(saved, min(len(saved), rng.choice((0, 1, 1, 2))))]
    routes = [(origin, destination, dates[0])] + ([(destination, origin, dates[1])] if len(dates) > 1 else [])
    found = [options(tools, o, d, date, cabin, len(people)) for o, d, date in routes]
    fits = [(p, h) for p in PREFER_BOOK for h in [single_out(found, p, rng)] if h is not False]
    if not fits:
        return None
    preference, hour = rng.choice(fits + 2 * [x for x in fits if "nonstop" in x[0]])  # official tasks favour nonstop
    chosen = [choose(f, preference, hour=hour) for f in found]
    return flight_type, origin, destination, dates, cabin, people, preference, hour, chosen, found


CAP = ("You will spend at most ${limit} on this booking; if it costs more for all of you, book it only for yourself.",
       "Your budget for the whole booking is ${limit}. If the total for everyone is higher, drop the others and book just "
       "for yourself.",
       "If the total for all passengers comes to more than ${limit}, you will book only for yourself.")
GIFT_CARDS = ("Pay with your gift cards, using up the one with the smallest balance first, then the next, and put the "
              "rest on {card}.",
              "Use all of your gift cards, smallest balance first, and charge whatever is left to {card}.",
              "Empty your gift cards first, starting with the smallest, and pay the remainder with {card}.")
CERTIFICATE_ONLY = ("Pay for the whole booking with {a} alone; nothing else should be charged.",
                    "You want {a} to cover the entire booking, with no other payment method.",
                    "Use only {a} for this booking.")
CERTIFICATE_GIFT = ("Use up {a} first, then {gifts}, and put whatever is left on {card}.",
                    "Pay with all of {a}, then {gifts}, and charge the remainder to {card}.",
                    "First {a}, then {gifts}; anything still owed goes on {card}.")
SUMS_FIRST = {1: ("Before anything is booked, you want to know the total balance of your {kind}.",
                  "First ask the agent how much your {kind} add up to; only then book."),
              2: ("Before anything is booked, you want to know the total balance of your gift cards and the total "
                  "balance of your travel certificates.",
                  "First ask the agent for the sum of your gift card balances and the sum of your certificate "
                  "balances; only then book.")}
GIFT_ONLY = ("Pay only with your gift cards, smallest balance first, taking from the last one just what is still owed; "
             "do not use a credit card.",
             "Use your gift cards and nothing else: the one with the smallest balance first, then the next, only as much "
             "as the booking costs.",
             "Cover the booking with your gift cards alone, starting with the smallest balance; no credit card.")


def book(db, user_id, rng, split=None, cap=None, sums=False):
    """Book flight: a reservation for at most five passengers, one cabin for all flights, free bags by membership and
    cabin, $30 per passenger for insurance, and at most one travel certificate, one credit card and three gift cards.
    With split, the user first asks to pay in a way the policy does not allow ("certificates" or "cards") and falls
    back to one it allows, or pays with all of their gift cards, smallest first, and a credit card ("gift_cards"); with
    one travel certificate that covers the whole price, the rest of it lost ("certificate"); with a travel certificate,
    then every gift card, then a credit card ("certificate_gift"); or with gift cards alone, smallest first, the last
    one only in part ("gift_only").
    With cap, the user books for companions too only if the total is within a limit: "within" or "over" it, in which
    case only the user is booked.
    With sums, the user first asks what their gift cards and travel certificates add up to, two or more of a kind,
    and then pays with some of them: the sums are those before the booking."""
    user = db.users[user_id]
    cards = sorted(k for k, m in user.payment_methods.items() if m.source == "credit_card" and pay(user, k))
    stored = sorted(k for k, m in user.payment_methods.items() if m.source != "credit_card" and pay(user, k))
    certs = [k for k in stored if user.payment_methods[k].source == "certificate"]
    gifts = sorted((k for k, m in user.payment_methods.items() if m.source == "gift_card"),
                   key=lambda k: user.payment_methods[k].amount)
    amounts = [user.payment_methods[k].amount for k in gifts]
    several = 2 <= len(gifts) <= 3 and len(set(amounts)) == len(amounts)  # "smallest first" names an order
    if ((not cards and split not in ("certificate", "gift_only")) or (split == "cards" and len(cards) < 2)
            or (split == "certificates" and len(certs) != 2) or (split in ("gift_cards", "gift_only") and not several)
            or (split == "certificate" and not certs)
            or (split == "certificate_gift" and not (certs and gifts and (len(gifts) == 1 or several)))):
        return None
    tools = AirlineTools(db)
    for _ in range(20):  # a random route and day often has no flights a preference singles out
        found = _itinerary(db, user, rng)
        if found is None:
            continue
        flight_type, origin, destination, dates, cabin, people, preference, hour, chosen, options_ = found
        if cap and len(people) < 2:
            continue
        if cap:  # the flights must be the same ones when the user flies alone
            routes = [(origin, destination, dates[0])] + ([(destination, origin, dates[1])] if len(dates) > 1 else [])
            alone = [choose(options(tools, o, d, date, cabin, 1), preference, hour=hour) for o, d, date in routes]
            if [a and a[0] for a in alone] != [c[0] for c in chosen]:
                continue
        free = FREE_BAGS[user.membership][cabin] * len(people)
        total_bags = 0 if cap else rng.randint(0, free + 2)
        nonfree = max(0, total_bags - free)
        asked_insurance = "no" if cap else rng.choice(("yes", "no"))
        insurance = asked_insurance  # what the user settles on: some change their mind when asked to confirm
        if not cap and not split and rng.random() < 0.25:
            insurance = "no" if asked_insurance == "yes" else "yes"
        price = sum(c[1] for c in chosen) * len(people) + (30 * len(people) if insurance == "yes" else 0) + 50 * nonfree
        if split == "certificates" and sum(user.payment_methods[c].amount for c in certs) >= price:
            continue
        if split == "gift_cards" and sum(amounts) >= price:
            continue
        if split == "certificate" and not any(user.payment_methods[c].amount >= price for c in certs):
            continue
        if split == "certificate_gift" and not any(user.payment_methods[c].amount + sum(amounts) < price for c in certs):
            continue
        if split == "gift_only" and not amounts[0] < price <= sum(amounts):  # two cards at least
            continue
        break
    else:
        return None
    each = " in each direction" if len(dates) > 1 else ""
    lines = [rng.choice(PREFER_BOOK[preference]).format(cabin=CABIN[cabin], each=each,
                                                        hour=clock(hour) if hour is not None else ""),
             rng.choice(BAGS).format(n=_count(total_bags, "bag")) if total_bags else rng.choice(NO_BAGS),
             rng.choice(INSURANCE[asked_insurance])]
    tags = {"book", "book_insurance"} | pref_tags(preference, chosen[0][0]) | ({"bag_allowance"} if total_bags else set())
    tags |= {"book_pax"} if len(people) > 1 else set()
    if insurance != asked_insurance:
        lines.append(rng.choice(INSURANCE_MIND[insurance]))
        tags.add("change_mind")
    if total_bags and rng.random() < 0.5:  # asked, not checked: a small number would match almost any reply
        lines.append(rng.choice(FREE_COUNT))
        tags.add("info_count")
    if rng.random() < (0.6 if preference == "fastest" else 0.2):
        lines.append(rng.choice(HOW_LONG))
        tags.add("info_attr")
    purpose = "booking the flights a preference singles out, with free bags by membership and cabin"
    asked = None
    if cap:
        alone_price = sum(c[1] for c in chosen)
        all_price = alone_price * len(people)
        limit = int(all_price * rng.uniform(1.1, 1.4)) if cap == "within" else int(rng.uniform(alone_price, all_price))
        limit -= limit % 10
        if not (limit >= all_price if cap == "within" else alone_price <= limit < all_price):
            return None
        card = rng.choice(cards)
        if cap == "over":
            people, price = people[:1], alone_price
        lines += [rng.choice(CAP).format(limit=limit), rng.choice(PAY_ONE).format(a=pay(user, card))]
        paid = [(card, price)]
        tags |= {"budget", "cond_value", "price_calc", "fallback"}
        purpose = f"booking for companions only within the user's limit: {cap} it"
    elif split == "gift_cards":
        card = rng.choice(cards)
        paid = [(k, int(user.payment_methods[k].amount)) for k in gifts]
        paid.append((card, price - sum(x for _, x in paid)))
        lines.append(rng.choice(GIFT_CARDS).format(card=pay(user, card)))
        tags |= {"book_pay", "pay_alloc"}
        purpose = "booking paid with every gift card, smallest balance first, and a credit card"
    elif split == "certificate":  # what the booking leaves of the certificate is not refunded
        a = rng.choice([c for c in certs if user.payment_methods[c].amount >= price])
        paid = [(a, price)]
        lines.append(rng.choice(CERTIFICATE_ONLY).format(a=pay(user, a)))
        tags.add("book_pay")
        purpose = "booking paid with one travel certificate, the rest of which is lost"
    elif split == "certificate_gift":
        a = rng.choice([c for c in certs if user.payment_methods[c].amount + sum(amounts) < price])
        card = rng.choice(cards)
        paid = [(a, int(user.payment_methods[a].amount))] + [(k, int(user.payment_methods[k].amount)) for k in gifts]
        paid.append((card, price - sum(x for _, x in paid)))
        named = pay(user, gifts[0]) if len(gifts) == 1 else "all of your gift cards, smallest balance first"
        lines.append(rng.choice(CERTIFICATE_GIFT).format(a=pay(user, a), gifts=named, card=pay(user, card)))
        tags |= {"book_pay", "pay_alloc"}
        purpose = "booking paid with a travel certificate, then every gift card, then a credit card"
    elif split == "gift_only":
        paid, left = [], price
        for k in gifts:
            if left:
                paid.append((k, min(left, int(user.payment_methods[k].amount))))
                left -= paid[-1][1]
        lines.append(rng.choice(GIFT_ONLY))
        tags |= {"book_pay", "pay_alloc"}
        purpose = "booking paid with gift cards alone, smallest balance first, the last one in part"
    elif split == "certificates":
        a = max(certs, key=lambda c: user.payment_methods[c].amount)
        card = rng.choice(cards)
        lines += [rng.choice(TWO_CERTIFICATES).format(a=pay(user, certs[0]), b=pay(user, certs[1]), card=pay(user, card)),
                  rng.choice(ONE_CERTIFICATE).format(a=pay(user, a), card=pay(user, card))]
        asked = [(c, int(user.payment_methods[c].amount)) for c in certs]
        asked.append((card, price - sum(x for _, x in asked)))
        paid = [(a, int(user.payment_methods[a].amount)), (card, price - int(user.payment_methods[a].amount))]
        tags |= {"book_pay", "pay_alloc", "fallback"}
        purpose = "booking paid with one travel certificate after the user asks to use two"
    elif split == "cards":
        a, b = rng.sample(cards, 2)
        lines += [rng.choice(TWO_CARDS).format(a=pay(user, a), b=pay(user, b)), rng.choice(ONE_CARD).format(a=pay(user, a))]
        asked = [(a, price // 2), (b, price - price // 2)]
        paid = [(a, price)]
        tags |= {"book_pay", "fallback"}
        purpose = "booking paid with one credit card after the user asks to use two"
    else:
        card = rng.choice(cards)
        partial = [k for k in stored if user.payment_methods[k].amount < price]
        if partial and rng.random() < 0.6:
            first = rng.choice(partial)
            paid = [(first, int(user.payment_methods[first].amount))]
            paid.append((card, price - paid[0][1]))
            lines.append(rng.choice(PAY_SPLIT).format(a=pay(user, first), b=pay(user, card)))
            tags.add("pay_alloc")
        else:
            paid = [(card, price)]
            lines.append(rng.choice(PAY_ONE).format(a=pay(user, card)))
    spec = {"family": "book", "user": user_id, "flight_type": flight_type, "origin": origin, "destination": destination,
            "dates": dates, "cabin": cabin, "passengers": people, "preference": preference, "bags": total_bags,
            "insurance": insurance, "payment": [k for k, _ in paid],
            **({"hour": hour} if hour is not None else {}), **({"cap": cap, "limit": limit} if cap else {})}
    shown = {c[1] for f in options_ for c in f} | {n for f in options_ for c in f for n, _ in c[0]}
    shown |= {p for c in chosen for n, d in c[0] for p in db.flights[n].dates[d].prices.values()}
    amount = sayable(price, db, user_id, shown)
    info, ask = ([amount], rng.choice(ASK_TOTAL)) if amount and rng.random() < 0.6 else ([], "")
    if info:
        tags |= {"price_calc", "info_amount"}
    if sums:
        totals = {}
        for source in ("gift_card", "certificate"):
            held = [m.amount for m in user.payment_methods.values() if m.source == source]
            said = sayable(sum(held), db, user_id, shown | {price} | {x for _, x in paid})
            if len(held) >= 2 and said:
                totals[source] = said
        if not set(totals) & {user.payment_methods[k].source for k, _ in paid}:
            return None
        kind = "gift cards" if "gift_card" in totals else "travel certificates"
        lines.insert(0, rng.choice(SUMS_FIRST[len(totals)]).format(kind=kind))
        info = list(totals.values()) + info
        tags |= {"balance_sum", "info_balance", "balance_pay"}
        spec["sums"] = totals
        purpose += "; the user first asks for the sums of the balances it spends"
    companions = found[5][1:]  # the ones the user asks to book, even if a cap leaves them out
    who = ["yourself"] + [f"{p['first_name']} {p['last_name']}" for p in companions]
    who = who[0] if len(who) == 1 else ", ".join(who[:-1]) + " and " + who[-1]
    request = rng.choice(BOOK).format(
        kind="a one-way trip" if flight_type == "one_way" else "a round trip", o=place(origin), d=place(destination),
        dates=f"on {day(dates[0])}" if len(dates) == 1 else f"leaving {day(dates[0])} and returning {day(dates[1])}",
        cabin=CABIN[cabin], who=who)
    if rng.random() < 0.35:  # the agent has to look the date of birth up
        known = [rng.choice(DOB_IN_PROFILE)]
        tags.add("book_pax")
    else:
        known = [rng.choice(KNOWN_SELF).format(dob=birthday(user.dob))]
    if companions and rng.random() < 0.5:
        known.append(rng.choice(OTHERS_IN_PROFILE))
    else:
        known += [rng.choice(KNOWN_OTHER).format(name=f"{p['first_name']} {p['last_name']}", dob=birthday(p["dob"]))
                  for p in companions]
    arguments = dict(user_id=user_id, origin=origin, destination=destination, flight_type=flight_type, cabin=cabin,
                     flights=_flights(leg for c in chosen for leg in c[0]), passengers=people,
                     total_baggages=total_bags, nonfree_baggages=nonfree, insurance=insurance)

    def booking(payments):
        return _call("book_reservation", **arguments, payment_methods=[{"payment_id": k, "amount": x} for k, x in payments])

    forbidden = booking(asked) if asked else None
    if cap == "over":  # booking everyone would break the user's limit
        forbidden = _call("book_reservation", **{**arguments, "passengers": found[5]},
                          payment_methods=[{"payment_id": paid[0][0], "amount": price * len(found[5])}])
    return Part(spec, purpose, request, lines, [booking(paid)], forbidden, info, ask, known=known, tags=tags,
                by_policy=cap != "over")


AGAIN = ("You want to book the exact same flight as on {trip} again, this time on {date}, in {cabin}, for yourself only.",
         "You'd like to fly the same flight as on {trip} once more, on {date}; book it in {cabin}, just for you.",
         "Book you on the same flight you have on {trip}, but on {date}, in {cabin}.")
AGAIN_UNKNOWN = ("You do not remember that flight's number; the agent can find it in that reservation.",
                 "You don't know the flight number, only the trip it was on.", "")


FRIEND = ("You want to book {friend} on exactly the same flight as {trip}, same day and cabin; only {friend} travels, "
          "not you.", "Book {friend}, who is in your profile, on the very same flight as {trip}, same date and cabin, "
                      "just for {friend}.")


def book_again(db, user_id, rng, friend=False):
    """Book flight, the same nonstop flight as on one of the user's reservations, on another day for the user, or with
    friend on the same day for a passenger saved in their profile: the agent finds the flight number in that
    reservation."""
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
        number = r.flights[0].flight_number
        if friend:
            on = {(p.first_name, p.last_name) for p in r.passengers}
            others = [p.model_dump() for p in user.saved_passengers if (p.first_name, p.last_name) not in on
                      and (p.first_name, p.last_name) != (me["first_name"], me["last_name"])]
            if not others:
                continue
            days = [int(r.flights[0].date[-2:])] if FIRST_DAY <= int(r.flights[0].date[-2:]) <= LAST_DAY else []
        else:
            days = [d for d in range(FIRST_DAY, LAST_DAY + 1) if f"2024-05-{d:02d}" != r.flights[0].date]
            rng.shuffle(days)
        for d in days:
            date = f"2024-05-{d:02d}"
            x = db.flights[number].dates.get(date)
            if text is None or x is None or x.status != "available" or x.available_seats[r.cabin] < 1:
                continue
            card = rng.choice(cards)
            price = x.prices[r.cabin]
            who = rng.choice(others) if friend else me
            spec = {"family": "book_again", "user": user_id, "reservation": rid, "date": date, "cabin": r.cabin,
                    "payment": [card], "passenger": who}
            lines = [x for x in (rng.choice(AGAIN_UNKNOWN), rng.choice(NO_BAGS), rng.choice(INSURANCE["no"]),
                                 rng.choice(PAY_ONE).format(a=pay(user, card))) if x]
            call = _call("book_reservation", user_id=user_id, origin=r.origin, destination=r.destination,
                         flight_type="one_way", cabin=r.cabin, flights=_flights([(number, date)]), passengers=[who],
                         payment_methods=[{"payment_id": card, "amount": price}], total_baggages=0, nonfree_baggages=0,
                         insurance="no")
            if friend:
                name = f"{who['first_name']} {who['last_name']}"
                return Part(spec, "booking a saved passenger on the flight of one of the user's reservations",
                            rng.choice(FRIEND).format(friend=name, trip=text), lines, [call],
                            known=[rng.choice(OTHERS_IN_PROFILE)],
                            tags={"book", "sel_same_as", "search_direct", "book_insurance", "book_pax", "find_res"})
            return Part(spec, "booking the flight of an earlier reservation on another day",
                        rng.choice(AGAIN).format(trip=text, date=day(date), cabin=CABIN[r.cabin]), lines, [call],
                        known=[rng.choice(DOB_IN_PROFILE)],
                        tags={"book", "sel_same_as", "search_direct", "book_insurance", "book_pax"})
    return None
