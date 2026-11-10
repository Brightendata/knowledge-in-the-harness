"""Change cabin, baggage, insurance and passengers: an upgrade or downgrade of the whole reservation (within a budget,
after asking for one journey or one passenger, or for every upcoming business class reservation); checked bags added,
or removed (refused); insurance after booking (refused); a passenger corrected or replaced, or removed (refused, with a
fallback); and several of these in the order the user gives."""
from rsi.data_preprocessing.airline.facts import (BACK, CABIN, COST, DOB_IN_PROFILE, FREE_BAGS, FREE_COUNT_ADDED,
                                                  PAY_CHANGE, PLANS, WHY, WORDS, _call, _count, _difference, _flights,
                                                  birthday, borderline, facts, pay, payer, sayable, trip)
from rsi.data_preprocessing.common import Part

# ---------------------------------------------------------------- change cabin

# The cabin never directly follows {trip}: a round trip's description ends with its return date.
UPGRADE = ("You want to upgrade to {cabin} on {trip}.", "You would like {cabin} for {trip}.",
           "You are calling to upgrade {trip}: you want {cabin}.")
DOWNGRADE = ("You want to save money by changing to {cabin} on {trip}.", "You would like to downgrade to {cabin} on {trip}.",
             "You are calling to move {trip} down: you want {cabin}.")
BUDGET = ("You will only upgrade if the extra cost for the whole reservation is at most ${budget}; otherwise you keep "
          "your current cabin.",
          "Your limit is ${budget} in total for the upgrade; if it costs more, you keep your current cabin.",
          "If the upgrade costs more than ${budget} altogether, you do not want it and keep your current cabin.")
ASK_UPGRADE = ("Ask the agent for the exact price difference.", "You want to know the exact amount before deciding.",
               "Ask how much the upgrade would cost in total.")
RANK = {"basic_economy": 0, "economy": 1, "business": 2}


def cabin_change(db, r, rng, budget=False):
    """Change cabin: any reservation whose flights are all still to come can change cabin for all of its flights, paying
    or being refunded the difference. With a budget, the user takes an upgrade only up to a price."""
    text = trip(db, r, rng)
    if text is None:
        return None
    user = db.users[r.user_id]
    free = {c: FREE_BAGS[user.membership][c] * len(r.passengers) for c in CABIN}
    # A downgrade keeps within the new free allowance and an upgrade has no paid bags, so that the agent has no reason
    # to touch the bags, which the policy says nothing about here.
    cabins = [c for c in CABIN if c != r.cabin and (RANK[c] > RANK[r.cabin] and r.nonfree_baggages == 0
                                                     or RANK[c] < RANK[r.cabin] and r.total_baggages <= free[c])]
    cabins = [c for c in cabins if not budget or RANK[c] > RANK[r.cabin]]
    if not cabins:
        return None
    new = rng.choice(cabins)
    flights = [(f.flight_number, f.date) for f in r.flights]
    if any(db.flights[n].dates[d].available_seats[new] < len(r.passengers) for n, d in flights):
        return None
    difference = _difference(db, r, new, flights)
    pid = payer(user, difference, rng)
    if pid is None or difference == 0:
        return None
    spec = {"family": "cabin", "reservation": r.reservation_id, "cabin": new, "payment": pid}
    up = RANK[new] > RANK[r.cabin]
    request = rng.choice(UPGRADE if up else DOWNGRADE).format(trip=text, cabin=CABIN[new])
    lines = [rng.choice(PAY_CHANGE).format(pay=pay(user, pid))]
    shown = {p for n, d in flights for p in db.flights[n].dates[d].prices.values()}
    amount = sayable(difference, db, r.user_id, shown)
    call = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=new, flights=_flights(flights),
                 payment_id=pid)
    if budget:
        within = rng.random() < 0.5
        limit = int(difference * rng.uniform(1.15, 1.5) if within else difference * rng.uniform(0.5, 0.85))
        limit -= limit % 10
        amount = sayable(difference, db, r.user_id, shown | {limit})
        # rounding down to tens can put a small upgrade's "within" budget below its price
        if amount is None or limit <= 0 or within != (limit > difference):
            return None
        spec["budget"] = limit
        lines = [rng.choice(BUDGET).format(budget=limit)] + lines
        tags = {"cabin_change", "cabin_diff", "budget", "cond_value", "price_calc", "info_amount"}
        if within:
            return Part(spec, "cabin upgrade within the user's budget", request, lines, [call], info=[amount],
                        ask=rng.choice(ASK_UPGRADE), tags=tags | {"mod_pay"})
        return Part(spec, "cabin upgrade over the user's budget: the user only asks for the price", request, lines,
                    forbidden=call, info=[amount], ask=rng.choice(ASK_UPGRADE), by_policy=False, tags=tags)
    part = Part(spec, f"cabin {'upgrade' if up else 'downgrade'} the policy allows", request, lines, [call],
                tags={"cabin_change", "cabin_diff", "mod_pay"})
    if amount and rng.random() < 0.6:
        part.ask, part.info = rng.choice(COST if difference > 0 else BACK), [amount]
        part.tags |= {"price_calc", "info_amount"}
    return part


SEGMENT = ("You want to upgrade to {cabin} only the {which} of {trip}.",
           "You'd like {cabin} just for the {which} of {trip}.", "For {trip}, you want {cabin} on the {which} only.")
ONE_PERSON = ("You want to upgrade only {name}'s seat to {cabin} on {trip}.",
              "You'd like to move just {name} up to {cabin} on {trip}.",
              "On {trip}, only {name} should fly {cabin}.")
WHOLE_IF = ("If that is not possible, you are willing to upgrade the whole reservation, as long as the extra cost is at "
            "most ${budget} in total; otherwise keep it as it is.",
            "Should only a part not be possible, upgrade everything instead, but only if that costs ${budget} or less "
            "altogether; if more, leave the reservation unchanged.",
            "If the agent says it has to be the whole reservation, you accept that for at most ${budget} in total, and "
            "otherwise keep your current cabin.")


def segment_upgrade(db, r, rng, within):
    """Change cabin, the whole reservation only: the user asks to upgrade one journey or one passenger and falls back
    on the whole reservation if it costs at most a limit (within) or keeps it as it is (not within)."""
    text = trip(db, r, rng)
    up = {"basic_economy": "economy", "economy": "business"}.get(r.cabin)
    if text is None or up is None or r.nonfree_baggages:
        return None
    user, flights = db.users[r.user_id], [(f.flight_number, f.date) for f in r.flights]
    if any(db.flights[n].dates[d].available_seats[up] < len(r.passengers) for n, d in flights):
        return None
    if r.flight_type == "round_trip" and (len(r.passengers) == 1 or rng.random() < 0.5):
        request = rng.choice(SEGMENT).format(which=rng.choice(("outbound flight", "return flight")), trip=text,
                                             cabin=CABIN[up])
    elif len(r.passengers) > 1:
        p = rng.choice(r.passengers)
        request = rng.choice(ONE_PERSON).format(name=f"{p.first_name} {p.last_name}", trip=text, cabin=CABIN[up])
    else:
        return None
    difference = _difference(db, r, up, flights)
    pid = payer(user, difference, rng)
    limit = int(difference * (rng.uniform(1.15, 1.5) if within else rng.uniform(0.5, 0.85)))
    limit -= limit % 10
    if pid is None or limit <= 0 or within != (limit >= difference):
        return None
    spec = {"family": "segment_upgrade", "reservation": r.reservation_id, "cabin": up, "budget": limit, "payment": pid}
    lines = [rng.choice(WHOLE_IF).format(budget=limit), rng.choice(PAY_CHANGE).format(pay=pay(user, pid))]
    call = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=up, flights=_flights(flights),
                 payment_id=pid)
    tags = {"cabin_whole", "fallback", "cabin_change", "cabin_diff", "budget", "cond_value"}
    if within:
        return Part(spec, "an upgrade of part of a reservation, then of all of it within the user's limit", request, lines,
                    [call], tags=tags | {"mod_pay"})
    return Part(spec, "an upgrade of part of a reservation, then of all of it over the user's limit", request, lines,
                forbidden=call, by_policy=False, tags=tags)


SCOPE_DOWN = ("You want to downgrade all of your upcoming business class reservations to economy, keeping the same "
              "flights and passengers.",
              "Money is tight: change every upcoming business class trip of yours to economy, same flights.",
              "You'd like all your upcoming business class bookings moved down to economy without changing flights.")
REFUND_EACH = ("Each refund can go back to the payment method used for that reservation.",
               "You are fine with every refund going to the original payment of its reservation.",
               "Refund each one to whatever paid for it.")
SAVED = ("Ask the agent how much you will get back in total.", "You want to know the total amount you save.",
         "Once the downgrades are done, ask for the total refund of all of them.")


def scope_downgrade(db, user_id, rng):
    """Change cabin, for every upcoming business class reservation: each downgrade refunds its own difference to the
    payment method that paid for it."""
    user = db.users[user_id]
    business = [db.reservations[x] for x in user.reservations if db.reservations[x].status != "cancelled"
                and db.reservations[x].cabin == "business"]
    hit = [r for r in business if facts(db, r)["upcoming"]]
    if len(hit) < 2 or any(borderline(db, r) for r in business):
        return None
    calls, total = [], 0
    for r in hit:
        flights = [(f.flight_number, f.date) for f in r.flights]
        free = FREE_BAGS[user.membership]["economy"] * len(r.passengers)
        pid = r.payment_history[0].payment_id
        if (r.total_baggages > free or r.nonfree_baggages or user.payment_methods[pid].source == "certificate"
                or any(db.flights[n].dates[d].available_seats["economy"] < len(r.passengers) for n, d in flights)):
            return None
        total += _difference(db, r, "economy", flights)
        calls.append(_call("update_reservation_flights", reservation_id=r.reservation_id, cabin="economy",
                           flights=_flights(flights), payment_id=pid))
    amount = sayable(-total, db, user_id)
    spec = {"family": "scope_downgrade", "user": user_id, "reservations": [r.reservation_id for r in hit]}
    part = Part(spec, "downgrading every upcoming business class reservation, each refunded to its own payment",
                rng.choice(SCOPE_DOWN), [rng.choice(REFUND_EACH)], calls,
                tags={"scope_all", "filter_res", "find_res", "cabin_change", "cabin_diff", "mod_pay"})
    if amount:
        part.ask, part.info = rng.choice(SAVED), [amount]
        part.tags |= {"price_calc", "info_amount"}
    return part


# ---------------------------------------------------------------- change baggage and insurance

ADD_BAGS = ("You want to add {k} to {trip}.", "You need {more} on {trip}.",
            "You want {total} checked bags in total on {trip}.")
PAY_BAGS = ("Pay any baggage fee with {pay}.", "If there is a fee for the bags, use {pay}.",
            "Charge any bag fees to {pay}.")
ASK_BAGS = ("Ask the agent how much the extra bags will cost.", "You want to know exactly what the extra bags cost.",
            "Before confirming, ask for the total baggage fee.")
# the count is asked for but not checked: a small number would match almost any reply
ASK_BAGS_COUNT = ("Ask the agent how many of your checked bags are free and how much the others will cost in total.",
                  "You want to know how many bags you can check for free on this trip and the total fee for the rest.",
                  "Before confirming, ask how many of the bags you will pay for and what that comes to.")
# {k} is the number of bags removed out of {total} on the reservation; the refund is asked for only where some are paid.
REMOVE_BAGS = ("You want to remove {k} of the {total} checked bags on {trip}{refund}.",
               "You no longer need {k} of the {total} checked bags on {trip}{refund}.",
               "You want to take {k} of the {total} checked bags off {trip}{refund}.")
REMOVE_REFUND = (" and want a refund for them", " and want your money back for them", " and would like to be refunded")


def bags(db, r, rng, remove=False):
    """Change baggage: bags can be added, the ones beyond the free allowance of the user's membership and the cabin at
    $50 each, but not removed."""
    text = trip(db, r, rng)
    if text is None:
        return None
    user = db.users[r.user_id]
    free = FREE_BAGS[user.membership][r.cabin] * len(r.passengers)
    if remove and r.total_baggages < 2:
        return None
    k = rng.randint(1, min(3, r.total_baggages - 1) if remove else 3)
    total = r.total_baggages - k if remove else r.total_baggages + k
    nonfree = max(0, total - free)
    fee = 50 * (nonfree - r.nonfree_baggages)
    pid = payer(user, max(fee, 0), rng)
    if pid is None:
        return None
    call = _call("update_reservation_baggages", reservation_id=r.reservation_id, total_baggages=total,
                 nonfree_baggages=nonfree, payment_id=pid)
    if remove:
        spec = {"family": "bags", "reservation": r.reservation_id, "remove": k}
        refund = rng.choice(REMOVE_REFUND) if r.nonfree_baggages else ""
        request = rng.choice(REMOVE_BAGS).format(k=WORDS[k - 1], total=r.total_baggages, trip=text, refund=refund)
        return Part(spec, "removing checked bags, which the policy does not allow", request, [], forbidden=call,
                    tags={"bags_no_remove"})
    spec = {"family": "bags", "reservation": r.reservation_id, "add": k, "payment": pid}
    request = rng.choice(ADD_BAGS).format(k=_count(k, "checked bag"), more=_count(k, "more checked bag"), total=total,
                                          trip=text)
    part = Part(spec, "adding checked bags: free allowance by membership and cabin", request,
                [rng.choice(PAY_BAGS).format(pay=pay(user, pid))], [call], tags={"bags_add", "bag_allowance", "mod_pay"})
    amount = sayable(fee, db, r.user_id)
    if fee and amount and rng.random() < 0.6:
        counted = rng.random() < 0.5
        part.ask, part.info = rng.choice(ASK_BAGS_COUNT if counted else ASK_BAGS), [amount]
        part.tags |= {"price_calc", "info_amount"} | ({"info_count"} if counted else set())
    elif rng.random() < 0.7:  # asked, not checked
        part.lines.append(rng.choice(FREE_COUNT_ADDED))
        part.tags.add("info_count")
    return part


INSURE = ("You thought you had added travel insurance to {trip}, but it is not there; you want it added now.",
          "You want to buy travel insurance for {trip}.", "Please add travel insurance to {trip}.")
UNINSURE = ("You want to cancel the travel insurance on {trip} and get your money back for it, but keep the flights.",
            "You no longer want the insurance you bought for {trip} and ask for a refund of it.",
            "You want a refund for the travel insurance on {trip}; the trip itself stays.")


def insurance(db, r, rng):
    """Change baggage and insurance: insurance cannot be added after booking, nor taken back. No tool could grant
    either, so this request only comes with one that writes."""
    text = trip(db, r, rng)
    if text is None or facts(db, r)["flown"]:
        return None
    add = r.insurance == "no"
    spec = {"family": "insurance", "reservation": r.reservation_id, "add": add}
    return Part(spec, "adding or refunding travel insurance after booking, which the policy does not allow",
                rng.choice(INSURE if add else UNINSURE).format(trip=text), [], needs_company=True,
                tags={"ins_no_add"} | (set() if add else {"deny_other"}))


# ---------------------------------------------------------------- change passengers

# The date of birth never directly follows {trip}: a round trip's description ends with its return date.
DOB = ("The date of birth of {name} on {trip} is wrong: it should be {dob}.",
       "You need to correct {name}'s date of birth to {dob} on {trip}.",
       "On {trip}, {name}'s date of birth was entered wrongly; the right one is {dob}.")
REPLACE = ("{old} can no longer travel on {trip}; you want {new} to take their place.",
           "You want to replace {old} with {new} on {trip}.", "{old} cannot come on {trip}, so {new} will travel instead.")
NEW_DOB = ("{new}'s date of birth is {dob}.", "{new} was born on {dob}.", "If asked, {new}'s date of birth is {dob}.")
SELF = ("{old} can no longer travel on {trip}; you will take their place yourself.",
        "You want to change the passenger {old} on {trip} to yourself.",
        "On {trip}, you will fly instead of {old}.")


def passengers(db, r, rng):
    """Change passengers: a passenger's details can be corrected, or a passenger replaced by someone else, as long as
    the number of passengers stays the same. The passenger is never the user, whose profile records a date of birth,
    and a date of birth is corrected only for someone not among the user's saved passengers, whose record would
    contradict the correction."""
    text = trip(db, r, rng)
    names = [(p.first_name, p.last_name) for p in r.passengers]
    user = db.users[r.user_id]
    saved = {(p.first_name, p.last_name) for p in user.saved_passengers}
    others = [k for k, n in enumerate(names) if n != (user.name.first_name, user.name.last_name)]
    if text is None or len(set(names)) < len(names) or not others:
        return None
    k = rng.choice(others)
    old = r.passengers[k]
    replacements = [p for p in user.saved_passengers if (p.first_name, p.last_name) not in names
                    and (p.first_name, p.last_name) != (user.name.first_name, user.name.last_name)]
    me = (user.name.first_name, user.name.last_name)
    if me not in names and rng.random() < 0.3:  # the user takes the seat, their date of birth in their profile
        new = {"first_name": me[0], "last_name": me[1], "dob": user.dob}
        request = rng.choice(SELF).format(old=f"{old.first_name} {old.last_name}", trip=text)
        lines = [rng.choice(DOB_IN_PROFILE)]
        purpose = "replacing a passenger with the user, whose date of birth is in the profile"
    elif replacements and (rng.random() < 0.5 or names[k] in saved):
        new = rng.choice(replacements).model_dump()
        request = rng.choice(REPLACE).format(old=f"{old.first_name} {old.last_name}", trip=text,
                                             new=f"{new['first_name']} {new['last_name']}")
        lines = [rng.choice(NEW_DOB).format(new=f"{new['first_name']} {new['last_name']}", dob=birthday(new["dob"]))]
        purpose = "replacing a passenger, which keeps the number of passengers"
    elif names[k] in saved:
        return None
    else:
        y, m, d = map(int, old.dob.split("-"))
        new_day = d + rng.choice((-2, -1, 1, 2)) if 3 <= d <= 26 else (d + 1 if d < 3 else d - 1)
        new = {"first_name": old.first_name, "last_name": old.last_name, "dob": f"{y:04d}-{m:02d}-{new_day:02d}"}
        request = rng.choice(DOB).format(name=f"{old.first_name} {old.last_name}", trip=text, dob=birthday(new["dob"]))
        lines = []
        purpose = "correcting a passenger's date of birth"
    updated = [p.model_dump() for p in r.passengers]
    updated[k] = new
    spec = {"family": "passengers", "reservation": r.reservation_id, "old": [old.first_name, old.last_name], "new": new}
    call = _call("update_reservation_passengers", reservation_id=r.reservation_id, passengers=updated)
    return Part(spec, purpose, request, lines, [call], tags={"pax_modify"})


DROP = ("You want to remove {name} from {trip}; they are no longer coming.",
        "{name} cannot travel anymore, so you want to take them off {trip}.",
        "You need {trip} without {name}: please remove them from it.")
DROP_DOWNGRADE = ("If a passenger cannot be removed, you want to save money another way: downgrade the whole reservation "
                  "to {cabin}, with the refund going to {pay}.",
                  "Should removing {name} be impossible, change the reservation to {cabin} instead and refund the "
                  "difference to {pay}.",
                  "If the agent cannot remove anyone, you would rather move everyone down to {cabin}; any refund goes to "
                  "{pay}.")
DROP_CANCEL = ("If a passenger cannot be removed, you want the agent to cancel the whole reservation instead; if that is "
               "not possible either, accept it.",
               "Should removing {name} be impossible, cancel the reservation altogether, and if the agent cannot do "
               "that, leave it.",
               "If {name} cannot be taken off, you'd rather cancel the whole trip; if even that is refused, you accept.")


def fewer_passengers(db, r, rng, fallback):
    """Change passengers: their number cannot change. The user falls back on downgrading the whole reservation, or on
    cancelling it, which the cancellation rules then allow or not."""
    text = trip(db, r, rng)
    user = db.users[r.user_id]
    others = [p for p in r.passengers if (p.first_name, p.last_name) != (user.name.first_name, user.name.last_name)]
    if text is None or len(r.passengers) < 2 or not others:
        return None
    p = rng.choice(others)
    name = f"{p.first_name} {p.last_name}"
    request = rng.choice(DROP).format(name=name, trip=text)
    spec = {"family": "fewer_passengers", "reservation": r.reservation_id, "drop": [p.first_name, p.last_name],
            "fallback": fallback}
    tags = {"pax_no_count", "fallback"}
    if fallback == "downgrade":
        down = {"business": "economy", "economy": "basic_economy"}.get(r.cabin)
        free = FREE_BAGS[user.membership].get(down, 0) * len(r.passengers)
        flights = [(f.flight_number, f.date) for f in r.flights]
        if down is None or r.total_baggages > free or r.nonfree_baggages:
            return None
        difference = _difference(db, r, down, flights)
        pid = payer(user, difference, rng)
        if pid is None or difference >= 0:
            return None
        spec.update({"cabin": down, "payment": pid})
        lines = [rng.choice(DROP_DOWNGRADE).format(name=name, cabin=CABIN[down], pay=pay(user, pid))]
        call = _call("update_reservation_flights", reservation_id=r.reservation_id, cabin=down,
                     flights=_flights(flights), payment_id=pid)
        part = Part(spec, "removing a passenger, which the policy does not allow, then a downgrade", request, lines,
                    [call], tags=tags | {"cabin_change", "cabin_diff", "mod_pay"})
        amount = sayable(difference, db, r.user_id, {x for n, d in flights for x in db.flights[n].dates[d].prices.values()})
        if amount and rng.random() < 0.6:
            part.ask, part.info = rng.choice(BACK), [amount]
            part.tags |= {"price_calc", "info_amount"}
        return part
    f = facts(db, r)
    if f["flown"]:
        return None
    lines = [rng.choice(DROP_CANCEL).format(name=name), rng.choice(WHY).format(why=rng.choice(PLANS))]
    call = _call("cancel_reservation", reservation_id=r.reservation_id)
    if f["age"] < 23 or r.cabin == "business" or f["airline"]:
        return Part(spec, "removing a passenger, which the policy does not allow, then a cancellation it allows",
                    request, lines, [call], tags=tags | {"cancel_ok"})
    if f["age"] > 25:
        return Part(spec, "removing a passenger, then a cancellation, neither of which the policy allows", request, lines,
                    forbidden=call, tags=tags | {"cancel_refuse"})
    return None


# ---------------------------------------------------------------- several changes to one reservation

STEPS = ("For {trip}, you want to: {steps}. Mention all of these at once and in this order.",
         "You have {n} changes for {trip}, to be done in this order: {steps}.",
         "On {trip}, please {steps} — in that order.")


def several(db, r, rng):
    """Two or three changes to one reservation, done in the order the user gives, since the order of payments matters:
    an upgrade to the next cabin, replacing a passenger, adding checked bags (free ones by the cabin at that point)."""
    text = trip(db, r, rng)
    user = db.users[r.user_id]
    up = {"basic_economy": "economy", "economy": "business"}.get(r.cabin)
    if text is None or not facts(db, r)["upcoming"] or r.nonfree_baggages:
        return None
    replacements = [p for p in user.saved_passengers if (p.first_name, p.last_name)
                    not in {(q.first_name, q.last_name) for q in r.passengers}]
    others = [k for k, q in enumerate(r.passengers)
              if (q.first_name, q.last_name) != (user.name.first_name, user.name.last_name)]
    ops = [op for op, ok in (("cabin", up is not None), ("passenger", replacements and others), ("bags", True)) if ok]
    if len(ops) < 2:
        return None
    ops = rng.sample(ops, 3 if len(ops) == 3 and rng.random() < 0.6 else 2)
    flights = [(f.flight_number, f.date) for f in r.flights]
    cabin, total_bags, nonfree, fee = r.cabin, r.total_baggages, 0, 0
    steps, calls, tags, spec = [], [], {"multi", "order"}, {"family": "several", "reservation": r.reservation_id,
                                                            "ops": ops}
    passengers = [q.model_dump() for q in r.passengers]
    lines = []
    for op in ops:
        if op == "cabin":
            if any(db.flights[n].dates[d].available_seats[up] < len(r.passengers) for n, d in flights):
                return None
            fee += max(0, _difference(db, r, up, flights))
            cabin = up
            steps.append(f"upgrade it to {CABIN[up]}")
            calls.append(("update_reservation_flights", dict(reservation_id=r.reservation_id, cabin=up,
                                                             flights=_flights(flights))))
            tags |= {"cabin_change", "cabin_diff", "mod_pay"}
        elif op == "passenger":
            k, new = rng.choice(others), rng.choice(replacements).model_dump()
            old = r.passengers[k]
            passengers[k] = new
            steps.append(f"replace {old.first_name} {old.last_name} with {new['first_name']} {new['last_name']}")
            calls.append(("update_reservation_passengers", dict(reservation_id=r.reservation_id,
                                                                passengers=[dict(x) for x in passengers])))
            spec["replace"] = [old.first_name, old.last_name, new]
            lines.append(rng.choice(NEW_DOB).format(new=f"{new['first_name']} {new['last_name']}",
                                                    dob=birthday(new["dob"])))
            tags.add("pax_modify")
        else:
            k = rng.randint(1, 3)
            free = FREE_BAGS[user.membership][cabin] * len(r.passengers)
            total_bags += k
            nonfree = max(0, total_bags - free)
            # Paid bags added before an upgrade would be free after it, and an agent might then mark them free.
            if nonfree and "cabin" in ops[ops.index(op):]:
                return None
            fee += 50 * nonfree
            steps.append(f"add {_count(k, 'checked bag')}")
            calls.append(("update_reservation_baggages", dict(reservation_id=r.reservation_id, total_baggages=total_bags,
                                                              nonfree_baggages=nonfree)))
            spec["add"] = k
            tags |= {"bags_add", "bag_allowance", "mod_pay"}
    pid = payer(user, fee, rng)
    if pid is None:
        return None
    spec["payment"] = pid
    calls = [_call(name, **(args if name == "update_reservation_passengers" else {**args, "payment_id": pid}))
             for name, args in calls]
    joined = ", ".join(steps[:-1]) + " and then " + steps[-1] if len(steps) > 2 else " and then ".join(steps)
    request = rng.choice(STEPS).format(trip=text, steps=joined, n=WORDS[len(steps) - 1])
    return Part(spec, "several changes to one reservation in the order the user gives", request,
                [rng.choice(PAY_CHANGE).format(pay=pay(user, pid))] + lines, calls, tags=tags)
