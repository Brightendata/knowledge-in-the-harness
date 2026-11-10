"""What the retail requests share: how a customer names an order, an item and a payment method, the payment method that
can settle an amount, the amounts the agent states, addresses, and sentences several requests use."""
import random
from itertools import combinations

RECEIVED = {"delivered", "exchange requested", "return requested"}
NOT_RECEIVED = {"pending", "processed", "pending (item modified)"}


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


# ---------------------------------------------------------------- how a customer names things

def product(name):
    """A product as a customer names it: 'mechanical keyboard', 'LED light bulb'."""
    return " ".join(w if w.isupper() else w.lower() for w in name.split())


def article(name):
    """A product with its indefinite article: 'a tea kettle', 'an e-reader', 'running shoes'."""
    p = product(name)
    return p if p.endswith("s") else f"{'an' if p[0] in 'aeiou' else 'a'} {p}"


def listed(names):
    names = [f"the {product(n)}" for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# The yes/no options in words, as official tasks put them ("the option without water resistance", "no backlight"): the
# user simulator reads "water resistance no" as water-resistant, and "you prefer none" as no preference.
YES_NO = {("water resistance", "yes"): "water resistance", ("water resistance", "no"): "no water resistance",
          ("waterproof", "yes"): "waterproofing", ("waterproof", "no"): "no waterproofing",
          ("backlight", "none"): "no backlight", ("armrest", "none"): "no armrest",
          ("features", "none"): "no extra features", ("connectivity", "none"): "no connectivity"}


def opts(options):
    return ", ".join(YES_NO.get((k, v), f"{k} {v}") for k, v in options.items())


# option values with an order, ranked, for "one size bigger", "the most storage" and the like
WORD_RANK = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5, "portable": 0, "small": 1, "medium": 2, "large": 3,
             "low": 1, "high": 3, "beginner": 1, "intermediate": 2, "expert": 3, "basic": 1, "professional": 2,
             "standard": 1, "high-back": 2, "A6": 1, "A5": 2, "A4": 3, "1080p": 1080, "2K": 1440, "4K": 2160,
             "5K": 2880}
WITH_NAME = {"type", "gender", "theme"}  # options whose value alone reads as something else


def said(options, variants, seed):
    """The options of a new variant as customers in the official tasks put them: about half of the values alone ("red",
    "1TB SSD", "polarized"), where no other option of the product has that value, so the agent maps each to its option;
    the others with the option's name ("2000 pieces", "size M", "brightness high", "frame color black"): a number, a
    graded word and a value two options share are always named. Deterministic in seed."""
    rng = random.Random(seed)
    keys = {}
    for v in variants.values():
        for k, x in v.options.items():
            keys.setdefault(x, set()).add(k)
    out = []
    for k, v in options.items():
        if (k, v) in YES_NO:
            out.append(YES_NO[k, v])
        elif v.isdigit():
            out.append(f"{v} {k}")
        elif v in WORD_RANK or k in WITH_NAME or len(keys[v]) > 1 or rng.random() < 0.5:
            out.append(f"{k} {v}")
        else:
            out.append(v)
    return ", ".join(out)


def money(x):
    """An amount as the COMMUNICATE check looks for it: '180.1' is found in both '$180.10' and '180.1'."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def address_text(a):
    return ", ".join(x for x in (a.address1, a.address2, a.city) if x) + f", {a.state} {a.zip}"


def address_args(a):
    return {k: getattr(a, k) for k in ("address1", "address2", "city", "state", "country", "zip")}


def pay_name(user, payment_id, order=None):
    """A payment method as its owner names it; the one an order was paid with is its original payment method."""
    if order is not None and payment_id == order.payment_history[0].payment_method_id:
        return "the original payment method"
    m = user.payment_methods[payment_id]
    if m.source == "credit_card":
        return f"your {m.brand.title()} credit card ending in {m.last_four}"
    return "your PayPal account" if m.source == "paypal" else "your gift card"


def peers(db, order):
    """The customer's other orders they would call the same way: received ones, or ones not received yet (a cancelled
    order never arrives either)."""
    kind = RECEIVED if order.status in RECEIVED else NOT_RECEIVED | {"cancelled"}
    return [db.orders[o] for o in db.users[order.user_id].orders if o != order.order_id and db.orders[o].status in kind]


def single_out(db, order, acted, everywhere=False):
    """Product names to mention so that the order is the only one of its kind (or, everywhere, of all the customer's
    orders, cancelled ones included) that has them all, starting with the products of the acted items; None if no set
    of names does."""
    own = list(dict.fromkeys(i.name for i in acted))
    others = sorted({i.name for i in order.items} - set(own))
    rivals = [db.orders[o] for o in db.users[order.user_id].orders if o != order.order_id] if everywhere else \
        peers(db, order)
    for k in range(len(others) + 1):
        for extra in combinations(others, k):
            named = set(own) | set(extra)
            if named and not any(named <= {i.name for i in p.items} for p in rivals):
                return own + list(extra)
    return None


def which_item(order, item):
    """How the customer tells the item from others of its product in the order: by price or by an option only it has
    ("the more expensive tablet", "the t-shirt in size M"); '' if it is the only one; None if no description does."""
    twins = [i for i in order.items if i.name == item.name and i is not item]
    if not twins:
        return ""
    if all(i.item_id == item.item_id for i in twins):
        return None
    for k, v in item.options.items():
        if all(i.options[k] != v for i in twins):
            return f"the {product(item.name)} with {opts({k: v})}"
    if len(twins) == 1 and twins[0].price != item.price:
        return f"the {'more' if item.price > twins[0].price else 'less'} expensive {product(item.name)}"
    return None


RECEIVED_ORDER = ("the order you received with {}", "your delivered order with {}", "the order with {} that you received")
PENDING_ORDER = ("your pending order with {}", "the order with {} that hasn't shipped yet",
                 "your order with {} that is still pending")
PROCESSED_ORDER = ("your order with {} that hasn't arrived yet", "the order with {} you are still waiting for")
# for items the request names already: the order, with any other products that tell it apart
RECEIVED_ITEMS = ("an order you received recently", "your delivered order", "an order that was delivered to you")
PENDING_ITEMS = ("your pending order", "an order that hasn't shipped yet", "your order that is still pending")
PROCESSED_ITEMS = ("an order that hasn't arrived yet", "the order you are still waiting for")
BY_ID = ("order {oid}", "your order {oid}", "order number {oid}")
ID_SHARE = 0.12  # official tasks give the order id in about one of nine
# without its status, for a customer who does not know whether the order has shipped
ANY_ORDER = ("your order with {}", "the order with {} that you placed recently", "your order containing {}")
# "recently" as the official tasks say it of an order that is often not the customer's last one: the products name it
ANY_ITEMS = ("your order", "the order you placed recently", "an order of yours")
# by a fact the agent has to look up in the customer's orders, as about one official task in six does
FACT_SHARE = 0.25
BY_CITY = ("your order that went to {}", "the order you had shipped to {}")
BY_COUNT = ("your order with {} items", "the order of yours that has {} items")
BY_PRICE = {True: ("the order with the pricier {} of the two you bought",
                   "the order where your {} cost more than the other one you bought"),
            False: ("the order with the cheaper {} of the two you bought",
                    "the order where your {} cost less than the other one you bought")}
BY_RECENT = ("the order you placed most recently", "your most recent order")  # the last of the customer's orders
NUMBER = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
NEUTRAL_SHARE = 0.65  # official tasks name an order by its products and its status about one time in three


def by_fact(db, order, rng, acted=()):
    """The order told apart from all the customer's other orders, cancelled ones included, by where it was shipped, how
    many items it has, which of two orders with a product paid more for it (of the acted items' products, or any of
    the order's), or as the one placed last (the official tasks read the customer's order list as placed in that
    order); (text, how) with how {"by", "value", "text"}, or None if no fact singles it out."""
    others = [db.orders[o] for o in db.users[order.user_id].orders if o != order.order_id]
    found = []
    if others and db.users[order.user_id].orders[-1] == order.order_id:
        found.append(("recent", None, rng.choice(BY_RECENT)))
    city = order.address.city
    if all(o.address.city != city for o in others):
        found.append(("city", city, rng.choice(BY_CITY).format(city)))
    n = len(order.items)
    if 2 <= n <= 10 and all(len(o.items) != n for o in others):
        found.append(("count", n, rng.choice(BY_COUNT).format(NUMBER[n])))
    for item in acted or order.items:
        holders = [o for o in others if any(i.name == item.name for i in o.items)]
        if len(holders) != 1 or sum(i.name == item.name for i in order.items) != 1 or \
                sum(i.name == item.name for i in holders[0].items) != 1:
            continue
        rival = next(i for i in holders[0].items if i.name == item.name)
        if abs(item.price - rival.price) >= 5:  # a difference no reading of the prices can miss
            more = item.price > rival.price
            found.append(("price", {"product": item.name, "more": more, "rival": holders[0].order_id},
                          rng.choice(BY_PRICE[more]).format(product(item.name))))
            break
    if not found:
        return None
    by, value, text = rng.choice(found)
    return text, {"by": by, "value": value, "text": text}


def facts_read(spec):
    """The orders a request's order fact compares it with, which no other request of the task may change, and the city
    no other request may ship an order to."""
    how = spec.get("named")
    if not isinstance(how, dict):
        return [], None
    return ([how["value"]["rival"]] if how["by"] == "price" else []), (how["value"] if how["by"] == "city" else None)


def order_name(db, order, rng, acted=(), neutral=False):
    """The order as its owner names it: by its id now and then, by a fact of the order (by_fact) about as often as the
    official tasks do, else by the products that single it out (at least the acted items', which the request names
    itself), mostly without its status when the products single it out among all the customer's orders that are not
    cancelled; None if nothing works. Returns (text, how): how is "id", the product names, or by_fact's. neutral: always
    without its status."""
    if rng.random() < ID_SHARE:
        return rng.choice(BY_ID).format(oid=order.order_id), "id"
    if rng.random() < FACT_SHARE:
        fact = by_fact(db, order, rng, acted)
        if fact:
            return fact
    names = single_out(db, order, acted, everywhere=True)
    if names is not None and (neutral or rng.random() < NEUTRAL_SHARE):
        extra = [n for n in names if n not in {i.name for i in acted}]
        if not acted:
            return rng.choice(ANY_ORDER).format(listed(names)), names
        return rng.choice(ANY_ITEMS) + (f" that also has {listed(extra)}" if extra else ""), names
    if neutral:
        return None
    names = single_out(db, order, acted)
    if names is None:
        return None
    if not acted:
        kind = RECEIVED_ORDER if order.status in RECEIVED else PENDING_ORDER if "pending" in order.status else \
            PROCESSED_ORDER
        return rng.choice(kind).format(listed(names)), names
    extra = [n for n in names if n not in {i.name for i in acted}]
    text = rng.choice(RECEIVED_ITEMS if order.status in RECEIVED else PENDING_ITEMS if "pending" in order.status else
                      PROCESSED_ITEMS)
    return text + (f" that also has {listed(extra)}" if extra else ""), names


def bought(db, order, item):
    """The item as a customer names it without knowing whether its order has shipped ("the tea kettle you ordered"), or
    None if another order of theirs, cancelled ones included, has that product too."""
    user = db.users[order.user_id]
    if any(o != order.order_id and any(i.name == item.name for i in db.orders[o].items) for o in user.orders):
        return None
    return f"the {product(item.name)} you ordered"


# ---------------------------------------------------------------- payments and amounts

def payer(db, order, amount, rng, gift_ok=True):
    """A payment method of the customer that can pay the amount (a gift card only with enough balance): the order's
    original one more often than not, else another. Returns its id."""
    user = db.users[order.user_id]
    fits = [k for k, m in user.payment_methods.items()
            if m.source != "gift_card" or (gift_ok and m.balance >= amount)]
    original = order.payment_history[0].payment_method_id
    if original in fits and rng.random() < 0.6:
        return original
    return rng.choice(sorted(fits)) if fits else None


def sayable(db, user_id, amount, shown=()):
    """The amount as the check looks for it, or None if a reply could contain it by chance: no cents, under four
    characters, or inside a price, total or balance the agent is likely to say anyway."""
    s = money(amount)
    if "." not in s or len(s) < 4:
        return None
    user = db.users[user_id]
    seen = set(shown) | {money(m.balance) for m in user.payment_methods.values() if m.source == "gift_card"}
    for o in user.orders:
        items = db.orders[o].items
        seen.add(money(sum(i.price for i in items)))
        for i in items:
            seen |= {money(v.price) for v in db.products[i.product_id].variants.values()}
    return None if any(s in x for x in seen) else s


# ---------------------------------------------------------------- addresses

STREETS = ("Maple Drive", "Oak Street", "Cedar Avenue", "Pine Lane", "Elm Street", "Willow Lane", "Lakeview Drive",
           "Hillcrest Road", "River Road", "Sunset Drive", "Highland Avenue", "Park Avenue", "Main Street", "Broadway")


def other_order_address(db, user, avoid, rng):
    """An address one of the customer's orders went to, other than the ones in avoid, as (address, how the customer
    names it, the orders that went there): by its city when no other order went to another address there, else by
    that order's id; or None."""
    places = {}
    for o in user.orders:
        a = db.orders[o].address
        if (a.address1, a.zip) not in avoid:
            places.setdefault((a.address1, a.zip), (a, o))
    if not places:
        return None
    key = rng.choice(sorted(places))
    a, o = places[key]
    sources = [x for x in user.orders if (db.orders[x].address.address1, db.orders[x].address.zip) == key]
    same_city = [k for k, (b, _) in places.items() if b.city == a.city]
    if len(same_city) == 1 and a.city != user.address.city:
        return address_args(a), rng.choice((f"your {a.city} address from one of your orders",
                                            f"the {a.city} address on one of your orders")), sources
    return address_args(a), f"the address on your order {o}", sources


def new_address(db, rng, avoid):
    """A street address in a city, state and zip code of the database, other than the ones in avoid."""
    cities = sorted({(u.address.city, u.address.state, u.address.zip) for u in db.users.values()})
    while True:
        city, state, zip_ = rng.choice(cities)
        a = {"address1": f"{rng.randint(10, 999)} {rng.choice(STREETS)}",
             "address2": rng.choice(("", f"Suite {rng.randint(100, 999)}", f"Apt {rng.randint(1, 40)}")),
             "city": city, "state": state, "country": "USA", "zip": zip_}
        if (a["address1"], a["zip"]) not in avoid:
            return a


# ---------------------------------------------------------------- sentences several requests use
# Each slot has a few wordings; a task picks one per slot with its own random.Random(task id).

REFUND_TO = ("The refund should go to {pay}.", "You want the money back on {pay}.", "Refund it to {pay}.")
# the two reasons the policy accepts, drawn about three to one as in the official tasks
REASONS = ("no longer needed", "no longer needed", "no longer needed", "ordered by mistake")
# the last two of each say what happened rather than the reason, which the agent maps to one of the two the policy
# accepts, as some official tasks have it
WHY = {"no longer needed": ("you no longer need it", "you don't need it anymore", "your plans changed and you no longer "
                                                                                    "need it",
                            "you already bought one somewhere else", "you won't be using it after all"),
       "ordered by mistake": ("you ordered it by mistake", "you placed it by mistake", "it was ordered by mistake",
                              "you clicked the wrong button when ordering", "you placed the order without meaning to")}
FIND_THEM = ("You do not remember your order numbers; let the agent find them.", "Let the agent look up which orders "
                                                                                 "these are.", "")
