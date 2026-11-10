"""Modify pending order: its shipping address (given by the customer, the default address, or another order's) and its
payment method (a gift card only if its balance covers the total, with a fallback); the address and then the items of
one order. Modify default user address."""
from rsi.data_preprocessing.common import Part
from rsi.data_preprocessing.retail.facts import (_call, address_args, address_text, new_address, order_name,
                                                 other_order_address, pay_name)
from rsi.data_preprocessing.retail.items import change_items

ADDRESS = {
    "given": ("You want {which} shipped to {address}.", "You'd like to change the shipping address of {which} to "
                                                        "{address}.", "Please send {which} to {address} instead."),
    "default": ("You want {which} shipped to your default address instead; you do not want to say it, but it is in your "
                "profile.", "You'd like {which} sent to the address in your profile, which you would rather not spell "
                            "out."),
    # both say the customer will not give the address: told only that the agent can look it up, the user simulator
    # answers that it does not have the address, and the agent cannot go on
    "order": ("You want {which} shipped to {other}; you prefer not to say the address yourself.",
              "You'd like {which} to go to {other}; you'd rather not spell out the address, the agent can look it up."),
}


def order_address(db, order, rng, source, neutral=False):
    """Modify pending order: its shipping address, given by the customer, their default address, or the address of
    another of their orders, which the agent looks up. neutral: the order named without its status."""
    if "pending" not in order.status:
        return None
    user = db.users[order.user_id]
    named = order_name(db, order, rng, neutral=neutral)
    if named is None:
        return None
    current = (order.address.address1, order.address.zip)
    tags = {"mod_addr"}
    sources = []  # the orders the address is taken from, which no other request of the task may readdress
    if source == "given":
        a = new_address(db, rng, {current})
        text = rng.choice(ADDRESS["given"]).format(which=named[0], address=address_text(type(order.address)(**a)))
    elif source == "default":
        a = address_args(user.address)
        if (a["address1"], a["zip"]) == current:
            return None
        text = rng.choice(ADDRESS["default"]).format(which=named[0])
        tags.add("addr_from_order")
    else:
        found = other_order_address(db, user, {current, (user.address.address1, user.address.zip)}, rng)
        if found is None:
            return None
        a, other, sources = found
        text = rng.choice(ADDRESS["order"]).format(which=named[0], other=other)
        tags.add("addr_from_order")
    spec = {"family": "address", "order": order.order_id, "source": source, "address": a, "named": named[1],
            "from": sources}
    return Part(spec, f"a new shipping address for a pending order ({source})", text, [],
                [_call("modify_pending_order_address", order_id=order.order_id, **a)], tags=tags)


HOME = ("You have moved there, so you also want it as your default address.",
        "That is where you live now: make it your default address as well.")
# with other pending orders: a customer who has moved might accept an offer to reroute them too
HOME_ALSO = ("You also want it as your default address.", "Make it your default address as well.")


def address_items(db, order, rng, source, n=1, held=False, home=False, said=None, mind=None):
    """Modify pending order, its shipping address and its items: the address first, since once the items are modified
    the order can no longer be changed. home: the customer has moved to that address and wants it as their default
    address too. said: "exchange", the word of a customer who does not know the order has not shipped; mind: how the
    customer changes their mind when asked to confirm the item change (both as in change_items)."""
    if home and source == "default":
        return None
    address = order_address(db, order, rng, source, neutral=bool(said))
    items = change_items(db, order, rng, n, held=held, mind=mind, which="that same order", said=said)
    if address is None or items is None:
        return None
    a = address.spec["address"]
    profile = [_call("modify_user_address", user_id=order.user_id, **a)] if home else []
    others = any(o != order.order_id and db.orders[o].status == "pending" for o in db.users[order.user_id].orders)
    spec = {"family": "address_items", "order": order.order_id, "source": source, "address": a, "home": home,
            "items": items.spec["items"], "payment": items.spec["payment"], "held": held, "mind": mind or False,
            "named": address.spec["named"], "from": address.spec["from"] + items.spec["from"], "said": said}
    return Part(spec, f"a new shipping address ({source}{', also the default' if home else ''}) and then {n} item(s) "
                      "changed in the same pending order",
                " ".join([address.request] + ([rng.choice(HOME_ALSO if others else HOME)] if home else []) +
                         [items.request]), items.lines,
                address.calls + profile + items.calls, items.forbidden, info=items.info, ask=items.ask,
                by_policy=items.by_policy,
                tags=address.tags | items.tags | {"multi"} | ({"user_addr"} if home else set()))


USER_ADDRESS = {
    "given": ("You want to change your default address to {address}.", "You moved and want your profile address "
                                                                        "updated to {address}."),
    "order": ("You want your default address changed to {other}; the agent can look it up, you would rather not say it.",
              "Please make {other} your default address; you do not want to read it out."),
}


def user_address(db, user_id, rng, source):
    """Modify default user address: to an address the customer gives, or to the one an order of theirs went to."""
    user = db.users[user_id]
    current = (user.address.address1, user.address.zip)
    sources = []  # as in order_address
    if source == "given":
        a = new_address(db, rng, {current})
        # a customer who has moved might accept an offer to reroute the pending orders too
        pending = any(db.orders[o].status == "pending" for o in user.orders)
        text = rng.choice(USER_ADDRESS["given"][:1] if pending else USER_ADDRESS["given"]).format(
            address=address_text(type(user.address)(**a)))
        tags = {"user_addr"}
    else:
        found = other_order_address(db, user, {current}, rng)
        if found is None:
            return None
        a, other, sources = found
        text = rng.choice(USER_ADDRESS["order"]).format(other=other)
        tags = {"user_addr", "addr_from_order"}
    spec = {"family": "user_address", "user": user_id, "source": source, "address": a, "from": sources}
    return Part(spec, f"a new default address ({source})", text, [],
                [_call("modify_user_address", user_id=user_id, **a)], tags=tags)


PAYMENT = ("You want to pay for {which} with {pay} instead.", "You'd like to switch the payment of {which} to {pay}.")
PAYMENT_GIFT = ("You want to pay for {which} with your gift card instead; if its balance is not enough, switch it to "
                "{pay}.", "You'd like the gift card to pay for {which}; if it does not cover the total, use {pay}.")


def payment(db, order, rng, gift_first=False):
    """Modify payment: a pending order paid once can switch to another of the customer's methods, a gift card only if
    its balance covers the total. gift_first: the gift card falls short and the customer falls back on another one."""
    if order.status != "pending" or len(order.payment_history) != 1:
        return None
    user = db.users[order.user_id]
    total = order.payment_history[0].amount
    original = order.payment_history[0].payment_method_id
    gift = next((k for k, m in user.payment_methods.items() if m.source == "gift_card"), None)
    others = [k for k, m in user.payment_methods.items() if k != original and m.source != "gift_card"]
    named = order_name(db, order, rng)
    if named is None:
        return None
    spec = {"family": "payment", "order": order.order_id, "named": named[1]}
    if gift_first:
        if gift is None or gift == original or user.payment_methods[gift].balance >= total or not others:
            return None
        pid = rng.choice(sorted(others))
        spec.update(payment=pid, gift_first=True)
        return Part(spec, "switching an order's payment to a gift card short of its total, then to another method",
                    rng.choice(PAYMENT_GIFT).format(which=named[0], pay=pay_name(user, pid)), [],
                    [_call("modify_pending_order_payment", order_id=order.order_id, payment_method_id=pid)],
                    tags={"mod_pay", "gc_balance", "fallback"})
    fits = others + ([gift] if gift and gift != original and user.payment_methods[gift].balance >= total else [])
    if not fits:
        return None
    pid = rng.choice(sorted(fits))
    spec["payment"] = pid
    tags = {"mod_pay"} | ({"gc_balance"} if pid == gift else set())
    return Part(spec, "switching a pending order's payment", rng.choice(PAYMENT).format(
        which=named[0], pay=pay_name(user, pid)), [],
                [_call("modify_pending_order_payment", order_id=order.order_id, payment_method_id=pid)], tags=tags)
