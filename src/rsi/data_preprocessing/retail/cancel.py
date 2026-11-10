"""Cancel pending order: the whole order, for one of the two reasons the policy accepts; refunded to a gift card
(refused: a cancellation refunds the original payment method); one item of it (refused: only whole orders), with a
fallback; items of two orders in one request; every pending order of the customer at once, or all of them shipped to a
new address."""
from collections import Counter

from rsi.data_preprocessing.common import Part
from rsi.data_preprocessing.retail.facts import (FIND_THEM, REASONS, WHY, _call, address_text, new_address,
                                                 order_name, other_order_address, pay_name, payer, product)

REASON_ASKED = ("If the agent asks for the reason, say {why}.", "Only give the reason when asked: {why}.",
                "When the agent asks why, explain that {why}.")


def reason_line(rng, reason, tags):
    """The cancellation reason as the customer gives it, up front or only when the agent asks; a task that turns on the
    reason (asked for, or the less usual 'ordered by mistake') is tagged cancel_reason."""
    why = rng.choice(WHY[reason])
    if rng.random() < 0.5:
        tags.add("cancel_reason")
        return rng.choice(REASON_ASKED).format(why=why)
    if reason == "ordered by mistake":
        tags.add("cancel_reason")
    return f"You want it cancelled because {why}."


CANCEL = ("You want to cancel {which}, the whole order.", "You'd like {which} cancelled, everything in it.",
          "Please cancel the whole of {which}.")
CANCEL_ACROSS = ("You no longer want {items} you ordered. If a single item cannot be removed, cancel each order they are "
                 "in as a whole.",
                 "You'd like {items} you ordered taken off your orders. If only whole orders can be cancelled, cancel "
                 "the orders they are in.")


def cancel_across(db, user_id, rng):
    """Cancel pending order, for items the customer names in one request that sit in two pending orders of several
    items: single items cannot be removed, so each whole order is cancelled, for the reason the customer gives. Each
    product is the only one of its name among the customer's orders, so its name tells which order holds it."""
    user = db.users[user_id]
    names = Counter(i.name for o in user.orders for i in db.orders[o].items)
    picks = []
    for o in rng.sample(user.orders, len(user.orders)):
        order = db.orders[o]
        unique = [i for i in order.items if names[i.name] == 1]
        if order.status == "pending" and len(order.items) > 1 and unique:
            picks.append((order, rng.choice(unique)))
        if len(picks) == 2:
            break
    if len(picks) < 2:
        return None
    picks.sort(key=lambda x: user.orders.index(x[0].order_id))
    reason = rng.choice(REASONS)
    tags = {"cancel", "cancel_partial_no", "fallback", "items_across"}
    lines = [reason_line(rng, reason, tags)]
    spec = {"family": "cancel_across", "user": user_id, "orders": [o.order_id for o, _ in picks],
            "items": [i.item_id for _, i in picks], "reason": reason}
    return Part(spec, "removing items of two pending orders named in one request: each whole order is cancelled",
                rng.choice(CANCEL_ACROSS).format(items=" and ".join(f"the {product(i.name)}" for _, i in picks)),
                lines, [_call("cancel_pending_order", order_id=o.order_id, reason=reason) for o, _ in picks],
                tags=tags)


def cancel(db, order, rng, reason=None):
    """Cancel pending order: the whole order, for one of the two reasons the policy accepts."""
    if order.status != "pending":
        return None
    named = order_name(db, order, rng)
    if named is None:
        return None
    reason = reason or rng.choice(REASONS)
    tags = {"cancel"}
    lines = [reason_line(rng, reason, tags)]
    spec = {"family": "cancel", "order": order.order_id, "reason": reason, "named": named[1]}
    return Part(spec, "cancelling a pending order", rng.choice(CANCEL).format(which=named[0]), lines,
                [_call("cancel_pending_order", order_id=order.order_id, reason=reason)], tags=tags)


REFUND_GIFT = {
    "keep": ("You want to cancel {which}, but only if the refund can go to your gift card; otherwise keep the order.",
             "You'd like {which} cancelled with the money put on your gift card; if the refund cannot go there, do not "
             "cancel it."),
    "cancel": ("You want to cancel {which} and have the refund put on your gift card; if that is not possible, cancel it "
               "anyway.", "You'd like {which} cancelled, refunded to your gift card if possible, and cancelled all the "
                          "same if not."),
}


def cancel_refund(db, order, rng, then):
    """Cancel pending order, refunded to a gift card: a cancellation refunds the original payment method only. The
    customer then keeps the order or cancels it all the same."""
    user = db.users[order.user_id]
    gift = next((k for k, m in user.payment_methods.items() if m.source == "gift_card"), None)
    if order.status != "pending" or gift is None or order.payment_history[0].payment_method_id == gift:
        return None
    named = order_name(db, order, rng)
    if named is None:
        return None
    reason = rng.choice(REASONS)
    call = _call("cancel_pending_order", order_id=order.order_id, reason=reason)
    spec = {"family": "cancel_refund", "order": order.order_id, "then": then, "reason": reason, "named": named[1]}
    tags = {"cancel_refund", "cancel", "fallback"}
    lines = [reason_line(rng, reason, tags)]
    request = rng.choice(REFUND_GIFT[then]).format(which=named[0])
    if then == "keep":
        return Part(spec, "a cancellation refunded to a gift card, which the policy does not allow; the order stays",
                    request, lines, forbidden=call, by_policy=False, tags=tags)
    return Part(spec, "a cancellation refunded to a gift card, which the policy does not allow, then as it allows",
                request, lines, [call], tags=tags)


CANCEL_ONE = ("You want to cancel just the {item} from {which}.", "You'd like to remove the {item} from {which}.",
              "You no longer want the {item} in {which} and want it taken off the order.")
PARTIAL = {
    "whole": ("If only the whole order can be cancelled, cancel all of it; {why}.",
              "If the agent cannot remove a single item, you are fine cancelling the entire order, since {why}."),
    "keep": ("If removing just that item is not possible, keep the order as it is; do not cancel the whole order.",
             "You do not want the whole order cancelled; if a single item cannot be removed, leave the order alone."),
    "cheapest": ("If removing just that item is not possible, change it to the cheapest {item} available instead, with "
                 "any refund going to {pay}; do not cancel the whole order.",
                 "If a single item cannot be removed, you want the {item} switched to the cheapest one available, the "
                 "difference going to {pay}; you do not want the whole order cancelled."),
}


def cancel_partial(db, order, rng, fallback):
    """Cancel pending order, one item of it: the tool cancels only whole orders. The customer falls back on cancelling
    all of it, keeping it, or changing that item to the cheapest variant in stock."""
    if order.status != "pending" or len({i.name for i in order.items}) < 2:
        return None
    item = rng.choice([i for i in order.items if sum(j.name == i.name for j in order.items) == 1])
    named = order_name(db, order, rng, [item])
    if named is None:
        return None
    reason = rng.choice(REASONS)
    request = rng.choice(CANCEL_ONE).format(item=product(item.name), which=named[0])
    spec = {"family": "cancel_partial", "order": order.order_id, "item": item.item_id, "fallback": fallback,
            "named": named[1]}
    whole = _call("cancel_pending_order", order_id=order.order_id, reason=reason)
    tags = {"cancel_partial_no", "fallback"}
    if fallback == "whole":
        spec["reason"] = reason
        return Part(spec, "removing one item of a pending order, then cancelling all of it", request,
                    [rng.choice(PARTIAL["whole"]).format(why=rng.choice(WHY[reason]))], [whole], tags=tags | {"cancel"})
    if fallback == "keep":
        return Part(spec, "removing one item of a pending order, which cannot be done; the customer keeps it", request,
                    [rng.choice(PARTIAL["keep"])], forbidden=whole, by_policy=False, tags=tags)
    variants = db.products[item.product_id].variants
    stock = sorted((v.price, k) for k, v in variants.items() if v.available and k != item.item_id)
    if len(stock) < 2 or stock[0][0] == stock[1][0] or stock[0][0] >= item.price:
        return None
    new = stock[0][1]
    pid = payer(db, order, 0, rng)
    spec.update(new=new, payment=pid)
    call = _call("modify_pending_order_items", order_id=order.order_id, item_ids=[item.item_id], new_item_ids=[new],
                 payment_method_id=pid)
    line_ = rng.choice(PARTIAL["cheapest"]).format(item=product(item.name),
                                                    pay=pay_name(db.users[order.user_id], pid, order))
    return Part(spec, "removing one item of a pending order, then changing it to the cheapest variant", request,
                [line_], [call], forbidden=whole, by_policy=False,
                tags=tags | {"mod_items", "var_extreme"})


# ---------------------------------------------------------------- every pending order at once

ALL_PENDING_CANCEL = ("You want to cancel all of your pending orders.", "Please cancel every order of yours that is "
                                                                         "still pending.")
ALL_PENDING_ADDRESS = ("You want all of your pending orders, and your default address, changed to {address}.",
                       "Please update the shipping address of every pending order of yours, and your profile address, "
                       "to {address}.")
ALL_PENDING_FROM = ("You want all of your pending orders, and your default address, changed to {other}; you would rather "
                    "not say the address, the agent can look it up.",
                    "Please ship every pending order of yours to {other}, and make it your profile address too; you "
                    "prefer not to spell it out.")


def all_pending(db, user_id, rng, kind):
    """Every pending order of the customer at once: cancel them all (only plain 'pending' ones can be), or ship them all,
    and the default address, to a new address ("address") or to the address one of their other orders went to
    ("address_order")."""
    user = db.users[user_id]
    mine = [db.orders[o] for o in user.orders if "pending" in db.orders[o].status]
    if len(mine) < 2 or any(o.status != "pending" for o in mine):
        return None
    lines = [x for x in (rng.choice(FIND_THEM),) if x]
    if kind == "address_order":
        found = other_order_address(db, user, {(user.address.address1, user.address.zip)}, rng)
        if found is None:
            return None
        a, other, sources = found
        if any((o.address.address1, o.address.zip) == (a["address1"], a["zip"]) for o in mine):
            return None  # an order already there would need no change
        spec = {"family": "all_pending_address", "user": user_id, "orders": [o.order_id for o in mine], "address": a,
                "source": "order", "from": sources}
        calls = [_call("modify_pending_order_address", order_id=o.order_id, **a) for o in mine]
        calls.append(_call("modify_user_address", user_id=user_id, **a))
        return Part(spec, "every pending order and the profile to the address another order went to",
                    rng.choice(ALL_PENDING_FROM).format(other=other), lines, calls,
                    tags={"scope_all", "list_status", "mod_addr", "user_addr", "addr_from_order"})
    if kind == "cancel":
        reason = rng.choice(REASONS)
        spec = {"family": "all_pending_cancel", "user": user_id, "orders": [o.order_id for o in mine],
                "reason": reason}
        tags = {"scope_all", "list_status", "cancel"}
        lines.append(reason_line(rng, reason, tags))
        return Part(spec, "cancelling every pending order", rng.choice(ALL_PENDING_CANCEL), lines,
                    [_call("cancel_pending_order", order_id=o.order_id, reason=reason) for o in mine], tags=tags)
    a = new_address(db, rng, {(user.address.address1, user.address.zip)})
    spec = {"family": "all_pending_address", "user": user_id, "orders": [o.order_id for o in mine], "address": a}
    calls = [_call("modify_pending_order_address", order_id=o.order_id, **a) for o in mine]
    calls.append(_call("modify_user_address", user_id=user_id, **a))
    return Part(spec, "a new address for every pending order and the profile", rng.choice(ALL_PENDING_ADDRESS).format(
        address=address_text(type(user.address)(**a))), lines, calls,
                tags={"scope_all", "list_status", "mod_addr", "user_addr"})
