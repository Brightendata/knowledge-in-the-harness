"""Return delivered order: the items the customer names, all of them, all but one, or one of two items of a product,
refunded to the original payment method or an existing gift card; another card first (refused), with a fallback; items
of two orders in one request; every delivered order of the customer at once."""
from collections import Counter

from rsi.data_preprocessing.common import Part
from rsi.data_preprocessing.retail.facts import (FIND_THEM, REFUND_TO, _call, listed, order_name, pay_name, product,
                                                 sayable, which_item)

ASK_REFUND = ("You also want to know the total amount you will get back.", "Ask the agent for the total refund.",
              "You want to know how much will be refunded in total.")
ASK_LEFT = ("After the return, ask the agent how much you paid in total for the items still in that order.",
            "Once the return is done, you want to know the total you paid for the rest of that order.")
# the order's tracking number asked along with a return from it, as official tasks do (25, 26, 30-32)
ASK_TRACKING = ("You also want to know the tracking number of that order.",
                "Ask the agent for the tracking number of that order as well.")
TRACKING_SHARE = 0.2  # of the returns that ask for no amount
RETURN = ("You want to return {items} from {which}.", "You'd like to send back {items} from {which}.",
          "You need to return {items} that came in {which}.")
RETURN_ALL = ("You want to return everything in {which}.", "You'd like to return all the items of {which}.")
RETURN_EXCEPT = ("You want to return everything in {which} except the {keep}.",
                 "From {which}, you want to return all of it but the {keep}.")
RETURN_LATE = ("When the agent asks you to confirm the return of {first}, do not say yes yet: you also want to return the "
               "{item} from that order.",
               "Before you confirm the return of {first}, add that you want to return the {item} from the same order as "
               "well.")
OTHER_CARD = ("You want the refund on {card}.", "The refund should go to {card}.")
CARD_FALLBACK = {
    "original": ("If the agent says that is not possible, have it go back to the original payment method.",
                 "If that is not possible, the original payment method is fine."),
    "gift": ("If that is not possible, you want the refund on your gift card instead.",
             "If the agent refuses, put the refund on your gift card."),
    "human": ("If the agent says that is not possible, you are annoyed and ask to speak to a human agent.",
              "If the refund cannot go there, ask to be transferred to a human agent."),
}


def return_items(db, order, rng, scope="named", refund="original"):
    """Return delivered order: the items the customer names, all of them, or all but one, refunded to the original
    payment method or an existing gift card; "held": one of the named items comes up only at confirmation and goes into
    the same return, since an order is returned once. refund "card:<fallback>": the customer first asks for another
    card, which the policy does not allow, then falls back on the original method, the gift card, or a human agent."""
    if order.status != "delivered":
        return None
    user = db.users[order.user_id]
    original = order.payment_history[0].payment_method_id
    gift = next((k for k, m in user.payment_methods.items() if m.source == "gift_card"), None)
    names = Counter(i.name for i in order.items)
    late = None
    if scope in ("named", "held"):
        unique = [i for i in order.items if names[i.name] == 1]
        if len(unique) < (2 if scope == "held" else 1):
            return None
        items = sorted(rng.sample(unique, rng.randint(2 if scope == "held" else 1, min(3, len(unique)))),
                       key=order.items.index)
        late = rng.choice(items) if scope == "held" else None
        first = [i for i in items if i is not late]
        named = order_name(db, order, rng, first)
        text = None if named is None else rng.choice(RETURN).format(items=listed([i.name for i in first]),
                                                                    which=named[0])
    elif scope == "twin":  # one of two items of a product, told apart by price or an option
        twins = [i for i in order.items if which_item(order, i)]
        if not twins:
            return None
        items = [rng.choice(twins)]
        named = order_name(db, order, rng, items)
        text = None if named is None else rng.choice(RETURN).format(items=which_item(order, items[0]), which=named[0])
    elif scope == "all":
        items = list(order.items)
        named = order_name(db, order, rng)
        text = None if named is None else rng.choice(RETURN_ALL).format(which=named[0])
    else:
        keep = [i for i in order.items if names[i.name] == 1]
        if len(order.items) < 3 or not keep:
            return None
        k = rng.choice(keep)
        items = [i for i in order.items if i is not k]
        named = order_name(db, order, rng, [k])
        text = None if named is None else rng.choice(RETURN_EXCEPT).format(which=named[0], keep=product(k.name))
    if text is None:
        return None
    tags = {"return"} | ({"scope_except"} if scope == "except" else set()) | ({"same_name"} if scope == "twin" else set())
    spec = {"family": "return", "order": order.order_id, "items": [i.item_id for i in items], "scope": scope,
            "named": named[1]}
    if late:  # added at confirmation, which the official tasks count as a change of mind
        tags |= {"items_once", "change_mind"}
        spec.update(late=late.item_id, held=True)
    ids = [i.item_id for i in items]
    if refund.startswith("card"):  # the return tool itself refuses another card
        then = refund.split(":")[1]
        cards = [k for k, m in user.payment_methods.items() if m.source != "gift_card" and k != original]
        if not cards or (then == "gift" and (gift is None or gift == original)):
            return None
        card = rng.choice(sorted(cards))
        lines = [rng.choice(OTHER_CARD).format(card=pay_name(user, card)), rng.choice(CARD_FALLBACK[then])]
        spec.update(asked=card, fallback=then)
        tags |= {"refund_rule", "fallback"}
        if then == "human":  # the customer wants no refund to the original method
            spec["transfer"] = True
            return Part(spec, "a return refunded to a card the policy does not allow; the customer asks for a human",
                        text, lines, forbidden=_call("return_delivered_order_items", order_id=order.order_id,
                                                     item_ids=ids, payment_method_id=original),
                        by_policy=False, tags=tags | {"transfer"}, transfer=True)
        pid = original if then == "original" else gift
        spec["payment"] = pid
        part = Part(spec, "a return refunded to a card the policy does not allow, then as it allows", text, lines,
                    [_call("return_delivered_order_items", order_id=order.order_id, item_ids=ids,
                           payment_method_id=pid)], tags=tags)
    else:
        if refund == "gift" and (gift is None or gift == original):
            return None
        pid = gift if refund == "gift" else original
        spec["payment"] = pid
        if pid != original:
            tags.add("refund_rule")
        part = Part(spec, f"a return ({scope}) refunded as the policy allows", text,
                    [rng.choice(REFUND_TO).format(pay=pay_name(user, pid, order))],
                    [_call("return_delivered_order_items", order_id=order.order_id, item_ids=ids,
                           payment_method_id=pid)], tags=tags)
    if late:
        part.lines.append(rng.choice(RETURN_LATE).format(first=listed([i.name for i in items if i is not late]),
                                                         item=product(late.name)))
    amount = sayable(db, order.user_id, sum(i.price for i in items)) if len(items) > 1 else None
    left = [i for i in order.items if i.item_id not in ids]
    rest = sayable(db, order.user_id, sum(i.price for i in left), [amount] if amount else ()) if left else None
    if amount and rng.random() < 0.5:
        part.ask, part.info = rng.choice(ASK_REFUND), [amount]
        part.tags |= {"compute", "info_amount"}
        if rest and rng.random() < 0.4:  # what the rest of the order cost as well, as some official tasks ask
            part.ask += " " + rng.choice(ASK_LEFT)
            part.info.append(rest)
            spec["remaining"] = rest
    elif rest and rng.random() < 0.25:
        part.ask, part.info = rng.choice(ASK_LEFT), [rest]
        part.tags |= {"compute", "info_amount"}
        spec["remaining"] = rest
    elif order.fulfillments and rng.random() < TRACKING_SHARE:  # the request itself asks: no part.ask
        part.lines.append(rng.choice(ASK_TRACKING))
        part.info = [order.fulfillments[0].tracking_id[0]]
        part.tags |= {"order_info", "info_tracking"}
        spec["tracking"] = part.info[0]
    return part


# "everything you have received, except the e-reader" was read by the user simulator as the rest of the e-reader's order
ALL_DELIVERED = ("You want to return the items of all your delivered orders{but}.",
                 "You'd like to send back all the items from your delivered orders{but}.")
EACH_REFUND = ("Each refund can go back to the payment method used for that order.",
               "Refund every order to whatever paid for it.")
# which orders hold the items is for the agent to find, as in the official tasks
RETURN_ACROSS = ("You want to return {items} you received.", "You'd like to send back {items}.")


def return_across(db, user_id, rng):
    """Return delivered order, for items the customer names in one request that came in two different orders: one return
    per order, each refunded to its own payment method. Each product is the only one of its name among the customer's
    orders, so its name tells which order holds it."""
    user = db.users[user_id]
    names = Counter(i.name for o in user.orders for i in db.orders[o].items)
    picks = []
    for o in rng.sample(user.orders, len(user.orders)):
        order = db.orders[o]
        unique = [i for i in order.items if names[i.name] == 1]
        if order.status == "delivered" and unique:
            picks.append((order, rng.choice(unique)))
        if len(picks) == 2:
            break
    if len(picks) < 2:
        return None
    picks.sort(key=lambda x: user.orders.index(x[0].order_id))
    calls = [_call("return_delivered_order_items", order_id=o.order_id, item_ids=[i.item_id],
                   payment_method_id=o.payment_history[0].payment_method_id) for o, i in picks]
    spec = {"family": "return_across", "user": user_id, "orders": [o.order_id for o, _ in picks],
            "items": [i.item_id for _, i in picks]}
    part = Part(spec, "returning items of two delivered orders named in one request",
                rng.choice(RETURN_ACROSS).format(items=listed([i.name for _, i in picks])), [rng.choice(EACH_REFUND)],
                calls, tags={"return", "items_across"})
    amount = sayable(db, user_id, sum(i.price for _, i in picks))
    if amount and rng.random() < 0.5:
        part.ask, part.info = rng.choice(ASK_REFUND), [amount]
        part.tags |= {"compute", "info_amount"}
    return part


def all_delivered(db, user_id, rng, but, refund="original"):
    """Return delivered order, for every delivered order of the customer at once, each refunded to its own payment
    method; with but, one item the customer names stays. refund "card:<fallback>": the customer first wants every
    refund on a card that paid for none of the orders, which the policy does not allow, then falls back on each order's
    own method, their gift card, or a human agent."""
    user = db.users[user_id]
    mine = [db.orders[o] for o in user.orders if db.orders[o].status == "delivered"]
    if len(mine) < 2:
        return None
    names = Counter(i.name for o in mine for i in o.items)
    keep = rng.choice(sorted(n for n, k in names.items() if k == 1)) if but and any(
        k == 1 for k in names.values()) else None
    if but and keep is None:
        return None
    calls = []
    for o in mine:
        ids = [i.item_id for i in o.items if i.name != keep]
        if ids:
            calls.append(_call("return_delivered_order_items", order_id=o.order_id, item_ids=ids,
                               payment_method_id=o.payment_history[0].payment_method_id))
    spec = {"family": "all_delivered", "user": user_id, "orders": [o.order_id for o in mine], "keep": keep}
    tags = {"scope_all", "list_status", "return"} | ({"scope_except"} if keep else set())
    text = rng.choice(ALL_DELIVERED).format(but=f", except the {product(keep)}" if keep else "")
    lines = [x for x in (rng.choice(FIND_THEM), rng.choice(EACH_REFUND)) if x]
    transfer, forbidden = False, None
    if refund.startswith("card"):  # the return tool itself refuses another card
        then = refund.split(":")[1]
        used = {o.payment_history[0].payment_method_id for o in mine}
        cards = sorted(k for k, m in user.payment_methods.items() if m.source != "gift_card" and k not in used)
        gift = next((k for k, m in user.payment_methods.items() if m.source == "gift_card"), None)
        if not cards or (then == "gift" and gift is None):
            return None
        card = rng.choice(cards)
        lines = lines[:-1] + [rng.choice(OTHER_CARD).format(card=pay_name(user, card)), rng.choice(CARD_FALLBACK[then])]
        spec.update(asked=card, fallback=then)
        tags |= {"refund_rule", "fallback"}
        if then == "gift":
            calls = [{**c, "arguments": {**c["arguments"], "payment_method_id": gift}} for c in calls]
        elif then == "human":  # the customer wants no refund to the orders' own methods
            calls, forbidden, transfer = [], calls[0], True
            spec["transfer"] = True
            tags.add("transfer")
    part = Part(spec, "returning everything delivered" + (", but one item" if keep else "") +
                (f", refunded to another card first ({refund})" if refund != "original" else ""), text, lines, calls,
                forbidden, by_policy=False, tags=tags, transfer=transfer)
    if transfer:
        return part
    total = sum(i.price for o in mine for i in o.items if i.name != keep)
    amount = sayable(db, user_id, total)
    if amount and rng.random() < 0.5:
        part.ask, part.info = rng.choice(ASK_REFUND), [amount]
        part.tags |= {"compute", "info_amount"}
    return part
