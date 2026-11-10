"""Requests whose next step turns on what the agent finds or works out: a pending order brought within the customer's
budget by switching its items to their cheapest variants, else a fallback; an item checked for an option and changed to
it unless that variant is out of stock or over a price, each with its own fallback; an exchange that, if it turns into a
return, takes a second order with it."""
from rsi.data_preprocessing.common import Part
from rsi.data_preprocessing.retail.facts import (REASONS, WHY, _call, money, opts, order_name, pay_name, product,
                                                 sayable)
from rsi.data_preprocessing.retail.items import difference, matching

# what a reply may write another way (as in info.py), so that no product name to state has it
SPELLED = ("-", "+", "grey", "smart watch")


def unique(order):
    """The items of the order whose product it holds only once."""
    return [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]


def cheapest(db, item):
    """The variant in stock the item would change to as "the cheapest": strictly the cheapest, not the item itself, and
    cheaper than it; else None."""
    stock = sorted((v.price, k) for k, v in db.products[item.product_id].variants.items() if v.available)
    if len(stock) < 2 or stock[0][0] == stock[1][0] or stock[0][1] == item.item_id or stock[0][0] >= item.price:
        return None
    return stock[0][1]


def margin(amount, above):
    """A round limit at least $10 above (or below) the amount."""
    return 10 * (int(amount // 10) + 2) if above else 10 * (int(amount // 10) - 1)


def capital(text):
    return text[0].upper() + text[1:]


# ---------------------------------------------------------------- a pending order within a budget

BUDGET = ("{which} costs more than you can afford right now: you want it to come to ${budget} or less.",
          "You can only spend ${budget} on {which}, which costs more than that at the moment.")
FIRST = {
    "remove": ("First ask the agent which item in it is the most expensive and what it costs, and whether that one item "
               "can be taken off the order.",
               "Start by asking which item in it costs the most and how much, and whether that item alone can be "
               "removed from the order."),
    "split": ("First ask whether the payment can be split between {a} and {b}.",
              "Start by asking the agent to split the payment of the order between {a} and {b}."),
}
SWITCH = ("{if_not}you want every item in it switched to its cheapest version in stock, but only if the order then "
          "comes to ${budget} or less.",
          "{if_not}ask to have each item in it changed to the cheapest version available, as long as that brings the "
          "order down to ${budget} or less.")
NEW_TOTAL = ("Ask the agent what the new total would be.", "You want to know what the order would cost after that.")
THEN = {"cancel": ("If it would still cost more than ${budget}, cancel the whole order instead, since {why}.",
                   "Should that still be over ${budget}, you'd rather cancel the entire order, as {why}."),
        "keep": ("If it would still cost more than ${budget}, leave the order as it is.",
                 "Should that still be over ${budget}, you keep the order unchanged.")}


def budget_chain(db, order, rng, within):
    """Modify pending order, within the customer's budget: every item to its cheapest variant in stock, in the one call
    an order allows (in the order's own order of items, as tau2's modify tool needs), if the new total is within the
    budget; else the order is cancelled (for the reason the customer gives) or kept. The customer may first ask for what
    the policy does not allow: to remove the most expensive item alone, or to split the payment. within: whether the
    new total is within the budget; the two tasks of a variant group differ only in the budget."""
    first = rng.choice(("remove", "split", "both", None))
    then = rng.choice(("cancel", "keep"))
    reason = rng.choice(REASONS)
    items = list(order.items)
    if order.status != "pending" or not 2 <= len(items) <= 4 or len(unique(order)) != len(items):
        return None
    new = [cheapest(db, i) for i in items]
    if None in new:
        return None
    user = db.users[order.user_id]
    total = round(sum(i.price for i in items), 2)
    after = round(sum(db.products[i.product_id].variants[n].price for i, n in zip(items, new)), 2)
    low, high = margin(after, False), margin(after, True)
    if low <= 0 or high > total - 10:  # both budgets must leave the current order over budget
        return None
    budget = high if within else low
    original = order.payment_history[0].payment_method_id
    named = order_name(db, order, rng)
    if named is None:
        return None
    priciest = max(items, key=lambda i: i.price)
    if first in ("remove", "both") and sum(i.price == priciest.price for i in items) > 1:
        return None
    others = sorted(k for k in user.payment_methods if k != original)
    if first in ("split", "both") and not others:
        return None
    sentences = [capital(rng.choice(BUDGET).format(which=named[0], budget=budget))]
    tags = {"budget", "cond_value", "cond_order", "fallback"}
    info = []
    if first in ("split", "both"):
        b = rng.choice(others)
        sentences.append(rng.choice(FIRST["split"]).format(a=pay_name(user, original), b=pay_name(user, b)))
        tags.add("deny")
    if first in ("remove", "both"):
        said = rng.choice(FIRST["remove"])
        sentences.append("If not, " + said[0].lower() + said[1:] if first == "both" else said)
        info.append(money(priciest.price))
        if "." not in info[0]:
            return None
        if product(priciest.name) not in named[0] and not any(s in priciest.name.lower() for s in SPELLED):
            info.append(priciest.name)
            tags.add("info_name")
        tags |= {"cancel_partial_no", "order_info", "info_amount"}
    sentences.append(rng.choice(SWITCH).format(if_not="If not, " if first else "", budget=budget))
    sentences[-1] = capital(sentences[-1])
    said = sayable(db, order.user_id, after, {x for x in info})
    if said and not any(x in said for x in info):
        sentences.append(rng.choice(NEW_TOTAL))
        info.append(said)
        tags |= {"compute", "info_amount"}
    sentences.append(rng.choice(THEN[then]).format(budget=budget, why=rng.choice(WHY[reason])))
    sentences.append(rng.choice(EITHER))
    modify = _call("modify_pending_order_items", order_id=order.order_id, item_ids=[i.item_id for i in items],
                   new_item_ids=new, payment_method_id=original)
    cancel = _call("cancel_pending_order", order_id=order.order_id, reason=reason)
    if within:
        calls, forbidden = [modify], (cancel if then == "cancel" else None)
        tags |= {"mod_items", "var_extreme", "items_once"}
    else:
        calls, forbidden = ([cancel] if then == "cancel" else []), modify
        tags |= {"cancel", "cancel_reason"} if then == "cancel" else set()
    spec = {"family": "budget_chain", "order": order.order_id, "items": [i.item_id for i in items], "new": new,
            "budget": budget, "within": within, "first": first, "then": then, "reason": reason, "payment": original,
            "named": named[1]}
    return Part(spec, f"a pending order brought within a budget by its cheapest variants ({within}), else {then}",
                sentences[0], sentences[1:], calls, forbidden, info=info, by_policy=False, tags=tags)


# ---------------------------------------------------------------- an option, its stock and its price

CHECK = ("Check whether the {item} in {which} has {option}.", "You want to know if the {item} from {which} has {option}.")
# the branches turn on that one variant's stock and price: read with the other options free, another might decide them
TO_IT = ("If it does not, {verb} it to one with {option}, everything else the same.",
         "If not, you want it {verbed} to the one with {option}, with all its other options unchanged.")
GONE_THEN = {"cancel": ("If that one is not available, cancel the whole order, since {why}.",
                        "Should that one be out of stock, cancel the entire order, as {why}."),
             "return": ("If that one is not available, return the {item} instead.",
                        "Should that one be out of stock, you'd rather return the {item}.")}
PRICE_THEN = {"cancel": ("If it is available but costs more than ${limit}, cancel the whole order, since {why}.",
                         "If it is in stock but priced over ${limit}, cancel the entire order, as {why}."),
              "return": ("If it is available but costs more than ${limit}, return the {item} instead.",
                         "If it is in stock but priced over ${limit}, you'd rather return the {item}."),
              "keep": ("If it is available but costs more than ${limit}, keep the {item} as it is.",
                       "If it is in stock but priced over ${limit}, leave the {item} as it is.")}
EITHER = ("Use the original payment method for any price difference or refund.",
          "Any price difference or refund goes to the original payment method.")


def option_chain(db, order, rng, case):
    """Modify pending order or exchange delivered order, by what the agent finds in the order and the catalog: the item
    lacks an option the customer wants; the variant with it is changed to if it is in stock and costs no more than the
    customer's limit; out of stock, one fallback (cancel the pending order for one reason, or return the item); over the
    limit, another (cancel for the other reason, return, or keep it). case: "under" or "over" the limit (a variant group,
    the two tasks differ only in the limit), or "gone" (the variant is out of stock)."""
    pending = order.status == "pending"
    if order.status not in ("pending", "delivered"):
        return None
    gone_then = "cancel" if pending else "return"
    price_then = rng.choice(("cancel", "keep") if pending else ("return", "keep"))
    reasons = rng.sample(("no longer needed", "ordered by mistake"), 2)
    for item in rng.sample(unique(order), len(unique(order))):
        variants = db.products[item.product_id].variants
        found = []
        for key in sorted(item.options):
            for value in sorted({v.options[key] for v in variants.values()} - {item.options[key]}):
                hit = matching(variants, {**item.options, key: value}, available=False)
                if len(hit) == 1 and (variants[hit[0]].available == (case != "gone")):
                    found.append((key, value, hit[0]))
        if not found:
            continue
        key, value, new = rng.choice(found)
        price = variants[new].price
        if margin(price, False) <= 0:  # both limits of a variant group must be possible
            continue
        limit = margin(price, case == "under") if case != "gone" else margin(price, rng.random() < 0.5)
        original = order.payment_history[0].payment_method_id
        method = db.users[order.user_id].payment_methods[original]
        diff = difference(db, [(item, new)])
        if method.source == "gift_card" and method.balance < diff:
            continue
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        name, option = product(item.name), opts({key: value})
        verb = ("change", "changed") if pending else ("exchange", "exchanged")
        sentences = [rng.choice(CHECK).format(item=name, which=named[0], option=option),
                     rng.choice(TO_IT).format(verb=verb[0], verbed=verb[1], option=option),
                     rng.choice(GONE_THEN[gone_then]).format(item=name, why=rng.choice(WHY[reasons[0]])),
                     rng.choice(PRICE_THEN[price_then]).format(item=name, limit=limit,
                                                              why=rng.choice(WHY[reasons[1]])),
                     rng.choice(EITHER)]
        change = _call("modify_pending_order_items" if pending else "exchange_delivered_order_items",
                       order_id=order.order_id, item_ids=[item.item_id], new_item_ids=[new], payment_method_id=original)
        drop = {"cancel": _call("cancel_pending_order", order_id=order.order_id, reason=reasons[0]),
                "return": _call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                                payment_method_id=original)}
        over = {"cancel": _call("cancel_pending_order", order_id=order.order_id, reason=reasons[1]),
                "return": drop["return"], "keep": None}[price_then]
        took = {"under": "change", "over": price_then, "gone": gone_then}[case]
        calls = {"under": [change], "over": [over] if over else [], "gone": [drop[gone_then]]}[case]
        forbidden = {"under": over or drop[gone_then], "over": change,
                     "gone": over if over and over != drop[gone_then] else None}[case]
        tags = {"cond_order", "cond_value", "avail", "fallback", "var_exact", "order_info"}
        tags |= {"change": {"mod_items" if pending else "exchange"}, "cancel": {"cancel", "cancel_reason"},
                 "return": {"return"}, "keep": set()}[took]
        spec = {"family": "option_chain", "order": order.order_id, "item": item.item_id, "key": key, "value": value,
                "new": new, "limit": limit, "case": case, "gone_then": gone_then, "price_then": price_then,
                "reasons": reasons, "named": named[1]}
        return Part(spec, f"an option the item lacks, changed to unless out of stock or over a price ({case})",
                    sentences[0], sentences[1:], calls, forbidden, by_policy=False, tags=tags)
    return None


# ---------------------------------------------------------------- a variant within the price paid, if there is one

WITHIN_PAID = ("{verb} the {item} in {which} for one with {option}, everything else the same, as long as it costs no more "
               "than what you paid for it.",
               "{verb} the {item} from {which} for a version with {option} and its other options unchanged, if that "
               "costs no more than what you paid.")
NONE_THEN = {"return": ("If there is no such version, return the {item} instead.",
                        "If none costs that little, you'll return the {item} instead."),
             "cancel": ("If there is no such version, cancel the whole order, since {why}.",
                        "If none costs that little, cancel the entire order, as {why}."),
             "keep": ("If there is no such version, keep the {item} as it is.",
                      "If none costs that little, leave the {item} as it is.")}


def bound_chain(db, order, rng, found):
    """Modify pending order or exchange delivered order, to the variant with one option changed and the others kept (as
    the official tasks read a request that names one option), if it costs less than what was paid for the item; if it
    costs more (it is in stock), a fallback: return the item (cancel the pending order) or keep it. found: whether it
    costs less; the two tasks of a variant group ask the same item for different options."""
    pending = order.status == "pending"
    if order.status not in ("pending", "delivered"):
        return None
    then = rng.choice(("cancel" if pending else "return", "keep"))
    reason = rng.choice(REASONS)
    for item in rng.sample(unique(order), len(unique(order))):
        variants = db.products[item.product_id].variants
        sides = {True: [], False: []}  # options -> the variant with them, everything else as the item
        for key in sorted(item.options):
            for value in sorted({v.options[key] for v in variants.values()} - {item.options[key]}):
                hit = matching(variants, {**item.options, key: value})
                # a price about the one paid could be read either way
                if len(hit) == 1 and abs(variants[hit[0]].price - item.price) >= 1:
                    sides[variants[hit[0]].price < item.price].append(({key: value}, hit[0]))
        if not (sides[True] and sides[False]):
            continue  # a variant group needs both, from the same draw
        picks = {s: rng.choice(sides[s]) for s in (True, False)}
        options, new = picks[found]
        original = order.payment_history[0].payment_method_id
        method = db.users[order.user_id].payment_methods[original]
        if method.source == "gift_card" and found and method.balance < variants[new].price - item.price:
            return None
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        name = product(item.name)
        verb = "You want to change" if pending else "You want to exchange"
        sentences = [rng.choice(WITHIN_PAID).format(verb=verb, item=name, which=named[0], option=opts(options)),
                     rng.choice(NONE_THEN[then]).format(item=name, why=rng.choice(WHY[reason])), rng.choice(EITHER)]
        change = _call("modify_pending_order_items" if pending else "exchange_delivered_order_items",
                       order_id=order.order_id, item_ids=[item.item_id], new_item_ids=[new],
                       payment_method_id=original)
        drop = {"cancel": _call("cancel_pending_order", order_id=order.order_id, reason=reason),
                "return": _call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                                payment_method_id=original), "keep": None}[then]
        tags = {"var_price_bound", "var_exact", "fallback", "cond_value"}
        if found:
            calls, forbidden = [change], drop
            tags.add("mod_items" if pending else "exchange")
        else:  # the dearer variant would break the customer's condition
            calls, forbidden = [drop] if drop else [], change
            tags |= {"cancel": {"cancel", "cancel_reason"}, "return": {"return"}, "keep": set()}[then]
        spec = {"family": "bound_chain", "order": order.order_id, "item": item.item_id, "options": options,
                "found": found, "new": new, "then": then, "reason": reason, "named": named[1]}
        return Part(spec, f"a variant within the price paid if there is one ({found}), else {then}", sentences[0],
                    sentences[1:], calls, forbidden, by_policy=False, tags=tags)
    return None


# ---------------------------------------------------------------- an exchange that may take a second order with it

EXCHANGE = {"variant": ("From {which}, you want to exchange the {item} for one with {option}, everything else the same.",
                        "You'd like to exchange the {item} from {which} for the one with {option}, all else unchanged."),
            "same": ("The {item} from {which} arrived damaged: you want to exchange it for a brand-new one exactly like "
                     "it.", "You want to exchange the {item} in {which} for an identical new one, since yours is "
                            "faulty.")}
OR_RETURN = ("If that cannot be done, return the {item} instead.", "If the agent cannot do that, you'll return the "
                                                                      "{item} instead.")
WITH_IT = {"cancel": ("If the {item} does end up being returned, also cancel {other}, since {why}; otherwise leave that "
                      "order alone.",
                      "Should the {item} be returned, you also want {other} cancelled, as {why}; if the exchange goes "
                      "through, keep that order."),
           "return": ("If the {item} does end up being returned, also return the {thing} from {other}, which goes with "
                      "it; otherwise keep the {thing}.",
                      "Should the {item} be returned, you also want to send back the {thing} from {other}, since it "
                      "goes with it; if the exchange goes through, keep the {thing}.")}


def linked(db, order, rng, how):
    """Exchange delivered order, and a second order that depends on how it ends: the customer asks for another variant
    of an item (in stock: "exchange"; out of stock: "gone") or for the very same item, which the policy does not allow
    ("same"); if the item is returned instead, another order of theirs goes too (a pending one cancelled, an item of a
    delivered one returned), else it stays. "exchange" and "gone" form a variant group: the same item, a variant in
    stock or not."""
    if order.status != "delivered":
        return None
    user = db.users[order.user_id]
    rivals = [db.orders[o] for o in user.orders if o != order.order_id and db.orders[o].status in ("pending",
                                                                                                    "delivered")]
    for item in sorted(unique(order), key=lambda i: i.item_id):
        variants = db.products[item.product_id].variants
        exact = {}  # in stock or not -> variants that differ from the item in one option
        for k, v in sorted(variants.items()):
            if k != item.item_id and sum(item.options[o] != x for o, x in v.options.items()) == 1:
                exact.setdefault(v.available, []).append(k)
        if how != "same" and not (exact.get(True) and exact.get(False)):
            continue  # a variant group needs one of each, from the same draw
        if how == "same" and not variants[item.item_id].available:
            continue
        picks = {s: rng.choice(exact[s]) for s in (True, False)} if how != "same" else {}
        if not rivals:
            return None
        other = rng.choice(sorted(rivals, key=lambda o: o.order_id))
        thing = None
        if other.status == "delivered":
            items = unique(other)
            if not items:
                return None
            thing = rng.choice(items)
            named_other = order_name(db, other, rng, [thing])
        else:
            named_other = order_name(db, other, rng)
        named = order_name(db, order, rng, [item])
        if named is None or named_other is None or item.name in {i.name for i in other.items}:
            return None
        reason = rng.choice(REASONS)
        original = order.payment_history[0].payment_method_id
        name = product(item.name)
        new = picks.get(how == "exchange") if how != "same" else item.item_id
        if how != "same":
            changed = {k: v for k, v in variants[new].options.items() if item.options[k] != v}
            diff = difference(db, [(item, new)])
            method = user.payment_methods[original]
            if method.source == "gift_card" and method.balance < diff:
                return None
            first = rng.choice(EXCHANGE["variant"]).format(which=named[0], item=name, option=opts(changed))
        else:
            first = rng.choice(EXCHANGE["same"]).format(which=named[0], item=name)
        then = "return" if thing else "cancel"
        sentences = [first, rng.choice(OR_RETURN).format(item=name),
                     rng.choice(WITH_IT[then]).format(item=name, other=named_other[0], why=rng.choice(WHY[reason]),
                                                      thing=product(thing.name) if thing else ""),
                     rng.choice(EITHER)]
        exchange = _call("exchange_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                         new_item_ids=[new], payment_method_id=original)
        back = _call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                     payment_method_id=original)
        follow = _call("cancel_pending_order", order_id=other.order_id, reason=reason) if not thing else \
            _call("return_delivered_order_items", order_id=other.order_id, item_ids=[thing.item_id],
                  payment_method_id=other.payment_history[0].payment_method_id)
        tags = {"exchange", "fallback", "cond_value", "multi"} | ({"cancel", "cancel_reason"} if not thing else set())
        if how == "exchange":
            calls, forbidden = [exchange], follow
            tags |= {"avail", "var_exact"}
        elif how == "gone":
            calls, forbidden = [back, follow], back
            tags |= {"avail", "var_exact", "return"}
        else:
            calls, forbidden = [back, follow], exchange
            tags |= {"same_item_no", "return"}
        spec = {"family": "linked", "order": order.order_id, "item": item.item_id, "how": how, "new": new,
                "orders": [other.order_id], "then": then, "thing": thing.item_id if thing else None, "reason": reason,
                "named": named[1]}
        return Part(spec, f"an exchange ({how}) that, if it becomes a return, takes another order with it", sentences[0],
                    sentences[1:], calls, forbidden, by_policy=how == "same", tags=tags)
    return None
