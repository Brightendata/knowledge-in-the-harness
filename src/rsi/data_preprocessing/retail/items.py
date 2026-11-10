"""Modify pending order items and exchange delivered order items: each item to a variant of the same product in stock,
described the way the official tasks describe variants; a variant out of stock (refused, with a fallback); requests the
order's status turns around; exchanges the policy forbids (the very same item, another product); an exchange or a return
by a price the agent looks up, in the catalog or in the order; a return and an exchange on one delivered order, of which
only one can be done."""
import json
import re
from collections import Counter
from itertools import combinations

from rsi.data_preprocessing.common import Part
from rsi.data_preprocessing.retail.facts import (REASONS, REFUND_TO, WHY, WORD_RANK, YES_NO, _call, article, bought,
                                                 listed, money, opts, order_name, pay_name, payer, product, said,
                                                 sayable, single_out, which_item)

# ---------------------------------------------------------------- variants

RELATIVE = {  # option: (one step up, one step down)
    "size": ("one size bigger", "one size smaller"), "frame size": ("a bigger frame", "a smaller frame"),
    "capacity": ("a bigger capacity", "a smaller capacity"), "pieces": ("{d} more pieces", "{d} fewer pieces"),
    "length": ("a longer one", "a shorter one"), "height": ("a taller one", "a shorter one"),
    "diameter": ("a bigger one", "a smaller one"), "resolution": ("a higher resolution", "a slightly lower resolution"),
    "zoom": ("a longer zoom", "a shorter zoom"), "brightness": ("a brighter one", "a less bright one"),
    "storage": ("more storage", "less storage"), "screen size": ("a bigger screen", "a smaller screen"),
    "battery life": ("a longer battery life", "a shorter battery life"), "pressure": ("a higher pressure", "a lower pressure"),
    "thickness": ("a thicker one", "a thinner one"), "weight range": ("a heavier weight range", "a lighter weight range"),
    "difficulty level": ("a harder one", "an easier one"), "ventilation": ("more ventilation", "less ventilation"),
}
EXTREME = {  # option: (the most, the least)
    "zoom": ("the longest zoom", "the shortest zoom"), "resolution": ("the highest resolution", "the lowest resolution"),
    "storage": ("the most storage", "the least storage"), "capacity": ("the biggest capacity", "the smallest capacity"),
    "pieces": ("the most pieces", "the fewest pieces"), "size": ("the biggest size", "the smallest size"),
    "battery life": ("the longest battery life", "the shortest battery life"),
    "screen size": ("the biggest screen", "the smallest screen"), "brightness": ("the highest brightness", "the lowest brightness"),
    "difficulty level": ("the hardest level", "the easiest level"), "pressure": ("the highest pressure", "the lowest pressure"),
}


def rank(value):
    """A number to order an option's values by, or None for a value without an order (a colour, a material)."""
    if value in WORD_RANK:
        return WORD_RANK[value]
    m = re.match(r"(\d+(?:\.\d+)?)\s*(TB)?", value)
    return None if m is None else float(m.group(1)) * (1000 if m.group(2) else 1)


def scale(variants, key):
    """The values of an option across all variants of the product, in order, or None if they have no order."""
    values = {v.options[key] for v in variants.values()}
    if any(rank(x) is None for x in values):
        return None
    return sorted(values, key=rank)


def option_name(key, name):
    """An option as the customer names it: "type" alone reads as the kind of product."""
    return f"{product(name)} type" if key == "type" else key


def matching(variants, fixed, available=True):
    return [k for k, v in variants.items() if (v.available or not available)
            and all(v.options[o] == x for o, x in fixed.items())]


def prefer(key, value, others, absent, rng):
    """A preference for value of an open option: alone, or now and then after a first choice not among the candidates
    in stock, so that the agent checks stock (others: the candidates' other values; absent: values none of them has).
    Returns (text, the preference order, tags)."""
    if absent and rng.random() < 0.5:
        order_ = [rng.choice(absent), value] + rng.sample(others, min(len(others), 1))
        return (f"for the {key} you prefer {' > '.join(YES_NO.get((key, x), x) for x in order_)}", order_,
                {"avail", "fallback"})
    return f"if several {key} options are available, you prefer {YES_NO.get((key, value), value)}", [value], set()


def ask_for(db, order, item, new_id, kind, rng):
    """How the customer asks for variant new_id of the item's product so that, among the variants in stock, only it
    fits: (text, tags, spec) or None. kind: "exact" (the changed options, the rest unchanged), "pref" (one option left
    open with a preference order), "relative" (one option a step up or down), "relative_pref" (one option a step up or
    down and another left open with a preference), "extreme" (cheapest, most expensive, or the most or least of an
    ordered option), "bound" (no more than the price paid), "same_as" (the same as an item in another of the customer's
    orders)."""
    variants = db.products[item.product_id].variants
    new = variants[new_id].options
    changed = {k: v for k, v in new.items() if item.options[k] != v}
    kept = {k: v for k, v in item.options.items() if k not in changed}
    if not variants[new_id].available or new_id == item.item_id:
        return None
    spec = {"kind": kind, "item": item.item_id, "new": new_id}
    seed = f"{item.item_id}-{new_id}-{kind}"  # the phrasing of the options, apart from the generator's random draws
    # the options left unsaid stay as they are, as in the official tasks: the agent applies that, the customer does not
    # spell it out
    if kind == "exact":
        if rng.random() < 0.25:  # as some official tasks do
            return f"to {json.dumps(changed).replace(chr(34), chr(39))}", {"var_exact"}, spec
        return f"for one with {said(changed, variants, seed)}", {"var_exact"}, spec
    if kind == "pref":
        for key in rng.sample(sorted(changed), len(changed)):
            stated = {k: v for k, v in changed.items() if k != key}
            fits = [x for x in matching(variants, {**kept, **stated}) if x != item.item_id]
            if len(fits) < 2 or new_id not in fits:
                continue
            values = sorted({variants[x].options[key] for x in fits} - {new[key]})
            absent = sorted({v.options[key] for v in variants.values()} - {variants[x].options[key] for x in fits}
                            - {item.options[key]})
            pref, order_, tags = prefer(key, new[key], values, absent, rng)
            head = f"for one with {said(stated, variants, seed)}" if stated else f"for a different {key}"
            spec.update(open=key, stated=stated, order=order_)
            # a first choice out of stock only with the other options kept: read with them free, it may be in stock
            keep = f", everything else the same apart from the {key}" if kept and "fallback" in tags else ""
            return f"{head}{keep}; {pref}", tags | {"var_pref"}, spec
        return None
    if kind == "relative_pref":
        for key in sorted(changed):
            steps = scale(variants, key)
            if key not in RELATIVE or steps is None or \
                    abs(steps.index(new[key]) - steps.index(item.options[key])) != 1:
                continue
            up = rank(new[key]) > rank(item.options[key])
            for open_ in sorted(changed):
                if open_ == key:
                    continue
                stated = {k: v for k, v in changed.items() if k not in (key, open_)}
                # "one size bigger" and "500 more pieces" say how far; "a brighter one" is any value on that side
                fits = [x for x in matching(variants, {**kept, **stated}) if x != item.item_id and (
                    variants[x].options[key] == new[key] if key in ("size", "pieces") else
                    variants[x].options[key] != item.options[key]
                    and (rank(variants[x].options[key]) > rank(item.options[key])) == up)]
                values = sorted({variants[x].options[open_] for x in fits} - {new[open_]})
                if not values or [x for x in fits if variants[x].options[open_] == new[open_]] != [new_id]:
                    continue
                absent = sorted({v.options[open_] for v in variants.values()} -
                                {variants[x].options[open_] for x in fits} - {item.options[open_]})
                pref, order_, tags = prefer(open_, new[open_], values, absent, rng)
                d = abs(int(rank(new[key]) - rank(item.options[key])))
                head = f"for {RELATIVE[key][0 if up else 1].format(d=d)}" + (
                    f" with {said(stated, variants, seed)}" if stated else "")
                spec.update(step=key, up=up, open=open_, stated=stated, order=order_)
                keep = f", everything else the same apart from the {open_}" if kept and "fallback" in tags else ""
                return f"{head}{keep}; {pref}", tags | {"var_relative", "var_pref"}, spec
        return None
    if kind == "relative":
        for key in sorted(changed):
            steps = scale(variants, key)
            if key not in RELATIVE or steps is None:
                continue
            i, j = steps.index(item.options[key]), steps.index(new[key])
            if abs(i - j) != 1:
                continue
            stated = {k: v for k, v in changed.items() if k != key}
            up = j > i
            # "a longer battery life" must fit no other variant in stock; "one size bigger" and "500 more pieces" say
            # how far
            side = [x for x in matching(variants, {**kept, **stated}) if x != item.item_id
                    and variants[x].options[key] != item.options[key]
                    and (rank(variants[x].options[key]) > rank(item.options[key])) == up]
            if key not in ("size", "pieces") and side != [new_id]:
                continue
            d = abs(int(rank(new[key]) - rank(item.options[key])))
            text = RELATIVE[key][0 if up else 1].format(d=d)
            spec.update(step=key, up=up, stated=stated)
            return f"for {text}" + (f" with {said(stated, variants, seed)}" if stated else ""), {"var_relative"}, spec
        return None
    if kind == "extreme":  # among the variants in stock, the current one included
        for keep_keys in ([], sorted(kept)):
            fixed = {k: kept[k] for k in keep_keys}
            fits = matching(variants, fixed)
            if len(fits) < 2:
                continue
            by_price = sorted(fits, key=lambda x: variants[x].price)
            keys = [option_name(k, item.name) for k in keep_keys]
            same = " and ".join([", ".join(keys[:-1]), keys[-1]] if len(keys) > 1 else keys)
            # "the cheapest one" alone is the cheapest of the product, as in the official tasks; an item told from its
            # twin by an option would seem to keep that option. "The one with the longest zoom" alone keeps the other
            # options in the official tasks: the free reading is said.
            hold = f" with the same {same} as the one you have" if keep_keys else ", whatever its options"
            price_hold = hold if keep_keys or which_item(order, item) else ""
            if by_price[0] == new_id and variants[by_price[1]].price > variants[new_id].price:
                spec.update(by="price", most=False, fixed=fixed)
                return f"for the cheapest one available{price_hold}", {"var_extreme"}, spec
            if by_price[-1] == new_id and variants[by_price[-2]].price < variants[new_id].price:
                spec.update(by="price", most=True, fixed=fixed)
                return f"for the most expensive one available{price_hold}", {"var_extreme"}, spec
            for key in EXTREME:
                if key in new and scale(variants, key):
                    ranks = sorted(fits, key=lambda x: rank(variants[x].options[key]))
                    for most, end, nxt in ((True, ranks[-1], ranks[-2]), (False, ranks[0], ranks[1])):
                        # not when the item already has that extreme: the customer would seem to ask for nothing
                        if end == new_id and variants[nxt].options[key] != new[key] and \
                                item.options[key] != new[key]:
                            spec.update(by=key, most=most, fixed=fixed)
                            return f"for the one with {EXTREME[key][0 if most else 1]}{hold}", {"var_extreme"}, spec
        return None
    if kind == "bound":
        stated = changed
        fits = [x for x in matching(variants, {**stated}) if x != item.item_id]
        cheap = [x for x in fits if variants[x].price <= item.price]
        if cheap == [new_id] and len(fits) > 1:
            spec.update(stated=stated)
            # its other options are free: unsaid, they would be read as kept, as the official tasks do
            return f"for one with {said(stated, variants, seed)}, whatever its other options, as long as it costs no " \
                   f"more than what you paid", {"var_price_bound"}, spec
        return None
    if kind == "same_as":
        user = db.users[order.user_id]
        holders = [db.orders[o] for o in user.orders if o != order.order_id
                   and any(i.name == item.name for i in db.orders[o].items)]
        if len(holders) != 1:
            return None
        same = [i for i in holders[0].items if i.name == item.name]
        if len(same) != 1 or same[0].item_id != new_id:
            return None
        spec.update(other=holders[0].order_id)
        return f"for the same one as the {product(item.name)} in your other order", {"var_same_as"}, spec
    if kind == "relaxed":  # the options said fit one variant in stock, and none in stock keeps the others as well
        for k in range(1, len(changed)):
            for keys in combinations(sorted(changed), k):
                stated = {x: changed[x] for x in keys}
                if not matching(variants, {**item.options, **stated}) and \
                        [x for x in matching(variants, stated) if x != item.item_id] == [new_id]:
                    spec.update(stated=stated)
                    return f"for one with {said(stated, variants, seed)}", {"var_relaxed"}, spec
        return None
    raise ValueError(kind)


KINDS = {"pref": 35, "extreme": 21, "exact": 18, "relative": 10, "relaxed": 10, "relative_pref": 9, "bound": 6,
         "same_as": 4}  # official tasks' counts
ASK_TWIN = 0.5
WHICH_TWIN = ("If the agent asks which {item} you mean, it is {which}.",
              "If asked which {item}, say {which}.")
RELAXED_OK = ("If the {item} you describe only comes with some of its other options changed, that is fine with you.",
              "You do not mind other options of the {item} changing if that is the only way to get what you describe.")
# an item told from another of its product in the order: the sentence names which one it is about
RELAXED_TWIN = "If what you describe for {name} only comes with some of its other options changed, that is fine with you."


def unsaid(db, rng, changes):
    """The sentences on the options a description leaves unsaid, for (item, new id, spec, how the item is named): only
    that they may change, for a relaxed description. Otherwise they stay as they are, the official tasks' default,
    which their scenarios do not spell out either."""
    return [RELAXED_TWIN.format(name=name) if name != f"the {product(item.name)}" else
            rng.choice(RELAXED_OK).format(item=product(item.name))
            for item, new, s, name in changes if s["kind"] == "relaxed"]


def pick_variant(db, order, item, rng, kinds=tuple(KINDS), fits=None):
    """A new variant of the item and how the customer asks for it, kinds drawn by the official tasks' counts: (new id,
    text, tags, spec) or None. fits(item, new id): whether a variant may be chosen."""
    pool = [k for k in kinds]
    weights = [KINDS[k] for k in pool]
    variants = db.products[item.product_id].variants
    ids = sorted(variants)
    while pool:
        kind = rng.choices(pool, weights)[0]
        for new_id in rng.sample(ids, len(ids)):
            if fits and not fits(item, new_id):
                continue
            found = ask_for(db, order, item, new_id, kind, rng)
            if found:
                return (new_id,) + found
        i = pool.index(kind)
        del pool[i], weights[i]
    return None


# ---------------------------------------------------------------- the price difference

PAY_DIFF = ("Any price difference should go to {pay}.", "Use {pay} for any price difference.",
            "Settle any price difference with {pay}.")
REFUND_OR_DIFF = ("Any refund or price difference goes to the original payment method.",
                  "Use the original payment method for any refund or price difference.")
GIFT_FIRST = ("You want to pay any price difference with your gift card; if the agent says that is not possible, use "
              "{pay}.", "You'd like the gift card to cover the difference, and {pay} if the gift card cannot.",
              "Pay the difference with your gift card if you can, otherwise with {pay}.")
ASK_DIFF = ("You want to know the price difference.", "Ask the agent how much the difference comes to.",
            "Before confirming, ask how much you will pay or get back.")
ASK_BALANCE = ("You want to know your gift card balance after the change.",
               "Once the change is made, ask the agent what is left on your gift card.")


def difference(db, pairs):
    return round(sum(db.products[i.product_id].variants[n].price - i.price for i, n in pairs), 2)


# ---------------------------------------------------------------- item changes

CHANGE = {  # (verb, sentence)
    "exchange": ("exchange", ("From {which}, you want to exchange {items}.", "For {which}, you'd like to exchange {items}.",
                              "In {which}, you want to exchange {items}.")),
    "modify": ("change", ("In {which}, you want to change {items}.", "For {which}, you'd like to modify {items}.",
                          "In {which}, you'd like to change {items}.")),
}
# after a request on the same order, for items no other order of the customer has: the official tasks seldom say it is the
# same order, and the agent finds which order holds them
CHANGE_ALSO = {"exchange": ("You also want to exchange {items}.", "You'd also like to exchange {items}.",
                            "On top of that, you want to exchange {items}."),
               "modify": ("You also want to change {items}.", "You'd also like to modify {items}.",
                          "On top of that, you'd like to change {items}.")}
# {item} and {rest} end with the variant's description, so nothing may follow them in the sentence.
ALSO_ITEM = ("In the same order, you also want to {verb} {item}.",
             "From that same order, you also want to {verb} {item}.")
ALSO_ANY = ("You also want to {verb} {item}.", "Besides, you want to {verb} {item}.")
HOLD_BACK = ("At first mention only {first}; when the agent asks whether there is anything else in that order or asks "
             "you to confirm the change to {first}, add that you also want to {verb} {rest}.",
             "Start with {first} alone; only when the agent asks if that is everything or asks you to confirm the change "
             "to {first}, add that in that same order you also want to {verb} {rest}.")
# after another change to the same order in the request, which "start with" would push back
HOLD_BACK_AFTER = ("Of the items, mention only {first} at first; when the agent asks whether there is anything else in "
                   "that order or asks you to confirm the change to {first}, add that you also want to {verb} {rest}.",)
# {items}: the items of this request, so that a confirmation of another request does not set it off
MIND = ("When the agent asks you to confirm the change to {items}, change your mind: {verb} only {keep} and leave the "
        "other item as it is.",
        "When asked to confirm the change to {items}, decide to {verb} just {keep}; you keep the other item as it is.")
MIND_PAY = ("When the agent asks you to confirm the change to {items}, change your mind about the payment: use {pay} for "
            "the price difference instead.",
            "When asked to confirm the change to {items}, decide to settle the difference with {pay} after all.")
MIND_KEEP_PAY = ("When the agent asks you to confirm the change to {items}, change your mind: {verb} only {keep}, leave "
                 "the other item as it is, and settle the difference with {pay} instead.",
                 "When asked to confirm the change to {items}, decide to {verb} just {keep}, keeping the other item as "
                 "it is, and to pay the difference with {pay} after all.")
MIND_RETURN = ("When the agent asks you to confirm the exchange of {items}, change your mind: return {items} instead of "
               "exchanging, with the refund to the original payment method.",
               "When asked to confirm the exchange of {items}, decide not to exchange after all but to return {items}; "
               "the refund goes to the original payment method.")


def change_items(db, order, rng, n=1, held=False, mind=False, gift=False, kinds=tuple(KINDS), twin=False, which=None,
                 said=None):
    """Modify items of a pending order or exchange items of a delivered one: each item to a variant of the same product
    in stock, all in the one call an order allows, the price difference paid or refunded with a method the customer
    names (a gift card only with enough balance). held: the customer names one item first and the others only when the
    agent asks for more. mind, at confirmation: the customer keeps only the first item ("keep"), settles the difference
    with another method ("payment"), both ("keep_payment"), or returns the delivered items instead ("return"). gift:
    True, the customer wants the gift card, which cannot cover the difference, and falls back on another method;
    "covers", the customer pays the
    difference with a gift card the order was not paid with, which covers it. kinds: how the new variants may be
    described (ask_for). twin: the first item has another of its product in the order, which the customer tells apart.
    which: how the customer names the order, when another request of theirs has named it already. said: "exchange" or
    "modify", the word of a customer who does not know whether the order has shipped and asks for the other action than
    its status allows; the order is then named without its status.

    modify_pending_order_items gives every item the last new variant's price and options, so an item change matches the
    reference only with the items in the reference's order: the order's own, in which the customer names them."""
    tool = {"pending": "modify_pending_order_items", "delivered": "exchange_delivered_order_items"}.get(order.status)
    if tool is None or (said and (mind or said == ("exchange" if order.status == "delivered" else "modify"))):
        return None
    user = db.users[order.user_id]
    gift_card = next((k for k, m in user.payment_methods.items() if m.source == "gift_card"), None)
    fits = None
    if gift == "covers":  # only variants whose difference the gift card, not the order's own method, covers
        if gift_card is None or gift_card == order.payment_history[0].payment_method_id:
            return None
        balance = user.payment_methods[gift_card].balance

        def covered(i, x):
            v = db.products[i.product_id].variants[x]
            return v.available and 0 < v.price - i.price <= balance
        fits = covered
    candidates = [i for i in order.items if which_item(order, i) is not None and (
        fits is None or any(fits(i, x) for x in db.products[i.product_id].variants))]
    twins = [i for i in candidates if which_item(order, i)]
    if len(candidates) < n or (twin and not twins):
        return None
    first = [rng.choice(twins)] if twin else []
    chosen = sorted(first + rng.sample([i for i in candidates if i not in first], n - len(first)),
                    key=order.items.index)
    pairs, asks, tags, specs, names, told = [], [], set(), [], [], []
    for item in chosen:
        found = pick_variant(db, order, item, rng, kinds, fits)
        if found is None:
            return None
        new_id, text, t, s = found
        pairs.append((item, new_id))
        twin_of = which_item(order, item)
        # now and then the customer names only the product and says which one when the agent asks, as official tasks
        # do; not for an item kept or held back, whose sentences tell it apart, nor next to its twin
        name = twin_of or f"the {product(item.name)}"
        if twin_of and not (held or mind) and sum(i.name == item.name for i in chosen) == 1 and \
                rng.random() < ASK_TWIN:
            name = f"the {product(item.name)}"
            told.append(rng.choice(WHICH_TWIN).format(item=product(item.name), which=twin_of))
            s["told"] = twin_of
        asks.append(f"{name} {text}")
        names.append(name)
        tags |= t
        specs.append(s)
        if twin_of:
            tags.add("same_name")
    named = (which, None) if which else order_name(db, order, rng, [chosen[0]] if held else chosen, bool(said))
    if named is None:
        return None
    if mind == "return" and tool != "exchange_delivered_order_items":
        return None
    kept = pairs[:1] if mind in ("keep", "keep_payment") else pairs
    diff = difference(db, kept)
    if gift == "covers":
        if diff <= 0 or user.payment_methods[gift_card].balance < diff:
            return None
        pid = gift_card
    elif gift:
        if gift_card is None or diff <= 0 or user.payment_methods[gift_card].balance >= diff:
            return None
        pid = payer(db, order, diff, rng, gift_ok=False)
        tags |= {"gc_balance", "fallback"}
    else:
        pid = payer(db, order, diff, rng)
    if pid is None:
        return None
    if user.payment_methods[pid].source == "gift_card":
        tags.add("gc_balance")
    kind = "exchange" if tool.startswith("exchange") else "modify"
    verb, sentences = CHANGE[said or kind]
    if said:
        tags.add("status")
    # items whose product no other of the customer's orders has, which then need not be said to be in this order
    everywhere = Counter(i.name for o in user.orders for i in db.orders[o].items)
    alone = [everywhere[i.name] == 1 for i in chosen]
    if which and all(alone):
        sentences = CHANGE_ALSO[said or kind]
    request = rng.choice(sentences).format(items=asks[0], which=named[0])
    head = which_item(order, chosen[0]) or f"the {product(chosen[0].name)}"  # the first item, told from its twin
    if held:  # added at confirmation, which the official tasks count as a change of mind
        lines = [rng.choice(HOLD_BACK_AFTER if which else HOLD_BACK).format(first=head, verb=verb,
                                                                           rest=" and ".join(asks[1:]))]
        tags |= {"items_once", "change_mind"}
    else:
        lines = [rng.choice(ALSO_ANY if a else ALSO_ITEM).format(verb=verb, item=x) for a, x in zip(alone[1:], asks[1:])]
        if n > 1:
            tags.add("items_once")
    lines += told + unsaid(db, rng, [(i, x, s, nm) for (i, x), s, nm in zip(pairs, specs, names)])
    pay = pay_name(user, pid, order)
    first = pid  # the method the customer names before confirming
    if mind in ("payment", "keep_payment"):  # it pays for all the customer asks for before confirming
        full = difference(db, pairs)
        others = sorted(k for k, m in user.payment_methods.items()
                        if k != pid and (m.source != "gift_card" or m.balance >= full))
        if not others:
            return None
        first = rng.choice(others)
    lines.append(rng.choice(GIFT_FIRST).format(pay=pay) if gift is True else
                 rng.choice(PAY_DIFF).format(pay=pay_name(user, first, order)))
    call = _call(tool, order_id=order.order_id, item_ids=[i.item_id for i, _ in kept],
                 new_item_ids=[x for _, x in kept], payment_method_id=pid)
    forbidden, by_policy = None, not mind
    if mind:
        tags.add("change_mind")
    heads = " and ".join(which_item(order, i) or f"the {product(i.name)}" for i, _ in pairs)
    if mind == "keep":
        lines.append(rng.choice(MIND).format(items=heads, verb=verb, keep=head))
        forbidden = _call(tool, order_id=order.order_id, item_ids=[i.item_id for i, _ in pairs],
                          new_item_ids=[x for _, x in pairs], payment_method_id=pid)
    elif mind == "payment":
        lines.append(rng.choice(MIND_PAY).format(items=heads, pay=pay))
        forbidden = _call(tool, order_id=order.order_id, item_ids=[i.item_id for i, _ in kept],
                          new_item_ids=[x for _, x in kept], payment_method_id=first)
    elif mind == "keep_payment":
        lines.append(rng.choice(MIND_KEEP_PAY).format(items=heads, verb=verb, keep=head, pay=pay))
        forbidden = _call(tool, order_id=order.order_id, item_ids=[i.item_id for i, _ in pairs],
                          new_item_ids=[x for _, x in pairs], payment_method_id=first)
        tags.add("mind_combo")
    elif mind == "return":
        lines.append(rng.choice(MIND_RETURN).format(items=heads))
        forbidden, call = call, _call("return_delivered_order_items", order_id=order.order_id,
                                      item_ids=[i.item_id for i, _ in pairs],
                                      payment_method_id=order.payment_history[0].payment_method_id)
        tags.add("return")
    spec = {"family": kind, "order": order.order_id, "items": specs, "payment": pid, "held": held, "mind": mind,
            "gift": gift, "named": named[1], "from": [s["other"] for s in specs if "other" in s], "said": said}
    part = Part(spec, f"{kind} of {n} item(s) to variants the customer describes", request, lines, [call], forbidden,
                by_policy=by_policy, tags=tags | {"mod_items" if kind == "modify" else "exchange"})
    method = user.payment_methods[pid]
    left = sayable(db, order.user_id, method.balance - diff) if method.source == "gift_card" and kind == "modify" \
        else None  # an item modification takes the difference from the gift card at once; an exchange does not
    amount = sayable(db, order.user_id, abs(diff)) if mind != "return" else None
    if left and rng.random() < 0.6:
        part.ask, part.info = rng.choice(ASK_BALANCE), [left]
        part.tags |= {"compute", "info_balance"}
    elif amount and rng.random() < 0.5:
        part.ask, part.info = rng.choice(ASK_DIFF), [amount]
        part.tags |= {"compute", "info_amount"}
    return part


SCOPE_ITEMS = {
    "pending": ("You want every {item} in your pending orders changed to {target}.",
                "Please modify each {item} you have in a pending order to {target}."),
    "delivered": ("You want to exchange every {item} you have received for {target}.",
                  "You'd like each {item} from your delivered orders exchanged for {target}."),
}
EACH_DIFF = ("Any price difference goes to the payment method of each order.",
             "Settle each order's price difference with the payment method used for that order.")
GIFT_THEN = ("Pay any price difference with your gift card; if it cannot cover the differences, use {pay} instead.",
             "You'd like your gift card to pay the differences, or {pay} if its balance is not enough.")
# paying: "today" ties the total to the charges; "the total you will pay" is read as the orders' new totals
ASK_EACH = {True: ("You want to know how much you need to pay today in total.",
                   "Ask the agent how much you will be charged today in total."),
            False: ("You want to know how much you will get back in total.", "Ask the agent for the total refund.")}


def scope_targets(variants, items):
    """The ways one description singles out a new variant in stock for each item (not the item itself): kind -> new item
    ids, for "exact" (one option set, the others kept), "most" / "least" (the most expensive or cheapest in stock),
    "pref" (one option set, one left open with a preferred value, the others kept; the preference picks among several
    for some item), "most_kept" / "least_kept" (the most expensive or cheapest in stock with an option of each item's)."""
    targets = {}
    for key in sorted(items[0].options):  # one option changed, the others kept
        for value in sorted({v.options[key] for v in variants.values()}):
            new = [matching(variants, {**i.options, key: value}) for i in items]
            if all(i.options[key] != value and len(n) == 1 for i, n in zip(items, new)):
                targets["exact", key, value] = [n[0] for n in new]
            for open_ in sorted(items[0].options):
                if open_ == key:
                    continue
                fits = [matching(variants, {**{o: x for o, x in i.options.items() if o != open_}, key: value})
                        for i in items]
                for prefer in sorted({variants[k].options[open_] for f in fits for k in f}):
                    new = [[k for k in f if variants[k].options[open_] == prefer] for f in fits]
                    if all(i.options[key] != value and len(n) == 1 for i, n in zip(items, new)) and \
                            any(len(f) > 1 for f in fits):
                        targets["pref", (key, value), (open_, prefer)] = [n[0] for n in new]
    stock = sorted((v.price, k) for k, v in variants.items() if v.available)
    for kind, end, nxt in (("most", -1, -2), ("least", 0, 1)):  # the most expensive or the cheapest in stock
        if len(stock) > 1 and stock[end][0] != stock[nxt][0] and all(i.item_id != stock[end][1] for i in items):
            targets[kind, None, None] = [stock[end][1]] * len(items)
        for key in sorted(items[0].options):
            new = []
            for i in items:
                own = sorted((variants[k].price, k) for k in matching(variants, {key: i.options[key]}))
                if len(own) > 1 and own[end][0] != own[nxt][0] and own[end][1] != i.item_id:
                    new.append(own[end][1])
            if len(new) == len(items) and len(set(new)) > 1:  # otherwise the plain kind says the same
                targets[kind + "_kept", key, None] = new
    return targets


def scope_items(db, user_id, rng, status):
    """Modify items of every pending order, or exchange items of every delivered order, that has a product: the same
    description of the new variant for each such item (scope_targets), one call per order, the difference going to each
    order's payment method, or to the gift card if it covers every difference and to another method if it covers none;
    the orders of another status with the product stay as they are, and there is one if only one order is in scope. Each
    order in scope has one item of the product, since tau2's modify tool writes several items as the last new
    variant."""
    user = db.users[user_id]
    tool = {"pending": "modify_pending_order_items", "delivered": "exchange_delivered_order_items"}[status]
    mine = [db.orders[o] for o in user.orders if db.orders[o].status != "cancelled"]
    names = sorted({i.name for o in mine for i in o.items})
    gift = next((k for k, m in user.payment_methods.items() if m.source == "gift_card"), None)
    for name in rng.sample(names, len(names)):
        holders = [o for o in mine if any(i.name == name for i in o.items)]
        scope = [o for o in holders if o.status == status]
        if not scope or len(holders) < 2 or any(sum(i.name == name for i in o.items) > 1 for o in scope):
            continue
        items = [next(i for i in o.items if i.name == name) for o in scope]
        variants = db.products[items[0].product_id].variants
        targets = scope_targets(variants, items)
        pays = [o.payment_history[0].payment_method_id for o in scope]
        if not targets:
            continue
        kind, key, value = rng.choice(sorted(targets, key=str))
        new = targets[kind, key, value]
        diffs = [round(variants[n].price - i.price, 2) for i, n in zip(items, new)]
        paying, other = None, sorted(k for k, m in user.payment_methods.items() if m.source != "gift_card")
        if gift and other and min(diffs) > 0 and rng.random() < 0.3:
            balance = user.payment_methods[gift].balance
            paying = rng.choice(other)
            if balance >= sum(diffs) or balance < min(diffs):  # in between, the order of the calls would decide
                pays = [gift if balance >= sum(diffs) else paying] * len(scope)
            else:
                paying = None
        if any(user.payment_methods[p].source == "gift_card" and user.payment_methods[p].balance < d
               for p, d in zip(pays, diffs)):
            continue
        what = product(name)
        target = {"exact": lambda: f"one with {opts({key: value})}, each keeping its other options",
                  "most": lambda: f"the most expensive {what} available",
                  "least": lambda: f"the cheapest {what} available",
                  "pref": lambda: f"one with {opts({key[0]: key[1]})}, each keeping its other options apart from the "
                                  f"{value[0]}; if several {value[0]} options are available, you prefer "
                                  f"{YES_NO.get(value, value[1])}",
                  "most_kept": lambda: f"the most expensive {what} available with the same {option_name(key, name)} as "
                                       f"the one it replaces",
                  "least_kept": lambda: f"the cheapest {what} available with the same {option_name(key, name)} as the "
                                        f"one it replaces",
                  }[kind]()
        spec = {"family": "scope_items", "user": user_id, "status": status, "product": name, "kind": kind,
                "key": list(key) if kind == "pref" else key, "value": list(value) if kind == "pref" else value,
                "orders": [o.order_id for o in scope], "untouched": [o.order_id for o in holders if o not in scope],
                "payments": pays, "paying": paying}
        calls = [_call(tool, order_id=o.order_id, item_ids=[i.item_id], new_item_ids=[n], payment_method_id=p)
                 for o, i, n, p in zip(scope, items, new, pays)]
        tags = {"scope_all", "scope_items", "list_status", "mod_items" if status == "pending" else "exchange",
                {"exact": "var_exact", "pref": "var_pref"}.get(kind, "var_extreme")}
        if paying:
            tags |= {"gc_balance", "fallback"}
        pay = rng.choice(GIFT_THEN).format(pay=pay_name(user, paying)) if paying else rng.choice(EACH_DIFF)
        part = Part(spec, f"the same change to the {name} of every {status} order, one call per order",
                    rng.choice(SCOPE_ITEMS[status]).format(item=f"pair of {what}" if what.endswith("s") else what,
                                                           target=target), [pay], calls, tags=tags)
        amount = sayable(db, user_id, abs(sum(diffs)))
        if amount and len({d > 0 for d in diffs}) == 1 and rng.random() < 0.4:
            part.ask, part.info = rng.choice(ASK_EACH[diffs[0] > 0]), [amount]
            part.tags |= {"compute", "info_amount"}
        return part
    return None


NOT_IN_STOCK = {
    "other": ("If that one is not available, {verb} it instead {text}.",
              "If it is out of stock, you'll settle for this: {verb} it {text}."),
    "cancel": ("If it is not available, cancel the whole order instead; {why}.",
               "If that one is out of stock, you'd rather cancel the entire order, since {why}."),
    "return": ("If it is not available, return the {item} instead.",
               "If the agent says it is out of stock, you want to return the {item} instead."),
    "nothing": ("If it is not available, leave it as it is.", "If it is out of stock, you do not want any change."),
}


def out_of_stock(db, order, rng, fallback):
    """Modify or exchange an item to a variant that is out of stock, which the policy does not allow; the customer
    falls back on another variant, cancelling the pending order, returning the item, or nothing."""
    tool = {"pending": "modify_pending_order_items", "delivered": "exchange_delivered_order_items"}.get(order.status)
    if tool is None or (fallback == "cancel" and order.status != "pending") or (
            fallback == "return" and order.status != "delivered"):
        return None
    items = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]
    rng.shuffle(items)
    for item in items:
        variants = db.products[item.product_id].variants
        gone = sorted(k for k, v in variants.items() if not v.available and k != item.item_id)
        if not gone:
            continue
        want = rng.choice(gone)
        changed = {k: v for k, v in variants[want].options.items() if item.options[k] != v}
        if not changed:
            continue
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        verb = "exchange" if tool.startswith("exchange") else "change"
        # read with the other options free, a variant in stock might fit, and the refusal would not follow
        request = (f"You want to {verb} the {product(item.name)} in {named[0]} for one with {opts(changed)}, the other "
                   f"options unchanged.")
        spec = {"family": "out_of_stock", "order": order.order_id, "item": item.item_id, "want": want,
                "fallback": fallback, "named": named[1]}
        tags = {"avail", "fallback", "mod_items" if verb == "change" else "exchange", "var_exact"}
        pid = payer(db, order, 0, rng)
        # every fallback takes the same steps up to here, so that the tasks of a variant group share the item
        found = pick_variant(db, order, item, rng, ("exact", "extreme"))
        if found is None:
            continue
        new_id, text, t, s = found
        diff = difference(db, [(item, new_id)])
        other_pid = payer(db, order, diff, rng)
        if other_pid is None:
            continue
        if fallback == "other":
            if s["kind"] == "exact":  # "everything else the same" could be read against the variant asked for first
                new = variants[new_id].options
                then = {k: v for k, v in new.items() if item.options[k] != v}
                text = f"for one with {opts(then)}" + (
                    ", with the other options as on the one in your order" if len(then) < len(new) else "")
            elif not s["fixed"] and not text.endswith("whatever its options"):  # not among the options asked for first
                text += ", whatever its options"
            pid = other_pid
            spec.update(new=new_id, then=s, payment=pid)
            line_ = rng.choice(NOT_IN_STOCK["other"]).format(verb=verb, text=text)
            call = _call(tool, order_id=order.order_id, item_ids=[item.item_id], new_item_ids=[new_id],
                         payment_method_id=pid)
            pay = rng.choice(PAY_DIFF).format(pay=pay_name(db.users[order.user_id], pid, order))
            return Part(spec, "an item change to a variant out of stock, then to another one", request, [line_, pay],
                        [call], tags=tags | t)
        if fallback == "cancel":
            reason = rng.choice(REASONS)
            spec["reason"] = reason
            line_ = rng.choice(NOT_IN_STOCK["cancel"]).format(why=rng.choice(WHY[reason]))
            # the less usual reason, as cancel.reason_line tags it
            return Part(spec, "an item change to a variant out of stock, then cancelling the order", request, [line_],
                        [_call("cancel_pending_order", order_id=order.order_id, reason=reason)],
                        tags=tags | {"cancel"} | ({"cancel_reason"} if reason == "ordered by mistake" else set()))
        if fallback == "return":
            original = order.payment_history[0].payment_method_id
            spec["payment"] = original
            line_ = rng.choice(NOT_IN_STOCK["return"]).format(item=product(item.name))
            return Part(spec, "an exchange to a variant out of stock, then a return", request,
                        [line_, rng.choice(REFUND_TO).format(pay="the original payment method")],
                        [_call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                               payment_method_id=original)], tags=tags | {"return"})
        # the tools refuse a variant out of stock, so no reply can grant the request
        return Part(spec, "an item change to a variant out of stock; the customer keeps the item", request,
                    [rng.choice(NOT_IN_STOCK["nothing"])], needs_company=True, tags=tags)
    return None


# ---------------------------------------------------------------- requests the status or the policy turns around

# the customer's word alone, as in the official tasks: nothing says the order's status may not fit it
EXCHANGE_PENDING = ("You want to exchange {item} for one with {options}.",
                    "You'd like to exchange {item}; you want one with {options}.")
CHANGE_DELIVERED = ("You want to change {item} to one with {options}.",
                    "You'd like to modify {item} so that it has {options}.")
RETURN_PENDING = ("You want to return {item}. If it cannot be returned, cancel the whole order instead, since {why}.",
                  "You'd like to return {item}; if that is not possible, you want the whole order cancelled, because "
                  "{why}.")
CANCEL_PROCESSED = ("You want to cancel your order with {items}.", "You'd like your order with {items} cancelled.")
CANCEL_DELIVERED = ("You want to cancel {items}. If that cannot be done, return {them} instead, with the refund to the "
                    "original payment method.",
                    "You'd like {items} cancelled; if the agent cannot cancel {them}, you want to return {them}, "
                    "refunded to the original payment method.")


def wrong_status(db, order, rng, kind):
    """A request the order's status turns around: an "exchange" of a pending order is an item modification; a "return"
    of a pending order cannot be done, and the customer falls back on cancelling it; a "cancel" of items already
    delivered cannot be done, and the customer falls back on returning them; a processed order can be neither
    cancelled nor changed."""
    user = db.users[order.user_id]
    if kind == "cancel_delivered":
        if order.status != "delivered":
            return None
        named = [(i, bought(db, order, i)) for i in order.items if sum(j.name == i.name for j in order.items) == 1]
        named = [x for x in named if x[1]]
        if not named:
            return None
        chosen = sorted(rng.sample(named, min(len(named), rng.choice((1, 1, 2)))),
                        key=lambda x: order.items.index(x[0]))
        items = [i for i, _ in chosen]
        original = order.payment_history[0].payment_method_id
        spec = {"family": "cancel_delivered", "order": order.order_id, "items": [i.item_id for i in items],
                "named": [i.name for i in items]}
        return Part(spec, "a 'cancel' of items already delivered, then returning them", rng.choice(
            CANCEL_DELIVERED).format(items=" and ".join(n for _, n in chosen),
                                     them="it" if len(items) == 1 else "them"),
                    [], [_call("return_delivered_order_items", order_id=order.order_id,
                               item_ids=[i.item_id for i in items], payment_method_id=original)],
                    tags={"status", "return", "fallback"})
    if kind in ("exchange_pending", "change_delivered"):  # the tool goes by the status, not the customer's word
        if order.status != ("pending" if kind == "exchange_pending" else "delivered"):
            return None
        items = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]
        for item in rng.sample(items, len(items)):
            found = pick_variant(db, order, item, rng, ("exact",))
            if found is None:
                continue
            new_id, _, _, asked = found
            changed = {k: v for k, v in db.products[item.product_id].variants[new_id].options.items()
                       if item.options[k] != v}
            named = bought(db, order, item)
            pid = payer(db, order, difference(db, [(item, new_id)]), rng)
            if named is None or pid is None:
                return None
            spec = {"family": kind, "order": order.order_id, "item": item.item_id, "new": new_id, "payment": pid,
                    "named": [item.name]}
            pending = kind == "exchange_pending"
            tool = "modify_pending_order_items" if pending else "exchange_delivered_order_items"
            return Part(spec, "an 'exchange' of an order not delivered yet: an item modification" if pending else
                        "a 'change' of an item already delivered: an exchange", rng.choice(
                            EXCHANGE_PENDING if pending else CHANGE_DELIVERED).format(item=named,
                                                                                      options=opts(changed)),
                        unsaid(db, rng, [(item, new_id, asked, named)])
                        + [rng.choice(PAY_DIFF).format(pay=pay_name(user, pid, order))],
                        [_call(tool, order_id=order.order_id, item_ids=[item.item_id], new_item_ids=[new_id],
                               payment_method_id=pid)],
                        tags={"status", "mod_items" if pending else "exchange", "var_exact"})
        return None
    if kind == "return_pending":
        if order.status != "pending":
            return None
        item = rng.choice(order.items)
        named = bought(db, order, item)
        if named is None or sum(i.name == item.name for i in order.items) > 1:
            return None
        reason = rng.choice(REASONS)
        spec = {"family": "return_pending", "order": order.order_id, "reason": reason, "named": [item.name]}
        return Part(spec, "a 'return' of an order not delivered yet, then cancelling it", rng.choice(
            RETURN_PENDING).format(item=named, why=rng.choice(WHY[reason])), [],
                    [_call("cancel_pending_order", order_id=order.order_id, reason=reason)],
                    tags={"status", "cancel", "fallback"})
    if order.status != "processed":
        return None
    names = single_out(db, order, [], everywhere=True)
    if names is None:
        return None
    spec = {"family": "cancel_processed", "order": order.order_id, "named": names}
    # the cancel tool refuses an order that is not pending, so no reply can grant the request
    return Part(spec, "cancelling an order already processed, which the policy does not allow", rng.choice(
        CANCEL_PROCESSED).format(items=listed(names)), ["You no longer need it."], needs_company=True,
                tags={"status", "cancel"})


SAME_ITEM = ("You want to exchange the {item} from {which} for a new one exactly like it, since yours is faulty.",
             "The {item} in {which} arrived damaged; you want to exchange it for the very same item.")
SAME_ITEM_THEN = {"variant": ("If the agent cannot exchange it for the same item, exchange it {text} instead.",
                              "If an identical replacement is not possible, you'd exchange it {text}."),
                  "return": ("If the agent cannot exchange it for the same item, return it instead.",
                             "If an identical replacement is not possible, you want to return it.")}
OTHER_TYPE = ("You want to exchange the {item} from {which} for {other} of about the same price.",
              "You'd like to swap the {item} in {which} for {other} instead.")
OTHER_TYPE_THEN = {"return": ("If that is not possible, return the {item} instead.",
                              "If the agent cannot do that, you want to return the {item}."),
                   "nothing": ("If that is not possible, keep it.", "If the agent says no, leave things as they are.")}


def against_exchange(db, order, rng, kind, then):
    """Exchange delivered order, asked in a way the policy forbids: for the very same item (exchanges are for a different
    option) or for another product (exchanges keep the product). The customer falls back on another variant, a return,
    or nothing."""
    if order.status != "delivered":
        return None
    items = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]
    if not items:
        return None
    item = rng.choice(items)
    named = order_name(db, order, rng, [item])
    if named is None:
        return None
    user = db.users[order.user_id]
    original = order.payment_history[0].payment_method_id
    variants = db.products[item.product_id].variants
    spec = {"family": kind, "order": order.order_id, "item": item.item_id, "then": then, "named": named[1]}
    if kind == "same_item":
        if not variants[item.item_id].available:
            return None
        forbidden = _call("exchange_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                          new_item_ids=[item.item_id], payment_method_id=original)
        request = rng.choice(SAME_ITEM).format(item=product(item.name), which=named[0])
        tags = {"same_item_no", "fallback"}
    else:
        others = sorted({p.name for p in db.products.values()} - {item.name})
        other = rng.choice(others)
        forbidden = None
        request = rng.choice(OTHER_TYPE).format(item=product(item.name), which=named[0], other=article(other))
        spec["other"] = other
        tags = {"same_type", "fallback"}
    if then == "variant":
        found = pick_variant(db, order, item, rng, ("exact", "pref", "relative"))
        if found is None:
            return None
        new_id, text, t, s = found
        pid = payer(db, order, difference(db, [(item, new_id)]), rng)
        if pid is None:
            return None
        spec.update(new=new_id, then_spec=s, payment=pid)
        lines = [rng.choice(SAME_ITEM_THEN["variant"]).format(text=text),
                 *unsaid(db, rng, [(item, new_id, s, f"the {product(item.name)}")]),
                 rng.choice(PAY_DIFF).format(pay=pay_name(user, pid, order))]
        calls = [_call("exchange_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                       new_item_ids=[new_id], payment_method_id=pid)]
        tags |= {"exchange"} | t
    elif then == "return":
        lines = [rng.choice(SAME_ITEM_THEN["return"] if kind == "same_item" else OTHER_TYPE_THEN["return"]).format(
            item=product(item.name)), rng.choice(REFUND_TO).format(pay="the original payment method")]
        calls = [_call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                       payment_method_id=original)]
        tags |= {"return"}
    else:
        lines, calls = [rng.choice(OTHER_TYPE_THEN["nothing"])], []
    part = Part(spec, f"an exchange the policy forbids ({kind}), then {then}", request, lines, calls, forbidden,
                tags=tags | {"exchange"})
    if not calls and forbidden is None:
        part.needs_company = True
    return part


UNDER = ("You want to know the price of the cheapest {item} available right now. If it is under ${limit}, you want to "
         "exchange the {item} from {which} for it; if not, return that {item} instead.",
         "Ask the agent how much the cheapest {item} in stock costs: below ${limit}, you exchange your {item} from "
         "{which} for it; otherwise you return yours.")
# the same in both branches, so that it does not tell which one applies
EITHER_WAY = ("Any price difference or refund should go to the original payment method.",
              "Use the original payment method for any price difference or refund.")


def price_condition(db, order, rng, below):
    """Exchange or return, by a price the agent looks up: the customer exchanges an item for the cheapest variant in
    stock if it costs under their limit, and returns the item otherwise. below: whether it does."""
    if order.status != "delivered":
        return None
    items = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]
    for item in rng.sample(items, len(items)):
        variants = db.products[item.product_id].variants
        stock = sorted((v.price, k) for k, v in variants.items() if v.available)
        if len(stock) < 2 or stock[0][1] == item.item_id or stock[0][0] == stock[1][0] or "." not in money(stock[0][0]):
            continue
        price, new = stock[0]
        low = int(price // 50) * 50  # both limits must work, so that the two tasks of a variant group share the item
        if low <= 0 or low == price:
            continue
        limit = low + 50 if below else low
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        original = order.payment_history[0].payment_method_id
        user = db.users[order.user_id]
        if user.payment_methods[original].source == "gift_card" and user.payment_methods[original].balance < price - item.price:
            continue
        exchange = _call("exchange_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                         new_item_ids=[new], payment_method_id=original)
        ret = _call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                    payment_method_id=original)
        spec = {"family": "price_condition", "order": order.order_id, "item": item.item_id, "new": new,
                "limit": limit, "below": below, "named": named[1]}
        request = rng.choice(UNDER).format(item=product(item.name), limit=limit, which=named[0])
        tags = {"catalog", "cond_value", "budget", "info_amount"} | ({"exchange", "var_extreme"} if below else {"return"})
        return Part(spec, f"an exchange for the cheapest variant in stock if under a limit, else a return ({below})",
                    request, [rng.choice(EITHER_WAY)], [exchange if below else ret], ret if below else exchange,
                    info=[money(price)], by_policy=False, tags=tags)
    return None


COUNT_RULE = ("Ask the agent how many {product} options are available right now. If there are at least {n}, you want to "
              "exchange the {item} from {which} for the cheapest one available; if there are fewer, return it instead.",
              "You want to know how many {product} variants are in stock: with {n} or more, exchange your {item} from "
              "{which} for the cheapest one in stock; with fewer, return yours.")


def count_condition(db, order, rng, enough):
    """Exchange or return, by how many variants of the item's product are in stock: at least the customer's number, an
    exchange for the cheapest one in stock, else a return, so that the count decides what is written. The number is the
    count (enough) or one more, so that a count one off decides wrongly, as the official tasks ask for the exact count;
    only products of ten or more variants, as theirs."""
    if order.status != "delivered":
        return None
    items = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1
             and len(db.products[i.product_id].variants) >= 10]
    for item in rng.sample(items, len(items)):
        stock = sorted((v.price, k) for k, v in db.products[item.product_id].variants.items() if v.available)
        if len(stock) < 3 or stock[0][1] == item.item_id or stock[0][0] == stock[1][0]:
            continue
        price, new = stock[0]
        n = len(stock) if enough else len(stock) + 1
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        original = order.payment_history[0].payment_method_id
        method = db.users[order.user_id].payment_methods[original]
        if method.source == "gift_card" and method.balance < price - item.price:
            continue
        exchange = _call("exchange_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                         new_item_ids=[new], payment_method_id=original)
        ret = _call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                    payment_method_id=original)
        spec = {"family": "count_condition", "order": order.order_id, "item": item.item_id, "new": new,
                "threshold": n, "enough": enough, "named": named[1]}
        request = rng.choice(COUNT_RULE).format(product=product(item.name), item=product(item.name), n=n,
                                                which=named[0])
        tags = {"catalog", "info_count", "cond_value"} | ({"exchange", "var_extreme"} if enough else {"return"})
        return Part(spec, f"an exchange for the cheapest variant in stock if there are enough variants, else a return "
                          f"({enough})", request, [rng.choice(EITHER_WAY)], [exchange if enough else ret],
                    ret if enough else exchange, by_policy=False, tags=tags)
    return None


PAID = {
    "delivered": ("You want to know how much you paid for the {item} in {which}. If it was more than ${limit}, you want "
                  "to return it; if not, exchange it for the cheapest {item} available.",
                  "Ask the agent what you paid for the {item} from {which}: more than ${limit}, and you return it; "
                  "otherwise you exchange it for the cheapest {item} in stock."),
    "pending": ("You want to know how much you are paying for the {item} in {which}. If it is more than ${limit}, cancel "
                "the whole order, since you ordered it by mistake; if not, change it to the cheapest {item} available.",
                "Ask the agent what the {item} in {which} costs you: more than ${limit}, and you cancel that order "
                "because you ordered it by mistake; otherwise you change it to the cheapest {item} in stock."),
}


def paid_condition(db, order, rng, over):
    """Act on what the customer paid for an item, which the agent reads from the order: over their limit, a return, or
    for a pending order its cancellation; otherwise an exchange, or a modification, for the cheapest variant in stock.
    over: whether the price paid is over the limit. The item and the variant do not depend on over, so that the two
    tasks of a variant group differ only in the limit."""
    pending = order.status == "pending"
    items = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]
    for item in rng.sample(items, len(items)):
        said = money(item.price)
        stock = sorted((v.price, k) for k, v in db.products[item.product_id].variants.items() if v.available)
        if (item.price < 50 or "." not in said or len(stock) < 2 or stock[0][1] == item.item_id
                or stock[0][0] == stock[1][0]):
            continue
        price, new = stock[0]
        original = order.payment_history[0].payment_method_id
        method = db.users[order.user_id].payment_methods[original]
        if method.source == "gift_card" and method.balance < price - item.price:
            continue
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        limit = int(item.price // 50) * 50 if over else (int(item.price // 50) + 1) * 50
        if pending:
            keep = _call("modify_pending_order_items", order_id=order.order_id, item_ids=[item.item_id],
                         new_item_ids=[new], payment_method_id=original)
            drop = _call("cancel_pending_order", order_id=order.order_id, reason="ordered by mistake")
        else:
            keep = _call("exchange_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                         new_item_ids=[new], payment_method_id=original)
            drop = _call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                         payment_method_id=original)
        spec = {"family": "paid_condition", "order": order.order_id, "item": item.item_id, "new": new,
                "limit": limit, "over": over, "named": named[1]}
        request = rng.choice(PAID[order.status]).format(item=product(item.name), limit=limit, which=named[0])
        tags = {"order_info", "cond_order", "cond_value", "info_amount"} | (
            ({"cancel", "cancel_reason"} if pending else {"return"}) if over else
            ({"mod_items" if pending else "exchange", "var_extreme"}))
        return Part(spec, f"a {'cancellation' if pending else 'return'} if the item cost more than a limit, else a "
                          f"change to the cheapest variant in stock ({over})", request, [rng.choice(EITHER_WAY)],
                    [drop if over else keep], keep if over else drop, info=[said], by_policy=False, tags=tags)
    return None


BOTH = ("From {which}, you want to return the {ret} and exchange the {exc} {text}. Mention both at once; if the agent "
        "says only one of them can be done, you prefer the {prefer}.",
        "You'd like to return the {ret} and exchange the {exc} {text}, both from {which}, asked together. If only one is "
        "possible, you'd rather do the {prefer}.")


def one_action(db, order, rng, prefer):
    """Return and exchange on one delivered order: after either, the order is no longer 'delivered', so only one can be
    done, and the customer says which they prefer."""
    if order.status != "delivered":
        return None
    unique = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]
    if len(unique) < 2:
        return None
    ret, exc = rng.sample(unique, 2)
    found = pick_variant(db, order, exc, rng, ("exact", "pref", "extreme", "relative"))
    named = order_name(db, order, rng, [ret, exc])
    if found is None or named is None:
        return None
    new_id, text, t, s = found
    original = order.payment_history[0].payment_method_id
    diff = difference(db, [(exc, new_id)])
    user = db.users[order.user_id]
    if user.payment_methods[original].source == "gift_card" and user.payment_methods[original].balance < diff:
        return None
    exchange_call = _call("exchange_delivered_order_items", order_id=order.order_id, item_ids=[exc.item_id],
                          new_item_ids=[new_id], payment_method_id=original)
    return_call = _call("return_delivered_order_items", order_id=order.order_id, item_ids=[ret.item_id],
                        payment_method_id=original)
    calls, forbidden = ([exchange_call], return_call) if prefer == "exchange" else ([return_call], exchange_call)
    spec = {"family": "one_action", "order": order.order_id, "return": ret.item_id, "exchange": s, "prefer": prefer,
            "named": named[1]}
    request = rng.choice(BOTH).format(which=named[0], ret=product(ret.name), exc=product(exc.name), text=text,
                                      prefer=prefer if prefer == "exchange" else "return")
    return Part(spec, f"a return and an exchange on one delivered order, of which only the {prefer} can be done",
                request, unsaid(db, rng, [(exc, new_id, s, f"the {product(exc.name)}")]) + [rng.choice(REFUND_OR_DIFF)],
                calls, forbidden,
                by_policy=False, tags={"one_action", "fallback", "return", "exchange", "out_partial"} | t)
