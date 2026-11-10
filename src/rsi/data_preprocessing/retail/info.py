"""Information the agent reads from the database and states, and requests the policy does not cover (refused)."""
import json
import re
from collections import Counter

from rsi.data_preprocessing.common import Part
from rsi.data_preprocessing.retail.facts import _call, article, money, order_name, product

TRACKING = ("You want the tracking number of {which}.", "Ask the agent for the tracking number of {which}.")
TOTAL = ("You want to know how much you paid in total for {which}.", "Ask the agent what the total of {which} came to.")
ATTRIBUTE = ("You want to know the {key} of the {item} in {which}.", "Ask the agent what {key} the {item} in {which} has.")
PRICIEST = ("You want to know which is the most expensive item in {which} and what it cost.",
            "Ask the agent for the most expensive item in {which} and its price.")
SHIPPED_TO = ("You want to know the full address {which} was shipped to.",
              "Ask the agent for the complete shipping address of {which}.")
CHEAPEST = ("You want to know the price of the cheapest {product} available right now.",
            "Ask the agent how much the cheapest {product} in stock costs.")
CARD = ("You want the last four digits of the credit card used to pay for {which}.",
        "Ask the agent for the last four digits of the credit card used to pay for {which}.")
# What a reply may write another way ("30 ml", "Wi-Fi and cellular", "space gray", "crewneck", "hardcover", "not water
# resistant", "smartwatch"), which the substring check would miss: no option value or product name to state has it.
SPELLED = re.compile(r"\d|[-+]|grey|crew neck|full size|hard cover|soft cover|auto shutoff|not resistant|smart watch",
                     re.I)


def info(db, user_id, rng, kind, order=None):
    """Information the agent reads from the database and states: an order's tracking number, total, an option of one of
    its items, its most expensive item, where it went or the card that paid for it, or the price of the cheapest variant
    in stock of a product."""
    user = db.users[user_id]
    if kind == "card":  # a credit card among the customer's payment methods, by digits nothing else of theirs shows
        if order is None:
            return None
        pid = order.payment_history[0].payment_method_id
        m = user.payment_methods[pid]
        rest = json.dumps([db.orders[o].model_dump(mode="json") for o in user.orders]) + json.dumps(
            [x.model_dump(mode="json") for k, x in user.payment_methods.items() if k != pid])
        if m.source != "credit_card" or len(user.payment_methods) < 2 or m.last_four in rest:
            return None
        named = order_name(db, order, rng)
        if named is None:
            return None
        return Part({"family": "card", "order": order.order_id, "named": named[1]}, "the card an order was paid with",
                    rng.choice(CARD).format(which=named[0]), [], info=[m.last_four],
                    tags={"order_info", "info_payment"})
    if kind == "tracking":
        if order is None or not order.fulfillments or len(order.fulfillments) != 1:
            return None
        named = order_name(db, order, rng)
        if named is None:
            return None
        said = order.fulfillments[0].tracking_id[0]
        return Part({"family": "tracking", "order": order.order_id, "named": named[1]}, "an order's tracking number",
                    rng.choice(TRACKING).format(which=named[0]), [], info=[said], tags={"order_info", "info_tracking"})
    if kind == "total":
        if order is None:
            return None
        named = order_name(db, order, rng)
        said = money(sum(p.amount for p in order.payment_history if p.transaction_type == "payment"))
        if named is None or "." not in said:
            return None
        return Part({"family": "total", "order": order.order_id, "named": named[1]}, "what an order cost",
                    rng.choice(TOTAL).format(which=named[0]), [], info=[said], tags={"order_info", "info_amount"})
    if kind == "attribute":  # a value no other item of the customer's has, of several words, so that a reply holds it
        if order is None:    # only if said
            return None
        seen = Counter(v for o in user.orders for i in db.orders[o].items for v in i.options.values())
        found = [(i, k, v) for i in order.items for k, v in i.options.items()
                 if seen[v] == 1 and " " in v and not SPELLED.search(v)
                 and sum(j.name == i.name for j in order.items) == 1]
        if not found:
            return None
        item, key, value = rng.choice(found)
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        return Part({"family": "attribute", "order": order.order_id, "item": item.item_id, "key": key,
                     "named": named[1]}, "an option of an item in an order",
                    rng.choice(ATTRIBUTE).format(key=key, item=product(item.name), which=named[0]), [], info=[value],
                    tags={"order_info", "info_attr"})
    if kind == "priciest":
        if order is None or len(order.items) < 2:
            return None
        prices = sorted(order.items, key=lambda i: i.price)
        said = money(prices[-1].price)
        named = order_name(db, order, rng)
        if named is None or prices[-1].price == prices[-2].price or "." not in said or SPELLED.search(prices[-1].name):
            return None
        text = rng.choice(PRICIEST).format(which=named[0])
        if prices[-1].name.lower() in text.lower():  # an order named by that product would have the agent say it anyway
            return None
        return Part({"family": "priciest", "order": order.order_id, "item": prices[-1].item_id, "named": named[1]},
                    "the most expensive item of an order", text, [], info=[said, prices[-1].name],
                    tags={"order_info", "info_name", "info_amount"})
    if kind == "shipped_to":  # all of the address, as the official tasks ask: the street alone leaves the rest unchecked
        a = order.address if order is not None else None
        if a is None or a.city == user.address.city or a.zip == user.address.zip:  # the customer's own city and zip
            return None                                                           # may be in the scenario
        named = order_name(db, order, rng)
        if named is None:
            return None
        return Part({"family": "shipped_to", "order": order.order_id, "named": named[1]}, "where an order went",
                    rng.choice(SHIPPED_TO).format(which=named[0]), [],
                    info=[x for x in (a.address1, a.address2, a.city, a.zip) if x], tags={"order_info", "info_address"})
    products = sorted(db.products.values(), key=lambda p: p.name)  # the cheapest variant in stock of a product
    p = rng.choice(products)
    stock = sorted(v.price for v in p.variants.values() if v.available)
    if len(stock) < 2 or stock[0] == stock[1] or "." not in money(stock[0]):
        return None
    return Part({"family": "cheapest", "product": p.product_id}, "the cheapest variant in stock",
                rng.choice(CHEAPEST).format(product=product(p.name)), [], info=[money(stock[0])],
                tags={"catalog", "info_amount"})


# ---------------------------------------------------------------- requests the policy does not cover

NEW_ORDER = ("You want the agent to place a new order for {product} for you.",
             "Ask the agent to order {product} for you.")
FAST_REFUND = ("Ask the agent to guarantee that your refund arrives within 3 days.",
               "You want the agent to promise the money will be back within two days.")
LOST = ("The {item} from {which} got lost after it arrived; you want the agent to refund it or send you a new one "
        "for free.",
        "You lost the {item} that came in {which}; ask the agent for your money back or a free replacement.")
LOST_THEN = ("If the agent can do neither, accept it and move on to your other request; you cannot send the item back.",
             "If neither is possible, let it go and continue with your other request; the item cannot be returned.")
UNDO = ("You cancelled an order by mistake and want the agent to undo the cancellation and ship everything as soon as "
        "possible. Insist; if the agent still cannot do it, ask for a human agent.",
        "You want the cancellation of your cancelled order reversed so that it ships after all. Be insistent, and if the "
        "agent will not do it, ask to be transferred to a human.")


def deny(db, user_id, rng, kind, order=None):
    """A request no tool grants and the policy does not offer: placing a new order, a faster refund, undoing a
    cancellation (the customer then asks for a human), a refund or replacement of a delivered item the customer lost (a
    return, which sends the item back, is not it). It only goes with a request that writes."""
    user = db.users[user_id]
    if kind == "lost":
        if order is None or order.status != "delivered":
            return None
        unique = [i for i in order.items if sum(j.name == i.name for j in order.items) == 1]
        if not unique:
            return None
        item = rng.choice(unique)
        named = order_name(db, order, rng, [item])
        if named is None:
            return None
        # a return is not what the customer asks for (the item cannot be sent back): no rule of the policy forbids it,
        # the customer's own situation does
        return Part({"family": "deny_lost", "order": order.order_id, "item": item.item_id, "named": named[1]},
                    "a refund or replacement of a lost item, which the policy does not cover",
                    rng.choice(LOST).format(item=product(item.name), which=named[0]), [rng.choice(LOST_THEN)],
                    forbidden=_call("return_delivered_order_items", order_id=order.order_id, item_ids=[item.item_id],
                                    payment_method_id=order.payment_history[0].payment_method_id),
                    by_policy=False, needs_company=True, tags={"deny", "deny_lost"})
    if kind == "new_order":
        text = rng.choice(NEW_ORDER).format(product=article(rng.choice(sorted(p.name for p in db.products.values()))))
    elif kind == "undo":
        if sum(db.orders[o].status == "cancelled" for o in user.orders) != 1:
            return None
        return Part({"family": "deny_undo", "user": user_id, "transfer": True}, "undoing a cancellation; the customer "
                    "then asks for a human", rng.choice(UNDO), [], needs_company=True, transfer=True,
                    tags={"deny", "pressure", "transfer"})
    else:
        text = rng.choice(FAST_REFUND)
    return Part({"family": f"deny_{kind}", "user": user_id}, f"a request the policy does not cover ({kind})", text, [],
                needs_company=True, tags={"deny"})
