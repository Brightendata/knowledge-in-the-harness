"""Generates the retail validation set V and the three training batches from tau2's retail database and policy, aimed at
the knowledge points of the official tasks.

Usage, from the repository root:
    python -m rsi.data_preprocessing.retail_generator
About 6 minutes on all cores; every run writes the same bytes. Then run python -m rsi.data_preprocessing.manual_fixes.

Reads tau2-bench/data/tau2/domains/retail/db.json, the official base tasks (their customers and reference writes are
left out) and data/test_annotation.json (the target share of each knowledge point). Writes
- data/processed/retail_generated.jsonl: every candidate, {"task", "source": "generated", "source_id", "db", "tags",
  "customer", "group"};
- data/processed/retail_final.jsonl and data/splits/retail.json: the tasks chosen for V (200) and the training batches
  round1-round3 (300 each), which rsi.benchmarks.run and the pipeline read;
- data/reports/retail_generate.json: candidates per family and tag, each split's counts against its targets, and per
  task what each request states (its spec) and, for a request the agent must refuse, the call that would grant it.

Then check every task with the tests, 8 processes in about 5 minutes; all 8 must print OK:
    for k in 0 1 2 3 4 5 6 7; do SHARD=$k/8 python -m unittest tests.test_retail_generate & done; wait

The requests are in retail/ (cancel.py, items.py, returns.py, modify.py, info.py); this file registers them as families
with quotas, combines them into tasks of several requests and writes each task's scenario.
"""
import json
import random
import re

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_tasks

from rsi.data_preprocessing.common import (CHUNK, ROOT, TRANSFER_LAST, Made, behaviours, leaks, line, make_task,
                                           outcome, outputs, owned, traits, write, writes_tag)
from rsi.data_preprocessing.retail.cancel import all_pending, cancel, cancel_across, cancel_partial, cancel_refund
from rsi.data_preprocessing.retail.chains import EITHER, bound_chain, budget_chain, linked, option_chain
from rsi.data_preprocessing.retail.facts import (FIND_THEM, REFUND_TO, facts_read, listed, pay_name, product, sayable,
                                                 which_item)
from rsi.data_preprocessing.retail.info import deny, info
from rsi.data_preprocessing.retail.items import (EITHER_WAY, PAY_DIFF, REFUND_OR_DIFF, against_exchange, change_items,
                                                 count_condition, one_action, out_of_stock, paid_condition,
                                                 price_condition, scope_items, wrong_status)
from rsi.data_preprocessing.retail.modify import address_items, order_address, payment, user_address
from rsi.data_preprocessing.retail.returns import EACH_REFUND, all_delivered, return_across, return_items

TAU2_RETAIL_DB = "tau2-bench/data/tau2/domains/retail/db.json"
GENERATED, FINAL, SPLITS, REPORT = outputs("retail")

FAMILIES = {  # family: (orders it applies to, builder)
    "cancel": ({"pending"}, cancel),
    "cancel_partial_whole": ({"pending"}, lambda db, o, rng: cancel_partial(db, o, rng, "whole")),
    "cancel_partial_keep": ({"pending"}, lambda db, o, rng: cancel_partial(db, o, rng, "keep")),
    "cancel_partial_cheapest": ({"pending"}, lambda db, o, rng: cancel_partial(db, o, rng, "cheapest")),
    "modify_1": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 1)),
    "modify_2": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 2)),
    "modify_held": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 2, held=True)),
    "modify_mind": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 2, mind="keep")),
    "modify_mind_pay": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 1, mind="payment")),
    "modify_mind_keep_pay": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 2, mind="keep_payment")),
    "modify_gift": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 1, gift=True)),
    "exchange_1": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 1)),
    "exchange_2": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 2)),
    "exchange_held": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 2, held=True)),
    "exchange_mind": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 2, mind="keep")),
    "exchange_mind_pay": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 1, mind="payment")),
    "exchange_mind_keep_pay": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 2, mind="keep_payment")),
    "exchange_mind_return": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 1, mind="return")),
    "exchange_gift": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 1, gift=True)),
    "modify_gift_covers": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 1, gift="covers")),
    "exchange_gift_covers": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 1, gift="covers")),
    "stock_other": ({"pending", "delivered"}, lambda db, o, rng: out_of_stock(db, o, rng, "other")),
    "stock_cancel": ({"pending"}, lambda db, o, rng: out_of_stock(db, o, rng, "cancel")),
    "stock_return": ({"delivered"}, lambda db, o, rng: out_of_stock(db, o, rng, "return")),
    "stock_nothing": ({"pending", "delivered"}, lambda db, o, rng: out_of_stock(db, o, rng, "nothing")),
    "return": ({"delivered"}, return_items),
    "return_all": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "all")),
    "return_except": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "except")),
    "return_held": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "held")),
    "return_gift": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "named", "gift")),
    "return_card_original": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "named", "card:original")),
    "return_card_gift": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "named", "card:gift")),
    "return_card_human": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "named", "card:human")),
    **{f"return_{scope}_card_{then}": ({"delivered"}, lambda db, o, rng, scope=scope, then=then: return_items(
        db, o, rng, scope, f"card:{then}")) for scope in ("except", "twin") for then in ("original", "gift", "human")},
    "address_given": ({"pending"}, lambda db, o, rng: order_address(db, o, rng, "given")),
    "address_default": ({"pending"}, lambda db, o, rng: order_address(db, o, rng, "default")),
    "address_order": ({"pending"}, lambda db, o, rng: order_address(db, o, rng, "order")),
    **{f"address_items_{s}": ({"pending"}, lambda db, o, rng, s=s: address_items(db, o, rng, s))
       for s in ("given", "default", "order")},
    "address_items_held": ({"pending"}, lambda db, o, rng: address_items(db, o, rng, "order", 2, held=True)),
    "address_items_two": ({"pending"}, lambda db, o, rng: address_items(db, o, rng, rng.choice(("order", "given")), 2)),
    # the official tasks' default address, two items, and at confirmation one item kept and the payment switched
    "address_items_mind": ({"pending"}, lambda db, o, rng: address_items(db, o, rng, "default", 2,
                                                                          mind="keep_payment")),
    **{f"address_items_home_{s}": ({"pending"}, lambda db, o, rng, s=s: address_items(db, o, rng, s, home=True))
       for s in ("given", "order")},
    "payment": ({"pending"}, payment),
    "payment_gift": ({"pending"}, lambda db, o, rng: payment(db, o, rng, gift_first=True)),
    "exchange_pending": ({"pending"}, lambda db, o, rng: wrong_status(db, o, rng, "exchange_pending")),
    "change_delivered": ({"delivered"}, lambda db, o, rng: wrong_status(db, o, rng, "change_delivered")),
    "return_pending": ({"pending"}, lambda db, o, rng: wrong_status(db, o, rng, "return_pending")),
    "cancel_delivered": ({"delivered"}, lambda db, o, rng: wrong_status(db, o, rng, "cancel_delivered")),
    "cancel_processed": ({"processed"}, lambda db, o, rng: wrong_status(db, o, rng, "cancel_processed")),
    "same_item_variant": ({"delivered"}, lambda db, o, rng: against_exchange(db, o, rng, "same_item", "variant")),
    "same_item_return": ({"delivered"}, lambda db, o, rng: against_exchange(db, o, rng, "same_item", "return")),
    "other_type_return": ({"delivered"}, lambda db, o, rng: against_exchange(db, o, rng, "other_type", "return")),
    "one_action_exchange": ({"delivered"}, lambda db, o, rng: one_action(db, o, rng, "exchange")),
    "one_action_return": ({"delivered"}, lambda db, o, rng: one_action(db, o, rng, "return")),
    "tracking": ({"delivered", "processed"}, lambda db, o, rng: info(db, o.user_id, rng, "tracking", o)),
    "total": ({"delivered", "processed", "pending"}, lambda db, o, rng: info(db, o.user_id, rng, "total", o)),
    "attribute": ({"delivered", "processed", "pending"}, lambda db, o, rng: info(db, o.user_id, rng, "attribute", o)),
    "priciest": ({"delivered", "processed", "pending"}, lambda db, o, rng: info(db, o.user_id, rng, "priciest", o)),
    "shipped_to": ({"delivered", "processed"}, lambda db, o, rng: info(db, o.user_id, rng, "shipped_to", o)),
    "card": ({"delivered", "processed", "pending"}, lambda db, o, rng: info(db, o.user_id, rng, "card", o)),
    "lost_item": ({"delivered"}, lambda db, o, rng: deny(db, o.user_id, rng, "lost", o)),
    "cancel_refund_keep": ({"pending"}, lambda db, o, rng: cancel_refund(db, o, rng, "keep")),
    "cancel_refund_cancel": ({"pending"}, lambda db, o, rng: cancel_refund(db, o, rng, "cancel")),
    "price_below": ({"delivered"}, lambda db, o, rng: price_condition(db, o, rng, True)),
    "price_above": ({"delivered"}, lambda db, o, rng: price_condition(db, o, rng, False)),
    "count_enough": ({"delivered"}, lambda db, o, rng: count_condition(db, o, rng, True)),
    "count_short": ({"delivered"}, lambda db, o, rng: count_condition(db, o, rng, False)),
    "paid_over": ({"pending", "delivered"}, lambda db, o, rng: paid_condition(db, o, rng, True)),
    "paid_under": ({"pending", "delivered"}, lambda db, o, rng: paid_condition(db, o, rng, False)),
    "return_twin": ({"delivered"}, lambda db, o, rng: return_items(db, o, rng, "twin")),
    **{f"{kind}_{how}": ({"pending" if kind == "modify" else "delivered"},
                         lambda db, o, rng, how=how: change_items(db, o, rng, 1, kinds=(how,)))
       for kind in ("modify", "exchange") for how in ("pref", "relative", "relative_pref", "bound", "same_as", "extreme")},
    **{f"{kind}_twin": ({"pending" if kind == "modify" else "delivered"},
                        lambda db, o, rng: change_items(db, o, rng, 1, twin=True)) for kind in ("modify", "exchange")},
    # the customer's word does not fit the order's status, with any description of the new variants
    "status_modify": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 1, said="exchange")),
    "status_modify_2": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 2, said="exchange")),
    "status_modify_gift": ({"pending"}, lambda db, o, rng: change_items(db, o, rng, 1, gift=True, said="exchange")),
    "status_exchange": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 1, said="modify")),
    "status_exchange_2": ({"delivered"}, lambda db, o, rng: change_items(db, o, rng, 2, said="modify")),
    **{f"address_items_status_{s}": ({"pending"}, lambda db, o, rng, s=s: address_items(db, o, rng, s, said="exchange"))
       for s in ("given", "default", "order")},
    # what the agent finds or works out decides the next step
    "budget_within": ({"pending"}, lambda db, o, rng: budget_chain(db, o, rng, True)),
    "budget_over": ({"pending"}, lambda db, o, rng: budget_chain(db, o, rng, False)),
    **{f"option_{case}": ({"pending", "delivered"}, lambda db, o, rng, case=case: option_chain(db, o, rng, case))
       for case in ("under", "over", "gone")},
    **{f"linked_{how}": ({"delivered"}, lambda db, o, rng, how=how: linked(db, o, rng, how))
       for how in ("exchange", "gone", "same")},
    **{f"bound_{side}": ({"pending", "delivered"}, lambda db, o, rng, side=side: bound_chain(db, o, rng, side == "found"))
       for side in ("found", "none")},
}
USER_FAMILIES = {
    "user_address_given": lambda db, u, rng: user_address(db, u, rng, "given"),
    "user_address_order": lambda db, u, rng: user_address(db, u, rng, "order"),
    "all_pending_cancel": lambda db, u, rng: all_pending(db, u, rng, "cancel"),
    "all_pending_address": lambda db, u, rng: all_pending(db, u, rng, "address"),
    "all_delivered": lambda db, u, rng: all_delivered(db, u, rng, False),
    "all_delivered_but": lambda db, u, rng: all_delivered(db, u, rng, True),
    "scope_items_pending": lambda db, u, rng: scope_items(db, u, rng, "pending"),
    "scope_items_delivered": lambda db, u, rng: scope_items(db, u, rng, "delivered"),
    "all_pending_address_order": lambda db, u, rng: all_pending(db, u, rng, "address_order"),
    **{f"refund_scope_{then}": lambda db, u, rng, then=then: all_delivered(db, u, rng, rng.random() < 0.5,
                                                                           f"card:{then}")
       for then in ("original", "gift", "human")},
    "return_across": return_across,
    "cancel_across": cancel_across,
    "cheapest": lambda db, u, rng: info(db, u, rng, "cheapest"),
    "new_order": lambda db, u, rng: deny(db, u, rng, "new_order"),
    "fast_refund": lambda db, u, rng: deny(db, u, rng, "fast_refund"),
    "undo": lambda db, u, rng: deny(db, u, rng, "undo"),
}
# Variant groups: the same customer and story with one condition changed, so that the reference changes with it.
VARIANTS = {
    "partial_group": ({"pending"}, ["cancel_partial_whole", "cancel_partial_keep"]),
    "one_action_group": ({"delivered"}, ["one_action_exchange", "one_action_return"]),
    "refund_group": ({"delivered"}, ["return_card_original", "return_card_gift"]),
    "stock_group": ({"pending"}, ["stock_other", "stock_cancel"]),
    "cancel_refund_group": ({"pending"}, ["cancel_refund_keep", "cancel_refund_cancel"]),
    "price_group": ({"delivered"}, ["price_below", "price_above"]),
    "count_group": ({"delivered"}, ["count_enough", "count_short"]),
    "paid_group": ({"pending", "delivered"}, ["paid_over", "paid_under"]),
    "budget_group": ({"pending"}, ["budget_within", "budget_over"]),
    "option_group": ({"pending", "delivered"}, ["option_under", "option_over"]),
    "linked_group": ({"delivered"}, ["linked_exchange", "linked_gone"]),
    "bound_group": ({"pending", "delivered"}, ["bound_found", "bound_none"]),
}
# Families a task of several requests draws from, one request each on its own order; a request that ends with a
# transfer only comes last.
PAIRED = ("cancel", "cancel_partial_whole", "cancel_partial_keep", "modify_1", "modify_2", "modify_held", "exchange_1",
          "exchange_2", "exchange_held", "stock_other", "stock_cancel", "stock_return", "stock_nothing", "return",
          "return_all", "return_except", "return_held", "return_gift", "return_card_original", "return_card_human",
          "address_given",
          "address_default", "address_order", "payment", "exchange_pending", "return_pending", "cancel_processed",
          "cancel_delivered", "same_item_variant", "same_item_return", "other_type_return", "tracking", "total",
          "attribute", "priciest", "shipped_to", "card", "user_address_given", "user_address_order", "all_pending_cancel",
          "all_pending_address", "all_pending_address_order", "refund_scope_original", "refund_scope_gift",
          "refund_scope_human", "return_except_card_original", "return_except_card_gift", "return_except_card_human",
          "return_twin_card_original", "return_twin_card_gift", "return_twin_card_human", "status_modify", "status_modify_2", "status_modify_gift", "status_exchange",
          "status_exchange_2", "address_items_status_given", "address_items_status_default",
          "address_items_status_order",
          "cheapest", "new_order", "fast_refund", "undo", "lost_item", "return_across", "cancel_across", "modify_gift",
          "exchange_gift", "cancel_refund_keep",
          "price_below", "price_above", "modify_pref", "exchange_pref", "modify_relative", "exchange_relative",
          "modify_relative_pref", "exchange_relative_pref",
          "modify_twin", "exchange_twin", "return_twin", "modify_same_as", "exchange_same_as", "modify_bound",
          "exchange_bound", "change_delivered", "all_delivered", "all_delivered_but", "cancel_refund_cancel",
          "modify_mind", "exchange_mind", "address_items_given", "address_items_default", "address_items_order",
          "address_items_held", "address_items_two", "address_items_mind", "address_items_home_given",
          "address_items_home_order", "modify_mind_pay",
          "exchange_mind_pay", "modify_mind_keep_pay", "exchange_mind_keep_pay", "exchange_mind_return", "count_enough",
          "count_short", "scope_items_pending",
          "scope_items_delivered", "paid_over", "paid_under")
# How often a family is tried first for a slot of a task of several requests. The official tasks of several requests
# combine status mix-ups, addresses taken from other orders and described variants far more often than requests over
# all of a customer's orders, which already have several writes each.
WEIGHT = {**{f: 3 for f in ("exchange_pending", "return_pending", "cancel_delivered", "change_delivered",
                            "address_order",
                            "address_default", "user_address_order", "address_items_given", "address_items_default",
                            "address_items_order", "address_items_held", "address_items_two", "address_items_mind",
                            "address_items_home_given",
                            "address_items_home_order", "status_modify", "status_modify_2", "status_modify_gift",
                            "status_exchange", "status_exchange_2", "address_items_status_given",
                            "address_items_status_default", "address_items_status_order")},
          **{f: 2 for f in ("modify_pref", "exchange_pref", "modify_relative_pref", "exchange_relative_pref",
                            "modify_twin", "exchange_twin", "return_twin",
                            "cancel_partial_whole", "cancel_partial_keep", "modify_held", "exchange_held", "return_held",
                            "modify_mind",
                            "exchange_mind", "stock_other", "stock_cancel", "stock_return", "modify_mind_pay",
                            "exchange_mind_pay", "modify_mind_keep_pay", "exchange_mind_keep_pay",
                            "exchange_mind_return")},
          # questions about an order, tried before the total, which fits every order and would crowd them out
          **{f: 4 for f in ("card", "shipped_to", "attribute", "priciest", "tracking")},
          **{f: 0.5 for f in ("all_delivered", "all_delivered_but", "all_pending_cancel", "all_pending_address",
                              "user_address_given", "scope_items_pending", "scope_items_delivered",
                              "all_pending_address_order", "refund_scope_original", "refund_scope_gift",
                              "refund_scope_human")}}
PROFILE = {"user_address_given", "user_address_order", "all_pending_address", "address_items_home_given",
           "address_items_home_order", "all_pending_address_order"}  # requests that set the default address
DEFAULT = {"address_default", "address_items_default", "address_items_status_default"}  # requests that ship to it
INFO = ("tracking", "total", "attribute", "priciest", "shipped_to", "cheapest", "card")
DENY = ("new_order", "fast_refund", "undo", "lost_item")  # refused, no write of their own
WRITING = tuple(f for f in PAIRED if f not in INFO + DENY)
ADDRESS_ITEMS = tuple(f for f in PAIRED if f.startswith("address_items"))
# the official tasks of three or more writes mostly ship one order to an address given or taken from another order,
# change its items (often in the wrong words), and return, cancel or exchange on another order (addr_write); the rest
# return, cancel or exchange on three orders (three_writes), or on all pending or delivered orders and one more
# (scope_write)
ADDR_WRITE = ("address_items_given", "address_items_order", "address_items_status_given", "address_items_status_order",
              "address_items_held", "address_items_two")
ONE_WRITE = ("return", "return_all", "return_except", "return_twin", "return_held", "cancel", "cancel_partial_whole",
             "exchange_1", "exchange_pending", "return_pending", "cancel_delivered", "change_delivered",
             "status_exchange")
SCOPE_WRITE = ("all_pending_cancel", "all_delivered", "all_delivered_but")
# slots of a task of several requests, each filled from its families: official tasks mostly ask for information along
# with a change. Customers rarely have three orders to act on, so the official tasks with three or more writes mostly
# change one order's address and items (and often the default address) along with another request (addr_pair).
MULTI = {"pair": [PAIRED] * 2, "triple": [PAIRED] * 3, "quad": [PAIRED] * 4, "info_pair": [WRITING, INFO],
         "info_triple": [WRITING, WRITING, INFO], "triple_write": [WRITING] * 3, "addr_pair": [ADDRESS_ITEMS, PAIRED],
         "addr_write": [ADDR_WRITE, ONE_WRITE], "three_writes": [ONE_WRITE] * 3, "scope_write": [SCOPE_WRITE, ONE_WRITE],
         # a lost item the policy has nothing for, refused, then the customer's other business (official tasks 31, 32)
         "lost_pair": [("lost_item",), WRITING],
         # the card that paid, where an order went and an option of an item, which few orders can be asked about,
         # along with a change
         "lookup_pair": [WRITING, ("card", "shipped_to", "attribute")]}
QUOTA = {
    # three orders acted on, and the questions few orders fit: before the requests of one order use up the customers'
    # orders
    "three_writes": 150, "scope_write": 100, "lookup_pair": 300, "lost_pair": 120,
    **{f: 30 for f in FAMILIES}, **{f: 30 for f in USER_FAMILIES},
    **{f: 60 for f in ("modify_1", "exchange_1", "return", "exchange_pending", "change_delivered", "return_pending",
                       "return_except", "return_held", "address_default", "address_order", "user_address_order",
                       "all_pending_cancel", "all_pending_address", "all_delivered", "all_delivered_but", "payment",
                       "other_type_return", "modify_twin", "exchange_twin", "return_twin", "return_gift",
                       "return_card_original", "return_card_human", "cancel_refund_keep", "cancel_refund_cancel",
                       "modify_mind", "exchange_mind", "modify_pref", "exchange_pref", "modify_relative_pref",
                       "exchange_relative_pref", "modify_same_as",
                       "exchange_same_as", "cancel_partial_whole", "cancel_partial_keep", "cancel",
                       "modify_mind_keep_pay", "exchange_mind_keep_pay", "return_across", "cancel_across",
                       "cancel_delivered")},
    **{f: 10 for f in INFO},
    "scope_items_pending": 60, "scope_items_delivered": 60,
    **{f: 30 for f in VARIANTS},
    # the official tasks ask for these along with other requests (info_pair, info_triple, lookup_pair), not on their own
    "card": 0, "priciest": 0, "attribute": 0, "shipped_to": 0,
    "address_items_two": 60, "address_items_mind": 60,
    **{f: 60 for f in ("status_modify", "status_exchange", "budget_group", "option_group", "modify_gift_covers",
                       "exchange_gift_covers")},
    **{f: 40 for f in ("status_modify_2", "status_exchange_2", "address_items_status_given",
                       "address_items_status_default", "address_items_status_order", "option_gone", "linked_same",
                       "linked_group", "all_pending_address_order", "bound_group")},
    # the two sides of a variant group only as the group, so that each story comes with its condition flipped
    **{f: 0 for f in ("budget_within", "budget_over", "option_under", "option_over", "linked_exchange", "linked_gone",
                      "bound_found", "bound_none", "cancel_partial_keep", "count_enough", "count_short",
                      "one_action_exchange", "one_action_return", "paid_over", "paid_under", "price_above",
                      "price_below", "stock_cancel", "stock_other")},
    # only as a request of a task of several: on their own the choice of V and the batches takes none of them
    **{f: 0 for f in ("address_given", "cancel", "cheapest", "exchange_gift", "exchange_pending", "modify_gift",
                      "return_card_gift", "return_pending", "stock_return", "total", "tracking", "user_address_given")},
    # one item of a pending order to cancel, then the whole order or nothing: no official task asks it alone, only with
    # other requests or in a chain (cancel_partial_cheapest, budget_group), and alone it would fill the knowledge point
    # with its easiest form
    "cancel_partial_whole": 0, "partial_group": 0,
    # before the other tasks of several requests use up the pending orders
    "addr_write": 200, "addr_pair": 150, "pair": 400, "triple": 150, "quad": 50, "info_pair": 450, "info_triple": 225,
    "triple_write": 175,
}
# How often a slot of a task of several requests may take a second writing request of a kind the task already has: the
# official tasks often exchange, return or cancel on two orders in one conversation (a quarter of them use one write
# tool twice)
REPEAT = 0.5
# Per customer and order across the candidates; the tasks of several requests come last and need room left
PER_USER = 30
PER_ORDER = 10
RATES = {"reactive": 0.2, "emotion": 0.12, "offtopic": 0.07}

# Each slot has a few wordings; a task picks one per slot with its own random.Random(task id).

INSIST = ("If the agent refuses, insist {n} times before you accept it.",
          "Do not take the first no for an answer: push back {n} times, then give up.")
ACCEPT_END = ("If the agent says it is not possible, accept that and end the conversation.",
              "If it cannot be done, thank the agent and end the call.")
ACCEPT_ON = ("If the agent says it is not possible, accept that and go on with your other request.",
             "If it cannot be done, accept it and move on to your other request.")
KNOWN_ZIP = ("You are {name} in zip code {zip}.", "Your name is {name} and your zip code is {zip}.",
             "You are {name}, and you live in {city}, {state} {zip}.", "You are {name}, residing in {city}, {zip}.")
KNOWN_EMAIL = ("Your name is {name} and your email is {email}.", "You are {name}, with email {email}.")
KNOWN_ID = ("You are {user_id}, living in zip code {zip}.",)
NO_EMAIL = ("You do not remember your email address.", "You don't remember your email.", "You do not know your email.")
WRONG_ZIP = "If the agent asks for your zip code, say {wrong} first; correct it to {zip} if the agent cannot find you."
PRODUCT_ID = ("You have a number noted for the {item}, {pid}, and read it out if the agent asks which item it is; if no "
              "item has that number, say you may have copied the product number instead.")
NO_ORDER_ID = ("You do not remember the order number.", "You don't know the order id.",
               "You don't have the order number at hand.")
ORDER = ("Mention all of your requests at the start of the conversation.",
         "Bring up each request only after the previous one has been dealt with.",
         "Make sure you mention your requests together, in the order given.")
CHANGE_MIND = ("When the agent asks you to confirm {what}, change your mind: you do not want that done after all.",
               "Once the agent is about to carry out {what}, tell it you have changed your mind and want it left as it "
               "is.")
ALSO = ("Also, {}", "In addition, {}", "On top of that, {}")
INFO_TAGS = {"compute", "info_amount", "info_balance"}
# paying: "today" ties the total to the charges; "the total you will pay" is read as the orders' new totals
TOTAL_ASK = {True: ("Once everything is done, ask the agent how much you will get back in total.",
                    "After all of this is done, you want to know the total amount refunded to you."),
             False: ("Once everything is done, ask the agent how much you need to pay today in total.",
                     "After all the changes, you want to know how much you will be charged today in total.")}
TOTAL_SHARE = 0.5  # of the tasks that could ask it
MOVES_MONEY = {"modify_pending_order_items", "exchange_delivered_order_items", "return_delivered_order_items",
               "cancel_pending_order", "modify_pending_order_payment"}
# The wording below is drawn with a task's own random.Random(f"{task id}/style"), apart from the draws that decide its
# tags, so that it changes no task's knowledge points. The official tasks script the user less than our sentences do:
# what to do after a refusal is mostly left to the user, a description that may change other options mostly does not say
# so, and a persona is often a kind of customer rather than a list of traits.
ACCEPT_SHARE = 0.25
RELAXED_SHARE = 0.2
PERSONA_SHARE = 0.4
PERSONAS = ("You are detail-oriented and want every part of your request handled properly.",
            "You keep your messages short and to the point.",
            "You are in a hurry and want this sorted out quickly.",
            "You are polite and thank the agent along the way.",
            "You tend to explain your situation at length.")
# half of the users who only answer what is asked are written as the official tasks' guarded kind of customer
GUARDED_SHARE = 0.5
GUARDED = ("You are guarded and reveal as little as you can.",
           "You keep details to yourself until the agent really needs them.",
           "You are not one to share much; you answer only what you must.")


def wrong_zip(db, user, rng):
    """A zip code one digit off the customer's under which no customer of that name lives, or None."""
    z = user.address.zip
    taken = {u.address.zip for u in db.users.values() if u.name == user.name}
    tries = [z[:i] + d + z[i + 1:] for i in (3, 4) for d in "0123456789" if d != z[i]]
    tries = [x for x in tries if x not in taken]
    return rng.choice(tries) if tries else None


def assemble(db, task_id, user_id, parts, rng, mind=None):
    """The task and its knowledge-point tags. Requests and their conditions go to reason_for_call and the persona and
    behaviours to task_instructions, as in the official retail tasks. A request asks for its amount only if no other
    request of the task moves money, which the question could be taken to be about. mind: the index of a request the
    user drops when asked to confirm it (parts already as drop() leaves them)."""
    user = db.users[user_id]
    style = random.Random(f"{task_id}/style")
    sentences, info, tags = [], [], set()
    for k, p in enumerate(parts):
        request = p.request if k == 0 else rng.choice(ALSO).format(p.request[0].lower() + p.request[1:])
        sentences += [request] + p.lines
        if p.transfer and len(parts) > 1:
            sentences.append(rng.choice(TRANSFER_LAST))
        if p.ask and not any(c["name"] in MOVES_MONEY for q in parts if q is not p for c in q.calls):
            sentences.append(p.ask)
            info += p.info
            tags |= p.tags
        else:
            tags |= p.tags - INFO_TAGS
            if p.info and not p.ask:  # the request itself asks for the information
                info += p.info
                tags |= p.tags & INFO_TAGS
        if (p.forbidden or p.needs_company) and not p.calls and p.by_policy and not p.transfer:
            if rng.random() < 0.2:
                sentences.append(rng.choice(INSIST).format(n=rng.choice((2, 3))))
                tags.add("pressure")
            # with other requests, the agent may answer this one first: the user does not hang up on them
            accept = rng.choice(ACCEPT_ON if len(parts) > 1 else ACCEPT_END)
            if style.random() < ACCEPT_SHARE:
                sentences.append(accept)
        if p.spec.get("named") == "id" and p.spec.get("order") and rng.random() < 0.3:
            digits = p.spec["order"][2:]
            # "that order": with other requests, the user simulator would give the number for their orders too
            sentences.append(f"When asked for that order's number, give {digits} first; if that does not work, say "
                             f"you forgot the W at the beginning.")
            tags.add("wrong_id")
        if p.spec.get("named") not in (None, "id") and p.spec.get("order") and rng.random() < 0.15:
            sentences.append(rng.choice(NO_ORDER_ID))
            tags.add("find_order")
        if isinstance(p.spec.get("named"), dict) and p.spec["named"]["by"] == "recent":
            tags.add("find_recent")
        if p.spec.get("family") in ("modify", "exchange") and rng.random() < 0.06:  # a product ID given as the item's
            order = db.orders[p.spec["order"]]
            item = next(i for i in order.items if i.item_id == p.spec["items"][0]["item"])
            if not which_item(order, item):
                sentences.append(PRODUCT_ID.format(item=product(item.name), pid=item.product_id))
                tags.add("id_confusion")
    # several requests that each move money the same way: the customer asks for the total over all of them, as official
    # tasks do; a task that ends with a transfer may not get that far
    flows = [[flow(db, c) for c in p.calls if flow(db, c)] for p in parts]
    signs = {x > 0 for f in flows for x in f}
    if sum(bool(f) for f in flows) > 1 and len(signs) == 1 and not any(p.transfer for p in parts) and \
            rng.random() < TOTAL_SHARE:
        total = sayable(db, user_id, abs(sum(x for f in flows for x in f)))
        if total:
            sentences.append(rng.choice(TOTAL_ASK[signs.pop()]))
            info.append(total)
            tags |= {"compute", "info_amount", "total_ops"}
    if len(parts) > 1:
        tags.add("multi")
        if rng.random() < 0.4:  # a customer holding an item back does not mention everything at once
            sentences.append(rng.choice(ORDER[1:2] if any(p.spec.get("held") or p.transfer for p in parts) else ORDER))
            tags.add("order")
    if mind is not None:
        sentences.append(rng.choice(CHANGE_MIND).format(what=dropped(parts[mind])))
    if len(user.payment_methods) == 1:  # the only method pays and is refunded: the official tasks leave it unsaid
        only = next(iter(user.payment_methods))
        said = {s.format(pay=p) for s in PAY_DIFF + REFUND_TO for p in ("the original payment method",
                                                                        pay_name(user, only))}
        sentences = [s for s in sentences if s not in said | set(REFUND_OR_DIFF + EITHER_WAY + EITHER + EACH_REFUND)]
    # the agent finds the orders without being told to; other options may change without being told so, mostly
    sentences = [s for s in sentences if s not in FIND_THEM and not (
        ("other options changed" in s or s.startswith("You do not mind other options")) and
        style.random() >= RELAXED_SHARE)]
    lines = []
    for b in behaviours(rng, RATES):
        lines.append(line(rng, b))
        if b == "reactive" and style.random() < GUARDED_SHARE:
            lines[-1] = style.choice(GUARDED)
        tags.add(b)
    persona = traits(rng)
    if style.random() < PERSONA_SHARE:
        persona = style.choice(PERSONAS)
    if persona:
        lines.insert(0, persona)
    name = f"{user.name.first_name} {user.name.last_name}"
    a = user.address
    how = rng.random()
    unknown = None
    if how < 0.25:
        known = rng.choice(KNOWN_EMAIL).format(name=name, email=user.email)
    else:
        known = rng.choice(KNOWN_ID if how > 0.95 else KNOWN_ZIP).format(
            name=name, user_id=user_id, zip=a.zip, city=a.city, state=a.state)
        unknown = rng.choice(NO_EMAIL)
        wrong = wrong_zip(db, user, rng) if how <= 0.95 and rng.random() < 0.06 else None
        if wrong:
            lines.append(WRONG_ZIP.format(wrong=wrong, zip=a.zip))
            tags.add("wrong_info")
    calls = [c for p in parts for c in p.calls]
    tags |= {"out_write" if calls else "out_nowrite", writes_tag(len(calls))}
    # a request a tool could grant is refused; one the user drops at confirmation is not, as in the official tasks
    if calls and any(not p.calls and p.forbidden and not p.spec.get("dropped") for p in parts):
        tags.add("out_partial")
    if any(p.transfer for p in parts):
        tags.add("out_transfer")
    return make_task("retail", task_id, parts, " ".join(sentences), known, unknown, lines, info), \
        sorted(tags | branches(db, parts))


def branches(db, parts):
    """Pseudo-tags for the branch of a policy rule, or the way of a request, a task takes, so that V and each batch have
    it (BRANCHES): an item change whose price difference a gift card pays that the order was not paid with, its balance
    covering it; how the customer changes their mind at confirmation; the wrong word the customer uses for the order's
    status; the request outside the policy that is refused. Also a task's shape the knowledge points do not show: one write
    tool called on two orders or more."""
    out = set()
    tools = [c["name"] for p in parts for c in p.calls]
    if any(tools.count(t) > 1 for t in tools):
        out.add("br_same_tool_twice")
    for p in parts:
        s = p.spec
        if s.get("mind") in ("keep", "payment", "return"):
            out.add(f"br_mind_{s['mind']}")
        if s.get("said"):
            out.add(f"br_status_said_{s['said']}")
        elif s.get("family") in ("exchange_pending", "change_delivered", "return_pending", "cancel_processed",
                                 "cancel_delivered"):
            out.add(f"br_status_{s['family']}")
        if s.get("family") in ("deny_new_order", "deny_fast_refund", "deny_undo"):
            out.add(f"br_{s['family']}")
        for c in p.calls:
            a = c["arguments"]
            if c["name"] in ("modify_pending_order_items", "exchange_delivered_order_items") and \
                    db.users[db.orders[a["order_id"]].user_id].payment_methods[a["payment_method_id"]].source == \
                    "gift_card" and a["payment_method_id"] != db.orders[a["order_id"]].payment_history[0].payment_method_id:
                items = {i.item_id: i for i in db.orders[a["order_id"]].items}
                if sum(db.products[items[x].product_id].variants[n].price - items[x].price
                       for x, n in zip(a["item_ids"], a["new_item_ids"])) > 0:
                    out.add("br_gift_pays")
    return out


# least in V and in each training batch, as in common.select; one write tool on two orders or more in a quarter of the
# official tasks (24.6%), which the choice does not match otherwise
BRANCHES = {"br_gift_pays": (6, 10), "br_same_tool_twice": (40, 45),
            **{f"br_{b}": (2, 6) for b in ("mind_keep", "mind_payment", "mind_return", "status_exchange_pending",
                                           "status_change_delivered", "status_return_pending",
                                           "status_cancel_processed", "status_cancel_delivered",
                                           "status_said_exchange", "status_said_modify", "deny_new_order",
                                           "deny_fast_refund", "deny_undo")}}


def dropped(part):
    """The request a user drops at confirmation, by what it is about: an ordinal could point at the wrong one, since a
    request that changes an order's address and items reads as two."""
    s = part.spec
    if s.get("family") == "user_address":
        return "the change of your default address"
    if s.get("family") == "scope_items":
        return f"the change to your {product(s['product'])}"
    if s.get("family") == "all_delivered":
        return "the return of what you have received"
    if isinstance(s.get("named"), dict):  # the order told apart by a fact of it
        return f"the request about {s['named']['text']}"
    if s.get("named") and s["named"] != "id":  # by the order: the products may name only part of what it acts on
        return f"the request about your order with {listed(s['named'])}"
    return f"the request about order {s['order']}"


def late_names(db, part):
    """The products a request brings up only when the agent asks to confirm it."""
    s = part.spec
    if not s.get("held"):
        return set()
    late = [s["late"]] if "late" in s else [x["item"] for x in s["items"][1:]]
    return {i.name for i in db.orders[s["order"]].items if i.item_id in late}


def slow_refund(db, call):
    """Whether a call sends money back to a method other than a gift card, which the policy says takes 5 to 7 business
    days (a gift card is refunded at once)."""
    a = call["arguments"]
    if "order_id" not in a:
        return False
    order = db.orders[a["order_id"]]
    if call["name"] == "cancel_pending_order":
        pid = order.payment_history[0].payment_method_id
    elif call["name"] == "return_delivered_order_items":
        pid = a["payment_method_id"]
    elif call["name"] in ("modify_pending_order_items", "exchange_delivered_order_items"):
        items = {i.item_id: i for i in order.items}
        if sum(db.products[items[x].product_id].variants[n].price - items[x].price
               for x, n in zip(a["item_ids"], a["new_item_ids"])) >= 0:
            return False
        pid = a["payment_method_id"]
    else:
        return False
    return db.users[order.user_id].payment_methods[pid].source != "gift_card"


def flow(db, call):
    """Money a call moves, from the customer's side: what a cancellation or a return refunds (positive), a price
    difference an item change refunds or charges (negative); 0 for a call that moves none."""
    a = call["arguments"]
    if call["name"] not in MOVES_MONEY or call["name"] == "modify_pending_order_payment":
        return 0
    items = {i.item_id: i for i in db.orders[a["order_id"]].items}
    if call["name"] == "cancel_pending_order":
        return round(sum(i.price for i in db.orders[a["order_id"]].items), 2)
    if call["name"] == "return_delivered_order_items":
        return round(sum(items[x].price for x in a["item_ids"]), 2)
    return round(-sum(db.products[items[x].product_id].variants[n].price - items[x].price
                      for x, n in zip(a["item_ids"], a["new_item_ids"])), 2)


def weighted(rng, names):
    """The families in a random order in which each comes early in proportion to its WEIGHT (Efraimidis-Spirakis)."""
    return sorted(names, key=lambda f: -rng.random() ** (1 / WEIGHT.get(f, 1)))


def orders_of(part):
    """The orders a request acts on, leaves as they are, takes an address or an item from, or compares its order with,
    so that no other request of the task changes what it names."""
    s = part.spec
    return ([s["order"]] if "order" in s else []) + s.get("orders", []) + s.get("untouched", []) + s.get("from", []) \
        + facts_read(s)[0]


def ships_to(part):
    """The cities a request ships an order to."""
    return {c["arguments"]["city"] for c in part.calls if c["name"] == "modify_pending_order_address"}


def changed_products(db, part):
    """The products whose items a request changes, which its sentences on the options left unsaid name alone."""
    return {i.name for c in part.calls + ([part.forbidden] if part.forbidden else [])
            if c["name"] in ("modify_pending_order_items", "exchange_delivered_order_items")
            for i in db.orders[c["arguments"]["order_id"]].items if i.item_id in c["arguments"]["item_ids"]}


def customers_of(db, task):
    """The customers an official task names, by an order or user id, an email, or a name and zip code."""
    text = json.dumps(task.model_dump(mode="json"))
    found = {db.orders["#" + o].user_id for o in re.findall(r"W\d{7}", text) if "#" + o in db.orders}
    for u in db.users.values():
        if u.user_id in text or u.email in text or (
                f"{u.name.first_name} {u.name.last_name}" in text and u.address.zip in text):
            found.add(u.user_id)
    return found


def official_users(db, test):
    """Customers of the official tasks."""
    return set().union(*(customers_of(db, t) for t in test))


def generate(db, official, taken):
    """Tasks per QUOTA family on customers of no evaluation task, each customer in at most PER_USER tasks and each order
    in at most PER_ORDER; a task whose reference writes are those of an evaluation task, or of a task made before, is
    skipped. Returns (family, task, parts, tags, customer, group) per task."""
    users = sorted(u for u, x in db.users.items() if u not in official and x.orders)
    orders = [db.orders[o] for u in users for o in db.users[u].orders]
    # a request whose items come out only at confirmation is not also dropped there, nor one that asks for its order's
    # tracking number, which a dropped request would still ask without its being checked
    made = Made("retail", db, taken, assemble, orders_of, (PER_USER, PER_ORDER),
                lambda p: not p.forbidden and not p.spec.get("held") and not p.spec.get("tracking"), 0.4, INFO_TAGS)

    def combine(family, slots, u, attempt):
        """A request of customer u per slot for a task of several, with their families, or None."""
        pick = random.Random(f"{family}_{u}_{attempt}")
        mine = [db.orders[o] for o in db.users[u].orders if made.uses[o] < PER_ORDER]
        parts, names = [], []
        for k, pool in enumerate(slots):
            # requests on an order first: the customer-level ones always fit and would crowd them out
            pool = weighted(pick, [f for f in pool if f in FAMILIES]) + \
                weighted(pick, [f for f in pool if f not in FAMILIES])
            for name in pool:
                kinds_so_far = [x.split("_")[0] for x in names]
                if kinds_so_far.count(name.split("_")[0]) > 1 or (
                        name.split("_")[0] in kinds_so_far and (name not in WRITING or pick.random() >= REPEAT)):
                    continue
                if name in USER_FAMILIES:
                    part = USER_FAMILIES[name](db, u, pick)
                else:
                    kinds, build = FAMILIES[name]
                    busy = {x for p in parts for x in orders_of(p)}
                    fits = [o for o in mine if o.status in kinds and o.order_id not in busy]
                    part = build(db, pick.choice(fits), pick) if fits else None
                # an order named by where it went stays the only one there; two requests do not change items of one
                # product, which the sentences on unsaid options name by the product alone
                if part is None or (part.transfer and k < len(slots) - 1) or any(
                        set(orders_of(part)) & set(orders_of(p)) or leaks(part, p) or leaks(p, part)
                        or facts_read(part.spec)[1] in ships_to(p) or facts_read(p.spec)[1] in ships_to(part)
                        or changed_products(db, part) & changed_products(db, p) for p in parts):
                    continue
                # both would set the default address, or one would change the one the other ships to
                if name in PROFILE and any(x in PROFILE | DEFAULT for x in names) or \
                        name in DEFAULT and any(x in PROFILE for x in names):
                    continue
                parts.append(part)
                names.append(name)
                break
        if len(parts) < len(slots):
            return None
        # a refund to a gift card arrives at once, so a faster one would not be refused
        # an item brought up only at confirmation is named by its product alone, which no other request may touch
        for p in parts:
            late = late_names(db, p)
            if late and any(late & {i.name for o in orders_of(q) for i in db.orders[o].items} for q in parts if q is not p):
                return None
        if family in ("addr_pair", "addr_write") and not parts[1].transfer and pick.random() < 0.5:  # either one first
            parts.reverse()
            names.reverse()
        # a guaranteed refund time is asked about a refund that takes days, which comes up before it
        k = next((k for k, p in enumerate(parts) if p.spec.get("family") == "deny_fast_refund"), None)
        if k is not None and not any(slow_refund(db, c) for p in parts[:k] for c in p.calls):
            return None
        return parts, names

    for family, quota in QUOTA.items():
        if not quota:
            continue
        rng = random.Random(family)
        if family in FAMILIES or family in VARIANTS:
            kinds, builds = (VARIANTS[family][0], [FAMILIES[f][1] for f in VARIANTS[family][1]]) \
                if family in VARIANTS else (FAMILIES[family][0], [FAMILIES[family][1]])
            candidates = [o for o in orders if o.status in kinds]
            rng.shuffle(candidates)
            made.singles(family, quota, candidates, lambda o: o.order_id[1:], lambda o: o.user_id, builds)
            continue
        candidates = list(users)
        rng.shuffle(candidates)
        if family in USER_FAMILIES:
            made.singles(family, quota, candidates, str, str, [USER_FAMILIES[family]])
            continue
        slots = MULTI[family]
        count = 0
        for attempt in range(4):  # several tasks of a kind per customer, each on other orders
            for start in range(0, len(candidates), CHUNK):
                if count == quota:
                    break
                # a customer's requests depend only on their own orders, so they are built ahead for a chunk
                chunk = [u for u in candidates[start:start + CHUNK] if made.free(u)]
                built = {u: combine(family, slots, u, attempt) for u in chunk}
                made.prefetch([b[0] for b in built.values() if b])
                for u in chunk:
                    if count == quota:
                        break
                    if built[u]:
                        parts, names = built[u]
                        task_id = f"generated_{family}_" + "+".join(names) + f"_{u}" + (f"_{attempt}" if attempt else "")
                        count += made.record(family, [task_id], u, [parts])
    made.pool.shutdown()
    return made.tasks


if __name__ == "__main__":
    db = RetailDB.load(ROOT / TAU2_RETAIL_DB)
    test = get_tasks("base")

    def owned_of(u):
        return owned(len(db.users[u].orders), "retail")

    # The official tasks' customers have more orders than the rest (4 or more for two in five tasks): half of the
    # customers with 4 or more orders are set aside for V, so that V can match them.
    heavy = {u for u, x in db.users.items() if len(x.orders) >= 4}
    write("retail", TAU2_RETAIL_DB, generate(db, official_users(db, test), {outcome(t, "retail") for t in test}), heavy,
          owned_of, {t.id: owned_of(min(customers_of(db, t))) for t in test}, BRANCHES)
