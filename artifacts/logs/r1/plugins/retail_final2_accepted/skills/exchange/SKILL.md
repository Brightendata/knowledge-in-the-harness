---
name: exchange
description: Exchange items of a delivered order for other options of the same product.
---

# Exchange delivered order

## Policy

An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.

For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.

## Rules

These rules add detail to the policy. If a rule ever conflicts with the policy, follow the policy.

- **[retail:exchange-1]** Before calling `exchange_delivered_order_items`, you must ensure that the new item has different product options than the original item. If the new item has the same product options, inform the user that an exchange is not possible.
