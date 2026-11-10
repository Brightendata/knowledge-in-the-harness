---
name: modify
description: Modify a pending order: shipping address, payment method, or product item options.
---

# Modify pending order

## Policy

An order can only be modified if its status is 'pending', and you should check its status before taking the action.

For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.

### Modify payment

The user can only choose a single payment method different from the original payment method.

If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.

After user confirmation, the order status will be kept as 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise it will be refunded within 5 to 7 business days.

### Modify items

This action can only be called once, and will change the order status to 'pending (items modifed)'. The agent will not be able to modify or cancel the order anymore. So you must confirm all the details are correct and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all the items they want to modify.

For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

## Rules

These rules add detail to the policy. If a rule ever conflicts with the policy, follow the policy.

- **[retail:modify-4]** When identifying which orders are pending, you must call `get_order_details` for every order ID in the list provided by `get_user_details` to verify its status.
- **[retail:modify-5]** When confirming details for a modification, only confirm the details you have access to from the tools and the policy; do not let the inability to provide other information prevent you from performing the database change if the user has confirmed the modification details.
- **[retail:modify-6]** Before asking for confirmation for a modification that results in a price increase, check if the payment method the user intends to use has enough balance. If the balance is insufficient, do not propose that modification.
- **[retail:modify-7]** When a user provides a description of an order that contradicts the order ID they provide, clarify with the user which order they wish to modify before asking for confirmation.
- **[retail:modify-8]** When proposing a modification to a pending order, you must remind the customer to confirm they have provided all the items they want to modify.
- **[retail:modify-9]** When the user asks to switch a product to the cheapest available option, you must call `get_product_details` for that product type to find all available options before responding.
- **[retail:modify-10]** When proposing a modification to a pending order, you must verify that the selected item matches all of the user's stated requirements (e.g., product type, material, color, capacity) from the product's variants before presenting it to the user.
- **[retail:modify-11]** When proposing a modification to a pending order where multiple variants meet the user's specified requirements, you must prioritize the variant that matches the original item's attributes unless the user specifies a different value for those attributes.
- **[retail:modify-12]** When the user asks to proceed with some of the requested modifications while others are still being decided, inform the user that performing the modifications now will change the order status to 'pending (items modified)' and prevent any further item modifications.
