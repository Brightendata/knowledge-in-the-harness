---
name: modify
description: Modify a reservation: flights, cabin, baggage and insurance, or passengers.
---

# Modify flight

## Policy

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 
- If the user doesn't know their reservation id, the agent should help locate it using available tools.

Change flights: 
- Basic economy flights cannot be modified.
- Other reservations can be modified without changing the origin, destination, and trip type.
- Some flight segments can be kept, but their prices will not be updated based on the current price.
- The API does not check these for the agent, so the agent must make sure the rules apply before calling the API!

Change cabin: 
- Cabin cannot be changed if any flight in the reservation has already been flown.
- In other cases, all reservations, including basic economy, can change cabin without changing the flights.
- Cabin class must remain the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.
- If the price after cabin change is higher than the original price, the user is required to pay for the difference.
- If the price after cabin change is lower than the original price, the user is should be refunded the difference.

Change baggage and insurance: 
- The user can add but not remove checked bags.
- The user cannot add insurance after initial booking.

Change passengers:
- The user can modify passengers but cannot modify the number of passengers.
- Even a human agent cannot modify the number of passengers.

Payment: 
- If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. The payment method must already be in user profile for safety reasons.

## Rules

These rules add detail to the policy. If a rule ever conflicts with the policy, follow the policy.

- **[airline:modify-1]** When calculating the price difference or refund for a modification, search for and determine the price of every flight in the new reservation, sum these prices, subtract the sum of the prices of all flights in the original reservation, and multiply the result by the number of passengers.
- **[airline:modify-2]** Before calling update_reservation_flights, check if the new flight(s) change the reservation's origin, destination, or trip type. If any of these attributes change, refuse the request and inform the user that modifications cannot change these details.
- **[airline:modify-3]** Before calling update_reservation_baggages, calculate the value for nonfree_baggages by subtracting the number of free bags from the total_baggages, and use 0 if the result is negative, where the number of free bags is the number of passengers multiplied by the allowance for their cabin class (regular: basic economy 0, economy 1, business 2; silver: basic economy 1, economy 2, business 3; gold: basic economy 2, economy 3, business 4).
- **[airline:modify-6]** When the user asks to change the cabin of a reservation, do not change the flight numbers unless the user specifically asks for new flights.
- **[airline:modify-7]** When the user asks to add a number of bags, calculate `total_baggages` as the current `total_baggages` plus the number the user wants to add, and `nonfree_baggages` as the number of bags in excess of the free allowance (minimum 0).
- **[airline:modify-8]** When the user provides explicit confirmation (yes) for a database update, call the corresponding tool to perform that update before addressing other requests or transferring the user.
- **[airline:modify-9]** Before calling update_reservation_baggages, always use the reservation's current cabin class and the user's membership level to calculate the baggage allowance.
