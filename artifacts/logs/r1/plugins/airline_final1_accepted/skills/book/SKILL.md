---
name: book
description: Book a new flight reservation.
---

# Book flight

## Policy

The agent must first obtain the user id from the user. 

The agent should then ask for the trip type, origin, destination.

Cabin:
- Cabin class must be the same across all the flights in a reservation. 

Passengers: 
- Each reservation can have at most five passengers. 
- The agent needs to collect the first name, last name, and date of birth for each passenger. 
- All passengers must fly the same flights in the same cabin.

Payment: 
- Each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards. 
- The remaining amount of a travel certificate is not refundable. 
- All payment methods must already be in user profile for safety reasons.

Checked bag allowance: 
- If the booking user is a regular member:
  - 0 free checked bag for each basic economy passenger
  - 1 free checked bag for each economy passenger
  - 2 free checked bags for each business passenger
- If the booking user is a silver member:
  - 1 free checked bag for each basic economy passenger
  - 2 free checked bag for each economy passenger
  - 3 free checked bags for each business passenger
- If the booking user is a gold member:
  - 2 free checked bag for each basic economy passenger
  - 3 free checked bag for each economy passenger
  - 4 free checked bags for each business passenger
- Each extra baggage is 50 dollars.

Do not add checked bags that the user does not need.

Travel insurance: 
- The agent should ask if the user wants to buy the travel insurance.
- The travel insurance is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.

## Rules

These rules add detail to the policy. If a rule ever conflicts with the policy, follow the policy.

- **[airline:book-6]** When calling `book_reservation`, the `payment_methods` objects must use the `payment_id` key for the payment's unique identifier and the `amount` key with an integer value.
