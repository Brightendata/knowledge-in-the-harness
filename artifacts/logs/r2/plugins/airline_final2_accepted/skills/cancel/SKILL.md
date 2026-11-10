---
name: cancel
description: Cancel a flight reservation.
---

# Cancel flight

## Policy

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 
- If the user doesn't know their reservation id, the agent should help locate it using available tools.

The agent must also obtain the reason for cancellation (change of plan, airline cancelled flight, or other reasons)

If any portion of the flight has already been flown, the agent cannot help and transfer is needed.

Otherwise, flight can be cancelled if any of the following is true:
- The booking was made within the last 24 hrs
- The flight is cancelled by airline
- It is a business flight
- The user has travel insurance and the reason for cancellation is covered by insurance.

The API does not check that cancellation rules are met, so the agent must make sure the rules apply before calling the API!

Refund:
- The refund will go to original payment methods within 5 to 7 business days.

## Rules

These rules add detail to the policy. If a rule ever conflicts with the policy, follow the policy.

- **[airline:cancel-2]** Before calling `cancel_reservation`, verify that the reservation was made within the last 24 hours, the flight was cancelled by the airline, it is a business flight, or the user has travel insurance and the reason for cancellation is a health or weather reason. If the user's reason for cancellation is based on insurance, ask the user to confirm that it is covered; if none of these conditions are met, inform the user that the cancellation is not allowed and do not call the tool.
- **[airline:cancel-3]** Before calling `cancel_reservation`, list the action details and obtain explicit user confirmation (yes) to proceed.
- **[airline:cancel-4]** When identifying reservations that meet a user's specified criteria (such as the number of passengers), the agent must call `get_reservation_details` for every reservation ID returned by `get_user_details` to verify the criteria for each one.
- **[airline:cancel-5]** Before calling `cancel_reservation`, if the user claims a flight was cancelled by the airline, call `get_flight_status` for the flight segments in the reservation to verify that the status is 'cancelled'.
- **[airline:cancel-6]** Before calling `cancel_reservation`, if the user's confirmation includes a request or condition (e.g., regarding insurance), ensure that the request is addressed and the user is informed of the result before calling the tool.
- **[airline:cancel-9]** Even if the user provides explicit confirmation to proceed with a cancellation, do not call cancel_reservation if the criteria in rule airline:cancel-2 are not met.
- **[airline:cancel-10]** Before calling cancel_reservation, check if any flight in the reservation has already been flown (i.e., its date is before the current date). If it has, do not call cancel_reservation and transfer the user to a human agent.
