---
name: compensation
description: Refunds and compensation for cancelled or delayed flights.
---

# Refunds and Compensation

## Policy

Do not proactively offer a compensation unless the user explicitly asks for one.

Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.

Always confirms the facts before offering compensation.

Only compensate if the user is a silver/gold member or has travel insurance or flies business.

- If the user complains about cancelled flights in a reservation, the agent can offer a certificate as a gesture after confirming the facts, with the amount being $100 times the number of passengers.

- If the user complains about delayed flights in a reservation and wants to change or cancel the reservation, the agent can offer a certificate as a gesture after confirming the facts and changing or cancelling the reservation, with the amount being $50 times the number of passengers.

Do not offer compensation for any other reason than the ones listed above.

## Rules

These rules add detail to the policy. If a rule ever conflicts with the policy, follow the policy.

- **[airline:compensation-1]** Before offering a compensation certificate, call `get_flight_status` and verify that the status confirms the flight was actually cancelled (for a cancellation complaint) or delayed (for a delay complaint).
