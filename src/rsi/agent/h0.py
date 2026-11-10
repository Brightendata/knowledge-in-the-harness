"""H0, the plugins evolution starts from, generated from each domain's policy alone: the entry quotes the policy up to
its first action section and holds the routing table; each skill quotes one action section. Together the quotes are
the whole policy.

    python -m rsi.agent.h0

Writes plugins/retail/ and plugins/airline/; a rerun writes the same bytes.
"""
import json
import shutil

from tau2.domains.airline.utils import AIRLINE_POLICY_PATH
from tau2.domains.retail.utils import RETAIL_POLICY_PATH

from rsi.agent.plugin import ROOT

ROUTING = ("Each request of the user belongs to one of the categories below. As soon as you know which category a "
           "request belongs to, load that skill, and always load it before making any change for that request.")
NO_SECTION = "The policy has no section specific to this action; its general rules apply."
# skill: (policy section or None, title, routing-table request, catalog description)
DOMAINS = {
    "retail": {
        "policy": RETAIL_POLICY_PATH,
        "title": "Retail customer service",
        "description": ("Retail customer service. Load it first for any retail request; it says which retail skill to "
                        "load next."),
        "skills": {
            "cancel": ("Cancel pending order", "Cancel pending order", "Cancel pending order",
                       "Cancel a pending order."),
            "modify": ("Modify pending order", "Modify pending order",
                       "Modify pending order (shipping address, payment method, or product item options)",
                       "Modify a pending order: shipping address, payment method, or product item options."),
            "return": ("Return delivered order", "Return delivered order", "Return delivered order",
                       "Return items of a delivered order."),
            "exchange": ("Exchange delivered order", "Exchange delivered order", "Exchange delivered order",
                         "Exchange items of a delivered order for other options of the same product."),
            "user-address": (None, "Modify default user address", "Modify default user address",
                             "Modify the user's default address."),
        },
    },
    "airline": {
        "policy": AIRLINE_POLICY_PATH,
        "title": "Airline customer service",
        "description": ("Airline customer service. Load it first for any flight request; it says which airline skill "
                        "to load next."),
        "skills": {
            "book": ("Book flight", "Book flight", "Book flight", "Book a new flight reservation."),
            "modify": ("Modify flight", "Modify flight",
                       "Modify flight (flights, cabin, baggage and insurance, or passengers)",
                       "Modify a reservation: flights, cabin, baggage and insurance, or passengers."),
            "cancel": ("Cancel flight", "Cancel flight", "Cancel flight", "Cancel a flight reservation."),
            "compensation": ("Refunds and Compensation", "Refunds and Compensation", "Refunds and compensation",
                             "Refunds and compensation for cancelled or delayed flights."),
        },
    },
}


def split(policy, titles):
    """The policy before the first of these '## ' sections, and each section's body; the sections must be the last
    ones of the policy, in this order."""
    at = [policy.index(f"\n## {t}\n") for t in titles]
    assert at == sorted(at), titles
    ends = at[1:] + [len(policy)]
    bodies = {t: policy[a + len(f"\n## {t}\n"):e] for t, a, e in zip(titles, at, ends)}
    return policy[:at[0] + 1], bodies


def document(name, description, body):
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n"


def write(domain, out):
    spec = DOMAINS[domain]
    policy = spec["policy"].read_text()
    sections = [s[0] for s in spec["skills"].values() if s[0]]
    general, bodies = split(policy, sections)
    table = "\n".join(f"| {request} | `{domain}:{skill}` |" for skill, (_, _, request, _) in spec["skills"].items())
    if out.exists():
        shutil.rmtree(out)
    (out / "agents").mkdir(parents=True)
    manifest = {"name": domain, "version": "0", "description": spec["description"], "author": "knora"}
    (out / "plugin.json").write_text(json.dumps(manifest, indent=2) + "\n")
    entry = (f"# {spec['title']}\n\n## Policy\n\n{general.strip()}\n\n## Routing\n\n{ROUTING}\n\n"
             f"| Request | Skill |\n| --- | --- |\n{table}")
    (out / "agents" / f"{domain}.md").write_text(document(domain, spec["title"], entry))
    for skill, (section, title, _, desc) in spec["skills"].items():
        body = f"# {title}\n\n" + (f"## Policy\n\n{bodies[section].strip()}" if section else NO_SECTION)
        (out / "skills" / skill).mkdir(parents=True)
        (out / "skills" / skill / "SKILL.md").write_text(document(skill, desc, body))


if __name__ == "__main__":
    for d in DOMAINS:
        write(d, ROOT / "plugins" / d)
        print(f"wrote plugins/{d}")
