"""Input and groundedness guardrails."""

import re

INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+all\s+rules",
    r"pretend\s+you\s+are",
)
REFUSAL = "I cannot follow instructions that attempt to override the support assistant's rules."
GROUNDEDNESS_REFUSAL = "I don't have enough information in the policy knowledge base to answer that reliably."


def is_prompt_injection(text: str) -> bool:
    """Detect common instruction-override attempts case-insensitively."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS)
