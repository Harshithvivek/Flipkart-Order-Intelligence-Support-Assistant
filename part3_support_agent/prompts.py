"""Prompt definitions with explicit 4S annotations."""

SYSTEM_PROMPT = """You are Flipkart's support assistant.
4S principles:
- Specific: answer only the user's concrete request.
- Short: use a concise, useful response.
- Surround: use the supplied retrieved evidence or tool result as context.
- Single: provide one clear next action.
Never invent a policy outside the supplied evidence.
"""

INTENT_FEW_SHOT = [
    ("What is the return window for shoes?", "policy"),
    ("Tell me the return probability for my order", "return_risk"),
    ("Classify this product photo", "product"),
    ("How long does a COD refund take?", "policy"),
]
