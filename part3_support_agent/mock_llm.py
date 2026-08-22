"""Deterministic, evidence-bound response generation."""

from part3_support_agent.guardrails import GROUNDEDNESS_REFUSAL, REFUSAL
from part3_support_agent.schemas import AgentResponse


def generate_response(
    intent: str,
    evidence: list[dict] | None = None,
    tool_result: dict | None = None,
    blocked: bool = False,
) -> dict:
    """Generate a deterministic structured answer without network or API keys."""
    if blocked:
        return AgentResponse(answer=REFUSAL, source="policy_kb", confidence=1.0).model_dump()
    if intent == "policy":
        if not evidence:
            return AgentResponse(answer=GROUNDEDNESS_REFUSAL, source="policy_kb", confidence=0.0).model_dump()
        best = evidence[0]
        return AgentResponse(
            answer=f"{best['document_text']} (Source: {best['document_id']})",
            source="policy_kb",
            confidence=float(best["score"]),
        ).model_dump()
    if intent == "return_risk":
        result = tool_result or {}
        return AgentResponse(
            answer=f"The estimated return probability is {result['return_probability']:.2f}, classified as {result['risk_bucket']} risk.",
            source="return_risk_tool",
            confidence=float(result["return_probability"]),
        ).model_dump()
    if intent == "product":
        result = tool_result or {}
        return AgentResponse(
            answer=f"The image is classified as {result['category']} with {result['confidence']:.2f} confidence.",
            source="image_classifier_tool",
            confidence=float(result["confidence"]),
        ).model_dump()
    raise ValueError(f"Invalid intent: {intent}")
