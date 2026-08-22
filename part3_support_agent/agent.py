"""Public support-agent entry point."""

from typing import Any

from part3_support_agent.graph import build_graph
from part3_support_agent.schemas import AgentResponse


def invoke_support_agent(
    user_message: str,
    *,
    order_features: dict[str, Any] | None = None,
    image_path: str | None = None,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke one graph conversation turn; omit previous_state for a fresh conversation."""
    state = dict(previous_state or {})
    state["user_message"] = user_message
    if order_features is not None:
        state["last_order_features"] = order_features
    if image_path is not None:
        state["last_image_path"] = image_path
    result = build_graph().invoke(state)
    return AgentResponse.model_validate(result["response"]).model_dump()
