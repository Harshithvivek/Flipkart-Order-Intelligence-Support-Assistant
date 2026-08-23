"""Short-term conversational state for the support graph."""

from typing import Any, TypedDict


class SupportState(TypedDict, total=False):
    user_message: str
    order_id: str
    intent: str
    retrieved: list[dict[str, Any]]
    tool_result: dict[str, Any]
    response: dict[str, Any]
    last_order_features: dict[str, Any]
    last_order_id: str
    last_image_path: str
