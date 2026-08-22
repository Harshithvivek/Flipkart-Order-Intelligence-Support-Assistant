"""Conditional LangGraph workflow for support requests."""

import re
from typing import Literal

from langgraph.graph import END, START, StateGraph

from part3_support_agent.guardrails import is_prompt_injection
from part3_support_agent.mock_llm import generate_response
from part3_support_agent.rag import PolicyRetriever
from part3_support_agent.state import SupportState
from part3_support_agent.tools import check_return_risk, classify_product_image


def intent_node(state: SupportState) -> SupportState:
    """Route using deterministic few-shot-aligned keyword intent classification."""
    message = state["user_message"]
    lowered = message.lower()
    if is_prompt_injection(message):
        return {"intent": "blocked"}
    if any(word in lowered for word in ("image", "photo", "picture", ".png", ".jpg")):
        return {"intent": "product"}
    if any(word in lowered for word in ("risk", "probability", "return likelihood")):
        return {"intent": "return_risk"}
    return {"intent": "policy"}


def route_after_intent(state: SupportState) -> Literal["retrieval", "tool_calling", "response_generation"]:
    if state["intent"] == "policy":
        return "retrieval"
    if state["intent"] in ("return_risk", "product"):
        return "tool_calling"
    return "response_generation"


def retrieval_node(state: SupportState) -> SupportState:
    retriever = PolicyRetriever()
    try:
        retriever.load()
    except FileNotFoundError:
        retriever.build()
    return {"retrieved": retriever.retrieve(state["user_message"])}


def tool_calling_node(state: SupportState) -> SupportState:
    if state["intent"] == "return_risk":
        features = state.get("last_order_features")
        if features is None:
            raise ValueError("Return-risk requests require last_order_features in state")
        return {"tool_result": check_return_risk(features)}
    image_path = state.get("last_image_path")
    if image_path is None:
        raise ValueError("Product requests require last_image_path in state")
    return {"tool_result": classify_product_image(image_path)}


def response_generation_node(state: SupportState) -> SupportState:
    return {"response": generate_response(state["intent"], state.get("retrieved"), state.get("tool_result"), state["intent"] == "blocked")}


def build_graph():
    """Build a graph with conditional intent edges."""
    graph = StateGraph(SupportState)
    graph.add_node("intent_node", intent_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("tool_calling_node", tool_calling_node)
    graph.add_node("response_generation_node", response_generation_node)
    graph.add_edge(START, "intent_node")
    graph.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "retrieval": "retrieval_node",
            "tool_calling": "tool_calling_node",
            "response_generation": "response_generation_node",
        },
    )
    graph.add_edge("retrieval_node", "response_generation_node")
    graph.add_edge("tool_calling_node", "response_generation_node")
    graph.add_edge("response_generation_node", END)
    return graph.compile()
