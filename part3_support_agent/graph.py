"""Conditional LangGraph workflow for support requests."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from part3_support_agent.guardrails import is_prompt_injection
from part3_support_agent.mock_llm import generate_response
from part3_support_agent.prompts import INTENT_FEW_SHOT
from part3_support_agent.rag import PolicyRetriever
from part3_support_agent.state import SupportState
from part3_support_agent.tools import check_return_risk, classify_product_image

RECALL_CUES = ("which order", "that order", "same order", "remind me", "earlier", "just assessed")
STOP_WORDS = {"the", "is", "a", "an", "for", "to", "my", "this", "what", "does", "i", "of"}


def _tokens(text: str) -> set[str]:
    """Normalize words for deterministic few-shot token-overlap routing."""
    return {word.strip("?!.,") for word in text.lower().split() if word not in STOP_WORDS}


def classify_intent(message: str) -> str:
    """Select the closest few-shot intent, defaulting unknown requests to policy."""
    query_tokens = _tokens(message)
    best_intent, best_score = "policy", 0.0
    for exemplar, intent in INTENT_FEW_SHOT:
        exemplar_tokens = _tokens(exemplar)
        score = len(query_tokens & exemplar_tokens) / max(len(query_tokens | exemplar_tokens), 1)
        if score > best_score:
            best_intent, best_score = intent, score
    return best_intent


def intent_node(state: SupportState) -> SupportState:
    """Route using deterministic few-shot-aligned keyword intent classification."""
    message = state["user_message"]
    lowered = message.lower()
    if is_prompt_injection(message):
        return {"intent": "blocked"}
    if any(cue in lowered for cue in RECALL_CUES):
        return {"intent": "recall"}
    if state.get("last_image_path"):
        return {"intent": "product"}
    if state.get("last_order_features"):
        return {"intent": "return_risk"}
    return {"intent": classify_intent(message)}


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
    return {"retrieved": retriever.retrieve(state["user_message"], threshold=0.35)}


def tool_calling_node(state: SupportState) -> SupportState:
    if state["intent"] == "return_risk":
        features = state.get("last_order_features")
        if features is None:
            raise ValueError("Return-risk requests require last_order_features in state")
        update: SupportState = {"tool_result": check_return_risk(features)}
        if state.get("order_id"):
            update["last_order_id"] = state["order_id"]
        return update
    image_path = state.get("last_image_path")
    if image_path is None:
        raise ValueError("Product requests require last_image_path in state")
    return {"tool_result": classify_product_image(image_path)}


def response_generation_node(state: SupportState) -> SupportState:
    if state["intent"] == "recall":
        order = state.get("last_order_id")
        answer = f"The order currently in context is {order}." if order else "There is no order in the current conversation context."
        return {"response": {"answer": answer, "source": "return_risk_tool" if order else "policy_kb", "confidence": 1.0}}
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
    return graph.compile(checkpointer=MemorySaver())
