"""Tests for intent routing, schema output, and conversation state."""


def test_agent_state_and_schema():
    from part3_support_agent.agent import invoke_support_agent

    features = {"product_category": "Fashion", "payment_method": "COD", "price_inr": 1000}
    first = invoke_support_agent("check return risk", order_features=features, order_id="A-1", thread_id="test-agent")
    carried = invoke_support_agent("which order did we assess?", thread_id="test-agent")
    fresh = invoke_support_agent("which order did we assess?", thread_id="test-fresh")
    assert first["source"] == "return_risk_tool"
    assert "A-1" in carried["answer"]
    assert "no order" in fresh["answer"].lower()
    assert set(first) == {"answer", "source", "confidence"}
