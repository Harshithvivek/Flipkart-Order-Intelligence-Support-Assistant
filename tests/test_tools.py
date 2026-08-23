"""Tests for saved-artifact tool integration."""


def test_return_risk_tool():
    from part3_support_agent.tools import check_return_risk

    result = check_return_risk({"product_category": "Fashion", "payment_method": "COD", "price_inr": 1000})
    assert set(result) == {"return_probability", "risk_bucket"}
    assert 0.0 <= result["return_probability"] <= 1.0
    assert result["risk_bucket"] in {"Low", "Medium", "High"}
