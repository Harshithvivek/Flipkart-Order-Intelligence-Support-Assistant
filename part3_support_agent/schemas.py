"""Strict response contracts for the support agent."""

from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["policy_kb", "return_risk_tool", "image_classifier_tool"]


class AgentResponse(BaseModel):
    answer: str = Field(min_length=1)
    source: Source
    confidence: float = Field(ge=0.0, le=1.0)
