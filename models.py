# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Message Recomendation System Environment.

This environment models iterative AI reply recommendation:
- a user provides an incoming message
- the AI infers sentiment
- the AI suggests a reply
- the user either accepts it or rejects it with feedback
- each rejection advances the step and triggers a new suggestion
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class InteractionState(str, Enum):
    NEW = "new"
    REPLIED = "replied"
    ACKNOWLEDGED = "acknowledged"
    PENDING = "pending"
    RESOLVED = "resolved"


class Message(BaseModel):
    """Message text plus optional AI-inferred sentiment."""

    text: str = Field(..., description="The message content")
    description: str = Field(..., description="Additional context or explanation")
    sentiment: Optional[SentimentType] = Field(
        default=None,
        description="Sentiment inferred by the AI from the message text",
    )
    liked: bool = Field(
        default=False,
        description="Whether the user liked the current recommended reply",
    )


class RefineMessageAction(Action):
    message: Message = Field(..., description="The message to refine")
    tone: Optional[str] = Field(
        default=None,
        description="Desired tone for the refined message",
    )
    language: str = Field(
        default="en",
        description="Language for the refined message (ISO 639-1 code)",
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum character length for the refined message",
    )
    preserve_meaning: bool = Field(
        default=True,
        description="Whether to preserve the core meaning of the original message",
    )


class RecommendMessageAction(Action):
    message: Message = Field(..., description="The message to get recommendations for")
    num_recommendations: int = Field(
        default=3,
        description="Number of recommendations to generate",
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context for more relevant recommendations",
    )
    user_preferences: Optional[dict] = Field(
        default=None,
        description="User preferences for style or formality",
    )


class MessageRecomendationSystemAction(Action):
    """User action for iterative acceptance/rejection of AI recommendations."""

    user_message: str = Field(
        ...,
        description="The original incoming message that needs an AI-generated reply",
    )
    liked_message: bool = Field(
        default=False,
        description="Whether the user accepted the current recommended reply",
    )
    improvement_review: str = Field(
        default="",
        description="Feedback to improve the next recommended reply if the current one was rejected",
    )
    current_step: int = Field(
        default=1,
        ge=1,
        description="Current refinement step",
    )
    max_steps: int = Field(
        default=6,
        ge=1,
        description="Maximum number of refinement steps allowed",
    )


class MessageRecomendationSystemObservation(Observation):
    """Observation with inferred user-message sentiment and AI-recommended reply."""

    message: Message = Field(..., description="The original user message with AI-inferred sentiment")
    autoreply: Optional[Message] = Field(
        default=None,
        description="AI-generated recommended reply",
    )
    interaction_state: InteractionState = Field(
        default=InteractionState.NEW,
        description="Current state of the interaction",
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence score of the autoreply recommendation (0.0-1.0)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this observation was created",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID associated with this interaction",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking conversation flow",
    )
    requires_human_review: bool = Field(
        default=False,
        description="Whether this interaction requires human review",
    )
    accepted: bool = Field(
        default=False,
        description="Whether the user accepted the current recommendation",
    )
    current_step: int = Field(
        default=1,
        ge=1,
        description="Current step in the iterative recommendation workflow",
    )
    max_steps: int = Field(
        default=6,
        ge=1,
        description="Maximum number of refinement steps allowed",
    )
    grader_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic benchmark score for the current task. The terminal score "
            "is always clamped strictly inside the interval (0, 1)."
        ),
    )
