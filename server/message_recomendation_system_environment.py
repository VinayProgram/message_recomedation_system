# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Message Recommendation System environment implementation."""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Optional, State

try:
    from ..grader import MessageRecommendationGrader
    from ..models import (
        InteractionState,
        Message,
        MessageRecomendationSystemAction,
        MessageRecomendationSystemObservation,
    )
    from .models.messaging.message_model import MessageSuggestionObservationAI
except ImportError:
    from grader import MessageRecommendationGrader
    from models import (
        InteractionState,
        Message,
        MessageRecomendationSystemAction,
        MessageRecomendationSystemObservation,
    )
    from server.models.messaging.message_model import MessageSuggestionObservationAI


class MessageRecomendationSystemEnvironment(
    Environment[MessageRecomendationSystemAction, MessageRecomendationSystemObservation, State]
):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._grader = MessageRecommendationGrader()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: object,
    ) -> MessageRecomendationSystemObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return MessageRecomendationSystemObservation(
            message=Message(
                text="Ready",
                description="Message recommendation system environment ready for iterative reply generation.",
                sentiment=None,
            ),
            autoreply=None,
            interaction_state=InteractionState.NEW,
            confidence_score=0.0,
            requires_human_review=False,
            accepted=False,
            current_step=1,
            max_steps=6,
            done=False,
            reward=0.0,
            metadata={"reset_count": self._reset_count},
            grader_score=0.01,
        )

    def step(self, action: MessageRecomendationSystemAction) -> MessageRecomendationSystemObservation:  # type: ignore[override]
        self._state.step_count += 1

        user_message = Message(
            text=action.user_message,
            description="Original incoming user message",
            sentiment=None,
            liked=action.liked_message,
        )

        suggestion = MessageSuggestionObservationAI.from_message(
            user_message,
            metadata={
                "original_message": action.user_message,
                "liked": action.liked_message,
                "improvement_review": action.improvement_review,
                "step": action.current_step,
            },
            improvement_review=action.improvement_review,
            accepted=action.liked_message,
            current_step=action.current_step,
            max_steps=action.max_steps,
        )

        grader_score = self._grader.grade_message(
            suggestion.autoreply.text if suggestion.autoreply else "",
            suggestion.confidence_score,
        )

        return MessageRecomendationSystemObservation(
            message=suggestion.message,
            autoreply=suggestion.autoreply,
            interaction_state=suggestion.interaction_state,
            confidence_score=suggestion.confidence_score,
            requires_human_review=suggestion.requires_human_review,
            accepted=suggestion.accepted,
            current_step=suggestion.current_step,
            max_steps=suggestion.max_steps,
            done=suggestion.done,
            reward=suggestion.reward,
            metadata={
                **(suggestion.metadata or {}),
                "original_message": action.user_message,
                "liked": action.liked_message,
                "improvement_review": action.improvement_review,
                "step": action.current_step,
            },
            grader_score=grader_score,
        )

    @property
    def state(self) -> State:
        return self._state
