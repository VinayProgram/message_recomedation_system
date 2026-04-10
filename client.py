# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Message Recomendation System Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import MessageRecomendationSystemObservation, MessageRecomendationSystemAction
except ImportError:
    from models import MessageRecomendationSystemObservation, MessageRecomendationSystemAction


class MessageRecomendationSystemEnv(
    EnvClient[MessageRecomendationSystemAction, MessageRecomendationSystemObservation, State]
):
    def _step_payload(self, action: MessageRecomendationSystemAction) -> Dict:
        return {
            "user_message": action.user_message,
            "liked_message": action.liked_message,
            "improvement_review": action.improvement_review,
            "current_step": action.current_step,
            "max_steps": action.max_steps,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MessageRecomendationSystemObservation]:
        obs_data = payload.get("observation", {})
        observation = MessageRecomendationSystemObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
