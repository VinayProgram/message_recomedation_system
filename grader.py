from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MessageRecommendationGrader:
    """
    Deterministic scorer for message recommendation quality.

    The score combines:
    - message quality signals
    - confidence_score from the observation or payload
    - simple safety / completeness checks
    """

    min_open_score: float = 0.01
    max_open_score: float = 0.99

    def grade_message(self, message: str, confidence_score: float) -> float:
        cleaned_message = (message or "").strip()
        confidence = self._clamp_closed(confidence_score)

        if not cleaned_message:
            return self.min_open_score

        quality = 0.0

        # Reward messages that are substantial enough to be useful.
        if len(cleaned_message) >= 8:
            quality += 0.20
        if len(cleaned_message) >= 20:
            quality += 0.15

        # Reward human-readable structure.
        if any(ch.isalpha() for ch in cleaned_message):
            quality += 0.20
        if cleaned_message[0].isupper():
            quality += 0.10
        if cleaned_message.endswith((".", "!", "?")):
            quality += 0.10

        # Penalize noisy or placeholder-like replies.
        lowered = cleaned_message.lower()
        if lowered in {"ok", "k", "hello", "test", "n/a"}:
            quality -= 0.25
        if cleaned_message.count("!") > 3:
            quality -= 0.10

        # Confidence_score is the strongest signal from the model/system.
        score = (quality * 0.45) + (confidence * 0.55)
        return self._clamp_open(score)

    def grade_observation(self, observation: Any) -> float:
        if observation is None:
            return 0.50

        existing_score = getattr(observation, "grader_score", None)
        if existing_score is not None:
            try:
                return self._clamp_open(float(existing_score))
            except (TypeError, ValueError):
                pass

        autoreply = getattr(observation, "autoreply", None)
        message_text = ""
        if autoreply is not None:
            message_text = getattr(autoreply, "text", "") or ""

        confidence_score = getattr(observation, "confidence_score", 0.0)
        return self.grade_message(message_text, confidence_score)

    def _clamp_closed(self, value: float) -> float:
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0

    def _clamp_open(self, value: float) -> float:
        return min(max(float(value), self.min_open_score), self.max_open_score)


def grade(observation=None, action=None, **kwargs) -> float:
    """
    Unified entrypoint expected by OpenEnv.
    Returns a score strictly inside the open interval (0, 1).
    """
    grader = MessageRecommendationGrader()

    if observation is not None:
        return grader.grade_observation(observation)

    if action is not None:
        message = getattr(action, "message", "")
        confidence_score = kwargs.get("confidence_score", 0.0)
        return grader.grade_message(message, confidence_score)

    return 0.50
