from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import PrivateAttr

try:
    from ....models import (
        InteractionState,
        Message,
        MessageRecomendationSystemObservation,
        SentimentType,
    )
    from ....promt import SYSTEM_PROMPT
except ImportError:
    from models import (
        InteractionState,
        Message,
        MessageRecomendationSystemObservation,
        SentimentType,
    )
    from promt import SYSTEM_PROMPT


class MessageSuggestionObservationAI(MessageRecomendationSystemObservation):
    """Observation model that can generate an autoreply suggestion using OpenAI."""

    _client: OpenAI = PrivateAttr()
    _model_name: str = PrivateAttr()
    _system_prompt: str = PrivateAttr()

    def __init__(self, **data: Any):
        load_dotenv()
        super().__init__(**data)

        api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
        model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name
        self._system_prompt = SYSTEM_PROMPT

    def generate_autosuggestion(
        self,
        message: Message | None = None,
        improvement_review: str = "",
    ) -> "MessageSuggestionObservationAI":
        source_message = message or self.message
        payload = {
            "text": source_message.text,
            "description": source_message.description,
            "liked_message": self.accepted,
            "improvement_review": improvement_review,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }

        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Generate an autoreply recommendation for this message and return "
                        "valid JSON only.\n"
                        f"{json.dumps(payload, ensure_ascii=True)}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=400,
            stream=False,
        )

        raw_content = (completion.choices[0].message.content or "").strip()
        parsed = self._parse_response(raw_content, source_message)
        parsed["accepted"] = self.accepted
        parsed["current_step"] = self.current_step
        parsed["max_steps"] = self.max_steps
        parsed["done"] = self.accepted or self.current_step >= self.max_steps
        parsed["metadata"] = {
            **(parsed.get("metadata") or {}),
            "improvement_review": improvement_review,
        }
        return self.__class__(**parsed)

    @classmethod
    def from_message(
        cls,
        message: Message,
        metadata: dict[str, Any] | None = None,
        improvement_review: str = "",
        accepted: bool = False,
        current_step: int = 1,
        max_steps: int = 6,
    ) -> "MessageSuggestionObservationAI":
        return cls(
            message=message,
            autoreply=None,
            interaction_state=InteractionState.NEW,
            confidence_score=0.0,
            requires_human_review=False,
            accepted=accepted,
            current_step=current_step,
            max_steps=max_steps,
            done=False,
            reward=0.0,
            metadata=metadata or {},
        ).generate_autosuggestion(improvement_review=improvement_review)

    def _parse_response(self, raw_content: str, source_message: Message) -> dict[str, Any]:
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            data = self._fallback_payload(source_message)

        fallback = self._fallback_payload(source_message)
        message_data = data.get("message") or fallback["message"]
        autoreply_data = data.get("autoreply") or fallback["autoreply"]

        return {
            "message": Message(
                sentiment=self._parse_sentiment(message_data.get("sentiment"), source_message.sentiment),
                text=message_data.get("text", source_message.text),
                description=message_data.get("description", source_message.description),
                liked=source_message.liked,
            ),
            "autoreply": Message(
                sentiment=self._parse_sentiment(autoreply_data.get("sentiment"), SentimentType.POSITIVE),
                text=autoreply_data.get("text", "Thank you for your message."),
                description=autoreply_data.get(
                    "description",
                    "Automatically generated reply suggestion.",
                ),
                liked=False,
            ),
            "interaction_state": self._parse_interaction_state(data.get("interaction_state")),
            "confidence_score": self._clamp_score(data.get("confidence_score", 0.5)),
            "requires_human_review": bool(data.get("requires_human_review", False)),
            "reward": 1.0 if bool(data.get("reward", 1)) else 0.0,
            "metadata": {
                "provider_model": self._model_name,
                "raw_response": raw_content,
            },
        }

    def _fallback_payload(self, source_message: Message) -> dict[str, Any]:
        return {
            "message": {
                "sentiment": SentimentType.NEUTRAL.value,
                "text": source_message.text,
                "description": source_message.description,
            },
            "autoreply": {
                "sentiment": SentimentType.POSITIVE.value,
                "text": "Thank you for your message. I will look into this and get back to you shortly.",
                "description": "Fallback autoreply used because the model response was not valid JSON.",
            },
            "interaction_state": InteractionState.REPLIED.value,
            "confidence_score": 0.3,
            "reward": 0,
            "requires_human_review": True,
        }

    def _parse_sentiment(self, value: Any, default: SentimentType | None) -> SentimentType:
        try:
            return SentimentType(str(value).lower())
        except ValueError:
            return default or SentimentType.NEUTRAL

    def _parse_interaction_state(self, value: Any) -> InteractionState:
        try:
            return InteractionState(str(value).lower())
        except ValueError:
            return InteractionState.REPLIED

    def _clamp_score(self, value: Any) -> float:
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0
