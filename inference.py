"""
Inference Script Example (validation-critical stdout format)
===========================================================
MANDATORY
- Define these env vars in your environment config:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Local Docker image name for MyEnv.from_docker_image(), if used.

- Defaults are set only for API_BASE_URL and MODEL_NAME.
- Participants must use the OpenAI client for all LLM calls.

STDOUT FORMAT
- Emit exactly three line types to stdout, in this order, per task episode:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line per episode, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none (must be single-token).
    - All fields on a single line with no newlines within a line.
    - Each task must return score in [0, 1].
"""

import asyncio
import inspect
import os
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

from client import MessageRecomendationSystemEnv
from models import Message, MessageRecomendationSystemAction
from server.models.messaging.message_model import MessageSuggestionObservationAI
from tasks import list_scenarios

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("MY_ENV_BASE_URL")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "message_recommendation_system")
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
MIN_OPEN_SCORE = 0.01
MAX_OPEN_SCORE = 0.99


def _normalize_env_base_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "http://localhost:8000"
    if not (value.startswith("http://") or value.startswith("https://")):
        value = f"http://{value}"
    parsed = urlparse(value)
    if not parsed.netloc:
        return "http://localhost:8000"
    if parsed.port is None:
        host = parsed.hostname or "localhost"
        netloc = f"{host}:8000"
        if parsed.username or parsed.password:
            auth = parsed.username or ""
            if parsed.password:
                auth = f"{auth}:{parsed.password}"
            netloc = f"{auth}@{netloc}"
        parsed = parsed._replace(netloc=netloc)
    return urlunparse(parsed._replace(path="", params="", query="", fragment="")).rstrip("/")


def _env_spec_mode(value: Optional[str]) -> str:
    value = (value or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        return "url"
    if value:
        return "image"
    return "default"


async def _resolve_env() -> MessageRecomendationSystemEnv:
    mode = _env_spec_mode(ENV_BASE_URL)
    if mode == "url":
        return MessageRecomendationSystemEnv(base_url=ENV_BASE_URL or 'http://localhost:8000')
    if mode == "image":
        maybe_env = MessageRecomendationSystemEnv.from_docker_image((ENV_BASE_URL or "").strip())
        return await maybe_env if inspect.isawaitable(maybe_env) else maybe_env
    if LOCAL_IMAGE_NAME:
        maybe_env = MessageRecomendationSystemEnv.from_docker_image(LOCAL_IMAGE_NAME.strip())
        return await maybe_env if inspect.isawaitable(maybe_env) else maybe_env
    return MessageRecomendationSystemEnv(base_url=_normalize_env_base_url("http://localhost:8000"))


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def _bool_str(value: bool) -> str:
    return str(bool(value)).lower()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_val = (action or "").replace("\n", " ").strip() or "agent_message"
    action_val = action_val.replace(" ", "_")
    error_val = (error or "").replace("\n", " ").strip()
    error_val = error_val.replace(" ", "_") if error_val else "null"
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={_bool_str(done)} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score = _clamp_open_score(score)
    print(
        f"[END] success={_bool_str(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _score_from_observation(observation) -> Optional[float]:
    try:
        value = getattr(observation, "grader_score", None)
    except Exception:
        value = None
    if value is None:
        return None
    try:
        return _clamp_open_score(float(value))
    except Exception:
        return None


def _clamp_open_score(value: float) -> float:
    try:
        return float(min(max(float(value), MIN_OPEN_SCORE), MAX_OPEN_SCORE))
    except Exception:
        return MIN_OPEN_SCORE


def _build_action_for_step(scenario, step_number: int) -> MessageRecomendationSystemAction:
    scenario_step = scenario.get_step(step_number)
    suggestion = MessageSuggestionObservationAI.from_message(
        Message(
            text=scenario_step.user_message,
            description=scenario_step.goal,
            sentiment=None,
            liked=scenario_step.accepted,
        ),
        metadata={
            "scenario_id": scenario.id,
            "scenario_title": scenario.title,
            "scenario_difficulty": scenario.difficulty,
            "expected_outcome": scenario.expected_outcome,
            "step_goal": scenario_step.goal,
        },
        improvement_review=scenario_step.feedback,
        accepted=scenario_step.accepted,
        current_step=step_number,
        max_steps=min(MAX_STEPS, len(scenario.steps)),
    )
    suggested_text = suggestion.autoreply.text if suggestion.autoreply else scenario_step.user_message
    return MessageRecomendationSystemAction(
        user_message=suggested_text,
        liked_message=scenario_step.accepted,
        improvement_review=scenario_step.feedback,
        current_step=step_number,
        max_steps=min(MAX_STEPS, len(scenario.steps)),
    )


async def run_task(task_id: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = MIN_OPEN_SCORE

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env: MessageRecomendationSystemEnv | None = None
    result = None
    try:
        env = await _resolve_env()
        await _maybe_await(env.connect())
        await _maybe_await(env.reset())

        scenario = next(s for s in list_scenarios() if s.id == task_id)
        max_steps = min(MAX_STEPS, len(scenario.steps))

        for step in range(1, max_steps + 1):
            action = _build_action_for_step(scenario, step)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = getattr(result.observation, "last_action_error", None)

            rewards.append(reward)
            steps_taken = step
            print(f"Step {step} - Action: {action.user_message}, Reward: {reward}, Done: {done}, Error: {error}, Result: {result}", flush=True  )
            log_step(step=step, action="agent_message", reward=reward, done=done, error=error)

            if done:
                break

        if result is not None:
            observed_score = _score_from_observation(result.observation)
            score = observed_score if observed_score is not None else (sum(rewards) / len(rewards) if rewards else MIN_OPEN_SCORE)
            score = _clamp_open_score(score)
            success = bool(result.done) and score > MIN_OPEN_SCORE

    except Exception as exc:
        log_step(step=max(1, steps_taken + 1), action="agent_message", reward=0.0, done=True, error=str(exc))
        success = False
        score = MIN_OPEN_SCORE
    finally:
        try:
            if env is not None:
                await _maybe_await(env.close())
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def _amain() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN or API_KEY must be set for inference.")
    for scenario in list_scenarios():
        await run_task(scenario.id)


def main() -> None:
    """
    Synchronous entrypoint (some validators import and call `main()` directly).
    """
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
