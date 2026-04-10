from __future__ import annotations

from typing import List

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class ScenarioStep(BaseModel):
    step_number: int = Field(..., description="1-based position of the scenario step")
    user_message: str = Field(..., description="Incoming user message that needs a suggested reply")
    accepted: bool = Field(..., description="Whether the user accepts the AI suggestion at this step")
    feedback: str = Field(..., description="Improvement feedback if the user rejects the suggestion")
    goal: str = Field(..., description="Expected improvement target for this step")


class ScenarioTaskAction(Action):
    scenario_id: str = Field(..., description="Unique scenario identifier")
    step_number: int = Field(..., ge=1, description="Current step number in the scenario")
    user_message: str = Field(..., description="Incoming user message")
    liked_message: bool = Field(..., description="Whether the current AI suggestion was accepted")
    improvement_review: str = Field(..., description="Feedback for the next suggestion if rejected")


class ScenarioTaskObservation(Observation):
    scenario_id: str = Field(..., description="Unique scenario identifier")
    title: str = Field(..., description="Scenario title")
    difficulty: str = Field(..., description="Difficulty label")
    current_step: ScenarioStep = Field(..., description="Current benchmark step")
    remaining_steps: int = Field(..., ge=0, description="Number of steps remaining")
    success_hint: str = Field(..., description="Short description of the desired end state")
    accepted: bool = Field(..., description="Whether the recommendation is accepted at this step")


class MessageScenario(BaseModel):
    id: str = Field(..., description="Scenario id such as easy or hard1")
    title: str = Field(..., description="Human-readable scenario title")
    difficulty: str = Field(..., description="Difficulty label")
    description: str = Field(..., description="Scenario story summary")
    steps: List[ScenarioStep] = Field(..., min_length=6, description="Ordered scenario steps")
    expected_outcome: str = Field(..., description="Successful end state for the scenario")

    def get_step(self, step_number: int) -> ScenarioStep:
        if step_number < 1 or step_number > len(self.steps):
            raise IndexError(f"step_number {step_number} is outside the scenario range")
        return self.steps[step_number - 1]

    def to_observation(self, step_number: int) -> ScenarioTaskObservation:
        current_step = self.get_step(step_number)
        remaining_steps = max(len(self.steps) - step_number, 0)
        return ScenarioTaskObservation(
            scenario_id=self.id,
            title=self.title,
            difficulty=self.difficulty,
            current_step=current_step,
            remaining_steps=remaining_steps,
            success_hint=self.expected_outcome,
            accepted=current_step.accepted,
            done=current_step.accepted or step_number >= len(self.steps),
            reward=1.0 if current_step.accepted else 0.0,
            metadata={
                "description": self.description,
                "total_steps": len(self.steps),
            },
        )


def build_step(
    step_number: int,
    user_message: str,
    accepted: bool,
    feedback: str,
    goal: str,
) -> ScenarioStep:
    return ScenarioStep(
        step_number=step_number,
        user_message=user_message,
        accepted=accepted,
        feedback=feedback,
        goal=goal,
    )
