---
title: Message Recomendation System Environment Server
emoji: "💬"
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Message Recomendation System Environment

This project is an iterative AI reply recommendation environment.

It is not a simple echo server. The intended story is:

1. A real incoming user message is given to the system.
2. The AI analyzes the message and infers the sentiment on its own.
3. The AI generates a recommended reply.
4. The user either accepts that recommendation or rejects it.
5. If the user rejects it, they provide improvement feedback.
6. Every rejection counts as a step.
7. On the next step, the AI must generate a better reply using that feedback.
8. The flow ends when the reply is accepted or the maximum step budget is reached.

## Core Behavior

The environment is designed to evaluate whether an AI system can:

- infer sentiment from plain user text without being given a sentiment label
- generate safe and useful suggested replies
- improve those replies after rejection
- use feedback to refine tone, empathy, clarity, and usefulness
- converge to an accepted reply within a limited number of steps

## Data Model

### Action

`MessageRecomendationSystemAction`

- `user_message`: the original incoming message that needs a suggested reply
- `liked_message`: whether the current AI suggestion was accepted
- `improvement_review`: feedback for the next suggestion if the current one was rejected
- `current_step`: current refinement step
- `max_steps`: maximum number of refinement attempts allowed

### Observation

`MessageRecomendationSystemObservation`

- `message`: the original message with AI-inferred sentiment
- `autoreply`: the AI-recommended reply
- `confidence_score`: model confidence in the recommendation
- `accepted`: whether the recommendation is accepted
- `current_step`: current refinement step
- `max_steps`: maximum number of refinement steps allowed
- `requires_human_review`: whether a human should review the case
- `grader_score`: deterministic quality score used for evaluation
- `reward`: reward for the current step
- `done`: whether the episode is complete

## Prompt Contract

The model must:

- infer sentiment from message text
- never expect the user to provide sentiment
- improve the reply when the previous recommendation is rejected
- avoid repeating rejected responses without meaningful changes
- return structured JSON matching the observation shape

## Scenario Benchmarks

The benchmark scenarios live in [tasks.py](/abs/path/c:/Users/vrtandale/Desktop/hackthon/message_recomendation_system/tasks.py) and use models from [task_classes.py](/abs/path/c:/Users/vrtandale/Desktop/hackthon/message_recomendation_system/task_classes.py).

Available scenario ids:

- `easy`
- `medium`
- `hard`
- `easy1`
- `medium1`
- `hard1`

Each scenario contains at least 6 steps. Early steps usually represent rejected drafts with improvement feedback. The final step represents an accepted reply.

## Example Flow

```python
from message_recomendation_system import MessageRecomendationSystemAction, MessageRecomendationSystemEnv

with MessageRecomendationSystemEnv(base_url="http://localhost:8000") as env:
    env.reset()

    result = env.step(
        MessageRecomendationSystemAction(
            user_message="I need help with my delayed order.",
            liked_message=False,
            improvement_review="Make the reply more empathetic and ask for my order number.",
            current_step=1,
            max_steps=6,
        )
    )

    print(result.observation.message.sentiment)
    print(result.observation.autoreply.text)
    print(result.observation.confidence_score)
```

## Project Structure

```text
message_recomendation_system/
|-- __init__.py
|-- client.py
|-- grader.py
|-- models.py
|-- promt.py
|-- task_classes.py
|-- tasks.py
|-- README.md
`-- server/
    |-- app.py
    |-- message_recomendation_system_environment.py
    `-- models/
        `-- messaging/
            `-- message_model.py
```
