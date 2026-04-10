from __future__ import annotations

from task_classes import MessageScenario, build_step


SCENARIOS: dict[str, MessageScenario] = {
    "easy": MessageScenario(
        id="easy",
        title="Order Help Refinement",
        difficulty="easy",
        description="A user needs help with an order. The AI suggestion is rejected several times until it becomes warm, clear, and actionable.",
        steps=[
            build_step(1, "Hi, I need help with my order.", False, "Make the reply warmer.", "Open politely."),
            build_step(2, "Hi, I need help with my order.", False, "Mention the order issue directly.", "Show understanding."),
            build_step(3, "Hi, I need help with my order.", False, "Add empathy.", "Improve tone."),
            build_step(4, "Hi, I need help with my order.", False, "Ask for the order number.", "Become actionable."),
            build_step(5, "Hi, I need help with my order.", False, "Keep it concise.", "Balance tone and brevity."),
            build_step(6, "Hi, I need help with my order.", True, "", "Final reply should be warm and ask for the order number."),
        ],
        expected_outcome="The user accepts a helpful order-support reply.",
    ),
    "medium": MessageScenario(
        id="medium",
        title="Refund Response Improvement",
        difficulty="medium",
        description="A refund request is refined across multiple steps until the user accepts a calm and accurate response.",
        steps=[
            build_step(1, "I want a refund for this charge.", False, "Acknowledge the refund request first.", "Start calmly."),
            build_step(2, "I want a refund for this charge.", False, "Sound more empathetic.", "Improve emotional tone."),
            build_step(3, "I canceled but still got billed.", False, "Do not promise a refund yet.", "Stay accurate."),
            build_step(4, "I canceled but still got billed.", False, "Ask for the billing email and charge date.", "Request useful details."),
            build_step(5, "I canceled but still got billed.", False, "Keep it professional and short.", "Avoid escalation."),
            build_step(6, "I canceled but still got billed.", True, "", "Final reply should be calm and verification-oriented."),
        ],
        expected_outcome="The user accepts a refund-verification reply with the right balance of empathy and accuracy.",
    ),
    "hard": MessageScenario(
        id="hard",
        title="Angry User De-escalation",
        difficulty="hard",
        description="An angry customer keeps rejecting drafts until the AI fully de-escalates and offers a controlled next step.",
        steps=[
            build_step(1, "This service is terrible and nobody helped me.", False, "The reply feels generic.", "Recognize frustration."),
            build_step(2, "This service is terrible and nobody helped me.", False, "Do not sound defensive.", "Lower tension."),
            build_step(3, "I have already explained this three times.", False, "Acknowledge the repeated effort.", "Validate frustration."),
            build_step(4, "I have already explained this three times.", False, "Offer a path forward, not just an apology.", "Become practical."),
            build_step(5, "Fix it now or cancel everything.", False, "Give two clear options.", "Restore control to the user."),
            build_step(6, "Fix it now or cancel everything.", True, "", "Final reply should offer resolution or cancellation calmly."),
        ],
        expected_outcome="The user accepts a clear de-escalation reply with concrete options.",
    ),
    "easy1": MessageScenario(
        id="easy1",
        title="Meeting Confirmation Draft",
        difficulty="easy",
        description="The user wants a polished meeting confirmation. The AI improves the message until it is concise and professional.",
        steps=[
            build_step(1, "Can you confirm tomorrow's meeting at 11 AM with the design team?", False, "Make it more professional.", "Improve polish."),
            build_step(2, "Can you confirm tomorrow's meeting at 11 AM with the design team?", False, "Make it shorter.", "Reduce filler."),
            build_step(3, "Can you confirm tomorrow's meeting at 11 AM with the design team?", False, "Mention the team name clearly.", "Include key details."),
            build_step(4, "Can you confirm tomorrow's meeting at 11 AM with the design team?", False, "Sound more natural.", "Improve phrasing."),
            build_step(5, "Can you confirm tomorrow's meeting at 11 AM with the design team?", False, "End on a positive note.", "Add warmth."),
            build_step(6, "Can you confirm tomorrow's meeting at 11 AM with the design team?", True, "", "Final reply should be concise and ready to send."),
        ],
        expected_outcome="The user accepts a professional meeting confirmation.",
    ),
    "medium1": MessageScenario(
        id="medium1",
        title="Feature Request Reply",
        difficulty="medium",
        description="The user suggests a feature and rejects early drafts until the AI produces an appreciative, honest, non-committal reply.",
        steps=[
            build_step(1, "It would be great if your app had dark mode.", False, "Thank the user for the suggestion.", "Show appreciation."),
            build_step(2, "It would be great if your app had dark mode.", False, "Mention the user's use case.", "Reflect the reason."),
            build_step(3, "I use the app at night and it feels too bright.", False, "Do not promise a release timeline.", "Stay realistic."),
            build_step(4, "I use the app at night and it feels too bright.", False, "Say you can share the feedback with the team.", "Offer a next step."),
            build_step(5, "Will your team add it soon?", False, "Be honest that there is no confirmed timeline.", "Set expectations."),
            build_step(6, "Will your team add it soon?", True, "", "Final reply should appreciate the idea without overpromising."),
        ],
        expected_outcome="The user accepts a thoughtful feature-request response that avoids unsupported promises.",
    ),
    "hard1": MessageScenario(
        id="hard1",
        title="Sensitive Application Support",
        difficulty="hard",
        description="A user facing a family emergency needs a careful support reply. The AI must refine the recommendation until it is empathetic and actionable.",
        steps=[
            build_step(1, "I missed the deadline because of a family emergency.", False, "Start with empathy.", "Acknowledge the situation."),
            build_step(2, "I missed the deadline because of a family emergency.", False, "Sound more human and less formal.", "Improve tone."),
            build_step(3, "Can anything still be done for my application?", False, "Offer a path forward.", "Be constructive."),
            build_step(4, "Can anything still be done for my application?", False, "Do not guarantee an exception.", "Stay accurate."),
            build_step(5, "I do not know what information to send.", False, "Explain exactly what details to provide.", "Reduce ambiguity."),
            build_step(6, "I do not know what information to send.", True, "", "Final reply should be empathetic, careful, and actionable."),
        ],
        expected_outcome="The user accepts a careful support reply that explains the next step clearly.",
    ),
}


def get_scenario(scenario_id: str) -> MessageScenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario_id '{scenario_id}'. Available: {available}") from exc


def list_scenarios() -> list[MessageScenario]:
    return [SCENARIOS[key] for key in sorted(SCENARIOS)]
