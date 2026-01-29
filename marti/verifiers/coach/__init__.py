"""
External Coach Module

Coach evaluates agent actions through external LLM API and provides process rewards.

Main classes:
- CoachFeedback: Evaluation result data structure
- BaseCoach: Abstract base class for Coach
- SimpleScalarCoach: Simple scalar reward coach
- create_coach: Factory function
"""

from marti.verifiers.coach.external_coach import (
    CoachFeedback,
    BaseCoach,
    SimpleScalarCoach,
    create_coach
)

__all__ = [
    "CoachFeedback",
    "BaseCoach",
    "SimpleScalarCoach",
    "create_coach"
]
