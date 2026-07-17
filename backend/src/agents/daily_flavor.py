"""
Daily flavor — random theme and emotion pickers for day_planner.

Each day an agent gets a random theme (what they focus on) and emotion
(their mood), injected into the planner prompts so the schedule doesn't
feel identical every day.
"""

from __future__ import annotations

import random

THEMES = [
    "Sports",
    "Academics",
    "Working",
    "Socialising",
    "Exploring",
    "Resting",
    "Entertainment",
    "Fitness",
    "Shopping",
    "Creative",
]

EMOTIONS = [
    "Happy",
    "Sad",
    "Melancholic",
    "Angry",
    "Peaceful",
    "Affectionate",
    "Excited",
    "Motivated",
    "Lonely",
    "Relaxed",
]


def pick_theme() -> str:
    return random.choice(THEMES)


def pick_emotion() -> str:
    return random.choice(EMOTIONS)
