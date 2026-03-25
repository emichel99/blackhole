"""
Monte Carlo simulation for estimating the probability that a news event will continue.

Model:
- An event has a base "decay rate" that depends on its category.
- Each simulated day, a random shock (drawn from a normal distribution)
  either prolongs or accelerates resolution.
- We run N simulations and count how many are still "active" after T days.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

# Base half-life (in days) by event category.
# Smaller half-life → event resolves faster on average.
CATEGORY_HALF_LIFE: Dict[str, float] = {
    "geopolitical": 30.0,
    "natural_disaster": 14.0,
    "economic": 21.0,
    "corporate": 7.0,
    "health": 60.0,
    "technology": 5.0,
    "political": 21.0,
    "social": 10.0,
    "default": 14.0,
}

# Volatility (std-dev of daily random shock) by category.
CATEGORY_VOLATILITY: Dict[str, float] = {
    "geopolitical": 0.12,
    "natural_disaster": 0.18,
    "economic": 0.10,
    "corporate": 0.15,
    "health": 0.08,
    "technology": 0.20,
    "political": 0.14,
    "social": 0.16,
    "default": 0.13,
}


@dataclass
class MonteCarloResult:
    probability_7d: float    # P(still active after 7 days)
    probability_14d: float   # P(still active after 14 days)
    probability_30d: float   # P(still active after 30 days)
    expected_duration_days: float
    simulations: int
    category: str
    severity: float
    daily_probs: list        # P(active) for days 1..30


def run_simulation(
    category: str = "default",
    severity: float = 5.0,      # 1–10 scale
    days_elapsed: int = 0,       # how many days the event has already been running
    simulations: int = 10_000,
    horizon_days: int = 30,
) -> MonteCarloResult:
    """
    Run a Monte Carlo simulation for event continuation probability.

    Parameters
    ----------
    category : str
        Event category (geopolitical, economic, etc.)
    severity : float
        Severity score from 1 (minor) to 10 (extreme)
    days_elapsed : int
        Days since the event started (0 = just occurred)
    simulations : int
        Number of Monte Carlo paths
    horizon_days : int
        Number of future days to simulate
    """
    np.random.seed(42)

    half_life = CATEGORY_HALF_LIFE.get(category, CATEGORY_HALF_LIFE["default"])
    volatility = CATEGORY_VOLATILITY.get(category, CATEGORY_VOLATILITY["default"])

    # Severity multiplies the half-life (higher severity → lasts longer)
    severity_factor = 0.5 + (severity / 10.0) * 1.5   # range [0.5, 2.0]
    adjusted_half_life = half_life * severity_factor

    # Daily decay probability from half-life
    base_daily_decay = 1.0 - np.exp(-np.log(2) / adjusted_half_life)

    # Condition on already having survived `days_elapsed`
    survival_given_elapsed = np.exp(-base_daily_decay * days_elapsed)

    # Simulate: each path tracks a "resilience" variable.
    # Each day, draw a shock; if cumulative decay crosses threshold, event ends.
    rng = np.random.default_rng(42)
    active = np.ones(simulations, dtype=bool)
    daily_probs = []

    for day in range(1, horizon_days + 1):
        # Random shock modulates that day's decay rate
        shocks = rng.normal(0.0, volatility, size=simulations)
        daily_decay = np.clip(base_daily_decay + shocks, 0.001, 0.999)

        # Each active simulation ends today with probability `daily_decay`
        ends_today = rng.random(size=simulations) < daily_decay
        active &= ~ends_today

        daily_probs.append(float(active.mean()))

    # Expected total duration (days from start)
    # We add days_elapsed because the simulation is forward-looking.
    # Approximate using the daily_probs survival curve.
    survival = np.array(daily_probs)
    expected_future = float(np.sum(survival))  # integral of survival curve ≈ E[remaining life]
    expected_total = days_elapsed + expected_future

    return MonteCarloResult(
        probability_7d=daily_probs[6] if len(daily_probs) >= 7 else daily_probs[-1],
        probability_14d=daily_probs[13] if len(daily_probs) >= 14 else daily_probs[-1],
        probability_30d=daily_probs[29] if len(daily_probs) >= 30 else daily_probs[-1],
        expected_duration_days=expected_total,
        simulations=simulations,
        category=category,
        severity=severity,
        daily_probs=daily_probs,
    )


def result_to_dict(r: MonteCarloResult) -> dict:
    return {
        "probability_7d": round(r.probability_7d * 100, 1),
        "probability_14d": round(r.probability_14d * 100, 1),
        "probability_30d": round(r.probability_30d * 100, 1),
        "expected_duration_days": round(r.expected_duration_days, 1),
        "simulations": r.simulations,
        "category": r.category,
        "severity": r.severity,
        "daily_probs": [round(p * 100, 1) for p in r.daily_probs],
    }
