"""Compatibility helpers for SciPy's ``seed`` -> ``rng`` keyword transition.

SciPy >= 1.15 introduced an ``rng`` keyword and is deprecating ``seed`` on
several functions. Passing whichever keyword the installed version supports
keeps results reproducible across SciPy versions without deprecation warnings.
"""

from __future__ import annotations

import inspect


def seed_kwargs(func, seed: int) -> dict:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {"seed": seed}
    if "rng" in params:
        return {"rng": seed}
    return {"seed": seed}
