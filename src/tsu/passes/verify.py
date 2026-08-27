"""Three-layer equivalence. Never fabricates: absent a reference, the field reads
"unavailable" with a reason (spec section 10)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Verification:
    energy_tv: float | None
    energy_note: str
    task_validity: float | None
    execution_tv: float | None
    execution_note: str
    execution_noise_floor: float | None
    cross_check_tv: float | None
    cross_check_note: str

    def to_dict(self):
        def f(v, note):
            return v if v is not None else note
        return {
            "energy_tv": f(self.energy_tv, self.energy_note),
            "task_validity": f(self.task_validity, "unavailable"),
            "execution_tv": f(self.execution_tv, self.execution_note),
            "execution_noise_floor": self.execution_noise_floor,
            "cross_check_tv": f(self.cross_check_tv, self.cross_check_note),
        }


def tv_noise_floor(probs: np.ndarray, n: int) -> float:
    """A TV number without its finite-sample floor beside it is uninterpretable."""
    return float(0.5 * np.sum(np.sqrt(2.0 * probs * (1 - probs) / (math.pi * n))))
