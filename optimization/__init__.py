"""Surrogate-model-based geometry optimization."""

from optimization.alternative_optimizer import run_dual_annealing, run_multistart_nelder_mead
from optimization.differential_evolution import run_differential_evolution

ALGORITHMS = {
    "differential_evolution": ("Differential Evolution (SciPy)", run_differential_evolution),
    "multistart_nelder_mead": ("LHS multi-start Nelder-Mead (SciPy)", run_multistart_nelder_mead),
    "dual_annealing": ("Dual Annealing / generalized simulated annealing (SciPy)", run_dual_annealing),
}

__all__ = ["ALGORITHMS", "run_differential_evolution", "run_multistart_nelder_mead", "run_dual_annealing"]
