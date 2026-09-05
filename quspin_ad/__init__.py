"""Automatic-differentiation rules for selected QuSpin callables.

This package is a sidecar: QuSpin remains the source of every primal value,
while this module only supplies explicit ChainRules JVP/VJP rules.  Importing
``quspin_ad`` registers rules for the sidecar callables exported here.  When a
QuSpin installation is available, the corresponding upstream callables are
also registered (``register_upstream_rules`` is idempotent).
"""

from .rules import (
    ED_state_vs_time,
    KL_div,
    anti_commutator,
    coherent_state,
    commutator,
    lin_comb_Q_T,
    project_op,
    register_upstream_rules,
    differentiable_drive,
    dynamic_trajectory,
    fixed_grid_trajectory,
    floquet_eigensystem,
)
from .rules import ad as _ad
from . import _chainrules as _extended_ad

ZERO = _ad.ZERO
grad = _ad.grad
jvp = _ad.jvp
value_and_grad = _ad.value_and_grad
vjp = _ad.vjp
hvp = getattr(_ad, "hvp", _extended_ad.hvp)
nested_jvp = getattr(_ad, "nested_jvp", _extended_ad.nested_jvp)
value_grad_and_hvp = getattr(_ad, "value_grad_and_hvp", _extended_ad.value_grad_and_hvp)

__all__ = [
    "ZERO",
    "ED_state_vs_time",
    "KL_div",
    "anti_commutator",
    "coherent_state",
    "commutator",
    "grad",
    "jvp",
    "lin_comb_Q_T",
    "project_op",
    "register_upstream_rules",
    "value_and_grad",
    "vjp",
    "hvp",
    "nested_jvp",
    "value_grad_and_hvp",
    "differentiable_drive",
    "dynamic_trajectory",
    "fixed_grid_trajectory",
    "floquet_eigensystem",
]
