"""The assumptions register: every hand-set number, its basis, a plausible range, and draws.

* **Addresses** name a document and a ``/``-separated path: ``goals:kappa``,
  ``powers:powers/USA/initial/economy``, ``dynamics:relations_step``,
  ``event:<id>/<path>``; ``code:<module>.<NAME>`` and ``engine:<field>`` name constants
  left in Python.
* **The register file** (``data/assumptions.yaml``) has ordered class rules, which cover
  every number (first match wins), and entries, which override single numbers. Each number
  gets a basis, a kind (which fixes the scale its range lives on), a group, and a range or
  a reason it is fixed.
* **A draw** sets each live number to a uniform point of its range on its natural scale
  (log for magnitudes, logit for probabilities and shares), from a counter-based uniform of
  the seed and the address, so adding a number never moves another's draw. Linked sets keep
  their total.
"""

from __future__ import annotations

from .doc import affects, render_markdown
from .plan import Plan, Sampling, base_inventory, check, code_value, numbers_of, plan, plan_for
from .register import (
    BASES,
    FIXED,
    GLOBAL_FILES,
    KINDS,
    LOG_KINDS,
    LOGIT_KINDS,
    REGISTER_FILE,
    Assumption,
    ClassRule,
    Entry,
    Number,
    RangeRule,
    Register,
    RegisterError,
    _glob,
    inventory,
    load_register,
    parse_register,
    pattern_of,
)

__all__ = [
    "BASES",
    "FIXED",
    "GLOBAL_FILES",
    "KINDS",
    "LOGIT_KINDS",
    "LOG_KINDS",
    "REGISTER_FILE",
    "Assumption",
    "ClassRule",
    "Entry",
    "Number",
    "Plan",
    "RangeRule",
    "Register",
    "RegisterError",
    "Sampling",
    "_glob",
    "affects",
    "base_inventory",
    "check",
    "code_value",
    "inventory",
    "load_register",
    "numbers_of",
    "parse_register",
    "pattern_of",
    "plan",
    "plan_for",
    "render_markdown",
]
