# Tractus — research archive orientation

Development stopped in October 2026. README.md and the final assessment
in docs/archive-2026-10-04/ supersede historical dispatch instructions.

The repository retains three packages: tractus_controller,
tractus_engine_advanced (TEA), and tractus_shared. Install their pinned
dependencies in a project virtual environment. GPU scripts require a
compatible CUDA-enabled JAX installation; lightweight checks can use CPU.

TEA precision defaults to fp32. Set TEA_PRECISION=fp64 before importing
JAX for isolated parity checks. Do not change the global precision policy
from other modules. Keep physical invariants and consent-bearing manifests.

Outputs, local datasets, machine inventories, credentials, and operational
session records must remain outside version control. Use relative paths or
explicit environment configuration. Run scripts/ci/lint_privacy.py before
committing. Historical workplans describe abandoned experiments.
