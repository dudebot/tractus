"""Inverse-fitting natural audio to articulator trajectories.

Back-end (post-dependency-removal): cached (T, 30) trajectory →
adapters.r1_inference (NN tract→tube) → tractus_engine_advanced
(JAX TDS) → audio → acoustic loss → Adam.

Front-end (text → trajectory) was removed with the C++ backend; see
plan/90-dependency-removal.md for what's needed to re-enable it.
"""
