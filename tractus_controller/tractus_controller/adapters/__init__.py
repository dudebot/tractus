"""Pure-JAX adapters that replace renderer C++ in the gradient hot path.

R1: ``the legacy C++ tract→tube map`` emulator (M1).
"""

from .r1_nn import R1Config, R1Norm, init_params, forward  # noqa: F401
