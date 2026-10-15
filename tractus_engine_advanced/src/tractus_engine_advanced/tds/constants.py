"""Physical + numerical constants used across the TDS port.

Standard transmission-line vocal-tract acoustic model constants.
and `TdsModel.cpp`. Units are CGS throughout (cm, g, s, dPa).
"""

from __future__ import annotations

# Acoustic medium (Constants.h:29-37).
STATIC_PRESSURE_CGS = 1.013e6   # dPa = µbar
AMBIENT_DENSITY_CGS = 1.14e-3   # g/cm^3
ADIABATIC_CONSTANT = 1.4
SOUND_VELOCITY_CGS = 3.5e4      # cm/s
AIR_VISCOSITY_CGS = 1.86e-4     # dyne-s/cm^2
SPECIFIC_HEAT_CGS = 0.24        # cal/g-K
HEAT_CONDUCTION_CGS = 0.055e-3  # cal/cm-s-K

# Sampling (Constants.h:43).
# TEA renders at 44.1 kHz — this is a hard engine constraint, not a project target.
# The project-wide output rate (22.05 kHz) is achieved by 2× decimation at the
# boundary *outside* the engine. Consumers wanting other rates must decimate after
# synthesis; re-deriving TIMESTEP_S would require rebuilding all MNA matrices.
# See `docs/incorrect_assumptions.md` A1 for the full disambiguation.
SAMPLING_RATE = 44100
TIMESTEP_S = 1.0 / SAMPLING_RATE

# THETA-scheme (TdsModel.cpp:35-36). 0.505 biases slightly toward implicit so
# resonances above 4 kHz get realistic damping; 0.5 would resonate too sharply.
THETA = 0.505
THETA1 = 1.0 - THETA

# Lower clamp for tube section area (TdsModel.cpp:39). 0.01 mm^2 = 1e-4 cm^2.
MIN_AREA_CM2 = 0.01e-2

# Elliptical cross-section threshold (prepareTimeStep:967). When the implied
# circular radius dips below this, the tube is treated as elliptical with the
# major half-axis pinned here — strongly increases viscous resistance for
# near-closure sections (matches a rectangular-slit limit).
MIN_RADIUS_CM = 0.6

# Stiffness-boost threshold for closed-section-stable wall vibration
# (TdsModel::setTube:746). For section areas below this, wallStiffness is
# multiplied by `CRITICAL_AREA_CM2 / area` (capped at 100x).
CRITICAL_AREA_CM2 = 1.0
MAX_STIFFNESS_BOOST = 100.0

# 1-pole LP for the per-branch noise-magnitude tracker (TdsModel.cpp:42).
# noiseMagnitude_new = (1-α)*flow + α*noiseMagnitude_old where α = exp(-2π*f_c*dt).
NOISE_CUTOFF_FREQ_HZ = 500.0
