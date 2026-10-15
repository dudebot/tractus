"""Parser for the tract-sequence text dumps consumed by TEA fixture
parity tests (ARCH §10).

Format::

    # 5 comment lines
    Geometric glottis
    <num_states>
    <11 glottis params separated by spaces>
    <19 tract params separated by spaces>
    ... (one glottis line + one tract line per state) ...

Frame step is 110 audio samples / 44100 Hz ≈ 2.494 ms (≈ 401 Hz).

Returns a numpy array of shape ``(T, 30)`` where the columns are 11
glottis params followed by 19 tract params.

Lives in ``tractus_shared`` and does not import from ``tractus_controller``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Hard-coded for the default speaker (11 glottis + 19 tract = 30 params).
NUM_GLOTTIS_PARAMS = 11
NUM_TRACT_PARAMS = 19
NUM_PARAMS = NUM_GLOTTIS_PARAMS + NUM_TRACT_PARAMS  # 30
FRAME_STEP_SAMPLES = 110
NATIVE_SAMPLING_RATE_HZ = 44100
NATIVE_FRAME_RATE_HZ = NATIVE_SAMPLING_RATE_HZ / FRAME_STEP_SAMPLES  # ≈ 400.91


@dataclass
class TractSequence:
    """One parsed tract-sequence dump.

    `data` columns: [0:11] glottis, [11:30] tract.
    `glottis_model_name` is whatever was written on the model line
    (typically "Geometric glottis").
    """

    glottis_model_name: str
    data: np.ndarray  # shape (T, 30)
    frame_rate_hz: float = NATIVE_FRAME_RATE_HZ

    @property
    def num_frames(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.frame_rate_hz


def parse_tract_sequence_text(text: str) -> TractSequence:
    lines = text.splitlines()
    # Drop comment lines.
    body = [ln for ln in lines if not ln.startswith("#")]
    if len(body) < 3:
        raise ValueError("tract sequence file too short to be valid")
    glottis_model = body[0].strip()
    try:
        num_states = int(body[1].strip())
    except ValueError as e:
        raise ValueError(f"expected integer state count on line 2; got {body[1]!r}") from e

    expected = 2 * num_states
    state_lines = body[2:]
    if len(state_lines) < expected:
        raise ValueError(
            f"truncated tract sequence: header says {num_states} states "
            f"({expected} lines) but body has {len(state_lines)} lines"
        )
    state_lines = state_lines[:expected]

    glottis = np.empty((num_states, NUM_GLOTTIS_PARAMS), dtype=np.float32)
    tract = np.empty((num_states, NUM_TRACT_PARAMS), dtype=np.float32)
    for i in range(num_states):
        g_vals = state_lines[2 * i].split()
        t_vals = state_lines[2 * i + 1].split()
        if len(g_vals) != NUM_GLOTTIS_PARAMS:
            raise ValueError(
                f"state {i}: expected {NUM_GLOTTIS_PARAMS} glottis params, "
                f"got {len(g_vals)}: {g_vals}"
            )
        if len(t_vals) != NUM_TRACT_PARAMS:
            raise ValueError(
                f"state {i}: expected {NUM_TRACT_PARAMS} tract params, "
                f"got {len(t_vals)}: {t_vals}"
            )
        glottis[i] = [float(x) for x in g_vals]
        tract[i] = [float(x) for x in t_vals]

    return TractSequence(
        glottis_model_name=glottis_model,
        data=np.concatenate([glottis, tract], axis=1),
    )


def read_tract_sequence(path: str | os.PathLike) -> TractSequence:
    return parse_tract_sequence_text(Path(path).read_text())


def resample_to_knot_rate(
    seq: TractSequence, target_hz: float
) -> tuple[np.ndarray, float]:
    """Down-/up-sample the per-frame array to a target knot rate.

    Uses simple linear interpolation. Returns ``(resampled, actual_hz)``
    where ``actual_hz`` is the achieved rate (may differ slightly from
    `target_hz` due to integer frame-count rounding).
    """
    if target_hz <= 0:
        raise ValueError("target_hz must be > 0")
    n_in = seq.num_frames
    n_out = max(1, int(round(seq.duration_sec * target_hz)))
    if n_out == n_in:
        return seq.data.copy(), seq.frame_rate_hz
    # Source times (uniform), target times (uniform).
    t_in = np.linspace(0, 1, n_in, dtype=np.float64)
    t_out = np.linspace(0, 1, n_out, dtype=np.float64)
    out = np.empty((n_out, seq.data.shape[1]), dtype=np.float32)
    for c in range(seq.data.shape[1]):
        out[:, c] = np.interp(t_out, t_in, seq.data[:, c]).astype(np.float32)
    actual_hz = n_out / seq.duration_sec
    return out, actual_hz
