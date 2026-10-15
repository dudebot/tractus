"""Static TDS topology — branch numbering, in/out wiring, MNA sparsity.

All tables are pure NumPy at module load time. They are speaker-independent
and shape-fixed: 93 tube sections, 97 branch currents (= 93 + 4 radiation).
Lift to `jnp.asarray(...)` at use site if you need JAX arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Section / branch counts (upstream Tube.h + TdsModel.h).
# ---------------------------------------------------------------------------

NUM_TRACHEA_SECTIONS = 23
NUM_GLOTTIS_SECTIONS = 2
NUM_PHARYNX_SECTIONS = 16
NUM_MOUTH_SECTIONS = 24
NUM_NOSE_SECTIONS = 19
NUM_SINUS_SECTIONS = 4
NUM_FOSSA_SECTIONS = 5
NUM_SECTIONS = (
    NUM_TRACHEA_SECTIONS + NUM_GLOTTIS_SECTIONS + NUM_PHARYNX_SECTIONS
    + NUM_MOUTH_SECTIONS + NUM_NOSE_SECTIONS + NUM_FOSSA_SECTIONS + NUM_SINUS_SECTIONS
)
assert NUM_SECTIONS == 93

NUM_RADIATION_BRANCHES = 4  # 2 mouth + 2 nostril (R, L pairs)
NUM_BRANCH_CURRENTS = NUM_SECTIONS + NUM_RADIATION_BRANCHES
assert NUM_BRANCH_CURRENTS == 97

# Section index ranges (FIRST_X is inclusive, LAST_X is inclusive).
FIRST_TRACHEA_SECTION = 0
LAST_TRACHEA_SECTION  = FIRST_TRACHEA_SECTION + NUM_TRACHEA_SECTIONS - 1   # 22
FIRST_GLOTTIS_SECTION = LAST_TRACHEA_SECTION + 1                            # 23
LOWER_GLOTTIS_SECTION = FIRST_GLOTTIS_SECTION                               # 23
UPPER_GLOTTIS_SECTION = FIRST_GLOTTIS_SECTION + 1                           # 24
FIRST_PHARYNX_SECTION = FIRST_GLOTTIS_SECTION + NUM_GLOTTIS_SECTIONS        # 25
LAST_PHARYNX_SECTION  = FIRST_PHARYNX_SECTION + NUM_PHARYNX_SECTIONS - 1    # 40
FIRST_MOUTH_SECTION   = LAST_PHARYNX_SECTION + 1                            # 41
LAST_MOUTH_SECTION    = FIRST_MOUTH_SECTION + NUM_MOUTH_SECTIONS - 1        # 64
FIRST_NOSE_SECTION    = LAST_MOUTH_SECTION + 1                              # 65
LAST_NOSE_SECTION     = FIRST_NOSE_SECTION + NUM_NOSE_SECTIONS - 1          # 83
FIRST_FOSSA_SECTION   = LAST_NOSE_SECTION + 1                               # 84
LAST_FOSSA_SECTION    = FIRST_FOSSA_SECTION + NUM_FOSSA_SECTIONS - 1        # 88
FIRST_SINUS_SECTION   = LAST_FOSSA_SECTION + 1                              # 89
LAST_SINUS_SECTION    = FIRST_SINUS_SECTION + NUM_SINUS_SECTIONS - 1        # 92

# Coupling offsets (upstream Tube.h + Tube.cpp).
FOSSA_COUPLING_SECTION = 3                               # offset within pharynx
SINUS_COUPLING_SECTION = (8, 9, 11, 12)                  # offsets within nose

# Cholesky envelope caps (upstream TdsModel.h).
MAX_ENVELOPE_ROW_LEN = 57
MAX_ENVELOPE_COL_LEN = 10


# ---------------------------------------------------------------------------
# Topology dataclass — frozen, NumPy-only, computed once.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TubeTopology:
    # Branch identity (length-97).
    source_section: np.ndarray   # int32, -1 = closed end / radiation source
    target_section: np.ndarray   # int32, -1 = free space (radiation)

    # Per-section wiring (length-93). -1 if absent.
    current_in: np.ndarray       # int32 (93,)
    current_out: np.ndarray      # int32 (93, 2)

    # Per-branch dispatch in calcMatrix.
    is_radiation: np.ndarray     # bool (97,) — target == -1
    is_branching: np.ndarray     # bool (97,) — source has 2 outflows AND target != -1
    is_simple: np.ndarray        # bool (97,) — everything else (incl. branch 0)
    branching_off: np.ndarray    # int32 (97,) — partner branch index for branching rows; -1 otherwise

    # Per-row LHS sparsity pattern (97x97 bool). Includes the diagonal.
    matrix_nonzero: np.ndarray   # bool (97, 97)

    # Symmetric-envelope (skyline) index tables. Padded with 0; use lengths to slice.
    envelope_row_idx: np.ndarray  # int32 (97, MAX_ENVELOPE_ROW_LEN), columns j < i in envelope
    envelope_row_len: np.ndarray  # int32 (97,)
    envelope_col_idx: np.ndarray  # int32 (97, MAX_ENVELOPE_COL_LEN), rows i > j in envelope
    envelope_col_len: np.ndarray  # int32 (97,)


def _build_branch_table() -> tuple[np.ndarray, np.ndarray]:
    """Mirror `TdsModel::initModel:108-152`: build (source, target) for each of 97 branches."""
    n_sec = NUM_SECTIONS
    n_br = NUM_BRANCH_CURRENTS
    src = np.full(n_br, -1, dtype=np.int32)
    tgt = np.full(n_br, -1, dtype=np.int32)

    # Default: branchCurrent[i] flows from section i-1 into section i.
    for i in range(n_sec):
        src[i] = i - 1
        tgt[i] = i

    # Source overrides for the bifurcations / boundary.
    src[FIRST_TRACHEA_SECTION] = -1                           # subglottal end is closed (pressure source injects)
    src[FIRST_NOSE_SECTION] = LAST_PHARYNX_SECTION            # velar coupling
    src[FIRST_FOSSA_SECTION] = FIRST_PHARYNX_SECTION + FOSSA_COUPLING_SECTION  # piriform coupling
    for i, off in enumerate(SINUS_COUPLING_SECTION):
        src[FIRST_SINUS_SECTION + i] = FIRST_NOSE_SECTION + off

    # Radiation branches: 2 from LAST_MOUTH (R, L), 2 from LAST_NOSE (R, L).
    src[NUM_SECTIONS + 0] = LAST_MOUTH_SECTION
    src[NUM_SECTIONS + 1] = LAST_MOUTH_SECTION
    src[NUM_SECTIONS + 2] = LAST_NOSE_SECTION
    src[NUM_SECTIONS + 3] = LAST_NOSE_SECTION
    # targets all stay -1 (free space)

    return src, tgt


def _build_section_wiring(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror `initModel:160-191`: for each section, which branch flows in / out."""
    current_in = np.full(NUM_SECTIONS, -1, dtype=np.int32)
    current_out = np.full((NUM_SECTIONS, 2), -1, dtype=np.int32)
    for i in range(NUM_BRANCH_CURRENTS):
        if src[i] != -1:
            ts = src[i]
            if current_out[ts, 0] == -1:
                current_out[ts, 0] = i
            else:
                current_out[ts, 1] = i
        if tgt[i] != -1:
            current_in[tgt[i]] = i
    return current_in, current_out


def _build_branch_masks(
    src: np.ndarray, tgt: np.ndarray, current_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror calcMatrix's branch-by-shape logic (`calcMatrix:1981-2213`)."""
    n_br = NUM_BRANCH_CURRENTS
    is_radiation = (tgt == -1)
    branching_off = np.full(n_br, -1, dtype=np.int32)
    is_branching = np.zeros(n_br, dtype=bool)
    for i in range(n_br):
        if is_radiation[i]:
            continue
        if src[i] == -1:
            continue  # branch 0 — sourceTs is NULL → simple
        s = src[i]
        out0, out1 = current_out[s]
        if out0 == i:
            other = out1
        else:
            other = out0
        if other != -1:
            is_branching[i] = True
            branching_off[i] = other
    is_simple = ~(is_radiation | is_branching)
    return is_radiation, is_branching, is_simple, branching_off


def _build_matrix_pattern(
    src: np.ndarray, tgt: np.ndarray, current_in: np.ndarray, current_out: np.ndarray
) -> np.ndarray:
    """Mirror `initModel:239-256`: which (i,j) entries can be non-zero post-calcMatrix."""
    n = NUM_BRANCH_CURRENTS
    mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        # Diagonal is always populated by calcMatrix (the row-i current term).
        mat[i, i] = True
        for sec in (src[i], tgt[i]):
            if sec == -1:
                continue
            ci = current_in[sec]
            co0, co1 = current_out[sec]
            for j in (ci, co0, co1):
                if j != -1:
                    mat[i, j] = True
    return mat


def _build_envelope_tables(matrix_nonzero: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror `initModel:262-313`: skyline index tables for symmetric Cholesky.

    For each row i, store the column indices `j < i` between the first non-zero
    and the diagonal (inclusive of fill-in zeros — `betweenNonZeros` at line 264).
    For each column j, store row indices `i > j` with non-zero entries.
    """
    n = NUM_BRANCH_CURRENTS
    row_idx = np.zeros((n, MAX_ENVELOPE_ROW_LEN), dtype=np.int32)
    row_len = np.zeros(n, dtype=np.int32)
    # First populate the envelope row-wise — also fills in interior zeros to mat.
    envelope_mat = matrix_nonzero.copy()
    for i in range(n):
        seen_nonzero = False
        for j in range(i):
            if envelope_mat[i, j] or seen_nonzero:
                seen_nonzero = True
                envelope_mat[i, j] = True  # fill-in
                k = row_len[i]
                if k >= MAX_ENVELOPE_ROW_LEN:
                    raise RuntimeError(
                        f"row {i} envelope overflow: needed > {MAX_ENVELOPE_ROW_LEN}"
                    )
                row_idx[i, k] = j
                row_len[i] = k + 1

    col_idx = np.zeros((n, MAX_ENVELOPE_COL_LEN), dtype=np.int32)
    col_len = np.zeros(n, dtype=np.int32)
    for j in range(n):
        for i in range(j + 1, n):
            if envelope_mat[i, j]:
                k = col_len[j]
                if k >= MAX_ENVELOPE_COL_LEN:
                    raise RuntimeError(
                        f"col {j} envelope overflow: needed > {MAX_ENVELOPE_COL_LEN}"
                    )
                col_idx[j, k] = i
                col_len[j] = k + 1
    return row_idx, row_len, col_idx, col_len


def build_topology() -> TubeTopology:
    src, tgt = _build_branch_table()
    current_in, current_out = _build_section_wiring(src, tgt)
    is_radiation, is_branching, is_simple, branching_off = _build_branch_masks(
        src, tgt, current_out
    )
    matrix_nonzero = _build_matrix_pattern(src, tgt, current_in, current_out)
    row_idx, row_len, col_idx, col_len = _build_envelope_tables(matrix_nonzero)
    return TubeTopology(
        source_section=src,
        target_section=tgt,
        current_in=current_in,
        current_out=current_out,
        is_radiation=is_radiation,
        is_branching=is_branching,
        is_simple=is_simple,
        branching_off=branching_off,
        matrix_nonzero=matrix_nonzero,
        envelope_row_idx=row_idx,
        envelope_row_len=row_len,
        envelope_col_idx=col_idx,
        envelope_col_len=col_len,
    )


# Module-level singleton: speaker-independent, computed once on import.
TOPOLOGY = build_topology()
