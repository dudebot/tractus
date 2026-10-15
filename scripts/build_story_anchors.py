"""Build the Story-1996 anchor NPZ at
``tractus_controller/outputs/fixtures/story_1996_anchors.npz``.

Source notes (auditable provenance — read carefully)
====================================================

The original target (per workplan/tasks/04_pretraining/002a) is the
Story, Titze, Hoffman 1996 paper ("Vocal tract area functions from
magnetic resonance imaging," J. Acoust. Soc. Am. 100, 537-554) which
reported 18 area functions: 12 vowels + 3 nasals + 3 voiced plosives.
The raw 1996 numerical tables are paywalled and were not freely
reachable when this fixture was built (May 2026).

What is reproduced here:

* **11 vowels** (/i ɪ e ɛ æ ʌ ɑ ɔ o ʊ u/) at 44 sections each, copied
  verbatim from **Table I of the open-access 2002 re-imaging paper**:

    Story, B. H. (2008). Comparison of magnetic resonance
    imaging-based vocal tract area functions obtained from the same
    speaker in 1994 and 2002. J. Acoust. Soc. Am. 123(1), 327-335.
    PMC2377017, Table I.

  The 2002 dataset uses the *same speaker* as STH96 (re-imaged 8 yr
  later) at the same 44-section equal-interval discretization. Per
  the 2008 paper's own analysis, the two datasets are quantitatively
  similar (modal patterns within ~5 % across the vowel space) — the
  2002 set is the closest publicly-fetchable analog of STH96 vowel
  data. We treat it as a verifiably-derived primary source.

  Per-vowel section length Δ (cm) and total VTL (cm) are also from
  Table I of the 2008 paper.

What is NOT reproduced (gaps, with rationale)
---------------------------------------------

* **/ɝ/ (rhotic schwa)**: present in STH96 (12th vowel) but NOT
  re-imaged in 2002. Would need access to the 1996 raw tables
  (paywalled). Marked as MISSING in `shape_status`.
* **3 nasals** (/m n ŋ/): in STH96 only. The 2005 Story parametric
  paper uses a vowel-substrate-plus-shaping-function model, not raw
  area tables. MISSING.
* **3 voiced plosives** (/b d g/, NOT /p t k/ as the task spec
  initially said — STH96's plosives are voiced because they need a
  glottal source for the MRI utterance). MISSING.

Total ingested: 11 of 18 STH96 shapes (61 %). The acceptance gate
(≥ 9 of 12 vowels passing F1+F2 within 15 %) becomes ≥ 9 of 11 here;
the gate is operationally equivalent because no fabrication is
permitted.

NPZ schema (per task spec)
--------------------------

* ``shape_names``       : (N,) string  — IPA-ish keys, e.g. "/i/"
* ``shape_status``      : (N,) string  — "ingested" or "missing"
* ``area_cm2_44sec``    : (N, 44) float — 44-section areas in cm²
* ``length_cm_per_44``  : (N,) float    — per-section length Δ (cm)
* ``total_length_cm``   : (N,) float    — VTL = 44 · Δ (cm)
* ``area_cm2_tea``      : (N, 40) float — resampled to TEA's 40 supra
                                          sections (FIRST_PHARYNX..LAST_MOUTH)
* ``length_cm_tea``     : (N, 40) float — per-section length under TEA
                                          (= VTL / 40, scalar broadcast)
* ``provenance``        : json string   — dict with citation, method, gaps

The "missing" entries have NaN-filled area arrays so the loader and
gate script can skip them cleanly without fabricating values.

Run from any CWD; writes to repo's
``tractus_controller/outputs/fixtures/``.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Story 2002 re-imaging — Table I (PMC2377017).
# Eleven vowels, 44 sections each. Areas in cm².
#
# Columns (left-to-right in source table):
#   i   ɪ   e   ɛ   æ   ʌ   ɑ   ɔ   o   ʊ   u
# --------------------------------------------------------------------------

VOWEL_KEYS_2002 = ("i", "I", "e", "E", "ae", "V", "A", "O", "o", "U", "u")
VOWEL_IPA_2002 = ("/i/", "/ɪ/", "/e/", "/ɛ/", "/æ/", "/ʌ/", "/ɑ/", "/ɔ/", "/o/", "/ʊ/", "/u/")

# 44 rows × 11 columns, copied verbatim from PMC2377017 Table I.
# Each row corresponds to one section index (1-indexed in source;
# 0-indexed here implicitly).
AREA_44_2002 = np.array(
    [
        [0.51, 0.28, 0.29, 0.37, 0.31, 0.23, 0.56, 0.27, 0.38, 0.37, 0.54],
        [0.59, 0.21, 0.26, 0.30, 0.21, 0.34, 0.62, 0.43, 0.45, 0.38, 0.61],
        [0.62, 0.21, 0.30, 0.24, 0.18, 0.47, 0.66, 0.54, 0.57, 0.49, 0.66],
        [0.72, 0.30, 0.40, 0.23, 0.23, 0.60, 0.78, 0.67, 0.77, 0.62, 0.75],
        [1.24, 0.47, 0.55, 0.29, 0.33, 0.77, 0.97, 0.83, 1.31, 0.85, 1.13],
        [2.30, 0.71, 0.74, 0.41, 0.50, 1.06, 1.16, 0.92, 1.92, 1.28, 1.99],
        [3.30, 1.12, 0.99, 0.58, 0.78, 1.26, 1.12, 0.89, 1.74, 1.62, 2.83],
        [3.59, 1.48, 1.09, 0.82, 0.96, 1.09, 0.82, 0.73, 1.11, 1.47, 2.90],
        [3.22, 1.35, 0.90, 0.97, 0.85, 0.80, 0.55, 0.55, 0.75, 1.04, 2.52],
        [2.86, 1.05, 0.69, 0.82, 0.63, 0.65, 0.45, 0.44, 0.59, 0.81, 2.40],
        [3.00, 0.92, 0.77, 0.62, 0.46, 0.56, 0.37, 0.37, 0.57, 1.03, 2.83],
        [3.61, 0.92, 1.31, 0.54, 0.36, 0.47, 0.29, 0.28, 0.68, 1.44, 3.56],
        [4.39, 1.19, 2.13, 0.54, 0.33, 0.37, 0.21, 0.18, 0.73, 1.49, 3.99],
        [4.95, 1.94, 2.74, 0.68, 0.46, 0.24, 0.15, 0.14, 0.67, 1.28, 3.89],
        [5.17, 2.83, 3.03, 1.09, 0.73, 0.17, 0.16, 0.15, 0.58, 1.06, 3.50],
        [5.16, 3.31, 3.23, 1.62, 1.00, 0.18, 0.25, 0.14, 0.49, 0.85, 3.04],
        [5.18, 3.48, 3.33, 2.03, 1.30, 0.23, 0.34, 0.13, 0.44, 0.69, 2.64],
        [5.26, 3.60, 3.27, 2.35, 1.66, 0.27, 0.43, 0.14, 0.42, 0.56, 2.44],
        [5.20, 3.64, 3.09, 2.51, 1.97, 0.28, 0.54, 0.18, 0.49, 0.42, 2.31],
        [5.02, 3.49, 2.84, 2.39, 2.06, 0.29, 0.61, 0.20, 0.53, 0.27, 2.07],
        [4.71, 3.20, 2.66, 2.22, 2.03, 0.33, 0.67, 0.23, 0.38, 0.27, 1.80],
        [4.13, 2.90, 2.46, 2.13, 2.01, 0.52, 0.98, 0.49, 0.30, 0.41, 1.52],
        [3.43, 2.59, 2.14, 2.00, 1.89, 0.97, 1.76, 1.03, 0.45, 0.51, 1.14],
        [2.83, 2.21, 1.79, 1.78, 1.66, 1.50, 2.75, 1.58, 0.61, 0.49, 0.74],
        [2.32, 1.87, 1.44, 1.58, 1.49, 1.91, 3.52, 2.06, 0.71, 0.47, 0.42],
        [1.83, 1.54, 1.17, 1.43, 1.42, 2.23, 4.08, 2.61, 0.79, 0.50, 0.22],
        [1.46, 1.20, 1.00, 1.31, 1.37, 2.65, 4.74, 3.35, 0.86, 0.58, 0.14],
        [1.23, 0.92, 0.88, 1.23, 1.34, 3.29, 5.61, 4.34, 1.01, 0.80, 0.20],
        [1.08, 0.74, 0.80, 1.24, 1.41, 4.13, 6.60, 5.51, 1.41, 1.19, 0.47],
        [0.94, 0.59, 0.81, 1.38, 1.58, 5.00, 7.61, 6.70, 2.09, 1.62, 0.89],
        [0.80, 0.52, 0.85, 1.61, 1.82, 5.77, 8.48, 7.75, 3.00, 2.27, 1.15],
        [0.67, 0.54, 0.84, 1.82, 2.19, 6.33, 9.06, 8.63, 4.10, 3.24, 1.42],
        [0.55, 0.59, 0.86, 1.96, 2.63, 6.61, 9.29, 9.29, 5.16, 4.16, 2.17],
        [0.46, 0.65, 0.96, 2.01, 2.97, 6.63, 9.26, 9.59, 6.22, 5.00, 3.04],
        [0.40, 0.71, 1.18, 2.00, 3.17, 6.45, 9.06, 9.42, 7.34, 5.70, 3.69],
        [0.36, 0.67, 1.35, 1.95, 3.40, 6.04, 8.64, 8.78, 8.15, 6.11, 4.70],
        [0.35, 0.61, 1.48, 1.77, 3.56, 5.39, 7.91, 7.82, 8.61, 6.21, 5.74],
        [0.35, 0.57, 1.62, 1.48, 3.57, 4.42, 6.98, 6.50, 8.37, 6.29, 5.41],
        [0.38, 0.50, 1.49, 1.30, 3.58, 3.29, 6.02, 4.95, 6.76, 6.24, 3.82],
        [0.51, 0.48, 1.29, 1.21, 3.44, 2.37, 5.13, 3.47, 4.37, 4.91, 2.34],
        [0.74, 0.54, 1.24, 1.10, 3.15, 1.74, 4.55, 2.15, 2.30, 2.61, 1.35],
        [0.92, 0.73, 1.17, 0.99, 3.38, 1.36, 4.52, 1.38, 1.06, 1.09, 0.65],
        [0.96, 0.93, 1.04, 0.91, 3.99, 1.17, 4.71, 1.11, 0.58, 0.63, 0.29],
        [0.91, 0.82, 0.95, 0.78, 4.17, 0.99, 4.72, 0.90, 0.47, 0.59, 0.16],
    ],
    dtype=np.float64,
)
assert AREA_44_2002.shape == (44, 11), AREA_44_2002.shape

# Per-vowel section length (Δ, cm) and total VTL — same row, source table.
DELTA_CM_2002 = np.array(
    [0.384, 0.376, 0.386, 0.393, 0.366, 0.390, 0.388, 0.395, 0.417, 0.440, 0.445],
    dtype=np.float64,
)
VTL_CM_2002 = np.array(
    [16.90, 16.55, 16.98, 17.30, 16.11, 17.14, 17.09, 17.40, 18.33, 19.34, 19.59],
    dtype=np.float64,
)
# Sanity: VTL ≈ 44 · Δ (table values are rounded; tolerance ~0.05 cm).
np.testing.assert_allclose(VTL_CM_2002, 44.0 * DELTA_CM_2002, atol=0.05)


# --------------------------------------------------------------------------
# Missing shapes (placeholders so schema stays uniform).
# --------------------------------------------------------------------------

MISSING_KEYS = ("er", "m", "n", "ng", "b", "d", "g")
MISSING_IPA = ("/ɝ/", "/m/", "/n/", "/ŋ/", "/b/", "/d/", "/g/")


# --------------------------------------------------------------------------
# 44-section → 40-section resampling.
# --------------------------------------------------------------------------

N_TEA_SUPRA = 40   # FIRST_PHARYNX_SECTION (25) .. LAST_MOUTH_SECTION (64)


def resample_44_to_40(area_44: np.ndarray) -> np.ndarray:
    """Resample one 44-section area function to 40 equal-length sections.

    The 44-section function is treated as a piecewise-constant area along
    a normalized arc-length [0, 1]. The 40-section output integrates the
    44-section function over each 40-section interval (preserves total
    cross-sectional volume per axial extent).

    For Story-style equal-Δ functions this is mathematically identical
    to placing 40 evenly-spaced sample points along the cumulative-area
    curve, which is the canonical resampling choice in the articulatory-
    synthesis literature (see Story 2008 §II for the same procedure
    applied between 1994/2002 datasets).

    Parameters
    ----------
    area_44 : (44,) float
        Areas in cm² at equal axial spacing.

    Returns
    -------
    area_40 : (40,) float
        Areas in cm² at equal axial spacing covering the same total tract
        length.
    """
    if area_44.shape != (44,):
        raise ValueError(f"expected (44,), got {area_44.shape}")
    # Normalized cumulative axial position at the END of each 44-section bin.
    edges_in = np.linspace(0.0, 1.0, 45)        # 44 intervals
    edges_out = np.linspace(0.0, 1.0, 41)       # 40 intervals

    # Cumulative integral A(x) dx along normalized arc length.
    # Width of each input bin = 1/44.
    bin_w_in = 1.0 / 44.0
    cum_in = np.zeros(45)
    cum_in[1:] = np.cumsum(area_44 * bin_w_in)

    # Interpolate cumulative at output edges.
    cum_out = np.interp(edges_out, edges_in, cum_in)
    bin_w_out = 1.0 / 40.0
    area_40 = np.diff(cum_out) / bin_w_out
    return area_40


# --------------------------------------------------------------------------
# NPZ assembly.
# --------------------------------------------------------------------------

def build_npz(out_path: Path) -> None:
    n_ingested = len(VOWEL_IPA_2002)
    n_missing = len(MISSING_IPA)
    n_total = n_ingested + n_missing

    shape_names = list(VOWEL_IPA_2002) + list(MISSING_IPA)
    shape_keys = list(VOWEL_KEYS_2002) + list(MISSING_KEYS)
    shape_status = ["ingested"] * n_ingested + ["missing"] * n_missing

    area_44 = np.full((n_total, 44), np.nan, dtype=np.float64)
    length_per_44 = np.full((n_total,), np.nan, dtype=np.float64)
    total_len = np.full((n_total,), np.nan, dtype=np.float64)
    area_tea = np.full((n_total, N_TEA_SUPRA), np.nan, dtype=np.float64)
    length_tea = np.full((n_total, N_TEA_SUPRA), np.nan, dtype=np.float64)

    # Source columns are (44, 11) -> transpose to (11, 44) for per-shape rows.
    area_44[:n_ingested, :] = AREA_44_2002.T
    length_per_44[:n_ingested] = DELTA_CM_2002
    total_len[:n_ingested] = VTL_CM_2002

    for i in range(n_ingested):
        area_tea[i, :] = resample_44_to_40(area_44[i, :])
        # Preserve per-vowel total length: each TEA section has length VTL/40.
        length_tea[i, :] = total_len[i] / N_TEA_SUPRA

    provenance = {
        "task": "pretraining/002a — Story-1996 anchor ingestion",
        "primary_citation": (
            "Story, B. H. (2008). Comparison of magnetic resonance "
            "imaging-based vocal tract area functions obtained from the "
            "same speaker in 1994 and 2002. J. Acoust. Soc. Am. 123(1), "
            "327-335. PMC2377017, Table I."
        ),
        "original_target_citation": (
            "Story, B. H., Titze, I. R., Hoffman, E. A. (1996). Vocal "
            "tract area functions from magnetic resonance imaging. "
            "J. Acoust. Soc. Am. 100(1), 537-554."
        ),
        "speaker": "Same adult-male subject as STH96, re-imaged 2002.",
        "section_count_source": 44,
        "section_count_tea": N_TEA_SUPRA,
        "resampling_method": (
            "Equal-arc-length cumulative-integral resampling: piecewise-"
            "constant area on [0,1] arc length, integrated over 40 "
            "uniform output bins. Per-shape total VTL is preserved by "
            "broadcasting length_cm_tea = VTL/40."
        ),
        "ingested_count": n_ingested,
        "ingested_keys": list(VOWEL_KEYS_2002),
        "missing_count": n_missing,
        "missing_keys": list(MISSING_KEYS),
        "missing_rationale": (
            "STH96 included 12 vowels + 3 nasals + 3 voiced plosives "
            "(/b d g/, NOT /p t k/). The 2002 re-imaging only covers "
            "11 vowels (/ɝ/ omitted). Nasals + plosives need access to "
            "STH96's raw tables, which were paywalled at ingest time. "
            "Missing rows are NaN; loader skips them."
        ),
        "ingest_date": date.today().isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        shape_names=np.array(shape_names),
        shape_keys=np.array(shape_keys),
        shape_status=np.array(shape_status),
        area_cm2_44sec=area_44,
        length_cm_per_44=length_per_44,
        total_length_cm=total_len,
        area_cm2_tea=area_tea,
        length_cm_tea=length_tea,
        provenance=np.array(json.dumps(provenance)),
    )
    print(f"Wrote {out_path} ({n_ingested} ingested, {n_missing} missing).")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "tractus_controller/outputs/fixtures/story_1996_anchors.npz"
    build_npz(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
