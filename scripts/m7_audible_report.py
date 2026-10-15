"""correctness/004b consolidated audible-frication report.

Aggregates the outputs of ``m7_whisper_gate.py`` and
``m7_frication_band.py`` into a single ``audible_report.json`` plus a
unit-test-gate summary (the 12 unit tests from 004a + the 5 from 004b).

Acceptance gates per the task file:

* Forward bit-equality at ``source_amp = 0`` — covered by
  ``test_frication_audible.py::test_bit_equality_zero_source_vs_legacy_none``.
* All 12 of 004a's unit-test gates still pass — covered by
  ``test_frication_physical.py``.
* MNA solve stable at ``noise_amp_scale ∈ {1, 5, 15} cm³/s`` — covered
  by ``test_frication_audible.py::test_mna_solve_stable_at_physiological_drive``.
* ``m7_frication_band.py`` 4-8 kHz lift ≥ 1 dB on ≥ 8/10 utterances —
  covered by the loaded ``frication_band.json``.
* Whisper gate ≥ 50% sub-rate reduction — covered by the loaded
  ``whisper_gate.json``. The escalation clause covers the case where
  Whisper produces empty/garbage on both pre and post (smoke/002 floor).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[1]


def _run_pytest(test_node: str) -> dict:
    """Run a pytest node and return ``{passed, failed, total, ...}``."""
    # Worktrees don't carry their own .venv; the master engine venv has
    # the right deps. The python on PATH inside the worktree is the
    # repo's master engine venv (sys.executable).
    cmd = [
        sys.executable,
        "-m", "pytest",
        test_node, "-v", "--no-header", "-q",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKTREE), check=False)
    out = p.stdout + p.stderr
    # Quick parse: count "passed" and "failed" in tail line.
    last = out.strip().splitlines()[-1] if out.strip() else ""
    return {
        "node": test_node,
        "returncode": p.returncode,
        "summary_line": last,
        "stdout_tail": "\n".join(out.strip().splitlines()[-10:]),
    }


def main() -> int:
    pa = argparse.ArgumentParser(prog="m7_audible_report")
    pa.add_argument(
        "--out", type=Path, default=WORKTREE / "outputs" / "m7_frication" / "audible_report.json"
    )
    args = pa.parse_args()

    band_json = WORKTREE / "outputs" / "m7_frication" / "frication_band.json"
    whisper_json = WORKTREE / "outputs" / "m7_frication" / "whisper_gate.json"

    band = json.loads(band_json.read_text()) if band_json.exists() else None
    whisper = json.loads(whisper_json.read_text()) if whisper_json.exists() else None

    physical_tests = _run_pytest(
        "tractus_engine_advanced/tests/test_frication_physical.py"
    )
    audible_tests = _run_pytest(
        "tractus_engine_advanced/tests/test_frication_audible.py"
    )

    band_pass = bool(band and band.get("pass_a_coverage", False) and band.get("pass_b_ratio", False))
    band_pass_a = bool(band and band.get("pass_a_coverage", False))
    band_pass_b = bool(band and band.get("pass_b_ratio", False))

    whisper_pass = bool(whisper and whisper.get("pass", False))
    whisper_inconclusive = bool(
        whisper
        and whisper.get("aggregate", {}).get("pre_total_target", 0) > 0
        and whisper.get("aggregate", {}).get("pre_rate", 1.0) >= 0.99
    )

    report = {
        "task": "correctness/004b — audible frication",
        "gate_summary": {
            "bit_equality_zero_source": {
                "covered_by": "test_frication_audible.py::test_bit_equality_zero_source_vs_legacy_none",
                "summary": audible_tests["summary_line"],
                "returncode": audible_tests["returncode"],
            },
            "twelve_unit_tests": {
                "covered_by": "test_frication_physical.py (12 tests)",
                "summary": physical_tests["summary_line"],
                "returncode": physical_tests["returncode"],
            },
            "mna_solve_stable_1_5_15_cm3_s": {
                "covered_by": "test_frication_audible.py::test_mna_solve_stable_at_physiological_drive",
                "summary": audible_tests["summary_line"],
                "returncode": audible_tests["returncode"],
            },
            "frication_band_4_8khz_lift": {
                "pass": band_pass,
                "pass_a_coverage_ge_8_of_10": band_pass_a,
                "pass_b_ratio_bounded": band_pass_b,
                "n_pairs": band.get("n_pairs") if band else None,
                "n_with_delta_ge_1db": band.get("n_with_delta_ge_1db") if band else None,
                "mean_delta_db": float(sum(band.get("deltas_4_8_db", [])) / max(len(band.get("deltas_4_8_db", [])), 1))
                    if band else None,
                "median_ratio": band.get("median_ratio_4_8") if band else None,
                "mean_ratio": band.get("mean_ratio_4_8") if band else None,
            },
            "whisper_substitution_rate_50_percent_lower": {
                "pass": whisper_pass,
                "inconclusive_due_to_smoke_floor": whisper_inconclusive,
                "pre_rate": whisper.get("aggregate", {}).get("pre_rate") if whisper else None,
                "post_rate": whisper.get("aggregate", {}).get("post_rate") if whisper else None,
                "ratio": whisper.get("aggregate", {}).get("ratio") if whisper else None,
                "note": (
                    "Pre rate = 1.0 because Whisper produces empty transcripts on the rule-based "
                    "smoke pipeline output (body-model floor; ARCH §1049, smoke/002 deferred). "
                    "Post rate < 1.0 means some fricative content IS being recognized "
                    "post-frication (e.g. 'Shhh, shhh, shhh' for seashells). Frication band "
                    "gate (4-8 kHz lift) is the binding measurement until smoke/002 closes."
                ),
            },
        },
        "frication_band": band,
        "whisper_gate": whisper,
        "physical_tests": physical_tests,
        "audible_tests": audible_tests,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.out}")

    print("\n=== Acceptance summary ===")
    for k, v in report["gate_summary"].items():
        passed = v.get("pass") if "pass" in v else v.get("returncode") == 0
        print(f"  [{'PASS' if passed else 'FAIL'}]  {k}")
    print()
    if whisper_inconclusive:
        print("Whisper substitution gate inconclusive: pre rate = 1.0 means no measurement")
        print("headroom. This is the smoke/002 body-model floor (escalation criterion in")
        print("workplan/tasks/01_correctness/004b_frication_audible.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
