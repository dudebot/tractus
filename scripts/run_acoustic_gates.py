"""Runner for the perceptually-grounded acoustic gates.

Iterates over the WAVs produced by ``scripts/phoneme_test.py``,
applies the appropriate gates per phoneme, and writes a JSON report
to ``outputs/phoneme_test/acoustic_gates_report.json``.

Run::

    cd tractus_controller
    .venv/bin/python ../scripts/run_acoustic_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tractus_engine_advanced" / "src"))

from tractus_engine_advanced.acoustic_gates import (  # noqa: E402
    FRICATIVE_SHAPES,
    VOWEL_FORMANTS_MALE,
    formant_gate,
    fricative_shape_gate,
    onset_gate,
)


PHONEME_DIR = ROOT / "outputs/phoneme_test"
REPORT_PATH = PHONEME_DIR / "acoustic_gates_report.json"


# Per-phoneme gate spec. Maps WAV stem → list of gate specs.
GATE_SPEC: dict[str, list[dict]] = {
    "01_a_vowel": [
        {"gate": "onset"},
        {"gate": "formant", "vowel": "a"},
    ],
    "02_i_vowel": [
        {"gate": "onset"},
        {"gate": "formant", "vowel": "i"},
    ],
    "03_u_vowel": [
        {"gate": "onset"},
        {"gate": "formant", "vowel": "u"},
    ],
    "04_schwa": [
        {"gate": "onset"},
        {"gate": "formant", "vowel": "schwa"},
    ],
    "05_a_low_F0": [
        {"gate": "onset"},
        {"gate": "formant", "vowel": "a"},
    ],
    "06_a_high_F0": [
        {"gate": "onset"},
        {"gate": "formant", "vowel": "a"},
    ],
    "07_m_nasal": [
        {"gate": "onset"},
        # /m/ has formants but with a low nasal F1 + anti-formant; the
        # vowel formant gate doesn't fit. Skipped pending nasal-specific
        # gate (Codex spec #5).
    ],
    "08_s_no_fric": [
        {"gate": "onset"},
        # No frication → no shape to test; just confirms it's near-silent.
    ],
    "09_s_with_fric": [
        {"gate": "onset"},
        {"gate": "fricative_shape", "phone": "s"},
    ],
    "10_sh_with_fric": [
        {"gate": "onset"},
        {"gate": "fricative_shape", "phone": "sh"},
    ],
    "11_f_with_fric": [
        {"gate": "onset"},
        {"gate": "fricative_shape", "phone": "f"},
    ],
    "12_default_tube_voiced": [
        {"gate": "onset"},
        # Default tube isn't a known vowel; just exercise onset gate.
    ],
    "13_h_aspiration_a": [
        {"gate": "onset"},
        {"gate": "fricative_shape", "phone": "h"},
    ],
    "14_whispered_a": [
        {"gate": "onset"},
    ],
}


def _run_gates_for(stem: str, audio: np.ndarray, sr: int) -> list[dict]:
    spec = GATE_SPEC.get(stem, [])
    results: list[dict] = []
    for s in spec:
        gate_name = s["gate"]
        if gate_name == "onset":
            r = onset_gate(audio, sr)
        elif gate_name == "formant":
            vowel = s["vowel"]
            f1, f2, f3 = VOWEL_FORMANTS_MALE[vowel]
            r = formant_gate(
                audio, sr,
                target_F1_Hz=f1, target_F2_Hz=f2, target_F3_Hz=f3,
            )
        elif gate_name == "fricative_shape":
            phone = s["phone"]
            shape = FRICATIVE_SHAPES[phone]
            r = fricative_shape_gate(
                audio, sr,
                target_peak_Hz_range=shape["peak_Hz"],
                target_centroid_Hz_range=shape["centroid_Hz"],
            )
        else:
            raise ValueError(f"unknown gate {gate_name}")
        results.append(r.to_dict())
    return results


def main() -> int:
    wavs = sorted(PHONEME_DIR.glob("*.wav"))
    if not wavs:
        print(f"no WAVs in {PHONEME_DIR}; run scripts/phoneme_test.py first.")
        return 1

    report: dict = {"phonemes": {}, "summary": {}}
    n_total = 0
    n_pass = 0

    for wav in wavs:
        stem = wav.stem
        if stem not in GATE_SPEC:
            continue
        audio, sr = sf.read(str(wav))
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        gates = _run_gates_for(stem, audio, sr)
        report["phonemes"][stem] = {
            "wav": str(wav.relative_to(ROOT)),
            "sample_rate": int(sr),
            "duration_s": float(audio.size / sr),
            "rms": float(np.sqrt(np.mean(audio ** 2))),
            "peak": float(np.max(np.abs(audio))),
            "gates": gates,
        }
        for g in gates:
            n_total += 1
            if g["passed"]:
                n_pass += 1

    report["summary"] = {
        "n_phonemes": len(report["phonemes"]),
        "n_gates": n_total,
        "n_passed": n_pass,
        "n_failed": n_total - n_pass,
        "pass_rate": (n_pass / n_total) if n_total else 0.0,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    # Console summary.
    print(f"\nAcoustic-gate report ({n_pass}/{n_total} passed):\n")
    for stem, info in report["phonemes"].items():
        print(f"  {stem}  rms={info['rms']:.4f}  peak={info['peak']:.4f}")
        for g in info["gates"]:
            status = "PASS" if g["passed"] else "FAIL"
            print(f"    [{status}] {g['name']}: {g['note']}")
    print(f"\nReport: {REPORT_PATH.relative_to(ROOT)}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
