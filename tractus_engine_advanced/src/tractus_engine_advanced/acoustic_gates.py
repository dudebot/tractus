"""Perceptually-grounded acoustic gates for TEA outputs.

Replaces the band-energy / Whisper-WER thresholds that gave us false
confidence on correctness/004-004b (the +8.2 dB 4-8 kHz lift gate
passed while the audio sounded like clicking — broadband transients
were satisfying the band-energy test). These gates anchor pass/fail
to *measurable acoustic targets* against literature-derived reference
shapes:

* :func:`onset_gate` — onset transient ≤ 1.5× steady-state RMS in
  the first 50 ms. Would have caught the click bug (commit 68196b8)
  on a single 100 ms /a/ render.
* :func:`formant_gate` — vowel F1/F2/F3 within tolerance of a
  literature target. Adult-male reference values from Peterson &
  Barney 1952 / Hillenbrand 1995.
* :func:`fricative_shape_gate` — log-PSD template match against a
  per-place target shape (sibilant peak, post-alveolar peak, dental
  diffuse) with both peak-frequency and spectral-centroid checks.

Each gate returns a :class:`GateResult` with quantitative measured
vs target values plus a one-line note. The runner script
``scripts/run_acoustic_gates.py`` (separate file) drives all gates
across the ``phoneme_test`` corpus and produces a JSON report.

Design constraints (from CLAUDE.md "Auto-tests beat ear-tests" and
the user's "gradient descent for target sounds" framing):

1. **Differentiable end-to-end.** Every gate's distance metric is a
   pure-numpy / pure-jnp function so it can serve as a loss for
   inverse-fitting a target sound through TEA via ``jax.grad``.
   See :func:`spectral_template_loss` for the canonical form.
2. **No ASR dependency.** Whisper is demoted to a smoke-stop-loss
   per Codex's design review; routine acceptance must work without
   it.
3. **Bounded compute.** All gates run on phone-scale windows
   (typically 100-500 ms); no full-utterance gradient descent at
   gate time. Inverse-fitting is opt-in for the per-task worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Gate result type.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Result of one gate measurement on one audio clip.

    Attributes
    ----------
    name : str
        Gate identifier (``"onset"``, ``"formant"``, ``"fricative_shape"``).
    passed : bool
        Whether the measurement satisfies the target.
    measured : dict
        Actual values from the rendered audio (e.g. ``F1_Hz=712``).
    target : dict
        Expected values (e.g. ``F1_target_Hz=730``).
    distance : float
        Single-scalar distance metric. ``0.0`` = exact match. Larger =
        worse. Sign convention: positive number always means "further
        from target", regardless of whether the underlying axis is
        Hz, dB, or ratio.
    note : str
        One-line human-readable explanation.
    """

    name: str
    passed: bool
    measured: dict
    target: dict
    distance: float
    note: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "measured": dict(self.measured),
            "target": dict(self.target),
            "distance": float(self.distance),
            "note": self.note,
        }


# --------------------------------------------------------------------------
# Onset transient gate.
# --------------------------------------------------------------------------


def onset_gate(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_ratio: float = 1.5,
    onset_ms: float = 50.0,
    steady_start_ms: float = 200.0,
    steady_window_ms: float = 100.0,
    silent_steady_threshold_rms: float = 1e-3,
    silent_onset_peak_floor: float = 5e-3,
) -> GateResult:
    """First-50-ms RMS / steady-state RMS ratio.

    Catches onset clicks. The pre-fix /a/ render had ratio ~3.3
    (transient peak 0.29 vs steady 0.09); post-fix is ~1.0.

    For near-silent clips (steady RMS below
    ``silent_steady_threshold_rms``) the ratio metric is meaningless
    — divide by ~zero blows up. Fall back to an absolute onset peak
    check: pass if the onset peak is also below
    ``silent_onset_peak_floor`` (i.e., the whole clip is genuinely
    quiet, not just steady-quiet-after-a-bang).
    """
    audio = np.asarray(audio, dtype=np.float64)
    onset_n = int(sample_rate * onset_ms / 1000.0)
    steady_n = int(sample_rate * steady_window_ms / 1000.0)
    steady_start = int(sample_rate * steady_start_ms / 1000.0)

    if audio.size < steady_start + steady_n:
        third = audio.size // 3
        onset_seg = audio[:third]
        steady_seg = audio[-third:]
    else:
        onset_seg = audio[:onset_n]
        steady_seg = audio[steady_start : steady_start + steady_n]

    onset_rms = float(np.sqrt(np.mean(onset_seg ** 2)))
    onset_peak = float(np.max(np.abs(onset_seg)))
    steady_rms = float(np.sqrt(np.mean(steady_seg ** 2)))

    if steady_rms < silent_steady_threshold_rms:
        # Near-silent clip — use absolute onset peak instead of ratio.
        passed = onset_peak <= silent_onset_peak_floor
        return GateResult(
            name="onset",
            passed=passed,
            measured={
                "onset_rms": onset_rms,
                "onset_peak": onset_peak,
                "steady_rms": steady_rms,
                "ratio": float("inf"),
            },
            target={
                "max_ratio": max_ratio,
                "silent_onset_peak_floor": silent_onset_peak_floor,
                "silent_steady_threshold_rms": silent_steady_threshold_rms,
            },
            distance=max(0.0, onset_peak - silent_onset_peak_floor),
            note=(
                f"Near-silent clip (steady RMS={steady_rms:.4f} below "
                f"{silent_steady_threshold_rms}); using absolute onset "
                f"peak={onset_peak:.4f} vs floor {silent_onset_peak_floor}. "
                f"{'OK' if passed else 'audible click on otherwise-silent clip'}."
            ),
        )

    ratio = onset_rms / max(steady_rms, 1e-9)
    return GateResult(
        name="onset",
        passed=(ratio <= max_ratio),
        measured={
            "onset_rms": onset_rms,
            "onset_peak": onset_peak,
            "steady_rms": steady_rms,
            "ratio": ratio,
        },
        target={"max_ratio": max_ratio},
        distance=max(0.0, ratio - 1.0),
        note=(
            f"Onset RMS / steady RMS = {ratio:.2f}. "
            f"{'OK' if ratio <= max_ratio else f'>{max_ratio} = audible click'}."
        ),
    )


# --------------------------------------------------------------------------
# Formant gate (vowels).
# --------------------------------------------------------------------------


def _spectral_envelope(
    audio: np.ndarray, sample_rate: int, *, fft_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs_Hz, log_magnitude_dB)`` of the windowed FFT.

    Uses a Hann window on the steady-state portion. Returns the
    one-sided spectrum from 0 to Nyquist.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size > fft_size:
        start = (audio.size - fft_size) // 2
        seg = audio[start : start + fft_size]
    else:
        seg = np.zeros(fft_size, dtype=np.float64)
        seg[: audio.size] = audio
    win = np.hanning(fft_size)
    spec = np.fft.rfft(seg * win)
    mag = np.abs(spec) + 1e-12
    log_mag_dB = 20.0 * np.log10(mag)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return freqs, log_mag_dB


def _preemphasize(audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """First-order high-pass pre-emphasis y[n] = x[n] - α·x[n-1].

    Cancels the ~-6 dB/oct of the typical glottal-source tilt
    (Liljencrants-Fant envelope is roughly -12 dB/oct overall;
    pre-emphasis at α=0.97 leaves a residual -6 dB/oct that's
    flat enough for formant peak-finding). Standard speech-analysis
    trick — see Rabiner & Schafer or any LPC text.
    """
    out = np.empty_like(audio)
    out[0] = audio[0]
    out[1:] = audio[1:] - alpha * audio[:-1]
    return out


def _cepstral_envelope(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fft_size: int = 4096,
    lifter_cutoff_ms: float = 2.0,
    preemphasis: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cepstrally-liftered spectral envelope.

    Pipeline: pre-emphasis → Hann window → FFT → log → cepstrum →
    quefrency lifter → FFT back. The result is the smooth spectral
    envelope formants live in. Standard speech-analysis primitive
    (Oppenheim & Schafer; Rabiner & Schafer).

    Parameters
    ----------
    lifter_cutoff_ms
        Quefrency cutoff in ms. For adult-male F0 100-150 Hz,
        F0 period 6-10 ms; cut at 2-3 ms keeps the smooth envelope
        (formants) and removes the harmonic comb.
    preemphasis
        Pre-emphasis coefficient. 0.97 is the textbook value;
        0.0 disables.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if preemphasis != 0.0:
        audio = _preemphasize(audio, alpha=preemphasis)
    if audio.size > fft_size:
        start = (audio.size - fft_size) // 2
        seg = audio[start : start + fft_size]
    else:
        seg = np.zeros(fft_size, dtype=np.float64)
        seg[: audio.size] = audio
    win = np.hanning(fft_size)
    seg = seg * win
    spec = np.fft.fft(seg)
    log_mag = np.log(np.abs(spec) + 1e-12)
    cepstrum = np.real(np.fft.ifft(log_mag))
    n_lifter = max(1, int(lifter_cutoff_ms * sample_rate / 1000.0))
    lifter = np.zeros_like(cepstrum)
    lifter[: n_lifter + 1] = 1.0
    lifter[-n_lifter:] = 1.0
    smoothed_log_mag = np.real(np.fft.fft(cepstrum * lifter))
    smoothed_log_mag_dB = smoothed_log_mag * (20.0 / np.log(10.0))
    half = fft_size // 2 + 1
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return freqs, smoothed_log_mag_dB[:half]


def _smoothed_envelope(log_mag_dB: np.ndarray, smoothing_bins: int = 32) -> np.ndarray:
    """Box-car smooth the log-magnitude spectrum.

    Less robust than cepstral liftering for vowel formant
    extraction (F0 harmonics leak through unless smoothing is much
    wider than 1/F0 in bins). Kept as a fallback / for noise-source
    analysis (fricatives) where there's no harmonic comb to remove.
    """
    if smoothing_bins <= 1:
        return log_mag_dB
    kernel = np.ones(smoothing_bins) / smoothing_bins
    return np.convolve(log_mag_dB, kernel, mode="same")


def _find_formant_peaks(
    freqs: np.ndarray,
    smoothed_dB: np.ndarray,
    n_formants: int,
    *,
    f_min_Hz: float = 200.0,
    f_max_Hz: float = 5000.0,
    min_separation_Hz: float = 200.0,
) -> list[float]:
    """Find the N largest peaks in the smoothed envelope between
    ``f_min`` and ``f_max``, enforcing a minimum frequency
    separation. Returns peak frequencies in ascending order."""
    mask = (freqs >= f_min_Hz) & (freqs <= f_max_Hz)
    fs = freqs[mask]
    ys = smoothed_dB[mask]
    # Peak = local max strictly greater than both neighbors.
    is_peak = np.zeros_like(ys, dtype=bool)
    is_peak[1:-1] = (ys[1:-1] > ys[:-2]) & (ys[1:-1] > ys[2:])
    peak_fs = fs[is_peak]
    peak_ys = ys[is_peak]
    # Sort by amplitude descending, then enforce min separation.
    order = np.argsort(-peak_ys)
    chosen: list[float] = []
    for i in order:
        f = float(peak_fs[i])
        if all(abs(f - c) >= min_separation_Hz for c in chosen):
            chosen.append(f)
        if len(chosen) >= n_formants:
            break
    chosen.sort()
    return chosen


def formant_gate(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_F1_Hz: float,
    target_F2_Hz: float,
    target_F3_Hz: float | None = None,
    tolerance_pct: float = 15.0,
    fft_size: int = 4096,
    lifter_cutoff_ms: float = 2.0,
    preemphasis: float = 0.0,
) -> GateResult:
    """Vowel-formant gate: F1, F2, (optional F3) within tolerance of
    target frequencies. Tolerance is a percentage of the target
    frequency.

    Targets are expected to come from speech-acoustics literature
    (Peterson & Barney 1952, Hillenbrand 1995). For adult male:

    +-------+----------+----------+----------+
    | vowel | F1 (Hz)  | F2 (Hz)  | F3 (Hz)  |
    +=======+==========+==========+==========+
    | /a/   | 730      | 1090     | 2440     |
    | /i/   | 270      | 2290     | 3010     |
    | /u/   | 300      | 870      | 2240     |
    | /e/   | 530      | 1840     | 2480     |
    | /o/   | 570      | 840      | 2410     |
    | /ə/   | 500      | 1500     | 2500     |
    +-------+----------+----------+----------+
    """
    freqs, smoothed = _cepstral_envelope(
        audio, sample_rate, fft_size=fft_size,
        lifter_cutoff_ms=lifter_cutoff_ms, preemphasis=preemphasis,
    )
    n_target = 3 if target_F3_Hz is not None else 2
    peaks = _find_formant_peaks(freqs, smoothed, n_formants=n_target)

    measured: dict = {}
    target_dict: dict = {"tolerance_pct": tolerance_pct}
    distances: list[float] = []
    targets = [("F1", target_F1_Hz), ("F2", target_F2_Hz)]
    if target_F3_Hz is not None:
        targets.append(("F3", target_F3_Hz))

    for i, (name, tgt) in enumerate(targets):
        if i < len(peaks):
            measured_f = peaks[i]
        else:
            measured_f = float("nan")
        measured[f"{name}_Hz"] = measured_f
        target_dict[f"{name}_target_Hz"] = float(tgt)
        if np.isnan(measured_f):
            distances.append(float("inf"))
        else:
            err_pct = abs(measured_f - tgt) / tgt * 100.0
            distances.append(err_pct)
            measured[f"{name}_err_pct"] = err_pct

    max_dist = max(distances) if distances else float("inf")
    passed = max_dist <= tolerance_pct
    return GateResult(
        name="formant",
        passed=passed,
        measured=measured,
        target=target_dict,
        distance=max_dist,
        note=(
            f"Formants @ {sample_rate} Hz: "
            + ", ".join(
                f"{name}={measured[f'{name}_Hz']:.0f}/{tgt:.0f} "
                f"({measured.get(f'{name}_err_pct', float('inf')):.1f}%)"
                for name, tgt in targets
            )
            + f". {'OK' if passed else f'>{tolerance_pct}% off target'}."
        ),
    )


# --------------------------------------------------------------------------
# Fricative shape gate.
# --------------------------------------------------------------------------


def _normalize_log_psd(
    audio: np.ndarray, sample_rate: int, *, fft_size: int = 4096,
    f_min_Hz: float = 100.0, f_max_Hz: float = 11000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, log_psd_dB) normalized so peak = 0 dB. Restricted
    to the analysis band [f_min, f_max]."""
    freqs, log_mag = _spectral_envelope(audio, sample_rate, fft_size=fft_size)
    # PSD ~ |X|^2; in log domain that's 2 × log_mag, but for shape
    # comparison the constant doesn't matter — keep log_mag.
    mask = (freqs >= f_min_Hz) & (freqs <= f_max_Hz)
    fs = freqs[mask]
    ys = log_mag[mask]
    smoothed = _smoothed_envelope(ys, smoothing_bins=64)
    smoothed -= smoothed.max()
    return fs, smoothed


def _spectral_centroid_Hz(freqs: np.ndarray, log_mag_dB: np.ndarray) -> float:
    """Centroid = sum(f * |X|) / sum(|X|), with |X| = 10^(dB/20)."""
    mag = np.power(10.0, log_mag_dB / 20.0)
    return float(np.sum(freqs * mag) / max(np.sum(mag), 1e-9))


def _peak_frequency_Hz(freqs: np.ndarray, log_mag_dB: np.ndarray) -> float:
    return float(freqs[int(np.argmax(log_mag_dB))])


def fricative_shape_gate(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_peak_Hz_range: tuple[float, float],
    target_centroid_Hz_range: tuple[float, float],
    fft_size: int = 4096,
) -> GateResult:
    """Fricative-shape gate: spectral peak in target range AND
    centroid in target range.

    Targets per place class (literature, e.g. Strevens 1960 /
    Stevens 1971):

    +----------+----------+--------------+--------------+
    | phone    | place    | peak_Hz      | centroid_Hz  |
    +==========+==========+==============+==============+
    | /s/      | alveolar | 4000-7000    | 5000-7000    |
    | /ʃ/      | palatal  | 2000-4000    | 3000-4500    |
    | /f/      | labio-d  | 1500-3500    | 2500-4500    |
    | /θ/      | dental   | 5500-7500    | 5000-7000    |
    | /h/      | glottal  | 500-2000     | 1500-3000    |
    +----------+----------+--------------+--------------+

    The band-energy gate that we previously used FAILED this case:
    a +8 dB lift in 4-8 kHz can come from broadband click energy
    just as easily as from a sibilant — both pass band-energy, only
    a real sibilant peaks at 5-6 kHz. This gate caught it.
    """
    fs, log_psd = _normalize_log_psd(audio, sample_rate, fft_size=fft_size)
    peak_f = _peak_frequency_Hz(fs, log_psd)
    centroid_f = _spectral_centroid_Hz(fs, log_psd)

    peak_lo, peak_hi = target_peak_Hz_range
    cen_lo, cen_hi = target_centroid_Hz_range
    peak_ok = peak_lo <= peak_f <= peak_hi
    centroid_ok = cen_lo <= centroid_f <= cen_hi
    passed = peak_ok and centroid_ok

    # Distance: how far outside the target ranges, normalized.
    def _outside(x: float, lo: float, hi: float) -> float:
        if lo <= x <= hi:
            return 0.0
        if x < lo:
            return (lo - x) / max(lo, 1.0)
        return (x - hi) / max(hi, 1.0)

    peak_dist = _outside(peak_f, peak_lo, peak_hi)
    cen_dist = _outside(centroid_f, cen_lo, cen_hi)
    return GateResult(
        name="fricative_shape",
        passed=passed,
        measured={
            "peak_Hz": peak_f,
            "centroid_Hz": centroid_f,
        },
        target={
            "peak_Hz_range": list(target_peak_Hz_range),
            "centroid_Hz_range": list(target_centroid_Hz_range),
        },
        distance=peak_dist + cen_dist,
        note=(
            f"Peak={peak_f:.0f} Hz "
            f"(target {peak_lo:.0f}-{peak_hi:.0f}, {'OK' if peak_ok else 'OFF'}); "
            f"centroid={centroid_f:.0f} Hz "
            f"(target {cen_lo:.0f}-{cen_hi:.0f}, {'OK' if centroid_ok else 'OFF'})."
        ),
    )


# --------------------------------------------------------------------------
# Differentiable spectral-template loss for inverse-fitting.
# --------------------------------------------------------------------------


def spectral_template_loss_np(
    rendered_audio: np.ndarray,
    target_log_psd_dB: np.ndarray,
    sample_rate: int,
    *,
    fft_size: int = 4096,
    f_min_Hz: float = 100.0,
    f_max_Hz: float = 11000.0,
) -> float:
    """L2 distance between the rendered audio's normalized log-PSD
    and a target template, restricted to the [f_min, f_max] band.

    The numpy version is for runtime gates; the jnp version below
    (``spectral_template_loss_jnp``) is differentiable and the
    inverse-fit demo's loss objective.
    """
    fs, rendered_log_psd = _normalize_log_psd(
        rendered_audio, sample_rate, fft_size=fft_size,
        f_min_Hz=f_min_Hz, f_max_Hz=f_max_Hz,
    )
    if rendered_log_psd.shape != target_log_psd_dB.shape:
        raise ValueError(
            f"target_log_psd_dB shape {target_log_psd_dB.shape} != "
            f"rendered shape {rendered_log_psd.shape}"
        )
    return float(np.mean((rendered_log_psd - target_log_psd_dB) ** 2))


# --------------------------------------------------------------------------
# JAX-differentiable spectral-template loss.
#
# The numpy version above is for runtime gates. The jnp version below is
# the inverse-fit objective — compose with TEA forward (which has
# jax.custom_vjp on step_one per correctness/008) to gradient-descend
# on tube/glottis params until the rendered spectrum matches a target.
# This is the user's "gradient descent for target sounds" hook.
# --------------------------------------------------------------------------


def make_target_log_psd(
    sample_rate: int,
    *,
    formant_peaks_Hz: Sequence[float] | None = None,
    formant_amps_dB: Sequence[float] | None = None,
    formant_bandwidths_Hz: Sequence[float] | None = None,
    spectral_tilt_dB_per_oct: float = -12.0,
    fft_size: int = 4096,
    f_min_Hz: float = 100.0,
    f_max_Hz: float = 11000.0,
) -> np.ndarray:
    """Construct a target log-PSD shape from formant peaks + tilt.

    Each formant is a bell-curve centered at ``formant_peaks_Hz[i]``
    with amplitude ``formant_amps_dB[i]`` and bandwidth
    ``formant_bandwidths_Hz[i]``. The result is a smooth
    log-magnitude spectrum on the same band [f_min, f_max] as
    :func:`_normalize_log_psd` produces. Output is peak-normalized
    to 0 dB so it's directly comparable.
    """
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    mask = (freqs >= f_min_Hz) & (freqs <= f_max_Hz)
    fs = freqs[mask]

    if formant_peaks_Hz is None:
        formant_peaks_Hz = ()
    n = len(formant_peaks_Hz)
    if formant_amps_dB is None:
        formant_amps_dB = [0.0] * n
    if formant_bandwidths_Hz is None:
        formant_bandwidths_Hz = [80.0] * n

    log_psd_dB = np.zeros_like(fs)
    # Spectral tilt: -12 dB/oct in dB-space is straight from log2.
    nz = fs > 1.0
    log_psd_dB[nz] = spectral_tilt_dB_per_oct * np.log2(fs[nz] / 1000.0)
    for f0, amp, bw in zip(formant_peaks_Hz, formant_amps_dB, formant_bandwidths_Hz):
        log_psd_dB += amp * np.exp(-0.5 * ((fs - f0) / bw) ** 2)
    log_psd_dB -= log_psd_dB.max()
    return log_psd_dB


def spectral_template_loss_jnp(
    rendered_audio,    # jnp.ndarray (N,)
    target_log_psd_dB,  # jnp.ndarray (N_band,) — output of make_target_log_psd
    sample_rate: int,
    *,
    fft_size: int = 4096,
    f_min_Hz: float = 100.0,
    f_max_Hz: float = 11000.0,
    preemphasis: float = 0.0,
):
    """JAX-differentiable spectral-template L2 loss.

    Mirrors :func:`spectral_template_loss_np` but uses jnp ops so
    it composes with ``jax.grad`` through TEA's forward path.
    Pre-emphasis and Hann window are inside the differentiable
    region; the band-restriction mask is precomputed via numpy
    indexing (constant across the optimization).
    """
    import jax.numpy as jnp

    audio = rendered_audio
    if preemphasis != 0.0:
        audio = jnp.concatenate(
            [audio[:1], audio[1:] - preemphasis * audio[:-1]]
        )
    # Center-trim or zero-pad to fft_size.
    n = audio.shape[0]
    if n >= fft_size:
        start = (n - fft_size) // 2
        seg = audio[start : start + fft_size]
    else:
        seg = jnp.concatenate([audio, jnp.zeros((fft_size - n,), dtype=audio.dtype)])
    win = jnp.asarray(np.hanning(fft_size), dtype=audio.dtype)
    seg = seg * win
    spec = jnp.fft.rfft(seg)
    log_mag = jnp.log(jnp.abs(spec) + 1e-12)
    # Convert to dB: log_mag is natural log; dB = 20*log10 = log_mag * (20/ln(10)).
    log_mag_dB = log_mag * (20.0 / float(np.log(10.0)))
    # Band-restrict via precomputed mask.
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    mask = (freqs >= f_min_Hz) & (freqs <= f_max_Hz)
    band_log_mag_dB = log_mag_dB[mask]
    # Peak-normalize so loss is on shape, not absolute level.
    band_log_mag_dB = band_log_mag_dB - jnp.max(band_log_mag_dB)
    target = jnp.asarray(target_log_psd_dB, dtype=band_log_mag_dB.dtype)
    return jnp.mean((band_log_mag_dB - target) ** 2)


# --------------------------------------------------------------------------
# Reference targets (literature-derived).
# --------------------------------------------------------------------------


# Adult-male formant targets (Peterson & Barney 1952 / Hillenbrand 1995 means).
VOWEL_FORMANTS_MALE: dict[str, tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),
    "i": (270.0, 2290.0, 3010.0),
    "u": (300.0, 870.0, 2240.0),
    "e": (530.0, 1840.0, 2480.0),
    "o": (570.0, 840.0, 2410.0),
    "schwa": (500.0, 1500.0, 2500.0),
}


# Fricative spectral-shape targets. Ranges, not point values, because
# fricative spectra are noise-like and have natural variability.
FRICATIVE_SHAPES: dict[str, dict[str, tuple[float, float]]] = {
    "s": {
        "peak_Hz": (4000.0, 7500.0),
        "centroid_Hz": (4500.0, 7000.0),
    },
    "sh": {
        "peak_Hz": (2000.0, 4000.0),
        "centroid_Hz": (2800.0, 4500.0),
    },
    "f": {
        "peak_Hz": (1500.0, 4000.0),
        "centroid_Hz": (2500.0, 4500.0),
    },
    "th": {
        "peak_Hz": (5000.0, 8000.0),
        "centroid_Hz": (5000.0, 7000.0),
    },
    "h": {
        "peak_Hz": (500.0, 2500.0),
        "centroid_Hz": (1500.0, 3500.0),
    },
}
