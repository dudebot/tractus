"""Build per-frame frication inputs from the smoke gesture list.

The body model emits ``place_index[T, 93]``, ``obstacle_proxy[T, 93]``,
``voicing[T]``, and (optionally) ``frication_color[T, 93]`` directly in
production. The rule-based smoke pipeline doesn't have a body model; this
module synthesises the same fields from the per-phone GestureEvent
templates so the smoke renderer can exercise correctness/004's frication
module end-to-end.

Only fricatives, aspiration, and stop bursts emit a non-NONE
``place_index``. All other phones leave the slot as PLACE_NONE → zero
gain at every section. The articulator-class section ranges are
hand-tuned to the standard 93-section topology (trachea 0-22, glottis
23-24, pharynx 25-40, mouth 41-64, nose 65-83, fossa 84-88, sinus 89-92):

* glottal /h/ + aspiration → sections 23-26 (glottis + first pharynx).
* velar /x/ + velar bursts /k/ /g/ → sections 38-44 (back of mouth).
* post-alveolar /ʃ/ /ʒ/ → sections 50-56 (mid mouth, blade-region).
* alveolar /s/ /z/ + alveolar bursts /t/ /d/ → sections 56-62 (front mouth).
* dental /θ/ /ð/ → sections 60-63.
* labio-dental /f/ /v/ → sections 62-64 (lip aperture).
* bilabial bursts /p/ /b/ → sections 62-64 with PLACE_BURST.

These ranges are smoke-grade approximations, not phonetics-grade
articulator-to-section mappings. The body model + R1 NN provide the
production mapping (correctness/004's body-model output channel).

obstacle_proxy is 1.0 at sibilant places (/s/ /z/ /ʃ/ /ʒ/) and during
labio-dental fricatives (lower-lip-against-upper-teeth obstacle), and
0.0 elsewhere.

voicing is 1.0 during voiced fricatives (/v/ /z/ /ʒ/ /ð/) and during
the prevoicing portion of voiced stops; 0.0 during voiceless segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..gestures import GestureEvent
from ..phonemes.arpabet import split_stress

# Place codes — must mirror tractus_engine_advanced.frication.PLACE_*.
# We re-export here so the smoke pipeline doesn't need to import TEA.
PLACE_NONE = 0
PLACE_LABIO_DENTAL = 1
PLACE_DENTAL = 2
PLACE_ALVEOLAR = 3
PLACE_POST_ALVEOLAR = 4
PLACE_VELAR = 5
PLACE_GLOTTAL = 6
PLACE_BURST = 7

# Per-phone (place, section_range_inclusive, obstacle, voicing, is_burst).
# section_range is half-open Python-slice-style for convenience.
@dataclass(frozen=True)
class FricationProfile:
    place: int
    sections: tuple[int, int]   # half-open
    obstacle: float
    voicing: float
    burst: bool = False         # True → only the first 30 ms of the phone


_FRICATION_PROFILES: dict[str, FricationProfile] = {
    # Voiceless fricatives.
    "S":  FricationProfile(PLACE_ALVEOLAR,      (56, 62), obstacle=1.0, voicing=0.0),
    "SH": FricationProfile(PLACE_POST_ALVEOLAR, (50, 56), obstacle=1.0, voicing=0.0),
    "F":  FricationProfile(PLACE_LABIO_DENTAL,  (62, 65), obstacle=1.0, voicing=0.0),
    "TH": FricationProfile(PLACE_DENTAL,        (60, 63), obstacle=0.5, voicing=0.0),
    "HH": FricationProfile(PLACE_GLOTTAL,       (23, 27), obstacle=0.0, voicing=0.0),
    # Voiced fricatives.
    "Z":  FricationProfile(PLACE_ALVEOLAR,      (56, 62), obstacle=1.0, voicing=1.0),
    "ZH": FricationProfile(PLACE_POST_ALVEOLAR, (50, 56), obstacle=1.0, voicing=1.0),
    "V":  FricationProfile(PLACE_LABIO_DENTAL,  (62, 65), obstacle=1.0, voicing=1.0),
    "DH": FricationProfile(PLACE_DENTAL,        (60, 63), obstacle=0.5, voicing=1.0),
    # Affricate components — release portion only (handled like a fricative).
    "CH": FricationProfile(PLACE_POST_ALVEOLAR, (50, 56), obstacle=1.0, voicing=0.0),
    "JH": FricationProfile(PLACE_POST_ALVEOLAR, (50, 56), obstacle=1.0, voicing=1.0),
    # Stop bursts — short transient at release. The burst window is only
    # the first ~30 ms after onset; sustained stops have zero gain.
    "P":  FricationProfile(PLACE_BURST, (62, 65), obstacle=0.5, voicing=0.0, burst=True),
    "T":  FricationProfile(PLACE_BURST, (56, 62), obstacle=0.8, voicing=0.0, burst=True),
    "K":  FricationProfile(PLACE_BURST, (38, 44), obstacle=0.4, voicing=0.0, burst=True),
    "B":  FricationProfile(PLACE_BURST, (62, 65), obstacle=0.5, voicing=1.0, burst=True),
    "D":  FricationProfile(PLACE_BURST, (56, 62), obstacle=0.8, voicing=1.0, burst=True),
    "G":  FricationProfile(PLACE_BURST, (38, 44), obstacle=0.4, voicing=1.0, burst=True),
}


# Burst window in seconds: stops emit broadband noise only for the first
# ~25-35 ms of the segment (release transient). After that the residual
# resonances dominate.
BURST_WINDOW_S: float = 0.030


def _profile_for(token: str) -> FricationProfile | None:
    bare, _ = split_stress(token)
    return _FRICATION_PROFILES.get(bare)


def build_frication_inputs(
    arpabet_tokens: list[str],
    durations_s: list[float],
    *,
    onset_offset_s: float,
    total_duration_s: float,
    frame_rate_hz: int,
) -> dict:
    """Stamp per-frame ``place_index``, ``obstacle_proxy``, ``voicing``.

    Returns a dict shaped to match ``synthesize_via_tea(frication_inputs=...)``.
    The caller is responsible for adding the ``frication`` config and
    ``noise_key`` (seeded for reproducibility).

    Returns
    -------
    dict with keys:
        ``place_index``    int32 ndarray (T, 93)
        ``obstacle_proxy`` float32 ndarray (T, 93)
        ``voicing``        float32 ndarray (T,)
    """
    T = max(2, int(round(total_duration_s * frame_rate_hz)) + 1)
    place = np.zeros((T, 93), dtype=np.int32)  # PLACE_NONE
    obstacle = np.zeros((T, 93), dtype=np.float32)
    voicing = np.zeros((T,), dtype=np.float32)

    t = onset_offset_s
    for token, dur in zip(arpabet_tokens, durations_s, strict=True):
        prof = _profile_for(token)
        if prof is None:
            t += dur
            continue
        if prof.burst:
            f0 = max(0, int(round(t * frame_rate_hz)))
            f1 = min(T, int(round((t + min(BURST_WINDOW_S, dur)) * frame_rate_hz)))
        else:
            f0 = max(0, int(round(t * frame_rate_hz)))
            f1 = min(T, int(round((t + dur) * frame_rate_hz)))
        if f1 <= f0:
            f1 = min(T, f0 + 1)
        s0, s1 = prof.sections
        place[f0:f1, s0:s1] = prof.place
        obstacle[f0:f1, s0:s1] = prof.obstacle
        # Voicing is per-frame scalar. If multiple voiced/voiceless
        # phones overlap (they don't in smoke; gestures are sequential)
        # the last write wins.
        voicing[f0:f1] = prof.voicing
        t += dur
    return {
        "place_index": place,
        "obstacle_proxy": obstacle,
        "voicing": voicing,
    }


__all__ = [
    "PLACE_NONE",
    "PLACE_LABIO_DENTAL",
    "PLACE_DENTAL",
    "PLACE_ALVEOLAR",
    "PLACE_POST_ALVEOLAR",
    "PLACE_VELAR",
    "PLACE_GLOTTAL",
    "PLACE_BURST",
    "BURST_WINDOW_S",
    "FricationProfile",
    "build_frication_inputs",
]
