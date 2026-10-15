"""Rule-based text -> audio smoke pipeline (P1, smoke/001).

Front-half scaffolding that lets the simulator produce *something*
audible from text before any learned modules exist:

    text  -> phonemes (espeak IPA -> ARPAbet)
          -> per-phone durations (rule-based JSON table)
          -> per-phone GestureEvents (template inventory below)
          -> 30-d trajectory (T, 30) by stamping channel targets
             across each phone's hold/release window
          -> R1 NN body model + TEA dynamic-tube renderer (TEA native
             44.1 kHz)
          -> WAV + sidecar JSON

Module-2 (controller) and Module-3 (body) coarticulation /
async-overlap composition is owned by ``correctness/004`` and
``pretraining/003``; smoke uses a single anchor gesture per phone with
no overlap. See ARCHITECTURE.md §3 (gesture-as-the-data-structure) and
the smoke task at ``workplan/tasks/02_smoke/001_text_audio_smoke.md``.

The intelligibility-WER gate (Whisper) is smoke/002 — explicitly NOT
run from this pipeline. P1 closes only when P0 closes.
"""

from .pipeline import (  # noqa: F401
    SmokeConfig,
    SmokeRender,
    render_text_to_audio,
)
from .templates import (  # noqa: F401
    NEUTRAL_TRAJECTORY,
    PHONE_TEMPLATES,
    PhoneTemplate,
)
