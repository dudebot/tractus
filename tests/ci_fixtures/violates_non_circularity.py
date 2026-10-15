"""DELIBERATELY VIOLATES non-circularity rule (infra/002 fixture).

The lint enforces the *grep contract* from the task spec literally:

    grep -rE "loss.*=.*synthesize_dynamic_tube|audio_loss.*body_model_output"

so the violation must happen on a single line.
"""

from tractus_engine_advanced.pipeline import synthesize_dynamic_tube  # noqa: F401


def train_step(body_model_output, target_audio):
    # Both grep patterns on one line each (lint catches either).
    loss = ((synthesize_dynamic_tube(body_model_output) - target_audio) ** 2).mean()
    audio_loss = ((body_model_output - target_audio) ** 2).mean()
    return loss + audio_loss
