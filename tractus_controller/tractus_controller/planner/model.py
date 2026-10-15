"""Small transformer planner.

Architecture (v0): FastSpeech-style phone encoder + length regulator +
frame decoder. Heads: per-phone duration, per-phone phrase-break, per-
frame log-F0 + voiced + loudness.

Sizing target: ≤ 10 M params. Default config sits at ~3.5 M trainable
params (verified by ``planner_param_count``); leaves ~2.6× headroom for
later ablations without crossing the cap.

Affect (D_affect = 8) is projected to ``d_model`` and added to every
phone embedding. Speaker F0 *range* (low/high in Hz) is encoded as a
2-scalar projection added to the same input. NO speaker_embed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokens import AFFECT_DIM, PAD_ID, STRESS_LEVELS, VOCAB_SIZE


@dataclass
class PlannerConfig:
    d_model: int = 192
    n_heads: int = 4
    ffn_dim: int = 768
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1
    max_phone_len: int = 256
    max_frame_len: int = 2048  # ~20 s at 100 Hz
    affect_dim: int = AFFECT_DIM
    vocab_size: int = VOCAB_SIZE
    stress_levels: int = STRESS_LEVELS
    pad_id: int = PAD_ID
    # Per-phone log-duration normalisation. We Huber-regress
    # log(dur_s + 1e-3); the head emits a single scalar in that space.
    log_dur_eps: float = 1e-3


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T]


class _TxLayerStack(nn.Module):
    """Pre-LN transformer encoder stack (used for both encoder + decoder).

    We don't need cross-attention; the decoder is an autoregressive-ish
    frame model conditioned on a length-regulated phone sequence (no
    causal mask at training time — F0/loud envelopes are predicted
    densely).
    """

    def __init__(self, cfg: PlannerConfig, n_layers: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.stack = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        return self.stack(x, src_key_padding_mask=key_padding_mask)


def length_regulator(
    encoded: torch.Tensor,  # [B, L_phone, d]
    durations_frames: torch.Tensor,  # [B, L_phone] long; sum per-row = T_frame
    max_frame_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each phone embedding ``durations_frames[i]`` times.

    Returns ``(expanded[B, T_max, d], frame_mask[B, T_max])`` where
    ``frame_mask`` is True for valid (non-padding) frames. ``T_max`` =
    max(sum_i durations_frames[b, i]) clipped to ``max_frame_len``.
    """
    B, L, D = encoded.shape
    device = encoded.device
    durs = durations_frames.clamp(min=0).long()
    total = durs.sum(dim=1).clamp(max=max_frame_len)
    T_max = int(total.max().item()) if total.numel() else 0
    T_max = max(T_max, 1)
    out = encoded.new_zeros(B, T_max, D)
    mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    for b in range(B):
        cursor = 0
        for i in range(L):
            d = int(durs[b, i].item())
            if d <= 0:
                continue
            end = min(cursor + d, T_max)
            if end <= cursor:
                break
            out[b, cursor:end] = encoded[b, i]
            cursor = end
        mask[b, :cursor] = True
    return out, mask


@dataclass
class PlannerOutput:
    log_dur_pred: torch.Tensor  # [B, L_phone]
    phrase_break_logit: torch.Tensor  # [B, L_phone]
    log_f0_pred: torch.Tensor  # [B, T_frame]
    voiced_logit: torch.Tensor  # [B, T_frame]
    loud_db_pred: torch.Tensor  # [B, T_frame]
    frame_mask: torch.Tensor  # [B, T_frame] bool

    def __iter__(self):
        return iter(
            (
                self.log_dur_pred,
                self.phrase_break_logit,
                self.log_f0_pred,
                self.voiced_logit,
                self.loud_db_pred,
                self.frame_mask,
            )
        )


class PlannerModel(nn.Module):
    """Module-1 planner. ARCHITECTURE.md §2.1.

    Input:
        token_ids:    long[B, L]     # phone vocab ids (incl. <bos>/<eos>/<sil>)
        stress:       long[B, L]     # 0/1/2
        affect:       float[B, D_affect]  # 8-dim; zeros = neutral
        speaker_f0_range_hz: float[B, 2]  # [low, high] Hz; not log
        durations_frames: long[B, L] | None   # if None → use predicted

    Output: per-phone log-dur + phrase-break + per-frame log-F0/voiced/loud.
    """

    def __init__(self, cfg: PlannerConfig | None = None):
        super().__init__()
        self.cfg = cfg or PlannerConfig()
        c = self.cfg
        self.token_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=c.pad_id)
        self.stress_emb = nn.Embedding(c.stress_levels, c.d_model)
        self.affect_proj = nn.Linear(c.affect_dim, c.d_model)
        self.f0_range_proj = nn.Linear(2, c.d_model)
        self.encoder_pe = _SinusoidalPE(c.d_model, c.max_phone_len)
        self.decoder_pe = _SinusoidalPE(c.d_model, c.max_frame_len)
        self.encoder = _TxLayerStack(c, c.n_encoder_layers)
        self.decoder = _TxLayerStack(c, c.n_decoder_layers)
        # heads
        self.dur_head = nn.Linear(c.d_model, 1)
        self.phrase_head = nn.Linear(c.d_model, 1)
        self.f0_head = nn.Linear(c.d_model, 1)
        self.voiced_head = nn.Linear(c.d_model, 1)
        self.loud_head = nn.Linear(c.d_model, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        stress: torch.Tensor,
        affect: torch.Tensor,
        speaker_f0_range_hz: torch.Tensor,
        durations_frames: torch.Tensor | None = None,
    ) -> PlannerOutput:
        c = self.cfg
        B, L = token_ids.shape
        # ---- encoder input -------------------------------------------------
        x = self.token_emb(token_ids) + self.stress_emb(stress)
        log_range = torch.log(speaker_f0_range_hz.clamp(min=1.0))
        bias = self.affect_proj(affect) + self.f0_range_proj(log_range)  # [B, d]
        x = x + bias.unsqueeze(1)
        x = self.encoder_pe(x)
        pad_mask = token_ids == c.pad_id  # [B, L] True = pad
        enc = self.encoder(x, key_padding_mask=pad_mask)
        # ---- per-phone heads ----------------------------------------------
        log_dur_pred = self.dur_head(enc).squeeze(-1)  # [B, L]
        phrase_break_logit = self.phrase_head(enc).squeeze(-1)  # [B, L]
        # ---- length regulator → frame decoder -----------------------------
        if durations_frames is None:
            # Inference: derive frames from predicted durations.
            dur_s = torch.exp(log_dur_pred) - c.log_dur_eps
            dur_s = dur_s.clamp(min=0.0)
            durations_frames = (dur_s * 100.0).round().long()  # 100 Hz
            durations_frames = durations_frames.masked_fill(pad_mask, 0)
        expanded, frame_mask = length_regulator(
            enc, durations_frames, c.max_frame_len
        )
        expanded = self.decoder_pe(expanded)
        # decoder padding mask is the inverse of frame_mask
        dec = self.decoder(expanded, key_padding_mask=~frame_mask)
        log_f0_pred = self.f0_head(dec).squeeze(-1)
        voiced_logit = self.voiced_head(dec).squeeze(-1)
        loud_db_pred = self.loud_head(dec).squeeze(-1)
        return PlannerOutput(
            log_dur_pred=log_dur_pred,
            phrase_break_logit=phrase_break_logit,
            log_f0_pred=log_f0_pred,
            voiced_logit=voiced_logit,
            loud_db_pred=loud_db_pred,
            frame_mask=frame_mask,
        )


def planner_param_count(cfg: PlannerConfig | None = None) -> int:
    """Trainable parameter count for the default (or supplied) config."""
    model = PlannerModel(cfg)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
