"""tractus-controller CLI. Subcommands defined here; heavy lifting in modules."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import __version__
from . import doctor as _doctor

app = typer.Typer(
    name="tractus-controller",
    help="Differentiable physics-first TTS controller.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def doctor(
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if any required check fails."
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color in output."),
) -> None:
    """Diagnose the environment (OS, Python, torch/CUDA, espeak-ng, libsndfile)."""
    results = _doctor.run_checks()
    typer.echo(_doctor.report(results, use_color=not no_color))
    if _doctor.has_failures(results, strict=strict):
        raise typer.Exit(code=1)


@app.command()
def phonemize(
    text: str = typer.Argument(..., help="Text to phonemize."),
    language: str = typer.Option("en-us", "--language", "-l", help="eSpeak language code."),
) -> None:
    """Phonemize text via eSpeak NG."""
    try:
        from .phonemes import phonemize as phon_mod
    except ImportError as e:
        typer.echo(f"phonemes module not available: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(phon_mod.phonemize(text, language=language))


@app.command()
def smoke(
    text: str = typer.Option(..., "--text", "-t", help="Text to render."),
    out: Path = typer.Option(..., "--out", "-o", help="Output WAV path."),
    sample_rate: int = typer.Option(
        22050, "--sample-rate", "-r",
        help="Output WAV sample rate (Hz). TEA renders at native 44.1 kHz "
             "and is resampled at output stage.",
    ),
    f0_hz: float = typer.Option(110.0, "--f0", help="Reference F0 (Hz)."),
    duration_table: Path | None = typer.Option(
        None, "--durations",
        help="Per-phone duration override JSON (seconds).",
    ),
    seed: int = typer.Option(0, "--seed", help="Deterministic seed."),
    consent_verified: bool = typer.Option(
        True, "--consent-verified/--no-consent-verified",
        help="Sidecar consent flag. Defaults true (smoke renders synthetic "
             "speech from rule-based templates, no real-speaker data).",
    ),
) -> None:
    """Rule-based text -> audio smoke (P1 front-half scaffolding).

    NOTE: P1 acceptance (Whisper-WER intelligibility gate) is gated on
    P0 close (correctness/004 + correctness/008 are still open). This
    verb produces a WAV + sidecar but does NOT claim P1 has passed.
    """
    try:
        from .smoke.pipeline import (
            SmokeConfig,
            render_text_to_audio,
            write_sidecar,
            write_wav,
        )
    except ImportError as e:
        typer.echo(f"smoke pipeline not available: {e}", err=True)
        raise typer.Exit(code=2) from e

    cfg = SmokeConfig(
        sample_rate=sample_rate,
        f0_hz=f0_hz,
        duration_table=duration_table,
        deterministic_seed=seed,
    )
    render = render_text_to_audio(text, cfg)
    wav_path = write_wav(out, render.audio, render.sample_rate)
    sidecar_path = write_sidecar(
        wav_path, render, consent_verified=consent_verified,
    )
    rms = float((render.audio.astype("float64") ** 2).mean() ** 0.5)
    typer.echo(
        f"wrote {wav_path} ({len(render.audio)} samp @ {render.sample_rate} Hz, "
        f"RMS={rms:.4f})"
    )
    typer.echo(f"sidecar -> {sidecar_path}")


@app.command(name="prepare-manifest")
def prepare_manifest(
    manifest_in: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    manifest_out: Path = typer.Argument(...),
) -> None:
    """Read/validate/round-trip a JSONL manifest."""
    try:
        from tractus_shared import manifest_natural as manifest_mod
    except ImportError as e:
        typer.echo(f"tractus_shared.manifest_natural module not available: {e}", err=True)
        raise typer.Exit(code=2) from e
    n = manifest_mod.round_trip(manifest_in, manifest_out)
    typer.echo(f"wrote {n} entries to {manifest_out}")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app() or 0)
