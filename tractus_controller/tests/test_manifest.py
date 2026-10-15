"""Phase-5 manifest: schema validation, JSONL round-trip, consent enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tractus_controller.cli import app
from tractus_shared import manifest_natural as m

runner = CliRunner()


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_round_trip_preserves_known_fields(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    _write_jsonl(
        src,
        [
            {
                "audio_path": "data/clips/0001.wav",
                "text": "hello world",
                "consent_verified": True,
                "phonemes": ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"],
                "speaker_id": "spk-001",
                "duration_sec": 1.234,
                "language": "en-us",
                "license": "CC0-1.0",
            },
            {
                "audio_path": "data/clips/0002.wav",
                "text": "another line",
                "consent_verified": True,
            },
        ],
    )

    out = tmp_path / "out.jsonl"
    n = m.round_trip(src, out)
    assert n == 2

    entries = m.read(out)
    assert len(entries) == 2
    assert entries[0].text == "hello world"
    assert entries[0].phonemes == ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"]
    assert entries[0].speaker_id == "spk-001"
    assert entries[1].audio_path == "data/clips/0002.wav"
    assert entries[1].phonemes is None


def test_consent_required_rejects_false(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    _write_jsonl(
        src, [{"audio_path": "x.wav", "text": "x", "consent_verified": False}]
    )
    with pytest.raises(m.ManifestError, match="consent_verified must be true"):
        m.read(src)


def test_consent_required_rejects_missing(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    _write_jsonl(src, [{"audio_path": "x.wav", "text": "x"}])
    with pytest.raises(m.ManifestError, match="missing required field"):
        m.read(src)


def test_unknown_fields_preserved_under_extra(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    _write_jsonl(
        src,
        [
            {
                "audio_path": "x.wav",
                "text": "x",
                "consent_verified": True,
                "future_field_we_havent_invented_yet": 42,
            }
        ],
    )
    [entry] = m.read(src)
    assert entry.extra["future_field_we_havent_invented_yet"] == 42

    out = tmp_path / "out.jsonl"
    m.write([entry], out)
    [round_tripped] = m.read(out)
    assert round_tripped.extra["future_field_we_havent_invented_yet"] == 42


def test_blank_lines_and_comments_skipped(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        "# this is a comment\n"
        "\n"
        + json.dumps({"audio_path": "a.wav", "text": "a", "consent_verified": True})
        + "\n"
        + "\n"
        + "# another\n"
        + json.dumps({"audio_path": "b.wav", "text": "b", "consent_verified": True})
        + "\n"
    )
    assert len(m.read(src)) == 2


def test_invalid_json_pinpoints_line(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps({"audio_path": "a.wav", "text": "a", "consent_verified": True}) + "\n"
        "not-json-at-all\n"
    )
    with pytest.raises(m.ManifestError, match="line 2"):
        m.read(src)


def test_iter_read_streams_one_entry_at_a_time(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    _write_jsonl(
        src,
        [
            {"audio_path": f"x{i}.wav", "text": f"x{i}", "consent_verified": True}
            for i in range(5)
        ],
    )
    entries = list(m.iter_read(src))
    assert [e.audio_path for e in entries] == [f"x{i}.wav" for i in range(5)]


def test_cli_prepare_manifest_round_trips(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    _write_jsonl(
        src,
        [
            {"audio_path": "a.wav", "text": "alpha", "consent_verified": True},
            {"audio_path": "b.wav", "text": "beta", "consent_verified": True},
        ],
    )
    out = tmp_path / "out.jsonl"
    result = runner.invoke(app, ["prepare-manifest", str(src), str(out)])
    assert result.exit_code == 0, result.output
    assert "wrote 2 entries" in result.output
    assert out.exists()
    assert len(m.read(out)) == 2
