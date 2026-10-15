"""Phase-1 smoke: package imports, CLI dispatches, doctor runs without crashing."""

from __future__ import annotations

from typer.testing import CliRunner

import tractus_controller
from tractus_controller import doctor as doctor_mod
from tractus_controller.cli import app

runner = CliRunner()


def test_package_version() -> None:
    assert tractus_controller.__version__


def test_cli_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for cmd in ["doctor", "phonemize", "prepare-manifest", "version"]:
        assert cmd in result.output, f"missing subcommand {cmd} in help"


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert tractus_controller.__version__ in result.output


def test_doctor_runs_non_strict() -> None:
    """Without --strict, doctor always exits 0 even if espeak is missing."""
    result = runner.invoke(app, ["doctor", "--no-color"])
    assert result.exit_code == 0, result.output
    # OS check is always green and always present.
    assert "os" in result.output


def test_doctor_check_results_have_tuple_shape() -> None:
    """Each check is (ok, value) — required by --strict scoring."""
    results = doctor_mod.run_checks()
    assert results, "no checks ran"
    for r in results:
        assert isinstance(r.ok, bool), r
        assert isinstance(r.value, str) and r.value, r


def test_doctor_strict_exits_nonzero_when_required_check_fails(tmp_path, monkeypatch) -> None:
    """If a required check is FAIL, --strict must exit non-zero."""
    fake_results = [
        doctor_mod.CheckResult("os", True, "ok", required=True),
        doctor_mod.CheckResult("required-missing", False, "boom", required=True),
    ]
    monkeypatch.setattr(doctor_mod, "run_checks", lambda: fake_results)
    result = runner.invoke(app, ["doctor", "--strict", "--no-color"])
    assert result.exit_code == 1, result.output


def test_doctor_strict_passes_when_only_informational_fails(monkeypatch) -> None:
    fake_results = [
        doctor_mod.CheckResult("os", True, "ok", required=True),
        doctor_mod.CheckResult("gpu", False, "no cuda", required=False),
    ]
    monkeypatch.setattr(doctor_mod, "run_checks", lambda: fake_results)
    result = runner.invoke(app, ["doctor", "--strict", "--no-color"])
    assert result.exit_code == 0, result.output
