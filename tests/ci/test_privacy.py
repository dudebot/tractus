"""Privacy guard regression tests use isolated repositories, never user files."""
from __future__ import annotations

import importlib.util
import hashlib
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/ci/lint_privacy.py"
SPEC = importlib.util.spec_from_file_location("privacy_guard", SOURCE)
privacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(privacy)


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(privacy, "BLOCKED_TOKEN_HASHES", {hashlib.sha256(b"privatefixture").hexdigest()})
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.invalid")
    return tmp_path


@pytest.mark.parametrize("payload", [
    b"/" + b"home/fixture/project",
    b"C:" + b"\\Users\\fixture\\project",
    b"192." + b"168.23.4",
    b"100." + b"72.1.2",
    b"ghp_" + b"a" * 32,
    b"privatefixture",
])
def test_rejects_private_payload_without_echoing_it(repo: Path, payload: bytes) -> None:
    (repo / "notes.txt").write_bytes(payload)
    git(repo, "add", "notes.txt")
    hits = privacy.scan(repo)
    assert hits
    assert payload.decode() not in repr(hits)


def test_history_scan_catches_deleted_leak(repo: Path) -> None:
    (repo / "notes.txt").write_bytes(b"/" + b"home/fixture/project")
    git(repo, "add", "notes.txt")
    git(repo, "commit", "-m", "fixture")
    git(repo, "rm", "notes.txt")
    git(repo, "commit", "-m", "remove")
    assert privacy.scan(repo) == []
    assert privacy.scan(repo, all_history=True)


def test_rejects_inventory_path(repo: Path) -> None:
    (repo / "machines").mkdir()
    (repo / "machines/fixture.md").write_text("inventory")
    git(repo, "add", "machines/fixture.md")
    assert privacy.scan(repo)


def test_public_urls_and_package_versions_are_allowed(repo: Path) -> None:
    (repo / "links.txt").write_text("https://api.github.com/users/example\nhttps://arxiv.org/abs/2406.12998")
    (repo / "uv.lock").write_text('version = "' + '10.' + '4.0.35"')
    git(repo, "add", ".")
    assert privacy.scan(repo) == []


def test_lockfile_exception_does_not_hide_credentials(repo: Path) -> None:
    (repo / "uv.lock").write_bytes(b"github_pat_" + b"a" * 30)
    git(repo, "add", "uv.lock")
    assert privacy.scan(repo)


def test_staged_leak_is_checked_even_if_worktree_is_clean(repo: Path) -> None:
    (repo / "notes.txt").write_bytes(b"/" + b"home/fixture/project")
    git(repo, "add", "notes.txt")
    (repo / "notes.txt").write_text("clean working copy")
    assert privacy.scan(repo)


def test_lockfile_private_url_host_is_rejected(repo: Path) -> None:
    (repo / "uv.lock").write_text('url = "https://' + '192.' + '168.23.4/package"')
    git(repo, "add", "uv.lock")
    assert privacy.scan(repo)
