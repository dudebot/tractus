#!/usr/bin/env python3
"""Reject private environment details in tracked files and optional Git history.

Offline best-effort guard: this cannot audit remote caches or other clones.
Reports paths and categories, never matching file contents or credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import os
import re
import subprocess
from pathlib import Path

BLOCKED_TOKEN_HASHES = set(filter(None, os.environ.get("TRACTUS_PRIVATE_IDENTIFIER_HASHES", "").split(",")))
PATTERNS = {
    "personal home path": re.compile(
        rb"(?<![A-Za-z0-9:/.])/(?:home|Users|root)/[A-Za-z0-9_.-]+|"
        rb"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+"
    ),
    "private session identifier": re.compile(rb"\b019e[0-9a-f]{4}-[0-9a-f-]{8,}\b", re.I),
    "credential": re.compile(
        rb"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE).*?KEY-----|"
        rb"\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|"
        rb"\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{24,}"
    ),
}
IP_RE = re.compile(rb"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
PRIVATE_NETWORKS = tuple(ipaddress.IPv4Network(item) for item in (
    (0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16), (0x64400000, 10),
))


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def categories(data: bytes, *, lockfile: bool = False, blocked_hashes: set[str] | None = None) -> set[str]:
    blocked_hashes = BLOCKED_TOKEN_HASHES if blocked_hashes is None else blocked_hashes
    found = {label for label, rx in PATTERNS.items() if rx.search(data)}
    for token in re.findall(rb"[a-z][a-z0-9]{2,}", data.lower()):
        if hashlib.sha256(token).hexdigest() in blocked_hashes:
            found.add("private identifier")
    # Package versions can look like IPs. Lockfile URL host addresses remain checked.
    for match in IP_RE.finditer(data):
        if not lockfile or data[max(0, match.start() - 3):match.start()] == b"://":
            try:
                addr = ipaddress.ip_address(match.group().decode("ascii"))
            except ValueError:
                continue
            if any(addr in network for network in PRIVATE_NETWORKS):
                found.add("private network address")
    return found


def scan(root: Path, all_history: bool = False) -> list[tuple[str, str]]:
    hits = set()
    blocked_hashes = set(BLOCKED_TOKEN_HASHES)
    local = subprocess.run(["git", "-C", str(root), "config", "--local", "--get", "privacy.blockedTokenHashes"], capture_output=True, text=True)
    if local.returncode == 0:
        blocked_hashes.update(filter(None, local.stdout.strip().split(",")))
    staged = git(root, "ls-files", "--stage", "-z").split(b"\0")
    entries = [item.split(b"\t", 1) for item in staged if item]
    request = b"".join(header.split()[1] + b"\n" for header, name in entries)
    result = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"], input=request, capture_output=True, check=True).stdout
    pos = 0
    for header, raw in entries:
        end = result.index(b"\n", pos)
        _, kind, size = result[pos:end].split()
        pos = end + 1
        data = result[pos:pos + int(size)]
        pos += int(size) + 1
        name = raw.decode("utf-8")
        for category in categories(raw + b"\n" + data, lockfile=name.endswith(".lock"), blocked_hashes=blocked_hashes):
            hits.add((name, category))
    for raw in git(root, "ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = root / name
        data = raw + b"\n"
        if path.is_symlink():
            data += str(path.readlink()).encode("utf-8")
        elif path.is_file():
            data += path.read_bytes()
        for category in categories(data, lockfile=name.endswith(".lock"), blocked_hashes=blocked_hashes):
            hits.add((name, category))
        if name.lower().startswith(("machines/", ".ssh/")) or path.name.lower() in {".env", "id_rsa", "id_ed25519"}:
            hits.add((name, "private operational file"))
    if all_history:
        # Object enumeration records only one name for shared blobs; inspect every
        # historical tree as well so duplicate contents cannot hide a private path.
        for commit in git(root, "rev-list", "--all").decode().splitlines():
            for raw in git(root, "ls-tree", "-r", "-z", "--name-only", commit).split(b"\0"):
                name = raw.decode("utf-8")
                for category in categories(raw, blocked_hashes=blocked_hashes):
                    hits.add((name, category))
                if name.lower().startswith(("machines/", ".ssh/")):
                    hits.add((name, "private operational file"))
        entries = git(root, "rev-list", "--objects", "--all").decode("utf-8").splitlines()
        request = "".join(line.split(" ", 1)[0] + "\n" for line in entries).encode()
        result = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"], input=request, capture_output=True, check=True).stdout
        pos = 0
        for entry in entries:
            end = result.index(b"\n", pos)
            oid, kind, size = result[pos:end].split()
            pos = end + 1
            data = result[pos:pos + int(size)]
            pos += int(size) + 1
            _, _, name = entry.partition(" ")
            if kind in (b"blob", b"commit", b"tag"):
                for category in categories(name.encode() + b"\n" + data, lockfile=name.endswith(".lock"), blocked_hashes=blocked_hashes):
                    hits.add((name or oid.decode(), category))
            if name.lower().startswith(("machines/", ".ssh/")):
                hits.add((name, "private operational file"))
    return sorted(hits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--all-history", action="store_true")
    args = parser.parse_args()
    hits = scan(args.root.resolve(), args.all_history)
    for path, category in hits:
        print(f"FAIL: {path}: {category}")
    if not hits:
        print("OK: privacy guard")
    return int(bool(hits))


if __name__ == "__main__":
    raise SystemExit(main())
