"""Hourly push of a paper bot's log folder to its GitHub branch, shared by all the bots."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

log = logging.getLogger("kalshi_btc_bot")

STALE_LOCK_SECS = 120


def git_push_logs(out: Path, message: str, unstage: list[str] | tuple[str, ...] = ()) -> None:
    """git add / commit / push of `out`. Raises RuntimeError with git's own message on failure.

    A bot killed in the middle of a push (restart, hard reset) leaves .git/index.lock behind, and every
    later `git add` fails with exit 128 until it is removed; a lock older than two minutes cannot belong
    to a running git, so it is cleared first. Paths in `unstage` (a file still being written) stay out
    of the commit."""
    git = ["git", "-C", str(out)]
    # the bot's folder is usually a subfolder of the log repo (/opt/kalshi-logs-range5/range5 inside
    # /opt/kalshi-logs-range5), so the lock lives in the repo's own .git, wherever that is
    try:
        gd = subprocess.run(git + ["rev-parse", "--git-dir"], capture_output=True, text=True, timeout=30)
        gitdir = Path(gd.stdout.strip()) if gd.returncode == 0 and gd.stdout.strip() else Path(".git")
        if not gitdir.is_absolute():
            gitdir = out / gitdir
        lock = gitdir / "index.lock"
        if lock.exists() and time.time() - lock.stat().st_mtime > STALE_LOCK_SECS:
            lock.unlink()
            log.warning("removed stale %s", lock)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("could not check the git lock: %s", e)

    def run(*args: str, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
        r = subprocess.run(git + list(args), capture_output=True, text=True, timeout=timeout)
        if check and r.returncode:
            raise RuntimeError(f"git {args[0]} exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
        return r

    run("add", "-A", ".")
    for p in unstage:
        run("reset", "-q", "--", p, check=False)
    run("commit", "-qm", message, check=False)   # nothing to commit is fine
    run("push", "-q", "origin", "HEAD", timeout=120)
