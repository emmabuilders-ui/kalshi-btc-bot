"""Hourly log push: a stale index.lock from a killed git is cleared, and git's own error is reported."""
import os
import subprocess
import time

import pytest

from kalshi_btc_bot.logpush import git_push_logs


def _git(*a, cwd):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    _git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    # as on the droplet: the repo root is /opt/kalshi-logs-x and the bot writes to its subfolder /opt/kalshi-logs-x/x
    root = tmp_path / "logs"
    out = root / "range5"
    out.mkdir(parents=True)
    _git("init", "-q", "-b", "paper-logs", cwd=root)
    _git("config", "user.email", "kbot@localhost", cwd=root)
    _git("config", "user.name", "kbot", cwd=root)
    _git("remote", "add", "origin", str(origin), cwd=root)
    (out / "trades.csv").write_text("a,b\n")
    return out, origin


def test_stale_lock_is_cleared_and_logs_are_pushed(repo):
    out, origin = repo
    lock = out.parent / ".git" / "index.lock"   # in the repo root, not the bot's subfolder
    lock.write_text("")
    old = time.time() - 600
    os.utime(lock, (old, old))
    git_push_logs(out, "paper log test")
    assert not lock.exists()
    assert "paper log test" in _git("log", "--oneline", "paper-logs", cwd=origin).stdout


def test_fresh_lock_is_left_alone_and_the_error_names_git(repo):
    out, _ = repo
    (out.parent / ".git" / "index.lock").write_text("")
    with pytest.raises(RuntimeError, match="git add exit 128.*index.lock"):
        git_push_logs(out, "paper log test")


def test_unstaged_file_stays_out_of_the_commit(repo):
    out, origin = repo
    (out / "prices").mkdir()
    (out / "prices" / "open.csv.gz").write_bytes(b"x")
    git_push_logs(out, "paper log test", ["prices/open.csv.gz"])
    files = _git("ls-tree", "-r", "--name-only", "paper-logs", cwd=origin).stdout.split()
    assert files == ["range5/trades.csv"]
