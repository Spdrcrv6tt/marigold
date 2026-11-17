"""Mythos's test (deoxylabs/marigold PR 1, Nov 17, 2026), adapted to master (marked ADAPTED): master's script takes MG_ROOT and MG_PY (not MARIGOLD_ROOT / MARIGOLD_PY) and ends its log with 'success 0' or 'success 1' (not '=== FAILED').

sw/intent/run_resume.sh tells its caller whether a run finished (review S5), and shell scripts keep Unix line
endings (review S9). Run: pytest sw/tests

The real script runs in an empty copy of the tree with a stand-in for the Python interpreter (MARIGOLD_PY) that
exits with the next code from a list, so only the script's own control flow is under test.
"""
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TAG = "t9"
STUB_PY = r'''#!/bin/bash
# stand-in for python: logs its arguments, leaves a checkpoint, exits with the next code in $STUB_CODES
echo "train $*" >> "$STUB_LOG"
: > "data/intent/t9.pt"
code=$(head -1 "$STUB_CODES"); rest=$(tail -n +2 "$STUB_CODES")
[ -n "$rest" ] && echo "$rest" > "$STUB_CODES"
exit ${code:-0}
'''


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "marigold"
    (root / "sw/intent").mkdir(parents=True)
    (root / "sw/intent/run_resume.sh").write_bytes((ROOT / "sw/intent/run_resume.sh").read_bytes())
    (root / "data/intent").mkdir(parents=True)
    (root / "data/logs").mkdir(parents=True)
    py = tmp_path / "python"
    py.write_text(STUB_PY)
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    return root


def run(root, codes, seconds_left=124):
    """Runs the script with `seconds_left` to its deadline; it stops starting runs 120 s before the deadline."""
    (root / "codes").write_text("\n".join(str(c) for c in codes) + "\n")
    (root / "stub.log").write_text("")
    env = dict(os.environ, MG_ROOT=str(root), MG_PY=str(root.parent / "python"),
               STUB_LOG=str(root / "stub.log"), STUB_CODES=str(root / "codes"))
    out = subprocess.run(["bash", str(root / "sw/intent/run_resume.sh"), TAG, "data/intent/start.pt",
                          str(int(time.time()) + seconds_left), "--qat"], env=env, capture_output=True, text=True, timeout=60)
    return out, (root / "stub.log").read_text().splitlines(), (root / f"data/logs/train_{TAG}.log").read_text()


def test_a_run_that_finishes_is_a_success(tree):
    out, starts, log = run(tree, [0])
    assert out.returncode == 0
    assert len(starts) == 1 and "--init data/intent/start.pt" in starts[0] and "--qat" in starts[0]
    assert "=== finished" in log


def test_a_crash_restarts_from_the_last_checkpoint(tree):
    out, starts, log = run(tree, [1, 0])
    assert out.returncode == 0
    assert len(starts) == 2 and f"--init data/intent/{TAG}-resume.pt" in starts[1]


def test_crashing_until_the_deadline_is_a_failure(tree):
    out, starts, log = run(tree, [1], seconds_left=122)
    assert out.returncode != 0
    assert len(starts) >= 1
    assert "success 0" in log and "success 1" not in log


def test_a_deadline_already_too_close_is_a_failure(tree):
    out, starts, log = run(tree, [0], seconds_left=60)
    assert out.returncode != 0
    assert starts == []
    assert "success 0" in log


def test_a_throughput_abort_is_passed_on(tree):
    out, starts, log = run(tree, [3])
    assert out.returncode == 3 and len(starts) == 1


def test_shell_scripts_have_unix_line_endings():
    listed = subprocess.run(["git", "ls-files", "*.sh"], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode == 0 and listed.stdout.split():
        scripts = listed.stdout.split()
    else:                                           # an exported tree without .git: every script outside data/
        scripts = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.sh") if "data" not in p.relative_to(ROOT).parts[:1]]
    assert scripts
    assert [s for s in scripts if b"\r" in (ROOT / s).read_bytes()] == []
