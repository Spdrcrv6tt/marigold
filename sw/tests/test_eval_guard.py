"""The held-out guard covers every eval set and is not fooled by apostrophes (review S3, S4). Run: pytest sw/tests

Written by Mythos (deoxylabs/marigold PR 1, Nov 17, 2026) and kept as written; renamed from test_textfilter.py,
which would clash with sw/test_textfilter.py in one pytest run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import EvalFilter  # noqa: E402


def first_line(path, containing=""):
    """The text of the first utterance in an eval file (without its ' | values' tail) that contains `containing`."""
    for line in open(ROOT / path, encoding="utf8"):
        text = line.strip().partition(" | ")[0]
        if text and not text.startswith("#") and containing in text:
            return text
    raise AssertionError(f"no line with {containing!r} in {path}")


def test_drops_a_line_of_the_blind_test_set_v3():
    assert not EvalFilter().ok(first_line("eval/text_v3/paraphrases.txt"))
    assert not EvalFilter().ok(first_line("eval/text_v3/noncommands.txt"))


def test_drops_a_line_of_the_calibration_set_dev2():
    assert not EvalFilter().ok(first_line("eval/text_dev2/paraphrases.txt"))
    assert not EvalFilter().ok(first_line("eval/text_dev2/noncommands.txt"))


def test_curly_apostrophe_matches_the_straight_one():
    line = first_line("eval/text/noncommands.txt", containing="'")
    assert not EvalFilter().ok(line.replace("'", "’"))


def test_keeps_a_line_that_is_in_no_eval_set():
    assert EvalFilter().ok("please recalibrate the left turbine at dawn")
