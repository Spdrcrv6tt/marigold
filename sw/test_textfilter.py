"""Review S3 / S4 checks for sw/textfilter.py (the training-text leak filter).
Run: python sw/test_textfilter.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw"))
import textfilter as TF  # noqa: E402


def first_line(rel):
    for line in open(ROOT / rel, encoding="utf8"):
        if line.strip() and not line.startswith("#"):
            return line.partition(" | ")[0].strip()


def test_curly_apostrophe():
    assert TF.norm("I don’t know.") == TF.norm("i don't know") == "i don't know"
    assert TF.norm("‘okay’") == "'okay'"


def test_every_eval_set_is_covered():
    f = TF.EvalFilter()
    for rel in ("eval/text/paraphrases.txt", "eval/text_v2/paraphrases.txt", "eval/text_v3/paraphrases.txt",
                "eval/text_dev2/paraphrases.txt", "eval/text_v3/noncommands.txt", "eval/text_dev2/noncommands.txt"):
        t = first_line(rel)
        assert not f.ok(t), (rel, t)
        assert not f.ok(t.replace("'", "’")), (rel, "curly")
    assert f.ok("please defrost the walrus at noon")


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
