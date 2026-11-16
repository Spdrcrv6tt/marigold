"""Teacher v4 training text: train_text_v3.json (72 products) plus the remaining products of
sw/gate0b/products_v3.json, written by local qwen2.5:14b with the v3 prompts (sw/gate0b/gen_train_v3.py).
Never fans, fan-like blowers or microwaves (asserted). Output data/gate0b/train_text_v4.json (resumable).
Every loader re-applies the eval leak filter (sw/textfilter.py: eval/text and eval/text_v2)."""
import json, re, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw"))
import gate0b.gen_train_v3 as G3  # noqa: E402

prods = [k for k in json.load(open(ROOT / "sw/gate0b/products_v3.json")) if not k.startswith("_")]
bad = [p for p in prods if re.search(r"\b(fan|fans|microwave|blower)\b", p)]
assert not bad, bad
V4 = ROOT / "data/gate0b/train_text_v4.json"
if not V4.exists():
    shutil.copy(ROOT / "data/gate0b/train_text_v3.json", V4)
G3.OUT = V4
sys.argv = [sys.argv[0], "--model", "qwen2.5:14b", "--threads", "4"]
G3.main()
