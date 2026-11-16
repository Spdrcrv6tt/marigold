"""Download candidate teacher bases (no account) into data/hf and print each model card's license line."""
import os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
from huggingface_hub import snapshot_download, hf_hub_download

for rid in sys.argv[1:]:
    card = Path(hf_hub_download(rid, "README.md"))
    lic = re.findall(r"^license:\s*(.+)$", card.read_text(encoding="utf8", errors="ignore"), re.M)
    print(rid, "license:", lic[:1], flush=True)
    p = snapshot_download(rid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "1_Pooling/*", "*.md", "tokenizer*", "merges.txt", "vocab*"],
                          ignore_patterns=["onnx/*", "openvino/*", "*.onnx", "*.bin", "*.pt", "*.h5", "*.msgpack"])
    sz = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9
    print(rid, "done", f"{sz:.1f} GB", p, flush=True)
