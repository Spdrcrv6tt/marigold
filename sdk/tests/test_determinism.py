"""The same config compiles to byte-identical blobs with the real teacher (Nov 17 fix for the bf16 run-to-run drift):
two separate processes, different caller batch sizes. Slow (loads the teacher twice): runs with MARIGOLD_SLOW=1.
Run: MARIGOLD_SLOW=1 python sdk/tests/test_determinism.py
"""
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk"))
import compile as C  # noqa: E402


def test_blob_bytes_identical_across_processes_and_batch_sizes():
    emb = C.newest_embedder()
    if os.environ.get("MARIGOLD_SLOW") != "1" or emb is None:
        print("SKIP test_blob_bytes_identical_across_processes_and_batch_sizes (needs MARIGOLD_SLOW=1 and a teacher)")
        return
    with tempfile.TemporaryDirectory() as d:
        blobs = []
        for bs in ("1000", "7"):
            out = Path(d) / f"fan-{bs}.mgc"
            r = subprocess.run([sys.executable, str(ROOT / "sdk/compile.py"), str(ROOT / "sdk/configs/fan.json"), "--embedder", str(emb), "-o", str(out),
                                "--enroll", "text"],   # the text path is what this guards (bf16 drift); audio: K2 tests
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env=dict(os.environ, MARIGOLD_EMBED_BATCH=bs, PYTHONIOENCODING="utf-8"))
            assert r.returncode == 0, r.stderr[-2000:]
            blobs.append(out.read_bytes())
        assert blobs[0] == blobs[1], "blob bytes differ between processes / batch sizes"
        print("identical blobs:", len(blobs[0]), "bytes, crc", C.parse(blobs[0])["crc32"])


if __name__ == "__main__":
    test_blob_bytes_identical_across_processes_and_batch_sizes()
    print("ok test_blob_bytes_identical_across_processes_and_batch_sizes")
