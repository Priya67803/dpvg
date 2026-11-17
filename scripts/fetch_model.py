"""Restore models/all-MiniLM-L6-v2 (only needed if you removed the bundled weights). Verifies SHA-256."""
import hashlib, pathlib, sys, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.embeddings import MODEL_DIR, MODEL_SHA256
REV = "751bff37182d3f1213fa05d7196b954e230abad9"
URLS = {"model.onnx": f"https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/{REV}/onnx/model.onnx",
        "tokenizer.json": f"https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/{REV}/tokenizer.json"}
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for name, url in URLS.items():
    dst = MODEL_DIR / name
    urllib.request.urlretrieve(url, dst)
    got = hashlib.sha256(dst.read_bytes()).hexdigest()
    assert got == MODEL_SHA256[name], f"{name}: checksum mismatch ({got})"
    print("ok", dst)
