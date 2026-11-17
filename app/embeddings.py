"""Embedding backends.

* OnnxMiniLMEmbedder - all-MiniLM-L6-v2 (384-d), the paper's model, run through onnxruntime from the weights
                       bundled in models/all-MiniLM-L6-v2/ (SHA-256 verified). No torch, no network, no download.
                       Verified to reproduce the sentence-transformers reference similarities (see tests).
* MiniLMEmbedder     - the same model through sentence-transformers (needs torch and a Hugging Face download).
"""
from __future__ import annotations
import hashlib, pathlib, re
from functools import lru_cache
import numpy as np
from . import config

MODEL_DIR = pathlib.Path(config.MODEL_DIR)
MODEL_SHA256 = {"model.onnx": "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e",
                "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"}

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OnnxMiniLMEmbedder:
    """all-MiniLM-L6-v2 via onnxruntime + the `tokenizers` WordPiece tokenizer (mean pooling, L2-normalised)."""
    name = "all-MiniLM-L6-v2"
    backend = "onnxruntime"
    default_tau = config.PAPER_TAU
    max_tokens = 256

    def __init__(self, model_dir: pathlib.Path = MODEL_DIR, verify: bool = True) -> None:
        import onnxruntime as ort                      # lazy: only needed for this backend
        from tokenizers import Tokenizer
        model_dir = pathlib.Path(model_dir)
        for fname, want in MODEL_SHA256.items():
            p = model_dir / fname
            if not p.exists():
                raise FileNotFoundError(f"{p} not found (run scripts/fetch_model.py)")
            if verify and _sha256_file(p) != want:
                raise ValueError(f"{p} failed its SHA-256 check - refusing to load a modified model")
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(self.max_tokens)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, (__import__("os").cpu_count() or 2)))
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self._names = {i.name for i in self.sess.get_inputs()}
        self._cache: dict[str, np.ndarray] = {}

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch([t if t.strip() else "." for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feed)[0]
        pooled = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
        return (pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            if len(self._cache) > 20000:
                self._cache.clear()
            v = self._cache[text] = self.encode_batch([text])[0]
        return v


class MiniLMEmbedder:
    name = "all-MiniLM-L6-v2"
    backend = "sentence-transformers"
    default_tau = config.PAPER_TAU

    def __init__(self, model) -> None:
        self.model = model
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)
        return self._cache[text]


@lru_cache(maxsize=4)
def get_embedder(kind: str | None = None):
    """onnx (default, bundled MiniLM) | st (sentence-transformers). The choice is explicit; if it cannot be loaded
    this raises - there is no automatic fallback to another embedder."""
    kind = (kind or config.EMBEDDER).lower()
    if kind in ("auto", "onnx", "minilm"):
        return OnnxMiniLMEmbedder()
    if kind == "st":
        from sentence_transformers import SentenceTransformer  # type: ignore
        return MiniLMEmbedder(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))
    raise ValueError(f"unknown embedder '{kind}' (onnx | st)")


def load_calibration() -> dict | None:
    try:
        import json
        return json.loads(pathlib.Path(config.CALIBRATION_FILE).read_text())
    except Exception:
        return None


def default_tau(embedder) -> float:
    """DPVG_TAU override > tau calibrated for THIS embedder (calibration.json, written by scripts/calibrate.py)
    > the paper default 0.45. A tau calibrated for one embedder is never applied to another."""
    if config.TAU_ENV:
        return float(config.TAU_ENV)
    cal = load_calibration()
    if cal and cal.get("embedder") == embedder.name and cal.get("tau") is not None:
        return float(cal["tau"])
    return float(embedder.default_tau)
