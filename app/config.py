from __future__ import annotations
import json
import os
from pathlib import Path

SPT_TTL = int(os.getenv("DPVG_SPT_TTL", "300"))       # seconds (paper default: 300)
MAX_DEPTH = int(os.getenv("DPVG_MAX_DEPTH", "6"))     # maximum delegation depth
TOKEN_BUDGET = int(os.getenv("DPVG_TOKEN_BUDGET", "3000"))
CONTEXT_SIMILARITY_CUTOFF = float(os.getenv("DPVG_CONTEXT_SIMILARITY_CUTOFF", "0.30"))
_CALIBRATION_PATH = os.getenv(
    "DPVG_CALIBRATION",
    str(Path(__file__).resolve().parent.parent / "calibration.json"),
)


def _calibrated_drift() -> dict:
    path = Path(_CALIBRATION_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("drift", {})


_DRIFT_CALIBRATION = _calibrated_drift()
DRIFT_THRESHOLD = float(os.getenv("DPVG_DRIFT_THRESHOLD", str(_DRIFT_CALIBRATION.get("tau_drift", 0.20))))
DRIFT_CONSECUTIVE_STEPS = int(os.getenv(
    "DPVG_DRIFT_CONSECUTIVE_STEPS", str(_DRIFT_CALIBRATION.get("consecutive_steps", 2)),
))
DRIFT_EMA_ALPHA = float(os.getenv("DPVG_DRIFT_EMA_ALPHA", str(_DRIFT_CALIBRATION.get("ema_alpha", 0.50))))
DRIFT_METRIC = os.getenv("DPVG_DRIFT_METRIC", "ema").lower()     # ema | mean
MAX_REPLANS = int(os.getenv("DPVG_MAX_REPLANS", "1"))
STRICT_MODE = os.getenv("DPVG_STRICT_MODE", "false").lower() in ("true", "1", "yes")
_LLM_SEED = os.getenv("DPVG_LLM_SEED")
LLM_SEED = int(_LLM_SEED) if _LLM_SEED else None
LLM_TEMPERATURE = float(os.getenv("DPVG_LLM_TEMPERATURE", "0"))
ISSUER = "dpvg-control-plane"
EMBEDDER = os.getenv("DPVG_EMBEDDER", "onnx")         # onnx (bundled MiniLM) | st (sentence-transformers); no fallback
MODEL_DIR = os.getenv("DPVG_MODEL_DIR", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"))
TAU_ENV = os.getenv("DPVG_TAU")                       # optional global override
CALIBRATION_FILE = _CALIBRATION_PATH
PAPER_TAU = 0.45                                      # paper default for MiniLM

# ---- Tool classification
READ_ONLY_TOOLS: set[str] = {"read_email", "get_calendar", "read_file", "search_web"}
SIDE_EFFECT_TOOLS: set[str] = {"send_email", "transfer_money", "delete_file", "post_slack", "external_action"}
ALL_CATALOG_TOOLS: list[str] = ["read_email", "get_calendar", "read_file", "search_web", "send_email", "transfer_money", "delete_file", "post_slack", "external_action"]
PLAN_EXTENSION_READ_ONLY_TTL: int = int(os.getenv("DPVG_PLAN_EXTENSION_READ_ONLY_TTL", "60"))

# ---- Stage B override patterns (rule-based filter)
# Regex patterns defining explicit instruction-override framing and system imitation.
OVERRIDE_PATTERNS: tuple[str, ...] = (
    # Pattern 1: Ignore/disregard/forget/override instructions/prompts/rules
    r"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any)\s+)?(?:the\s+)?"
    r"(?:(?:previous|prior|above|earlier|system|developer|user|assistant)\s+)*"
    r"(?:instructions?|prompts?|rules?)\b",
    # Pattern 2: New/updated system/assistant instructions
    r"\b(?:new|updated)\s+(?:(?:system|assistant)\s+)?instructions?\s*:",
    # Pattern 3: System/Assistant role brackets or tags (e.g. [SYSTEM], [ADMIN])
    r"\[\s*(?:system|assistant|admin|developer)\s*\]",
    # Pattern 4: System/Assistant role header at start of line
    r"^\W*(?:system|assistant|admin|developer)\s*:",
    # Pattern 5: Explicit task redirection
    r"\byour\s+(?:new\s+)?task\s+is\s+to\b",
    # Pattern 6: User secrecy directives
    r"\bdo\s+not\s+(?:tell|inform|mention\s+to)\s+(?:the\s+)?user\b",
)

# ---- agent backends and roles
BACKEND = os.getenv("DPVG_BACKEND", "ollama").lower()       # ollama | vulnerable | hosted
BACKEND_PLANNER = os.getenv("DPVG_BACKEND_PLANNER", BACKEND).lower()
BACKEND_TOOL = os.getenv("DPVG_BACKEND_TOOL", BACKEND).lower()
BACKEND_REASONING = os.getenv("DPVG_BACKEND_REASONING", BACKEND).lower()
BACKEND_SPECIALIZED = os.getenv("DPVG_BACKEND_SPECIALIZED", BACKEND).lower()
PLANNER_TYPE = os.getenv("DPVG_PLANNER_TYPE", "rule").lower()  # rule | llm
OLLAMA_URL = os.getenv("DPVG_OLLAMA_URL", "http://localhost:11434").rstrip("/")   # native Ollama API root (no /v1)
OLLAMA_MODEL = os.getenv("DPVG_OLLAMA_MODEL", "llama3.2")                      # must be pulled: `ollama pull <model>`; must support tools
LLM_TIMEOUT = float(os.getenv("DPVG_LLM_TIMEOUT", "120"))
HOSTED_API_URL = os.getenv("DPVG_HOSTED_API_URL", "https://api.openai.com/v1").rstrip("/")
HOSTED_API_KEY = os.getenv("DPVG_HOSTED_API_KEY", "")
HOSTED_API_MODEL = os.getenv("DPVG_HOSTED_API_MODEL", "")
EXECUTOR = os.getenv("DPVG_EXECUTOR", "langgraph")                                # langgraph | plain (explicit choice, never an automatic fallback)
