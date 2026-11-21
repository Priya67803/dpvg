"""FastAPI backend: REST API + static dashboard."""
from __future__ import annotations
import json, pathlib
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import __version__, config
from .agents import get_llm
from .alignment import AlignmentEngine, extract_candidates, has_hidden_content, sanitize
from .dataset import CASES, SCENARIOS
from .embeddings import default_tau, get_embedder
from .evaluation import evaluate, sweep_tau
from .llm_backends import LLMError
from .embeddings import load_calibration
from .gateway import DEFAULT_MODE, MODES, Gateway, RuntimeBlockException
from .registry import VectorRegistry
from .token_opt import TokenOptimizer, count_tokens, get_token_counter_source

app = FastAPI(title="Dual-Plane Runtime Verification Gateway", version=__version__)
FRONTEND = pathlib.Path(__file__).resolve().parent.parent / "frontend"


RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def _gw(tau: float | None = None, backend: str | None = None) -> Gateway:
    """Create the requested model backend and fail explicitly if it is not configured or reachable."""
    try:
        llm = get_llm(backend=backend)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    gw = Gateway(embedder=get_embedder(), tau=tau, llm=llm)
    try:
        check = getattr(gw.llm, "check", None)
        if check:
            check()
    except LLMError as e:
        raise HTTPException(503, str(e))
    return gw


def _ollama_status() -> dict:
    from .llm_backends import OllamaBackend
    b = OllamaBackend(timeout=3, retries=0)
    try:
        b.check()
        return {"url": b.base_url, "model": b.model, "ready": True, "error": None}
    except LLMError as e:
        return {"url": b.base_url, "model": b.model, "ready": False, "error": str(e)}


@app.get("/api/health")
def health():
    e = get_embedder()
    cal = load_calibration()
    return {"status": "ok", "version": __version__, "embedder": e.name, "embedder_backend": e.backend,
            "default_tau": default_tau(e), "default_mode": DEFAULT_MODE, "executor": Gateway(embedder=e).executor,
            "calibration": {k: cal[k] for k in ("tau", "dataset", "n_cases", "date", "at_edge")} if cal and cal.get("embedder") == e.name else None,
            "llm": config.BACKEND, "backend": config.BACKEND, "ollama": _ollama_status(),
            "hosted_configured": bool(config.HOSTED_API_KEY and config.HOSTED_API_MODEL),
            "spt_ttl": config.SPT_TTL, "max_depth": config.MAX_DEPTH, "algorithm": "EdDSA (Ed25519)",
            "token_counter": "tiktoken" if _tiktoken() else "approximate"}


@app.post("/api/ollama/configure")
def configure_ollama(body: dict = Body(...)):
    new_url = body.get("url")
    new_model = body.get("model")
    if new_url:
        config.OLLAMA_URL = str(new_url).rstrip("/")
    if new_model:
        config.OLLAMA_MODEL = str(new_model).strip()
    status = _ollama_status()
    return {"status": "ok", "ollama": status}


def _tiktoken() -> bool:
    from . import token_opt
    return token_opt._ENC is not None


@app.get("/api/modes")
def modes():
    return [{"key": m.key, "label": m.label, "stage_a": m.stage_a, "stage_b": m.stage_b,
             "stage_b_rules": m.stage_b_rules, "stage_b_sem": m.stage_b_sem,
             "scope": m.scope, "token_opt": m.token_opt, "trusted_plan_only": m.trusted_plan_only,
             "argument_checks": m.argument_checks, "plan_extension": m.plan_extension,
             "spotlight": m.spotlight} for m in MODES.values()]


@app.get("/api/backends")
def backends():
    return {
        "default": "ollama",
        "available": [
            {"key": "ollama", "label": "Ollama (llama3.2)"},
            {"key": "compromised", "label": "Compromised Agent (follows injection)"},
        ],
    }


@app.get("/api/cases")
def cases():
    return {"scenarios": SCENARIOS, "cases": CASES}


@app.post("/api/run")
def run(body: dict = Body(...)):
    case, mode = body.get("case"), body.get("mode", DEFAULT_MODE)
    if not case or "query" not in case:
        raise HTTPException(422, "case.query is required")
    case.setdefault("goal", case["query"])
    case.setdefault("is_attack", bool(case.get("injection") or case.get("rogue")))
    strict_mode = body.get("strict_mode")
    try:
        return _gw(body.get("tau"), body.get("backend")).run(case, mode, strict_mode=strict_mode)
    except KeyError as e:
        raise HTTPException(422, f"unknown mode {e}")
    except LLMError as e:
        raise HTTPException(502, str(e))
    except RuntimeBlockException as e:
        raise HTTPException(403, detail={"error": "RuntimeBlockException", "reason": e.reason,
                                         "score": e.score, "stage": e.stage, "code": e.code})


@app.post("/api/compare")
def compare(body: dict = Body(...)):
    case = body.get("case")
    if not case or "query" not in case:
        raise HTTPException(422, "case.query is required")
    case.setdefault("goal", case["query"])
    case.setdefault("is_attack", bool(case.get("injection") or case.get("rogue")))
    gw = _gw(body.get("tau"), body.get("backend"))
    out = []
    is_attack = bool(case.get("is_attack"))
    for k in body.get("modes") or list(MODES):
        try:
            r = gw.run(case, k)
        except LLMError as e:
            raise HTTPException(502, str(e))
        outcome = r.get("outcome", {})
        injection_detected = any(c.get("stage") in ("rules", "sem", "action_class", "drift") and not c.get("allowed") for c in r.get("checks", [])) or (r.get("metrics", {}).get("detected") and is_attack)
        attacker_goal_blocked = bool(is_attack and not r.get("metrics", {}).get("attack_success") and (injection_detected or any(not c.get("allowed") for c in r.get("checks", []))))
        benign_blocked = bool(not is_attack and any(not c.get("allowed") for c in r.get("checks", [])))
        out.append({
            "mode": k, "label": MODES[k].label, **r["metrics"],
            "outcome": outcome,
            "injection_detected": injection_detected,
            "attacker_goal_blocked": attacker_goal_blocked,
            "benign_blocked": benign_blocked,
            "blocked_at": next((f"{c['stage']}:{c['code']}" for c in r["checks"] if not c["allowed"]), None)
        })
    return {"tau": gw.tau, "results": out}


@app.get("/api/results")
def results():
    """Stored real-data results written by scripts/evaluate.py (InjecAgent, AgentDojo, B1 sensitivity, prototype set)."""
    out = {}
    for name in ("proto", "injecagent", "agentdojo", "spotlight", "token_opt", "plan_gating_utility", "results", "latency_benchmark"):
        f = RESULTS / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


@app.post("/api/evaluate")
def evaluate_api(body: dict = Body(default={})):
    _gw(body.get("tau"))                                   # fail fast if Ollama is down
    try:
        return evaluate(modes=body.get("modes"), tau=body.get("tau"), split=body.get("split", "all"),
                        repeats=max(1, min(int(body.get("repeats", 1)), 10)))
    except LLMError as e:
        raise HTTPException(502, str(e))


@app.post("/api/sweep")
def sweep_api(body: dict = Body(default={})):
    split = body.get("split", "val")
    if split != "val":
        raise HTTPException(422, "tau selection is restricted to the validation split")
    _gw()
    try:
        return sweep_tau(
            body.get("mode", "G-AB"), float(body.get("lo", 0.00)), float(body.get("hi", 1.00)),
            float(body.get("step", 0.01)), split, max_fpr=float(body.get("max_fpr", 0.10)),
        )
    except LLMError as e:
        raise HTTPException(502, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/optimize")
def optimize(body: dict = Body(...)):
    try:
        cutoff = float(body.get("context_similarity_cutoff", config.CONTEXT_SIMILARITY_CUTOFF))
        opt = TokenOptimizer(
            get_embedder(),
            total_budget=int(body.get("budget", config.TOKEN_BUDGET)),
            min_sim=cutoff,
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(422, f"invalid token optimization settings: {e}") from e
    ctx = [c for c in body.get("context", []) if c.strip()]
    user_prefs = list(body.get("user_preferences", [])) or list(body.get("preferences", []))
    protected = [x for x in [body.get("goal")] + list(body.get("constraints", [])) + list(body.get("policy", [])) + user_prefs if x]
    r = opt.optimize(body["query"], ctx, protected, min_similarity=cutoff)
    return {"query": r.query, "subtasks": r.subtasks, "budget": r.budget,
            "context_by_subtask": r.context_by_subtask, "context_similarity_cutoff": cutoff,
            "kept_context": r.context,
            "dropped_context": r.dropped_context, "tokens_before": r.tokens_before, "tokens_after": r.tokens_after,
            "r_tok": round(r.r_tok, 4), "protected": protected,
            "counter": get_token_counter_source()}


@app.post("/api/align")
def align(body: dict = Body(...)):
    e = get_embedder()
    tau = float(body.get("tau", default_tau(e)))
    sess = VectorRegistry(e).register({"goal": body["intent"]})
    res = AlignmentEngine(e).check_payload(sess.e_root, body["text"], tau)
    for c in res["candidates"]:
        if c.get("score") is not None:
            c["margin"] = round(c["score"] - tau, 4)
            c["near_threshold"] = abs(c["margin"]) < 0.03
            c["label"] = "aligned but instruction-like (sanitized)" if c["allowed"] else "Blocked"
    return {**res, "tau": tau, "root_hash": sess.root_hash, "hidden_content": has_hidden_content(body["text"]),
            "sanitized": sanitize(body["text"], [(c["text"], c["allowed"]) for c in res["candidates"]])}


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")
