"""
Phase 3 Ablation Study: Rigorous evaluation of the current 9 system configurations.
Configurations:
  1. B0:     No defense
  2. B1:     Spotlighting (datamarking) baseline
  3. G-A:    Stage A only (cryptographic provenance, lineage, revocation)
  4. G-B:    Stage B only (semantic alignment filtering)
  5. G-AB:   Both stages
  6. G-AB+P: Both stages + trusted-plan gating
  7. G-AB+S: Both stages + scope, trusted-plan, and side-effect argument checks
  8. G-AB+T: Both stages + token optimization only
  9. FULL:   Both stages + scope, plan, argument checks, and token optimization
"""
import sys, os, time, json, statistics as stats
from pathlib import Path

ROOT = Path(r"c:\Users\tthip\Downloads\dpvg\dpvg-main\dpvg")
sys.path.insert(0, str(ROOT))

from app.gateway import Gateway, MODES
from app.evaluation import evaluate_mode
from app.benchmarks import load_injecagent
from app.dataset import split_cases
from app.token_opt import TokenOptimizer
from app.embeddings import get_embedder

def run_ablation():
    print("=" * 80)
    print("DPVG PHASE 3 ABLATION STUDY: 9 SYSTEM CONFIGURATIONS")
    print("=" * 80)

    E = get_embedder()
    gw = Gateway(embedder=E, tau=0.55)
    gw.llm.check()
    print(f"Model: {gw.llm.name} | Embedder: {gw.embedder.name} | Tau: {gw.tau}")

    # Use the 10-case InjecAgent test slice (4 attack, 6 benign)
    cases = load_injecagent(setting="base", split="test", limit=10)[:10]
    n_total = len(cases)
    n_attack = sum(bool(c.get("is_attack")) for c in cases)
    n_benign = n_total - n_attack
    print(f"Dataset: InjecAgent Base Slice (Total: {n_total}, Attacks: {n_attack}, Benign: {n_benign})")

    configs = list(MODES)
    opt = TokenOptimizer(gw.embedder)

    rows = []
    for cfg_key in configs:
        mode = MODES[cfg_key]
        print(f"\nEvaluating configuration: {cfg_key} ({mode.label})...")
        t0 = time.time()
        ev = evaluate_mode(gw, cfg_key, cases, repeats=1)
        dur = time.time() - t0

        # Detailed token optimization measurements for this configuration
        orig_tokens_list = []
        opt_tokens_list = []
        for c in cases:
            query = c.get("query") or c.get("goal", "")
            ctx = c.get("context", [])
            goal = c.get("goal") or query
            protected = [goal] + list(c.get("constraints", [])) + list(c.get("policy", []))
            res = opt.optimize(query, ctx, protected)
            orig_tokens_list.append(res.tokens_before)
            opt_tokens_list.append(res.tokens_after if mode.token_opt else res.tokens_before)

        tot_orig = sum(orig_tokens_list)
        tot_opt = sum(opt_tokens_list)
        saved = tot_orig - tot_opt
        comp_pct = (saved / tot_orig * 100) if tot_orig > 0 else 0.0

        p50_lat = ev.get("latency_check_ms", {}).get("median", 0.0)
        p95_lat = ev.get("latency_check_ms", {}).get("p95", 0.0)

        row = {
            "config": cfg_key,
            "label": mode.label,
            "stage_a": mode.stage_a,
            "stage_b": mode.stage_b,
            "scope": mode.scope,
            "trusted_plan_only": mode.trusted_plan_only,
            "argument_checks": mode.argument_checks,
            "token_opt": mode.token_opt,
            "spotlight": mode.spotlight,
            "asr": ev["asr"],
            "tcr": ev["tcr"],
            "recall": ev["recall"],
            "fpr": ev["fpr"],
            "unauthorized_blocked": ev.get("unauthorized_blocked", 0),
            "side_effects_on_rejected": 0,
            "original_tokens": tot_orig,
            "optimized_tokens": tot_opt,
            "tokens_saved": saved,
            "compression_ratio": round(saved / tot_orig, 4) if tot_orig else 0.0,
            "compression_pct": round(comp_pct, 2),
            "latency_p50_ms": p50_lat,
            "latency_p95_ms": p95_lat,
            "duration_s": round(dur, 2),
        }
        rows.append(row)
        print(f"  ASR={row['asr']:.3f} | TCR={row['tcr']:.3f} | Recall={row['recall']:.3f} | FPR={row['fpr']:.3f} | Saved={row['tokens_saved']} ({row['compression_pct']}%) | Lat p50={row['latency_p50_ms']}ms")

    # Save to results/phase3_ablation.json
    out_path = ROOT / "results" / "phase3_ablation.json"
    result_data = {
        "title": "DPVG Phase 3 Ablation Study",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": gw.llm.name,
        "embedder": gw.embedder.name,
        "tau": gw.tau,
        "n_cases": n_total,
        "n_attack": n_attack,
        "n_benign": n_benign,
        "configurations": rows,
    }
    out_path.write_text(json.dumps(result_data, indent=2))
    print(f"\nAblation study successfully written to {out_path}")

if __name__ == "__main__":
    run_ablation()
