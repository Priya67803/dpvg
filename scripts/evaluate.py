"""Evaluation CLI. Results are written to results/*.json.

  python scripts/evaluate.py proto      [--repeats 3]                 35-case prototype set (all configurations)
  python scripts/evaluate.py injecagent [--setting base|enhanced] [--split test|val|all] [--limit N]
  python scripts/evaluate.py agentdojo  [--suites workspace ...] [--limit-user N] [--limit-inj N]
  python scripts/evaluate.py spotlight  [--setting base]               B1 with delimit, datamark AND encode vs B0 / gateway, on the Ollama model
  python scripts/evaluate.py token-opt  [--dataset injecagent|agentdojo|proto|val|all] [--limit N]
  python scripts/evaluate.py all
"""
import argparse, json, pathlib, statistics as stats, sys, time
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.benchmarks import load_injecagent, run_agentdojo
from app.dataset import split_cases
from app.evaluation import evaluate, evaluate_mode
from app.gateway import MODES, Gateway
from app.token_opt import TokenOptimizer, count_tokens

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
ORDER = list(MODES)


def table(rows, title):
    print(f"\n== {title}")
    print(f"{'mode':8} {'ASR':>6} {'TCR':>6} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'chk p50 ms':>11} {'Rtok':>6}")
    for r in rows:
        print(f"{r['mode']:8} {r['asr']:6.3f} {r['tcr']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} "
              f"{r['fpr']:6.3f} {r['latency_check_ms']['median']:11.3f} {r['r_tok']:6.2f}")


def slim(rows):
    return [{k: v for k, v in r.items() if k != "failures"} for r in rows]


def proto(a):
    out = evaluate(modes=ORDER, tau=a.tau, split="all", repeats=a.repeats)
    table(out["rows"], f"prototype set ({out['embedder']}, tau={out['tau']})")
    (OUT / "proto.json").write_text(json.dumps(out, indent=2))


def injecagent(a):
    res = {}
    for setting in (["base", "enhanced"] if a.setting == "both" else [a.setting]):
        cases = load_injecagent(setting=setting, split=a.split, limit=a.limit)
        gw = Gateway(tau=a.tau)
        gw.llm.check()
        n_atk = sum(c["is_attack"] for c in cases)
        t = time.time()
        rows = [evaluate_mode(gw, m, cases, a.repeats) for m in ORDER]
        table(rows, f"InjecAgent {setting} / {a.split}: {n_atk} attacks + {len(cases) - n_atk} benign "
                    f"({gw.embedder.name}, tau={gw.tau}, {time.time() - t:.0f}s)")
        for r in rows:
            if r["mode"] in ("G-AB", "FULL"):
                print("   ", r["mode"], {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r["by_scenario"].items()})
        res[setting] = {"n_attack": n_atk, "n_benign": len(cases) - n_atk, "rows": slim(rows)}
    out = {"dataset": "InjecAgent (real data)", "split": a.split, "tau": gw.tau, "embedder": gw.embedder.name,
           "agent": gw.llm.name, "settings": res}
    (OUT / "injecagent.json").write_text(json.dumps(out, indent=2))


def spotlight(a):
    """B1 with every marking (delimit, datamark, encode) next to B0 and the gateway, on the same cases and model."""
    from app import spotlight as sp
    cases = load_injecagent(setting=a.setting, split=a.split, limit=a.limit)
    gw = Gateway(tau=a.tau)
    gw.llm.check()
    modes = ["B0"] + [{"key": f"B1-{k}", "label": f"Spotlighting ({k})", "spotlight": k} for k in sp.KINDS] + ["G-AB", "FULL"]
    rows = [evaluate_mode(gw, m, cases) for m in modes]
    print(f"\n== B1 spotlighting (all markings) vs the gateway, measured on {gw.llm.name}")
    print(f"{'mode':12} {'ASR':>6} {'TCR':>6}")
    for r in rows:
        print(f"{r['mode']:12} {r['asr']:6.3f} {r['tcr']:6.3f}")
    (OUT / "spotlight.json").write_text(json.dumps({"model": gw.llm.name, "setting": a.setting, "split": a.split,
                                                    "rows": [{"mode": r["mode"], "asr": r["asr"], "tcr": r["tcr"]} for r in rows]}, indent=2))


def agentdojo(a):
    t = time.time()
    try:
        out = run_agentdojo(modes=a.modes, suites=a.suites, tau=a.tau, limit_user=a.limit_user, limit_inj=a.limit_inj,
                            progress=lambda m, s, r: print(f"  [{m}] {s}: {r}", flush=True))
    except Exception as e:
        print(f"AgentDojo evaluation skipped: {e}")
        out = {"benchmark": "AgentDojo v1.2.1", "status": "skipped", "reason": str(e), "rows": []}
        (OUT / "agentdojo.json").write_text(json.dumps(out, indent=2))
        return
    print(f"\n== AgentDojo {out['benchmark']} ({out['embedder']}, tau={out['tau']}, {time.time() - t:.0f}s)")
    print(f"{'mode':7} {'ASR':>6} {'util':>6} {'util|atk':>9} {'clean flagged':>14} {'inj flagged':>12} {'p50 ms':>7}")
    for r in out["rows"]:
        print(f"{r['mode']:7} {r['asr']:6.3f} {r['utility_clean']:6.3f} {r['utility_under_attack']:9.3f} "
              f"{r['clean_runs_flagged']:14.3f} {r['injected_runs_flagged']:12.3f} {r['check_ms_median']:7.2f}")
    (OUT / "agentdojo.json").write_text(json.dumps(out, indent=2))


def token_opt(a):
    """Reproducible evaluation of token optimization on real benchmark tasks."""
    dataset_name = (a.dataset or "injecagent").lower()
    gw = Gateway(tau=a.tau)
    gw.llm.check()
    cases = []

    if dataset_name in ("injecagent", "all"):
        cases += load_injecagent(setting=a.setting, split=a.split, limit=a.limit)
    if dataset_name in ("proto", "val", "all") or not cases:
        cases += split_cases("val" if dataset_name == "val" else "all")

    if a.limit and len(cases) > a.limit:
        cases = cases[:a.limit]

    print(f"\n== Evaluating Token Optimization on {dataset_name} ({len(cases)} cases, model: {gw.llm.name})")

    opt_modes = ["B0", "G-AB", "G-AB+T", "FULL"]
    results = []

    for m in opt_modes:
        t0 = time.time()
        ev = evaluate_mode(gw, m, cases, repeats=1)
        dur = time.time() - t0

        # Detailed token optimization measurements
        orig_tokens_list = []
        opt_tokens_list = []
        opt_latencies = []

        opt = TokenOptimizer(gw.embedder)
        for c in cases:
            query = c.get("query") or c.get("goal", "")
            ctx = c.get("context", [])
            goal = c.get("goal") or query
            protected = [goal] + list(c.get("constraints", [])) + list(c.get("policy", []))

            # Measure optimization on prompt
            res = opt.optimize(query, ctx, protected)
            orig_tokens_list.append(res.tokens_before)
            opt_tokens_list.append(res.tokens_after if MODES[m].token_opt else res.tokens_before)
            opt_latencies.append(res.latency_ms)

        tot_orig = sum(orig_tokens_list)
        tot_opt = sum(opt_tokens_list)
        saved = tot_orig - tot_opt
        comp_ratio = (saved / tot_orig) if tot_orig > 0 else 0.0

        lat_sorted = sorted(opt_latencies)
        p50_lat = lat_sorted[int(len(lat_sorted) * 0.50)] if lat_sorted else 0.0
        p95_lat = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))] if lat_sorted else 0.0
        avg_lat = stats.mean(opt_latencies) if opt_latencies else 0.0

        row = {
            "mode": m,
            "label": MODES[m].label,
            "original_tokens": tot_orig,
            "optimized_tokens": tot_opt,
            "tokens_saved": saved,
            "compression_ratio": round(comp_ratio, 4),
            "task_completion_rate": ev["tcr"],
            "asr": ev["asr"],
            "recall": ev["recall"],
            "fpr": ev["fpr"],
            "opt_latency_avg_ms": round(avg_lat, 3),
            "opt_latency_p50_ms": round(p50_lat, 3),
            "opt_latency_p95_ms": round(p95_lat, 3),
            "duration_s": round(dur, 2),
        }
        results.append(row)

    print(f"\n{'Mode':8} {'Orig Tok':>9} {'Opt Tok':>8} {'Saved':>6} {'Comp%':>7} {'TCR':>6} {'ASR':>6} {'Recall':>6} {'FPR':>6} {'p50(ms)':>8} {'p95(ms)':>8}")
    print("-" * 92)
    for r in results:
        print(f"{r['mode']:8} {r['original_tokens']:9d} {r['optimized_tokens']:8d} {r['tokens_saved']:6d} "
              f"{r['compression_ratio']*100:6.1f}% {r['task_completion_rate']:6.3f} {r['asr']:6.3f} "
              f"{r['recall']:6.3f} {r['fpr']:6.3f} {r['opt_latency_p50_ms']:8.2f} {r['opt_latency_p95_ms']:8.2f}")

    out_data = {
        "dataset": dataset_name,
        "n_cases": len(cases),
        "embedder": gw.embedder.name,
        "model": gw.llm.name,
        "tau": gw.tau,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": results,
    }
    (OUT / "token_opt.json").write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved token optimization results to {OUT / 'token_opt.json'}")


ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["proto", "injecagent", "agentdojo", "spotlight", "token-opt", "all"])
ap.add_argument("--tau", type=float); ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--setting", default="both", choices=["base", "enhanced", "both"]); ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int); ap.add_argument("--suites", nargs="*"); ap.add_argument("--modes", nargs="*")
ap.add_argument("--limit-user", type=int); ap.add_argument("--limit-inj", type=int)
ap.add_argument("--dataset", default="injecagent", choices=["injecagent", "agentdojo", "proto", "val", "all"])
a = ap.parse_args()

if a.what in ("proto", "all"):
    a.repeats = 3 if a.what == "all" else a.repeats; proto(a)
if a.what in ("injecagent", "all"):
    a.repeats = 1; injecagent(a)
if a.what in ("spotlight", "all"):
    a.setting = "base" if a.setting == "both" else a.setting; spotlight(a)
if a.what in ("agentdojo", "all"):
    agentdojo(a)
if a.what in ("token-opt", "all"):
    token_opt(a)
