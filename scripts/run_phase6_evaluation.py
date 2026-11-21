"""Calibrate on InjecAgent validation data and run the fixed-tau Phase 6 evaluation matrix."""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import statistics
import sys
import time
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config
from app.benchmarks import load_adaptive_attacks, load_injecagent, run_agentdojo
from app.dataset import split_cases
from app.embeddings import get_embedder
from app.evaluation import evaluate_mode, sweep_candidate_tau
from app.gateway import Gateway, MODES

MODE_ORDER = list(MODES)
DEFAULT_BACKENDS = ["ollama"]
DEFAULT_SEEDS = [17, 29, 43]
METRICS = ["asr", "tcr", "precision", "recall", "f1", "fpr", "r_tok"]
OUT = ROOT / "results"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 4) if values else 0.0,
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }


def _new_gateway(backend: str, seed: int, tau: float, temperature: float):
    _set_seed(seed)
    gateway = Gateway(backend=backend, seed=seed, temperature=temperature, tau=tau)
    check = getattr(gateway.llm, "check", None)
    if check:
        check()
    return gateway


def _preflight_backends(backends: list[str], seed: int, tau: float, temperature: float) -> None:
    for backend in backends:
        gateway = _new_gateway(backend, seed, tau, temperature)
        client = getattr(gateway.llm, "client", None)
        if client is not None:
            client.close()


def _without_failures(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "failures"}


def evaluate_matrix(cases: list[dict], backend: str, seeds: list[int], tau: float,
                    modes: list[str] = MODE_ORDER, temperature: float = 0.0) -> dict:
    per_seed = []
    model_names = set()
    for seed in seeds:
        gateway = _new_gateway(backend, seed, tau, temperature)
        model_names.add(gateway.llm.name)
        for mode in modes:
            row = evaluate_mode(gateway, mode, cases)
            per_seed.append({
                **_without_failures(row), "backend": backend, "model": gateway.llm.name,
                "tau": tau, "seed": seed,
            })

    summaries = []
    for mode in modes:
        mode_rows = [row for row in per_seed if row["mode"] == mode]
        mode_config = MODES[mode]
        baseline_rows = {row["seed"]: row for row in per_seed if row["mode"] == "B0"}
        summary = {
            "mode": mode,
            "components": {
                "stage_a": mode_config.stage_a, "stage_b": mode_config.stage_b,
                "stage_b_rules": mode_config.stage_b_rules, "stage_b_sem": mode_config.stage_b_sem,
                "scope": mode_config.scope, "trusted_plan_only": mode_config.trusted_plan_only,
                "argument_checks": mode_config.argument_checks, "token_opt": mode_config.token_opt,
                "plan_extension": mode_config.plan_extension, "spotlight": mode_config.spotlight,
            },
            "metrics": {metric: _mean_std([row[metric] for row in mode_rows]) for metric in METRICS},
            "n_cases": mode_rows[0]["n_cases"] if mode_rows else 0,
        }
        median_deltas, p95_deltas = [], []
        if mode != "B0":
            for row in mode_rows:
                baseline = baseline_rows[row["seed"]]
                median_deltas.append(row["latency_run_ms"]["median"] - baseline["latency_run_ms"]["median"])
                p95_deltas.append(row["latency_run_ms"]["p95"] - baseline["latency_run_ms"]["p95"])
        summary["added_e2e_latency_ms_vs_B0"] = {
            "median": _mean_std(median_deltas),
            "p95": _mean_std(p95_deltas),
        }
        summary["score_distributions"] = {
            label: [score for row in mode_rows for score in row["score_distributions"][label]]
            for label in ("benign", "injected")
        }
        summaries.append(summary)
    return {
        "backend": backend, "models": sorted(model_names), "tau": tau, "temperature": temperature,
        "seeds": seeds, "rows": summaries, "per_seed": per_seed,
    }


def _write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_matrix(dataset_name: str, result: dict) -> None:
    for row in result["rows"]:
        metric = row["metrics"]
        added = row["added_e2e_latency_ms_vs_B0"]
        print(
            f"  {row['mode']:8} ASR {metric['asr']['mean']:.3f}+/-{metric['asr']['std']:.3f} "
            f"TCR {metric['tcr']['mean']:.3f}+/-{metric['tcr']['std']:.3f} "
            f"P {metric['precision']['mean']:.3f}+/-{metric['precision']['std']:.3f} "
            f"R {metric['recall']['mean']:.3f}+/-{metric['recall']['std']:.3f} "
            f"F1 {metric['f1']['mean']:.3f}+/-{metric['f1']['std']:.3f} "
            f"FPR {metric['fpr']['mean']:.3f}+/-{metric['fpr']['std']:.3f} "
            f"Rtok {metric['r_tok']['mean']:.3f}+/-{metric['r_tok']['std']:.3f} "
            f"delta-p50 {added['median']['mean']:.2f}+/-{added['median']['std']:.2f} ms "
            f"delta-p95 {added['p95']['mean']:.2f}+/-{added['p95']['std']:.2f} ms"
        )


def _case_sets(dataset_names: list[str], setting: str, limit: int | None) -> dict[str, list[dict]]:
    case_sets = {}
    if "adaptive" in dataset_names:
        case_sets["adaptive_test"] = load_adaptive_attacks(setting=setting, split="test", limit=limit)
    if "injecagent" in dataset_names:
        settings = ["base", "enhanced"] if setting == "both" else [setting]
        for name in settings:
            case_sets[f"injecagent_{name}_test"] = load_injecagent(setting=name, split="test", limit=limit)
    if "proto" in dataset_names:
        case_sets["proto_test"] = split_cases("test")
        if limit is not None:
            case_sets["proto_test"] = case_sets["proto_test"][:limit]
    return case_sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", choices=("grid", "agentdojo", "all"), default="all")
    parser.add_argument("--datasets", nargs="+", choices=("adaptive", "injecagent", "proto"),
                        default=["adaptive", "injecagent"])
    parser.add_argument("--setting", choices=("base", "enhanced", "both"), default="both")
    parser.add_argument("--backends", nargs="+", choices=("ollama",),
                        default=DEFAULT_BACKENDS)
    parser.add_argument("--modes", nargs="+", choices=MODE_ORDER, default=MODE_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tau", type=float, help="Use a fixed tau instead of validation calibration.")
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--limit", type=int, help="Limit attack cases per evaluation dataset.")
    parser.add_argument("--lo", type=float, default=0.00)
    parser.add_argument("--hi", type=float, default=1.00)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--max-fpr", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--agentdojo-suites", nargs="*")
    parser.add_argument("--agentdojo-limit-user", type=int)
    parser.add_argument("--agentdojo-limit-inj", type=int)
    parser.add_argument("--output", type=pathlib.Path, default=OUT / "phase6_evaluation.json")
    args = parser.parse_args()

    if "B0" not in args.modes:
        parser.error("B0 must be included to calculate added end-to-end latency")
    if not args.seeds:
        parser.error("at least one seed is required")
    if args.max_fpr < 0 or args.max_fpr > 1:
        parser.error("--max-fpr must be between 0 and 1")
    if args.step <= 0 or args.hi < args.lo:
        parser.error("invalid tau grid")
    if args.tau is not None and not 0 <= args.tau <= 1:
        parser.error("--tau must be between 0 and 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.benchmarks in ("agentdojo", "all") and "ollama" not in args.backends:
        parser.error("AgentDojo's current adapter requires the Ollama backend in --backends")
    if args.tau is None:
        val_cases = load_injecagent(setting="base", split="val", limit=args.calibration_limit)
        calibration = sweep_candidate_tau(
            val_cases, get_embedder(), args.lo, args.hi, args.step, args.max_fpr, split="val",
        )
        _write(OUT / "phase6_calibration_sweep.json", {
            "embedder": get_embedder().name, "source_split": "validation",
            "n_cases": calibration["n_cases"], "grid": [args.lo, args.hi, args.step],
            "max_fpr": args.max_fpr, "selection_status": calibration["selection_status"],
            "decision_rule": calibration["decision_rule"],
            "best": calibration["best"], "points": calibration["points"],
            "fpr_recall_curve": calibration["fpr_recall_curve"],
            "score_distributions": calibration["score_distributions"],
        })
        if calibration["best"] is None:
            raise RuntimeError(calibration["selection_status"])
        tau = calibration["best"]["tau"]
        calibration_record = {
            "source": "InjecAgent base validation split (dh+ds) only",
            "method": f"{calibration['decision_rule']}; select maximum validation F1 under the configured FPR cap; test split is never used",
            "tau": tau, "best": calibration["best"], "grid": [args.lo, args.hi, args.step],
            "score_distributions": calibration["score_distributions"],
            "n_cases": calibration["n_cases"], "max_fpr": args.max_fpr,
            "fpr_recall_curve": calibration["fpr_recall_curve"],
        }
        _write(pathlib.Path(config.CALIBRATION_FILE), {
            "embedder": get_embedder().name, "tau": tau, "mode": "G-B candidate scoring",
            "dataset": "InjecAgent base, dh+ds, validation split",
            "selection": calibration_record["method"], "n_cases": calibration["n_cases"],
            "grid": [args.lo, args.hi, args.step], "max_fpr": args.max_fpr,
            "plateau": calibration["best"]["plateau"], "at_edge": calibration["best"]["at_edge"],
            "date": datetime.date.today().isoformat(),
            "points": calibration["points"], "fpr_recall_curve": calibration["fpr_recall_curve"],
            "score_distributions": calibration["score_distributions"],
        })
    else:
        tau = args.tau
        calibration_record = {
            "source": "fixed CLI override; no validation calibration performed",
            "tau": tau, "score_distributions": None,
        }

    _preflight_backends(args.backends if args.benchmarks != "agentdojo" else ["ollama"],
                        args.seeds[0], tau, args.temperature)
    if args.benchmarks in ("agentdojo", "all"):
        from app.benchmarks import _need_agentdojo
        _need_agentdojo()

    output = {
        "title": "DPVG Phase 6 evaluation",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedder": get_embedder().name,
        "calibration": calibration_record,
        "configuration": {
            "modes": args.modes, "backends": args.backends, "seeds": args.seeds,
            "temperature": args.temperature, "setting": args.setting,
            "limit_per_dataset": args.limit, "calibration_limit": args.calibration_limit,
        },
        "datasets": {}, "agentdojo": [],
    }
    _write(args.output, output)

    if args.benchmarks in ("grid", "all"):
        for dataset_name, cases in _case_sets(args.datasets, args.setting, args.limit).items():
            output["datasets"][dataset_name] = {}
            for backend in args.backends:
                output["datasets"][dataset_name][backend] = evaluate_matrix(
                    cases, backend, args.seeds, tau, args.modes, args.temperature,
                )
                _write(args.output, output)
                print(f"{dataset_name:24} {backend:10} tau={tau:.3f} n={len(cases)}")
                _print_matrix(dataset_name, output["datasets"][dataset_name][backend])

    if args.benchmarks in ("agentdojo", "all"):
        dojo_modes = [mode for mode in args.modes if mode in {"B0", "G-A", "G-B", "G-AB", "FULL"}]
        if not dojo_modes:
            parser.error("AgentDojo supports B0, G-A, G-B, G-AB, and FULL in its current adapter")
        for seed in args.seeds:
            result = run_agentdojo(
                modes=dojo_modes,
                suites=args.agentdojo_suites, tau=tau,
                limit_user=args.agentdojo_limit_user, limit_inj=args.agentdojo_limit_inj,
                seed=seed, temperature=args.temperature,
            )
            result.update({"backend": "ollama", "seed": seed, "model": result["agent"]})
            output["agentdojo"].append(result)
            _write(OUT / f"phase6_agentdojo_seed_{seed}.json", result)
            _write(args.output, output)
            print(f"AgentDojo seed={seed} tau={tau:.3f} model={result['agent']}")

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
