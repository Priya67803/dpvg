"""Stage B Ablation Sweep: B-rules alone vs B-sem alone vs combined G-B on leak-free validation data."""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmarks import load_injecagent
from app.embeddings import get_embedder
from app.registry import spec_to_text
from app.alignment import (
    AlignmentEngine, extract_candidates, has_high_confidence_injection_marker,
)


def run_ablation_sweep(split: str = "val", max_fpr: float = 0.10, step: float = 0.01) -> dict:
    embedder = get_embedder()
    engine = AlignmentEngine(embedder)

    results_by_setting = {}
    distributions_by_setting = {}

    for setting in ["base", "enhanced"]:
        cases = load_injecagent(split=split, setting=setting)
        scored_cases = []
        score_distributions = {"benign": [], "injected": []}

        # Pre-compute root embeddings, candidates, scores, and rule markers
        for case in cases:
            spec = case.get("intent_spec") or {
                "goal": case.get("goal") or case.get("query", ""),
                "constraints": case.get("constraints", []), "policy": case.get("policy", []),
            }
            root = embedder.encode(spec_to_text(spec))
            payload = "\n".join(str(value) for value in (case.get("fixtures") or {}).values())

            # For rules: extracted with strip_obfuscation=True
            cands_rules = extract_candidates(payload, strip_obfuscation=True)
            marker_detected = (
                any(has_high_confidence_injection_marker(c) for c in cands_rules) or
                has_high_confidence_injection_marker(payload)
            )

            # For sem: extracted with strip_obfuscation=False (no rules)
            cands_sem = extract_candidates(payload, strip_obfuscation=False)
            scores_sem = [engine.score(root, c) for c in cands_sem]

            # Also scores with strip_obfuscation=True (for combined G-B)
            scores_combined = [engine.score(root, c) for c in cands_rules]

            is_attack = bool(case.get("is_attack"))
            label = "injected" if is_attack else "benign"
            score_distributions[label].extend(round(s, 6) for s in scores_combined)

            scored_cases.append({
                "is_attack": is_attack,
                "marker": marker_detected,
                "scores_sem": scores_sem,
                "scores_combined": scores_combined,
            })

        distributions_by_setting[setting] = score_distributions

        # 1. B-rules alone (no semantic thresholding)
        tp_r = sum(c["is_attack"] and c["marker"] for c in scored_cases)
        fp_r = sum((not c["is_attack"]) and c["marker"] for c in scored_cases)
        fn_r = sum(c["is_attack"] and (not c["marker"]) for c in scored_cases)
        tn_r = sum((not c["is_attack"]) and (not c["marker"]) for c in scored_cases)
        prec_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) else 0.0
        rec_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) else 0.0
        f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) else 0.0
        fpr_r = fp_r / (fp_r + tn_r) if (fp_r + tn_r) else 0.0

        b_rules_res = {
            "mode": "B-rules",
            "setting": setting,
            "precision": round(prec_r, 4),
            "recall": round(rec_r, 4),
            "f1": round(f1_r, 4),
            "fpr": round(fpr_r, 4),
            "tp": tp_r, "fp": fp_r, "tn": tn_r, "fn": fn_r,
            "n_attack": tp_r + fn_r,
            "n_benign": fp_r + tn_r,
        }

        # 2. B-sem alone sweep (across tau in [0.00, 1.00])
        steps = int(round(1.0 / step))
        sem_points = []
        best_sem = None

        for i in range(steps + 1):
            tau = round(i * step, 4)
            tp = sum(c["is_attack"] and any(s < tau for s in c["scores_sem"]) for c in scored_cases)
            fp = sum((not c["is_attack"]) and any(s < tau for s in c["scores_sem"]) for c in scored_cases)
            fn = sum(c["is_attack"] and not any(s < tau for s in c["scores_sem"]) for c in scored_cases)
            tn = sum((not c["is_attack"]) and not any(s < tau for s in c["scores_sem"]) for c in scored_cases)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0

            pt = {
                "tau": tau, "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4), "fpr": round(fpr, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
            sem_points.append(pt)

            if fpr <= max_fpr:
                if (best_sem is None or
                    pt["recall"] > best_sem["recall"] or
                    (pt["recall"] == best_sem["recall"] and pt["f1"] > best_sem["f1"])):
                    best_sem = pt.copy()

        # 3. Combined G-B sweep (rules + semantic thresholding)
        gb_points = []
        best_gb = None

        for i in range(steps + 1):
            tau = round(i * step, 4)
            tp = sum(c["is_attack"] and (c["marker"] or any(s < tau for s in c["scores_combined"])) for c in scored_cases)
            fp = sum((not c["is_attack"]) and (c["marker"] or any(s < tau for s in c["scores_combined"])) for c in scored_cases)
            fn = sum(c["is_attack"] and not (c["marker"] or any(s < tau for s in c["scores_combined"])) for c in scored_cases)
            tn = sum((not c["is_attack"]) and not (c["marker"] or any(s < tau for s in c["scores_combined"])) for c in scored_cases)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0

            pt = {
                "tau": tau, "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4), "fpr": round(fpr, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
            gb_points.append(pt)

            if fpr <= max_fpr:
                if (best_gb is None or
                    pt["f1"] > best_gb["f1"] or
                    (pt["f1"] == best_gb["f1"] and pt["recall"] > best_gb["recall"])):
                    best_gb = pt.copy()

        results_by_setting[setting] = {
            "b_rules": b_rules_res,
            "b_sem_best": best_sem,
            "b_sem_points": sem_points,
            "gb_best": best_gb,
            "gb_points": gb_points,
        }

    return {
        "split": split,
        "max_fpr": max_fpr,
        "step": step,
        "embedder": embedder.name,
        "results": results_by_setting,
        "distributions": distributions_by_setting,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep B-rules alone, B-sem alone, and G-B on validation split.")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--max-fpr", type=float, default=0.10)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Stage B ablation sweep (split={args.split}, max_fpr={args.max_fpr}, step={args.step})...")
    t0 = time.perf_counter()
    res = run_ablation_sweep(split=args.split, max_fpr=args.max_fpr, step=args.step)
    dt = time.perf_counter() - t0
    print(f"Completed sweep in {dt:.2f}s.")

    # Save ablation summary and curves
    ablation_out = out_dir / "stage_b_ablation.json"
    curves_to_save = {
        "split": res["split"],
        "max_fpr": res["max_fpr"],
        "embedder": res["embedder"],
        "results": res["results"],
    }
    with open(ablation_out, "w", encoding="utf-8") as f:
        json.dump(curves_to_save, f, indent=2)
    print(f"Saved ablation results to {ablation_out}")

    # Save score distributions
    dist_out = out_dir / "stage_b_score_distributions.json"
    with open(dist_out, "w", encoding="utf-8") as f:
        json.dump(res["distributions"], f, indent=2)
    print(f"Saved score distributions to {dist_out}")

    # Print summary table
    print("\n" + "=" * 80)
    print("STAGE B DECOMPOSITION: VALIDATION ABLATION SUMMARY (FPR <= 0.10)")
    print("=" * 80)
    print(f"{'Setting':<10} | {'Mode':<10} | {'Tau':<6} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'FPR':<8}")
    print("-" * 80)
    for setting, data in res["results"].items():
        br = data["b_rules"]
        bs = data["b_sem_best"]
        gb = data["gb_best"]
        print(f"{setting:<10} | {'B-rules':<10} | {'N/A':<6} | {br['recall']:<8.4f} | {br['precision']:<10.4f} | {br['f1']:<8.4f} | {br['fpr']:<8.4f}")
        print(f"{setting:<10} | {'B-sem':<10} | {bs['tau']:<6.2f} | {bs['recall']:<8.4f} | {bs['precision']:<10.4f} | {bs['f1']:<8.4f} | {bs['fpr']:<8.4f}")
        print(f"{setting:<10} | {'G-B (comb)':<10} | {gb['tau']:<6.2f} | {gb['recall']:<8.4f} | {gb['precision']:<10.4f} | {gb['f1']:<8.4f} | {gb['fpr']:<8.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
