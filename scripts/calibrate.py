"""Calibrate tau on the VALIDATION split of the real InjecAgent data (paper Section VI-B) and record it.

    python scripts/calibrate.py [--lo 0.00] [--hi 1.00] [--step 0.01] [--max-fpr 0.10] [--setting base]

Writes calibration.json and results/calibration_sweep.json, including the complete validation curve and
benign/injected candidate-score distributions. The test split is never used for threshold selection.
"""
import argparse, datetime, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app import config
from app.benchmarks import load_injecagent
from app.embeddings import get_embedder
from app.evaluation import sweep_candidate_tau

ap = argparse.ArgumentParser()
ap.add_argument("--lo", type=float, default=0.00); ap.add_argument("--hi", type=float, default=1.00)
ap.add_argument("--step", type=float, default=0.01); ap.add_argument("--max-fpr", type=float, default=0.10)
ap.add_argument("--setting", default="base", choices=["base", "enhanced", "both"])
a = ap.parse_args()

emb = get_embedder()
cases = load_injecagent(setting=a.setting, split="val")
sw = sweep_candidate_tau(cases, emb, a.lo, a.hi, a.step, a.max_fpr, split="val")
print(f"embedder={emb.name}  cases={len(cases)} (attacks={sum(c['is_attack'] for c in cases)})")
print(f"{'tau':>5} {'P':>5} {'R':>5} {'F1':>5} {'FPR':>5}")
for p in sw["points"]:
    print(f"{p['tau']:5.2f} {p['precision']:5.2f} {p['recall']:5.2f} {p['f1']:5.2f} {p['fpr']:5.2f}")
print("FPR-recall curve (FPR, recall):")
print("; ".join(f"({p['fpr']:.4f}, {p['recall']:.4f})" for p in sw["fpr_recall_curve"]))
result_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
result_dir.mkdir(parents=True, exist_ok=True)
result_path = result_dir / "calibration_sweep.json"
result_payload = {
    "embedder": emb.name, "source_split": "validation", "setting": a.setting,
    "n_cases": sw["n_cases"], "grid": [a.lo, a.hi, a.step], "max_fpr": a.max_fpr,
    "decision_rule": sw["decision_rule"],
    "selection_status": sw["selection_status"], "best": sw["best"],
    "points": sw["points"], "fpr_recall_curve": sw["fpr_recall_curve"],
    "score_distributions": sw["score_distributions"],
}
result_path.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
print("wrote", result_path)
b = sw["best"]
if b is None:
    raise SystemExit(f"{sw['selection_status']}; no tau was written to calibration.json")
print("calibrated tau =", b["tau"], "| F1 plateau", b["plateau"], "| touches grid edge:", b["at_edge"])
out = {"embedder": emb.name, "tau": b["tau"], "mode": "G-B candidate scoring",
       "dataset": f"InjecAgent {a.setting}, dh+ds, validation split",
       "selection": f"{sw['decision_rule']}; select maximum validation F1 under the configured FPR cap; test split is never used",
       "n_cases": len(cases), "grid": [a.lo, a.hi, a.step], "max_fpr": a.max_fpr, "plateau": b["plateau"], "at_edge": b["at_edge"],
       "date": datetime.date.today().isoformat(), "points": sw["points"],
       "fpr_recall_curve": sw["fpr_recall_curve"],
       "score_distributions": sw["score_distributions"]}
pathlib.Path(config.CALIBRATION_FILE).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("wrote", config.CALIBRATION_FILE)
