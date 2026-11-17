"""Calibrate semantic drift controls using only scenarios marked split=val."""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config
from app.embeddings import cosine, get_embedder
from app.registry import spec_to_text


def _drift_triggered(scores: list[float], threshold: float, steps: int, alpha: float) -> bool:
    ema = None
    below = 0
    for score in scores:
        ema = score if ema is None else alpha * score + (1 - alpha) * ema
        below = below + 1 if ema < threshold else 0
        if below >= steps:
            return True
    return False


def select_drift_parameters(
    scenarios: list[dict],
    embedder,
    *,
    max_fpr: float = 0.10,
    threshold_values: list[float] | None = None,
    step_values: list[int] | None = None,
    alpha_values: list[float] | None = None,
) -> dict:
    splits = {scenario.get("split") for scenario in scenarios}
    if not scenarios or splits != {"val"}:
        raise ValueError("drift parameter selection accepts validation scenarios only")
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be between 0 and 1")
    threshold_values = threshold_values or [round(i / 100, 2) for i in range(5, 61, 5)]
    step_values = step_values or [1, 2, 3, 4]
    alpha_values = alpha_values or [round(i / 10, 1) for i in range(2, 9)]

    trajectories = []
    for scenario in scenarios:
        root = embedder.encode(spec_to_text({"goal": scenario["root_intent"]}))
        scores = [cosine(embedder.encode(step), root) for step in scenario["steps"]]
        trajectories.append((scenario, scores))

    points = []
    for threshold in threshold_values:
        for n_steps in step_values:
            for alpha in alpha_values:
                decisions = [
                    _drift_triggered(scores, threshold, n_steps, alpha)
                    for _, scores in trajectories
                ]
                benign = [
                    decision for (scenario, _), decision in zip(trajectories, decisions)
                    if scenario["label"] == "benign"
                ]
                attacks = [
                    decision for (scenario, _), decision in zip(trajectories, decisions)
                    if scenario["label"] == "attack"
                ]
                false_blocks = sum(benign) / len(benign) if benign else 0.0
                detection = sum(attacks) / len(attacks) if attacks else 0.0
                points.append({
                    "tau_drift": threshold,
                    "consecutive_steps": n_steps,
                    "ema_alpha": alpha,
                    "detection_rate": round(detection, 4),
                    "false_block_rate": round(false_blocks, 4),
                    "detected_attacks": sum(attacks),
                    "attack_scenarios": len(attacks),
                    "false_blocked_benign": sum(benign),
                    "benign_scenarios": len(benign),
                })

    eligible = [point for point in points if point["false_block_rate"] <= max_fpr]
    if not eligible:
        raise ValueError(f"no drift parameters satisfy validation FPR <= {max_fpr:.3f}")
    best = min(
        eligible,
        key=lambda point: (
            -point["detection_rate"],
            point["false_block_rate"],
            -point["consecutive_steps"],
            point["tau_drift"],
            point["ema_alpha"],
        ),
    )
    return {
        "selection_split": "val",
        "max_fpr": max_fpr,
        "scenario_count": len(scenarios),
        "grid": {
            "tau_drift": threshold_values,
            "consecutive_steps": step_values,
            "ema_alpha": alpha_values,
        },
        "best": best,
        "points": points,
        "selection_rule": "maximize validation detection rate under the false-block cap; then minimize false-block rate, prefer larger N, then lower threshold and alpha",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fpr", type=float, default=0.10)
    parser.add_argument("--data", type=pathlib.Path, default=ROOT / "data" / "drift_scenarios.json")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "drift_calibration_sweep.json")
    args = parser.parse_args()
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    validation = [scenario for scenario in payload["scenarios"] if scenario.get("split") == "val"]
    embedder = get_embedder()
    result = select_drift_parameters(validation, embedder, max_fpr=args.max_fpr)
    result.update({
        "embedder": embedder.name,
        "source_data": str(args.data.relative_to(ROOT)),
        "date": datetime.date.today().isoformat(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    best = result["best"]
    calibration_path = pathlib.Path(config.CALIBRATION_FILE)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["drift"] = {
        "tau_drift": best["tau_drift"],
        "consecutive_steps": best["consecutive_steps"],
        "ema_alpha": best["ema_alpha"],
        "selection_split": "val",
        "max_fpr": args.max_fpr,
        "detection_rate": best["detection_rate"],
        "false_block_rate": best["false_block_rate"],
        "source_data": str(args.data.relative_to(ROOT)),
        "date": result["date"],
    }
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected drift params on validation only: tau_drift={best['tau_drift']:.2f}, "
        f"N={best['consecutive_steps']}, alpha={best['ema_alpha']:.1f}, "
        f"detection={best['detection_rate']:.3f}, false-block={best['false_block_rate']:.3f}"
    )
    print(f"Saved full sweep to {args.output} and updated {calibration_path}")


if __name__ == "__main__":
    main()
