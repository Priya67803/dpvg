"""Tests for Phase 4: Hard-Benign Dataset & Expanded Drift Scenarios.

Verifies:
1. Hard-benign dataset integrity (>= 300 items across 7 required categories, val/test split).
2. Drift dataset integrity (>= 150 multi-step scenarios across 5 categories, val/test split).
3. Calibration.json separation between active drift calibration and superseded legacy estimate.
4. Drift parameter sensitivity and held-out test split evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app import config

ROOT = Path(__file__).resolve().parents[1]


def test_hard_benign_dataset_integrity():
    """Hard-benign set must have >= 300 items, 7 categories, and clean val/test splits."""
    path = ROOT / "data" / "hard_benign.json"
    assert path.exists(), f"Missing hard-benign dataset: {path}"

    items = json.loads(path.read_text(encoding="utf-8"))
    assert len(items) >= 300, f"Expected at least 300 items, got {len(items)}"

    required_groups = {
        "imperatives",
        "polite_requests",
        "recipes",
        "meeting_agendas",
        "todo_lists",
        "human_instructions",
        "quoted_commands",
    }
    actual_groups = {it.get("group") for it in items}
    assert required_groups.issubset(actual_groups)

    # Every item is benign
    assert all(it.get("is_attack") is False for it in items)
    assert all(it.get("injection") is None for it in items)

    # Both validation and test splits are present
    splits = {it.get("split") for it in items}
    assert splits == {"val", "test"}
    val_count = sum(1 for it in items if it.get("split") == "val")
    test_count = sum(1 for it in items if it.get("split") == "test")
    assert val_count >= 100
    assert test_count >= 100


def test_drift_dataset_integrity():
    """Drift scenarios set must have >= 150 multi-step scenarios across 5 categories."""
    path = ROOT / "data" / "drift_scenarios.json"
    assert path.exists(), f"Missing drift dataset: {path}"

    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    assert len(scenarios) >= 150, f"Expected at least 150 scenarios, got {len(scenarios)}"

    required_categories = {
        "benign_long_task",
        "benign_topic_shift",
        "gradual_drift",
        "sudden_jump",
        "slow_drip",
    }
    actual_categories = {s.get("category") for s in scenarios}
    assert required_categories.issubset(actual_categories)

    # All scenarios have at least 4 steps
    assert all(len(s.get("steps", [])) >= 4 for s in scenarios)

    # Both validation and test splits present
    splits = {s.get("split") for s in scenarios}
    assert splits == {"val", "test"}


def test_calibration_json_separates_legacy_and_new_drift():
    """Calibration.json must keep legacy 11-scenario estimate marked superseded and new 160-scenario calibration."""
    cal_path = Path(config.CALIBRATION_FILE)
    assert cal_path.exists()
    calibration = json.loads(cal_path.read_text(encoding="utf-8"))

    # Active drift calibration
    assert "drift" in calibration
    active = calibration["drift"]
    assert "tau_drift" in active
    assert "consecutive_steps" in active
    assert "ema_alpha" in active
    assert active.get("selection_split") == "val"
    assert "heldout_test_detection_rate" in active
    assert "heldout_test_false_block_rate" in active
    assert active.get("scenario_count_total", 0) >= 150

    # Legacy 11-scenario calibration marked superseded
    assert "drift_legacy_11_scenarios" in calibration
    legacy = calibration["drift_legacy_11_scenarios"]
    assert "superseded" in legacy.get("status", "").lower()
    assert legacy.get("scenario_count") == 11


def test_drift_sensitivity_reports_exist():
    """Verify that sensitivity and hard-benign results artifacts were generated."""
    sens_md = ROOT / "results" / "drift_sensitivity.md"
    sens_json = ROOT / "results" / "drift_sensitivity.json"
    hb_md = ROOT / "results" / "hard_benign_summary.md"
    hb_json = ROOT / "results" / "hard_benign_results.json"

    assert sens_md.exists()
    assert sens_json.exists()
    assert hb_md.exists()
    assert hb_json.exists()

    hb_data = json.loads(hb_json.read_text(encoding="utf-8"))
    assert "stage_b_fpr" in hb_data
    assert "mode_tcr" in hb_data
