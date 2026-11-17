from __future__ import annotations

import csv
import json
import pathlib
import pytest

from scripts.run_matrix import (
    get_git_commit,
    set_random_seed,
    mean_std,
    percentile,
    categorize_case,
    check_backend_available,
    load_workload_cases,
    evaluate_single_mode,
    run_matrix,
    generate_summary_tables,
    wilson_interval,
)
from app.gateway import Gateway


def test_git_commit_and_provenance():
    """get_git_commit returns a non-empty string and does not crash."""
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_mean_std_and_percentile():
    """Statistics functions compute expected statistical properties."""
    empty = mean_std([])
    assert empty == {"mean": 0.0, "std": 0.0}

    single = mean_std([5.0])
    assert single == {"mean": 5.0, "std": 0.0}

    multi = mean_std([10.0, 20.0, 30.0])
    assert multi["mean"] == 20.0
    assert multi["std"] == 10.0

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p50 = percentile(data, 50)
    assert p50 == 5.0 or p50 == 6.0
    p95 = percentile(data, 95)
    assert p95 == 10.0


def test_categorize_case_scenarios():
    """Case categorizer identifies each designated scenario row."""
    # 1. Benign
    assert categorize_case({"is_attack": False, "scenario": "benign"}) == "benign"

    # 2. Static injection
    assert categorize_case({"is_attack": True, "scenario": "static"}) == "static injection"

    # 3. Token attacks
    assert categorize_case({"is_attack": True, "rogue": {"token_attack": "forged"}}) == "forged token"
    assert categorize_case({"is_attack": True, "rogue": {"token_attack": "tampered"}}) == "tampered token"
    assert categorize_case({"is_attack": True, "rogue": {"token_attack": "expired"}}) == "expired token"
    assert categorize_case({"is_attack": True, "rogue": {"token_attack": "replayed"}}) == "replayed token"
    assert categorize_case({"is_attack": True, "rogue": {"token_attack": "wrong_root"}}) == "wrong_root token"

    # 4. Adaptive
    assert categorize_case({"is_attack": True, "scenario": "adaptive"}) == "adaptive/camouflaged"
    assert categorize_case({"is_attack": True, "scenario": "adaptive", "attack_type": "hidden_text"}) == "adaptive/camouflaged"

    # 5. Propagation
    assert categorize_case({"is_attack": True, "scenario": "propagation"}) == "multi-hop handoff propagation"


def test_wilson_interval_reports_bounded_binomial_ci():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(5, 10) == {"low": 0.2366, "high": 0.7634}


def test_drift_matrix_workload_uses_heldout_multistep_cases():
    cases, skip = load_workload_cases("drift")

    assert skip is None
    assert len(cases) == 3
    assert {case["split"] for case in cases} == {"test"}
    assert all(len(case["subtasks"]) >= 4 for case in cases)


def test_check_backend_available():
    """check_backend_available identifies availability without unhandled errors."""
    avail, reason = check_backend_available("vulnerable")
    assert avail is True
    assert reason == "ok"

    avail_bogus, reason_bogus = check_backend_available("nonexistent_backend")
    assert avail_bogus is False
    assert "Unknown backend" in reason_bogus


def test_load_workload_cases_and_skip():
    """Workload loader returns cases or clean skip reasons."""
    cases, skip = load_workload_cases("demo", limit=5)
    assert skip is None
    assert isinstance(cases, list)
    assert len(cases) == 5

    # AgentDojo skipping when not installed
    cases_dojo, skip_dojo = load_workload_cases("agentdojo")
    if skip_dojo is not None:
        assert "agentdojo package is not installed" in skip_dojo
        assert cases_dojo is None


def test_evaluate_single_mode():
    """evaluate_single_mode evaluates cases and computes all required metrics."""
    from app.dataset import split_cases
    gw = Gateway(backend="vulnerable")
    cases = split_cases("test")[:4]

    res = evaluate_single_mode(gw, "B0", cases)
    assert res["mode"] == "B0"
    assert "asr" in res
    assert "tcr_benign" in res
    assert "tcr_attack" in res
    assert "precision" in res
    assert "recall" in res
    assert "f1" in res
    assert "fpr" in res
    assert "r_tok" in res
    assert res["token_source"] not in (None, "unavailable", "none")
    assert isinstance(res["backend_tokens_sum"], int)
    assert res["token_sources"][0] in ("tiktoken", "wordpiece", "documented-estimate")
    assert "llm_prompt_tokens_sum" in res
    assert "stopped_by_layer" in res
    assert set(res["counts"]) >= {"tp", "fp", "tn", "fn"}
    assert "latency_ms" in res
    assert "scenarios" in res


def test_run_matrix_end_to_end(tmp_path):
    """run_matrix executes full matrix workflow, writes individual and summary JSONs, and exports tables."""
    out_file = tmp_path / "matrix_test.json"
    runs_dir = tmp_path / "runs"

    res = run_matrix(
        modes=["B0", "FULL"],
        backends=["vulnerable"],
        workloads=["demo"],
        seeds=[17, 29],
        limit=4,
        output_path=out_file,
        runs_dir=runs_dir,
    )

    # Check output JSON files
    assert out_file.exists()
    assert runs_dir.exists()
    run_files = list(runs_dir.glob("*.json"))
    assert len(run_files) == 2  # 2 seeds for vulnerable demo

    # Check matrix structure
    assert "provenance" in res
    assert "workloads" in res
    assert "demo" in res["workloads"]
    demo_vulnerable = res["workloads"]["demo"]["vulnerable"]
    assert "summary" in demo_vulnerable
    assert "B0" in demo_vulnerable["summary"]
    assert "FULL" in demo_vulnerable["summary"]

    b0_summary = demo_vulnerable["summary"]["B0"]
    full_summary = demo_vulnerable["summary"]["FULL"]
    assert "asr" in full_summary
    assert "added_latency_ms" in full_summary
    assert "median" in full_summary["added_latency_ms"]
    assert "p95" in full_summary["added_latency_ms"]
    assert full_summary["wilson_95"]["tcr_benign"] is not None
    assert "f1" in full_summary["wilson_95"]
    assert "token_sources" in full_summary
    assert "stopped_by_layer" in full_summary

    # Check generated summary tables
    md_file = tmp_path / "summary_table.md"
    csv_file = tmp_path / "summary_table.csv"
    sc_md_file = tmp_path / "scenarios_table.md"
    sc_csv_file = tmp_path / "scenarios_table.csv"
    layer_md_file = tmp_path / "layer_stops_table.md"
    layer_csv_file = tmp_path / "layer_stops_table.csv"
    type_md_file = tmp_path / "attack_types_table.md"
    type_csv_file = tmp_path / "attack_types_table.csv"

    assert md_file.exists()
    assert csv_file.exists()
    assert sc_md_file.exists()
    assert sc_csv_file.exists()
    assert layer_md_file.exists()
    assert layer_csv_file.exists()
    assert type_md_file.exists()
    assert type_csv_file.exists()

    md_content = md_file.read_text(encoding="utf-8")
    assert "Overall Configuration Performance" in md_content
    assert "FULL" in md_content
    assert "B0" in md_content
    assert "Wilson" in md_content

    # Read and validate CSV
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 3  # Header + B0 + FULL
        header = reader[0]
        assert "Mode" in header
        assert "ASR" in header
        assert "Δ Med Latency" in header
        assert "R_tok" in header


def test_generate_tables_only(tmp_path):
    """generate_summary_tables parses JSON results and outputs markdown and CSV tables."""
    dummy_data = {
        "provenance": {"git_commit": "abc1234", "timestamp": "2026-11-17T00:00:00", "tau": 0.12},
        "workloads": {
            "demo": {
                "vulnerable": {
                    "summary": {
                        "B0": {
                            "asr": {"mean": 1.0, "std": 0.0},
                            "tcr_benign": {"mean": 1.0, "std": 0.0},
                            "tcr_attack": {"mean": 1.0, "std": 0.0},
                            "precision": {"mean": 0.0, "std": 0.0},
                            "recall": {"mean": 0.0, "std": 0.0},
                            "f1": {"mean": 0.0, "std": 0.0},
                            "fpr": {"mean": 0.0, "std": 0.0},
                            "r_tok": {"mean": 0.0, "std": 0.0},
                            "added_latency_ms": {"median": {"mean": 0.0, "std": 0.0}, "p95": {"mean": 0.0, "std": 0.0}},
                            "scenarios": {},
                        },
                        "FULL": {
                            "asr": {"mean": 0.0, "std": 0.0},
                            "tcr_benign": {"mean": 1.0, "std": 0.0},
                            "tcr_attack": {"mean": 1.0, "std": 0.0},
                            "precision": {"mean": 1.0, "std": 0.0},
                            "recall": {"mean": 1.0, "std": 0.0},
                            "f1": {"mean": 1.0, "std": 0.0},
                            "fpr": {"mean": 0.0, "std": 0.0},
                            "r_tok": {"mean": 0.73, "std": 0.01},
                            "added_latency_ms": {"median": {"mean": 12.5, "std": 1.2}, "p95": {"mean": 18.0, "std": 2.0}},
                            "scenarios": {
                                "static injection": {"n": 5, "asr": {"mean": 0.0, "std": 0.0}, "detected": {"mean": 1.0, "std": 0.0}, "tcr_attack": {"mean": 1.0, "std": 0.0}},
                            },
                        },
                    }
                }
            }
        }
    }

    md_path, csv_path = generate_summary_tables(dummy_data, tmp_path)
    assert md_path.exists()
    assert csv_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "abc1234" in content
    assert "FULL" in content
    assert "static injection" in content
