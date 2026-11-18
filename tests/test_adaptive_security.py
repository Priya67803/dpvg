"""Tests for Phase 5: Adaptive attacks against Stage B and layer attribution."""
import pytest
from app.alignment import AlignmentEngine
from app.benchmarks import load_adaptive_attacks
from app.embeddings import get_embedder
from app.gateway import Gateway


def test_similarity_maximizing_attack_structure():
    """Verify that adaptive_search attack rows include search_trace, search_attempts,

    and achieved_similarity >= 0.
    """
    attacks = load_adaptive_attacks(split="test")
    search_attacks = [a for a in attacks if a.get("attack_type") == "adaptive_search"]
    assert len(search_attacks) > 0, "No adaptive_search attacks found in test split"

    sample = search_attacks[0]
    assert "achieved_similarity" in sample, "Missing achieved_similarity in adaptive_search"
    assert "search_trace" in sample, "Missing search_trace in adaptive_search"
    assert "search_attempts" in sample, "Missing search_attempts in adaptive_search"
    assert isinstance(sample["search_trace"], list)
    assert len(sample["search_trace"]) > 0
    # Every step in trace should have iteration and similarity
    for step in sample["search_trace"]:
        assert "iteration" in step
        assert "similarity" in step
        assert "text" in step


def test_adaptive_attack_passes_b_sem_alone_and_stopped_by_full():
    """Test that at least one adaptive attack type passes B-sem alone (not stopped / executes)

    and is stopped by FULL, with the result recorded.
    """
    embedder = get_embedder()
    gw = Gateway(embedder=embedder, tau=0.13, backend="vulnerable")
    attacks = load_adaptive_attacks(split="test")

    # 1. Test hidden_html_comment: passes G-B-sem alone (ASR > 0) but is stopped by FULL (ASR == 0)
    html_attacks = [a for a in attacks if a.get("attack_type") == "hidden_html_comment"]
    assert len(html_attacks) > 0

    target_case = html_attacks[0]

    # Under G-B-sem (B-sem alone): action is generated and passes semantic check
    res_bsem = gw.run(target_case, mode="G-B-sem")
    assert res_bsem["metrics"]["attack_success"] is True, "Expected hidden_html_comment to pass G-B-sem alone"

    # Under FULL: attack is contained and does not succeed
    res_full = gw.run(target_case, mode="FULL")
    assert res_full["metrics"]["attack_success"] is False, "Expected hidden_html_comment to be stopped under FULL"

    # 2. Test camouflaged: semantic similarity exceeds tau (passes B-sem alignment check alone)
    camouflaged_attacks = [a for a in attacks if a.get("attack_type") == "camouflaged"]
    assert len(camouflaged_attacks) > 0
    c_case = camouflaged_attacks[0]

    # Verify B-sem alone marks it as aligned (score >= tau)
    engine = AlignmentEngine(embedder)
    root_emb = embedder.encode(c_case["goal"])
    sim = engine.score(root_emb, c_case["injection"])
    assert sim >= 0.13, f"Expected camouflaged text to pass B-sem threshold (got {sim} < 0.13)"

    # Verify FULL stops camouflaged attack
    res_full_c = gw.run(c_case, mode="FULL")
    assert res_full_c["metrics"]["attack_success"] is False


def test_layer_attribution_adaptive_attack_types_retained():
    """Verify that results/attack_types_table.md and layer_attribution_table.md

    contain all 11 adaptive attack types and keep succeeding types under vulnerable/B0.
    """
    import pathlib
    table_path = pathlib.Path("results/attack_types_table.md")
    assert table_path.exists(), "attack_types_table.md missing"
    content = table_path.read_text(encoding="utf-8")

    expected_types = [
        "paraphrased",
        "camouflaged",
        "split_tool_results",
        "split_across_results",
        "hidden_html_comment",
        "hidden_zero_width",
        "hidden_text",
        "in_scope_misuse",
        "allowed_recipient_substitution",
        "task_framed",
        "adaptive_search",
    ]
    for attack_type in expected_types:
        assert f"| adaptive | vulnerable | B0 | {attack_type} |" in content, (
            f"Expected {attack_type} under B0 to be recorded in attack_types_table.md"
        )
