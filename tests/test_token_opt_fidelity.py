"""Tests for Phase 6: Token optimization fidelity, protected user preferences,

counter source tracking, and dynamic budget truncation.
"""
import pytest
from app.embeddings import get_embedder
from app.gateway import Gateway
from app.token_opt import (
    PROTECTED_CLASSES,
    TokenOptimizer,
    count_tokens,
    get_token_counter_source,
    is_protected_chunk,
    is_user_preference,
)


def test_user_preference_detection():
    """Verify is_user_preference detects style, formatting, and language preferences."""
    pref = "The user prefers concise answers written in British English"
    assert is_user_preference(pref) is True

    pref_dict = {"class": "user_preference", "text": "Format output in markdown tables."}
    assert is_user_preference(pref_dict) is True

    # Negative controls
    assert is_user_preference("Earlier today the user asked about the weather in Mumbai.") is False
    assert is_user_preference("The cricket match ended in a draw.") is False
    assert is_user_preference("Reminder: the kitchen coffee machine is broken.") is False


def test_user_preference_preserved_while_distractors_pruned():
    """Test that 'The user prefers concise answers written in British English'

    is kept by context pruning while weather, cricket, and kitchen chunks are pruned.
    """
    embedder = get_embedder()
    opt = TokenOptimizer(embedder, total_budget=3000, min_sim=0.30)

    query = "Summarize the quarterly financial report for Q3."
    user_pref = "The user prefers concise answers written in British English"
    weather = "The weather forecast in London predicts cloudy skies with intermittent drizzle."
    cricket = "Today's cricket match between India and Australia ended in a draw after rain interruption."
    kitchen = "The kitchen recipe requires two cups of flour, one cup of sugar, and three eggs."

    chunks = [user_pref, weather, cricket, kitchen]

    # Test direct pruning with min_sim cutoff
    kept, dropped = opt.prune(chunks, query, budget=500, min_similarity=0.30)

    assert user_pref in kept, "Protected user preference was unexpectedly dropped"
    assert weather in dropped, "Weather chunk should be pruned"
    assert cricket in dropped, "Cricket chunk should be pruned"
    assert kitchen in dropped, "Kitchen chunk should be pruned"

    # Test full optimize call
    result = opt.optimize(query, chunks, protected=[query, user_pref])
    assert user_pref in result.context, "User preference missing from optimized context"
    assert weather in result.dropped_context
    assert cricket in result.dropped_context
    assert kitchen in result.dropped_context


def test_token_counter_source_reporting():
    """Verify token counter source is reported correctly (tiktoken, wordpiece, or approximate)."""
    source = get_token_counter_source()
    assert source in {"tiktoken", "wordpiece", "approximate"}

    # Test Gateway records the active token counter source
    gw = Gateway(backend="ollama")
    case = {
        "id": "t_pref_01",
        "query": "What is the capital of France?",
        "context": [
            "The user prefers concise answers written in British English",
            "Random kitchen notes on coffee recipes.",
        ],
        "is_attack": False,
    }
    res = gw.run(case, mode="G-AB+T")
    metrics = res["metrics"]
    assert "token_counter_source" in metrics or "llm_token_source" in metrics
    recorded = metrics.get("token_counter_source") or metrics.get("llm_token_source")
    assert recorded in {"tiktoken", "wordpiece", "documented-estimate", "approximate"}


def test_multi_subtask_dynamic_budget_truncation():
    """Verify that when total budget is smaller than the sum of sub-task needs,

    the dynamic budget truncates context and records dropped chunks.
    """
    embedder = get_embedder()
    # Very small total budget (e.g. 15 tokens) where subtasks consume most/all budget
    tight_budget = 25
    opt = TokenOptimizer(embedder, total_budget=tight_budget, min_sim=0.0)

    # Multi-subtask query with 3 subtasks
    query = "Search for recent financial news; and then draft a summary email; and then file the report."

    relevant_ctx1 = "Financial markets showed solid growth across tech sectors in Q3."
    relevant_ctx2 = "Summary email should highlight key revenue milestones."
    relevant_ctx3 = "Filing report requires archiving into the internal database."

    chunks = [relevant_ctx1, relevant_ctx2, relevant_ctx3]

    result = opt.optimize(query, chunks, protected=[query])

    # Decomposed into subtasks
    assert len(result.subtasks) >= 2, f"Expected multi-subtask decomposition, got {result.subtasks}"

    # Verify allocations exist for each subtask
    for i in range(len(result.subtasks)):
        assert f"subtask_{i}" in result.budget
        assert result.budget[f"subtask_{i}"] >= 0

    # Total allocated budget should not exceed total_budget
    assert sum(result.budget.values()) <= tight_budget

    # Because total budget is tight (25 tokens) and query subtasks consume tokens,
    # at least some relevant context chunks must be dropped due to budget truncation
    assert len(result.dropped_context) > 0, "Expected context to be truncated under tight budget"

    # Verify dropped_context is explicitly reported
    for dropped_item in result.dropped_context:
        assert isinstance(dropped_item, str)
        assert dropped_item in chunks
