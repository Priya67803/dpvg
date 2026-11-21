"""Unit and integration tests for AgentDojo adapter, authorizing executor, and mode wiring."""
import pytest
from app.benchmarks import (
    make_authorizing_executor_class,
    make_filter_class,
    DPVGDojoContext,
    is_write_tool,
)
from app.gateway import Gateway
from app.embeddings import get_embedder

E = get_embedder()


def test_is_write_tool_classification():
    assert is_write_tool("send_email") is True
    assert is_write_tool("transfer_money") is True
    assert is_write_tool("send_money") is True
    assert is_write_tool("delete_file") is True
    assert is_write_tool("create_calendar_event") is True
    assert is_write_tool("get_unread_emails") is False
    assert is_write_tool("search_files_by_filename") is False
    assert is_write_tool("get_iban") is False


def test_authorizing_executor_plan_extension_approved():
    """G-AB+P+E allows benign read-only tool extension."""
    gw = Gateway(embedder=E, tau=0.55)
    ctx = DPVGDojoContext(gw, "Search for invoices in files and read them", allowed_tools=["search_files_by_filename"])
    AuthorizerClass = make_authorizing_executor_class()
    auth = AuthorizerClass(
        gw, context=ctx, enforce_scope=True, allow_plan_extension=True,
        suite_tools=["search_files_by_filename", "get_file_by_id", "send_email"]
    )

    class MockRuntime:
        def run_function(self, env, fn, args):
            return f"content of {args.get('file_id')}", None

    from types import SimpleNamespace
    call = SimpleNamespace(id="tc-ext-1", function="get_file_by_id", args={"file_id": "inv_123"})
    msgs = [{"role": "assistant", "tool_calls": [call]}]

    _, _, _, out_msgs, _ = auth.query("Search for invoices in files and read them", MockRuntime(), None, msgs, {})
    assert len(out_msgs) == 2
    # Verify plan extension event occurred and tool was allowed
    ext_events = [l for l in auth.ledger if l.get("event") == "plan_extension"]
    assert len(ext_events) == 1
    assert ext_events[0]["approved"] is True
    assert ext_events[0]["tool"] == "get_file_by_id"
    tool_dec = [l for l in auth.ledger if l.get("tool") == "get_file_by_id" and "allowed" in l]
    assert len(tool_dec) == 1
    assert tool_dec[0]["allowed"] is True


def test_authorizing_executor_plan_extension_denied_for_unauthorized_side_effect():
    """G-AB+P+E denies un-cited side-effect extension (injection attempt)."""
    gw = Gateway(embedder=E, tau=0.55)
    ctx = DPVGDojoContext(gw, "Read my calendar for today", allowed_tools=["get_day_calendar_events"])
    AuthorizerClass = make_authorizing_executor_class()
    auth = AuthorizerClass(
        gw, context=ctx, enforce_scope=True, allow_plan_extension=True,
        suite_tools=["get_day_calendar_events", "send_email", "transfer_money"]
    )

    class MockRuntime:
        def __init__(self): self.executed = False
        def run_function(self, env, fn, args):
            self.executed = True
            return "sent", None

    from types import SimpleNamespace
    call = SimpleNamespace(id="tc-ext-inj", function="send_email", args={"recipients": ["evil@attacker.com"], "body": "stolen"})
    msgs = [{"role": "assistant", "tool_calls": [call]}]

    rt = MockRuntime()
    _, _, _, out_msgs, _ = auth.query("Read my calendar for today", rt, None, msgs, {})
    assert rt.executed is False
    ext_events = [l for l in auth.ledger if l.get("event") == "plan_extension"]
    assert len(ext_events) == 1
    assert ext_events[0]["approved"] is False


def test_authorizing_executor_argument_checks():
    """FULL mode blocks unauthorized recipients on side-effect tool calls."""
    gw = Gateway(embedder=E, tau=0.55)
    ctx = DPVGDojoContext(gw, "Send the report to alice@corp.com", allowed_tools=["send_email"])
    AuthorizerClass = make_authorizing_executor_class()
    auth = AuthorizerClass(
        gw, context=ctx, enforce_scope=True, allow_plan_extension=False,
        argument_checks=True, suite_tools=["send_email"]
    )

    class MockRuntime:
        def __init__(self): self.calls = []
        def run_function(self, env, fn, args):
            self.calls.append((fn, args))
            return "sent", None

    from types import SimpleNamespace
    # Injected recipient not mentioned in query
    bad_call = SimpleNamespace(id="tc-arg-1", function="send_email", args={"recipients": ["attacker@evil.com"], "body": "secret"})
    msgs = [{"role": "assistant", "tool_calls": [bad_call]}]
    rt = MockRuntime()
    _, _, _, out_msgs, _ = auth.query("Send the report to alice@corp.com", rt, None, msgs, {})
    assert len(rt.calls) == 0
    assert auth.ledger[-1]["allowed"] is False
    assert auth.ledger[-1]["code"] == "unauthorized_argument"

    # Authorized recipient mentioned in query
    good_call = SimpleNamespace(id="tc-arg-2", function="send_email", args={"recipients": ["alice@corp.com"], "body": "report"})
    msgs_good = [{"role": "assistant", "tool_calls": [good_call]}]
    _, _, _, out_msgs_good, _ = auth.query("Send the report to alice@corp.com", rt, None, msgs_good, {})
    assert len(rt.calls) == 1
    assert auth.ledger[-1]["allowed"] is True


def test_b1_spotlighting_formatter():
    """Verify B1 spotlighting datamarks tool output strings without spaces."""
    from agentdojo.agent_pipeline import ToolsExecutor
    from app import spotlight
    try:
        from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    except ImportError:
        from agentdojo.agent_pipeline.tools_executor import tool_result_to_str

    def spotlight_formatter(r):
        return spotlight.apply(tool_result_to_str(r), "datamark")

    executor = ToolsExecutor(tool_output_formatter=spotlight_formatter)

    class MockRuntime:
        def __init__(self):
            from types import SimpleNamespace
            self.functions = {"get_balance": SimpleNamespace(name="get_balance")}
        def run_function(self, env, fn, args):
            return "Sensitive account balance is 5000 USD", None

    from types import SimpleNamespace
    call = SimpleNamespace(id="tc-sp-1", function="get_balance", args={})
    msgs = [{"role": "assistant", "tool_calls": [call]}]

    _, _, _, out_msgs, _ = executor.query("Get balance", MockRuntime(), None, msgs, {})
    assert len(out_msgs) == 2
    tool_resp = out_msgs[1]
    content_text = tool_resp["content"][0]["content"] if isinstance(tool_resp, dict) else tool_resp.content[0].content
    assert " " not in content_text
    assert spotlight.MARK in content_text
    assert spotlight.invert(content_text, "datamark") == "Sensitive account balance is 5000 USD"

