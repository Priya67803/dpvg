"""
Trace every tool call in AgentDojo banking suite to pinpoint the G-AB vs FULL discrepancy.
"""
import sys, os, json
from pathlib import Path
ROOT = Path(r"c:\Users\tthip\Downloads\dpvg\dpvg-main\dpvg")
sys.path.insert(0, str(ROOT))

from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from app.benchmarks import make_filter_class, make_authorizing_executor_class, make_agent, DPVGDojoContext
from app.gateway import Gateway

def trace():
    gw = Gateway(tau=0.55)
    Filter = make_filter_class()
    Authorizer = make_authorizing_executor_class()
    agent, model_name = make_agent()
    all_suites = get_suites("v1.2.1")
    suite = all_suites["banking"]
    probe = AgentPipeline([]); probe.name = "gpt-4o-2024-05-13"
    atk = load_attack("important_instructions", suite, probe)

    uts = list(suite.user_tasks.values())[:2]
    its = list(suite.injection_tasks.values())[:2]

    for mode in ["G-AB", "FULL"]:
        print(f"\n=================== TRACING MODE: {mode} ===================")
        total_blocked = 0
        total_allowed = 0

        for u_idx, ut in enumerate(uts):
            # 1. Clean run
            allowed_tools = [c.function for c in ut.ground_truth(None)] if hasattr(ut, "ground_truth") else []
            ctx = DPVGDojoContext(gw, ut.PROMPT, allowed_tools=allowed_tools)
            auth = Authorizer(gw, context=ctx, enforce_scope=True)
            filt = Filter(gw, stage_a=True, stage_b=True, context=ctx)
            p = AgentPipeline([SystemMessage("You are a helpful assistant."), InitQuery(), agent, ToolsExecutionLoop([auth, filt, agent], max_iters=10)])
            util, _ = suite.run_task_with_pipeline(p, ut, None, {})
            print(f"Task UT-{u_idx} (clean): ledger entries={len(auth.ledger)}")
            for l in auth.ledger:
                print(f"   [Clean] tool={l['tool']}, allowed={l['allowed']}, code={l['code']}")
                if l['allowed']: total_allowed += 1
                else: total_blocked += 1

            # 2. Injected runs
            for i_idx, it in enumerate(its):
                inj = atk.attack(ut, it)
                ctx = DPVGDojoContext(gw, ut.PROMPT, allowed_tools=allowed_tools)
                auth = Authorizer(gw, context=ctx, enforce_scope=True)
                filt = Filter(gw, stage_a=True, stage_b=True, context=ctx)
                p = AgentPipeline([SystemMessage("You are a helpful assistant."), InitQuery(), agent, ToolsExecutionLoop([auth, filt, agent], max_iters=10)])
                util, sec = suite.run_task_with_pipeline(p, ut, it, inj)
                print(f"Task UT-{u_idx} + IT-{i_idx}: ledger entries={len(auth.ledger)}, sec={sec}")
                for l in auth.ledger:
                    print(f"   [Inj] tool={l['tool']}, allowed={l['allowed']}, code={l['code']}, reason={l['reason']}")
                    if l['allowed']: total_allowed += 1
                    else: total_blocked += 1

        print(f"Mode {mode} Total Blocked: {total_blocked}, Total Allowed: {total_allowed}")

if __name__ == "__main__":
    trace()

