"""Diagnostic: how many CLEAN AgentDojo runs does Stage B flag?  (no injection, no LLM involved)

Flagging depends only on the tool outputs the agent receives, so we replay each user task's ground-truth tool calls
through the gateway filter and record which candidate sentences were quarantined. This is a measurement of the filter,
not an agent: it needs no Ollama server.

    python scripts/stage_b_clean_fpr.py [--suites workspace ...] [--limit-user N] [--show 40] [--tau T]
"""
import argparse, collections, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string
from app.benchmarks import make_filter_class
from app.embeddings import get_embedder
from app.gateway import Gateway


class Replay(BasePipelineElement):
    name = "replay"
    def __init__(self, ut): self.ut, self.i, self.plan = ut, 0, None
    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        if self.plan is None: self.plan = list(self.ut.ground_truth(env))
        if self.i < len(self.plan):
            c = self.plan[self.i]; self.i += 1
            m = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")],
                                     tool_calls=[FunctionCall(function=c.function, args=dict(c.args), id=f"c{len(messages)}")])
        else:
            m = ChatAssistantMessage(role="assistant", tool_calls=None, content=[text_content_block_from_string("Done.")])
        return query, runtime, env, [*messages, m], extra_args or {}


ap = argparse.ArgumentParser()
ap.add_argument("--suites", nargs="*"); ap.add_argument("--limit-user", type=int); ap.add_argument("--show", type=int, default=0)
ap.add_argument("--tau", type=float); ap.add_argument("--inj", action="store_true", help="also replay injected runs (important_instructions)"); ap.add_argument("--limit-inj", type=int)
a = ap.parse_args()
gw = Gateway(embedder=get_embedder(), tau=a.tau, llm=object())          # llm unused here
Filter = make_filter_class()
suites = get_suites("v1.2.1")
flagged = total = 0
per, examples = {}, []
for name in a.suites or list(suites):
    suite, n, f = suites[name], 0, 0
    for ut in list(suite.user_tasks.values())[:a.limit_user]:
        filt = Filter(gw, stage_a=False, stage_b=True)
        p = AgentPipeline([SystemMessage("x"), InitQuery(), Replay(ut), ToolsExecutionLoop([ToolsExecutor(), filt, Replay(ut)], max_iters=60)])
        p.name = "replay"
        # the replay element is stateful, so use a single instance
        rep = Replay(ut)
        p = AgentPipeline([SystemMessage("x"), InitQuery(), rep, ToolsExecutionLoop([ToolsExecutor(), filt, rep], max_iters=60)]); p.name = "replay"
        suite.run_task_with_pipeline(p, ut, None, {})
        n += 1
        bad = any(r["blocked"] for r in filt.records)
        f += bad
        if bad and len(examples) < max(a.show, 0):
            examples.append((name, ut.ID, ut.PROMPT[:90]))
    per[name] = (f, n); flagged += f; total += n
for k, (f, n) in per.items():
    print(f"{k:10} clean runs flagged {f}/{n} = {f / n:.1%}")
if a.inj:
    from agentdojo.attacks.attack_registry import load_attack
    probe = AgentPipeline([]); probe.name = "gpt-4o-2024-05-13"
    nf = nt = 0
    for name in a.suites or list(suites):
        suite = suites[name]; atk = load_attack("important_instructions", suite, probe); f = n = 0
        for ut in list(suite.user_tasks.values())[:a.limit_user]:
            for it in list(suite.injection_tasks.values())[:a.limit_inj]:
                inj = atk.attack(ut, it)
                filt = Filter(gw, stage_a=False, stage_b=True); rep = Replay(ut)
                p = AgentPipeline([SystemMessage("x"), InitQuery(), rep, ToolsExecutionLoop([ToolsExecutor(), filt, rep], max_iters=60)]); p.name = "replay"
                suite.run_task_with_pipeline(p, ut, it, inj)
                n += 1; f += any(r["blocked"] for r in filt.records)
        print(f"{name:10} injected runs flagged {f}/{n} = {f / n:.1%}"); nf += f; nt += n
    print(f"ALL        injected runs flagged {nf}/{nt} = {nf / nt:.1%}")
print(f"ALL        clean runs flagged {flagged}/{total} = {flagged / total:.1%}   (tau={gw.tau})")
