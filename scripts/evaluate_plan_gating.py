from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation import evaluate_plan_gating_utility, evaluate_planner_reliability
from app.gateway import Gateway
from app.agents import ProposedAction
from app.orchestrator import RulePlanner


DEFAULT_MODES = ["G-AB", "G-AB+P", "G-AB+S", "G-AB+P+E"]


class DynamicUtilityBackend:
    """Deterministic simulated planner/agent for benign, data-driven follow-up tool use."""

    name = "simulated-dynamic-utility"

    def __init__(self) -> None:
        self.planner = RulePlanner()

    def plan(self, user_query: str, tool_catalog=None) -> str:
        plan = self.planner.plan(user_query, tool_catalog=tool_catalog)
        return json.dumps([
            {"text": task.text, "tool": task.tool, "args": task.args}
            for task in plan
        ])

    def decide(self, observation: str, downstream: bool = False, **ctx) -> list[ProposedAction]:
        if "verify" not in str(ctx.get("goal", "")).lower():
            return []
        identifier = re.search(r"(?:citation|source) identifier:\s*([\w-]+)", observation, re.I)
        if not identifier:
            return []
        value = identifier.group(1)
        return [ProposedAction(
            "search_web", {"query": value},
            f"Verify the retrieved citation using search_web for {value}.",
        )]

    def summarize(self, observation: str, token_budget: int | None = None) -> str:
        return observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure utility costs and planner reliability for plan gating.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "dynamic_tool_use.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "plan_gating_utility.json")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.0,
                        help="Use 0.0 to isolate plan-gating utility from semantic filtering.")
    args = parser.parse_args()

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    backend = DynamicUtilityBackend()
    gateway = Gateway(llm=backend, planner_type="llm", seed=args.seed, temperature=0.0, tau=args.tau)
    utility = evaluate_plan_gating_utility(gateway, cases, tuple(args.modes))
    reliability = evaluate_planner_reliability(gateway.planner, cases)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "planner_type": gateway.planner_type,
        "tau": gateway.tau,
        "dataset": str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
        "seed": args.seed,
        "planner_reliability": reliability,
        **utility,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
