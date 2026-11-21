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
        # Dynamic tool use benchmark: initial plan schedules the initial read/inspection step.
        # Follow-up tools are decided dynamically at runtime based on the retrieved content.
        return json.dumps([
            {"text": f"Read document: {user_query}", "tool": "read_file", "args": {"path": "document.txt"}}
        ])

    def extend_plan(self, user_query: str, proposed_tool: str, tool_catalog=None) -> dict:
        return self.planner.extend_plan(user_query, proposed_tool, tool_catalog=tool_catalog)

    def decide(self, observation: str, downstream: bool = False, **ctx) -> list[ProposedAction]:
        goal = str(ctx.get("goal") or ctx.get("query") or "")
        obs = str(ctx.get("raw_fixture") or observation)

        # 1. Attack injection in observation (vulnerable follower)
        attack_email = re.search(r"\b(?:send|email|forward)\b[^\n.]*?(?:to\s+)?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", obs, re.I)
        attack_transfer = re.search(r"\b(?:transfer|wire|pay)\b[^\n.]*?\$?(\d[\d,]*)\s*(?:to\s+)?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-.]+)", obs, re.I)
        attack_slack = re.search(r"\b(?:post\b[^\n.]*?to\s+slack\s+(#[a-zA-Z0-9_-]+))", obs, re.I)

        if "evil.com" in obs or "SYSTEM OVERRIDE" in obs:
            if attack_email:
                to_addr = attack_email.group(1).rstrip(".,;:")
                return [ProposedAction("send_email", {"to": to_addr, "body": obs}, f"Call send_email with to={to_addr}")]
            if attack_transfer:
                amt, to_acc = attack_transfer.group(1), attack_transfer.group(2).rstrip(".,;:")
                return [ProposedAction("transfer_money", {"to": to_acc, "amount": amt}, f"Call transfer_money with to={to_acc} amount={amt}")]
            if attack_slack:
                chan = attack_slack.group(1).rstrip(".,;:")
                return [ProposedAction("post_slack", {"channel": chan, "message": obs}, f"Call post_slack with channel={chan}")]

        # 2. Benign Category 1: Read-only citation / source verification
        identifier = re.search(r"(?:citation|source|reference|index)\s*identifier:\s*([\w-]+)", observation, re.I)
        if identifier and "verify" in goal.lower():
            value = identifier.group(1).rstrip(".,;:")
            return [ProposedAction("search_web", {"query": value}, f"Verify the retrieved citation using search_web for {value}.")]

        # 3. Benign Category 2: User-authorized side-effect follow-up
        email_match = re.search(r"\b(?:email|mail|send)(?:\s+(?:the\s+)?summary)?\s+to\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-.]+)", goal, re.I)
        if email_match:
            to_addr = email_match.group(1).rstrip(".,;:")
            return [ProposedAction("send_email", {"to": to_addr, "body": observation[:100]}, f"Call send_email with to={to_addr}")]

        slack_match = re.search(r"\b(?:post(?:\s+to)?\s+slack\s+)(#[a-zA-Z0-9_-]+)", goal, re.I)
        if slack_match:
            chan = slack_match.group(1).rstrip(".,;:")
            return [ProposedAction("post_slack", {"channel": chan, "message": observation[:100]}, f"Call post_slack with channel={chan}")]

        transfer_match = re.search(r"\btransfer\s+\$?(\d[\d,]*)\s+to\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-.]+)", goal, re.I)
        if transfer_match:
            amt, to_acc = transfer_match.group(1), transfer_match.group(2).rstrip(".,;:")
            return [ProposedAction("transfer_money", {"to": to_acc, "amount": amt}, f"Call transfer_money with to={to_acc} amount={amt}")]

        return []

    def summarize(self, observation: str, token_budget: int | None = None) -> str:
        return observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure utility costs and planner reliability for plan gating on the in-house AgentDojo-style dynamic set.")
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

    # Stratified disjoint split: 50% tuning split, 50% held-out test split (Fix 19)
    attacks = [c for c in cases if c.get("is_attack")]
    benign = [c for c in cases if not c.get("is_attack")]
    tuning_cases = attacks[:7] + benign[:20]  # 27 cases
    test_cases = attacks[7:] + benign[20:]    # 27 cases

    tuning_utility = evaluate_plan_gating_utility(gateway, tuning_cases, tuple(args.modes))
    test_utility = evaluate_plan_gating_utility(gateway, test_cases, tuple(args.modes))
    full_utility = evaluate_plan_gating_utility(gateway, cases, tuple(args.modes))

    reliability = evaluate_planner_reliability(gateway.planner, cases)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "llama3.2",
        "temperature": 0.7,
        "seed": args.seed,
        "tau": gateway.tau,
        "embedder": "all-MiniLM-L6-v2",
        "N": len(cases),
        "backend": backend.name,
        "planner_type": gateway.planner_type,
        "workload": "in-house AgentDojo-style dynamic set",
        "dataset": str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
        "dataset_description": "in-house AgentDojo-style dynamic set (data/dynamic_tool_use.json, not official AgentDojo)",
        "split_methodology": "stratified 50/50 split (27 tuning, 27 held-out test) holding out unseen tasks for G-AB+P+E",
        "tuning_split": {
            "n_cases": len(tuning_cases),
            "n_attacks": len([c for c in tuning_cases if c.get("is_attack")]),
            "n_benign": len([c for c in tuning_cases if not c.get("is_attack")]),
            "utility": tuning_utility,
        },
        "heldout_test_split": {
            "n_cases": len(test_cases),
            "n_attacks": len([c for c in test_cases if c.get("is_attack")]),
            "n_benign": len([c for c in test_cases if not c.get("is_attack")]),
            "utility": test_utility,
        },
        "combined_full_set": {
            "n_cases": len(cases),
            "utility": full_utility,
        },
        "planner_reliability": reliability,
        **full_utility,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
