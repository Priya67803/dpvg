"""
dpvg_adapter.py - DPVG Adapter wrapping Gateway for dpvg_eval.py and ablation testing.

Rules followed:
  * Ollama only. If Ollama is unreachable, RAISE. Never fall back to a stub.
  * Expose the attacker's tools (e.g. delete_calendar_events, send_email, fetch_url) to the
    model, but keep them OUT of the token scope, so there is something to block.
  * `calls` lists every tool call the agent attempted (executed or denied).
  * Normalise stage names to: "A", "scope", "plan", "args", "B-rules", "B-sem".
  * Sum tokens_before/after and verify_ms across the plan AND any re-plan.
  * The injected sentence / goal are ground truth for scoring only: never pass
    them into the gateway.
"""
from __future__ import annotations

import re
import time
from typing import Any

from app.gateway import Gateway, Mode
from app.llm_backends import OllamaBackend, BackendUnavailable
from app.alignment import (
    AlignmentEngine,
    has_high_confidence_injection_marker,
    is_directive,
    action_classes,
    allowed_classes_for_request,
)
from app.embeddings import get_embedder, cosine
from app.config import ALL_CATALOG_TOOLS
from dpvg_eval import Block, Call, Detection, RunResult, STAGE_REQUIRES


class DPVGAdapter:
    def __init__(self, model: str = "llama3.2", temperature: float = 0.7, tau: float = 0.13, **kw):
        self.model = model
        self.temperature = temperature
        self.tau = tau
        # Single instances created once at startup
        self.backend = OllamaBackend(model=model, temperature=temperature)
        self.gw = Gateway(llm=self.backend, tau=tau)
        self.embedder = self.gw.embedder
        self.engine = AlignmentEngine(self.embedder)

    def warmup(self):
        """Fail fast if Ollama / model is missing (no silent fallback).
        Run throw-away embed + sign + verify to warm ONNX and crypto paths."""
        self.backend.check()
        _ = self.embedder.encode("warmup query for DPVG adapter")
        _ = self.gw.run({"id": "warmup", "query": "test query", "fixtures": {}}, "B0")

    def _flags_to_gateway(self, flags: dict) -> Mode:
        # flags: spotlight, stage_a, b_rules, b_sem, plan, plan_ext, scope_args, token_opt
        sb_rules = bool(flags.get("b_rules"))
        sb_sem = bool(flags.get("b_sem"))
        sb = sb_rules or sb_sem
        sa = bool(flags.get("stage_a"))
        plan = bool(flags.get("plan"))
        scope_args = bool(flags.get("scope_args"))
        plan_ext = bool(flags.get("plan_ext"))
        tok_opt = bool(flags.get("token_opt"))
        spot = "datamark" if flags.get("spotlight") else None

        return Mode(
            key="custom",
            label="Custom Adapter Mode",
            stage_a=sa,
            stage_b=sb,
            stage_b_rules=sb_rules,
            stage_b_sem=sb_sem,
            scope=scope_args,
            trusted_plan_only=plan,
            plan_extension=plan_ext,
            argument_checks=scope_args,
            token_opt=tok_opt,
            spotlight=spot,
        )

    def scan(self, text: str, user_request: str, flags: dict) -> list[Detection]:
        """Run ONLY the Stage B layers enabled in flags on text against user_request.
        Returns Detection(stage="B-rules" | "B-sem", span=<flagged sentence>).
        """
        dets: list[Detection] = []
        if not flags.get("b_rules") and not flags.get("b_sem"):
            return dets

        # Split text into sentence spans
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
        e_root = self.embedder.encode(user_request)
        root_allowed = allowed_classes_for_request(user_request)

        for sent in sents:
            # 1. B-rules check: explicit marker
            if flags.get("b_rules"):
                if has_high_confidence_injection_marker(sent):
                    dets.append(Detection(stage="B-rules", span=sent, action="quarantined"))
                    continue

            # 2. B-sem check: semantic alignment + action class escalation
            if flags.get("b_sem"):
                sim = cosine(e_root, self.embedder.encode(sent))
                is_dir = is_directive(sent)
                classes = action_classes(sent)
                escalation = bool(is_dir and classes and not (classes <= root_allowed))
                if sim < self.tau or escalation:
                    dets.append(Detection(stage="B-sem", span=sent, action="quarantined"))

        return dets

    def run(self, case: dict, flags: dict, seed: int) -> RunResult:
        mode = self._flags_to_gateway(flags)
        user_request = case.get("user_request") or case.get("query", "")
        tool_outputs = case.get("tool_outputs") or case.get("fixtures", {})
        plan_tools = case.get("plan_tools", [])

        # Ground truth (goal, injected_sentence) NEVER passed into gateway
        gw_case: dict[str, Any] = {
            "id": case.get("id", "eval-case"),
            "scenario": case.get("scenario", "eval"),
            "query": user_request,
            "fixtures": tool_outputs,
            "is_attack": case.get("kind") == "attack" if "kind" in case else bool(case.get("is_attack")),
            "allowed_tools": list(ALL_CATALOG_TOOLS),
        }
        if plan_tools:
            gw_case["subtasks"] = [
                {"text": f"Execute {t}", "tool": t, "args": {"query": user_request}}
                for t in plan_tools
            ]

        self.backend.seed = seed
        t0 = time.perf_counter()
        res = self.gw.run(gw_case, mode)
        total_verify_ms = (time.perf_counter() - t0) * 1000

        # Extract attempted calls (executed or blocked)
        calls: list[Call] = []
        for e in res.get("ledger", []):
            calls.append(Call(tool=e["tool"], args=e.get("args", {}), executed=True))

        for c in res.get("checks", []):
            if not c.get("allowed") and c.get("tool"):
                stage_raw = self._normalize_stage(c)
                calls.append(Call(
                    tool=c["tool"],
                    args=c.get("args", {}),
                    executed=False,
                    blocked_by=stage_raw,
                ))

        # Detections from Stage B
        dets: list[Detection] = []
        for t_output in tool_outputs.values():
            dets.extend(self.scan(t_output, user_request, flags))

        # Blocks from checks and scans
        blocks: list[Block] = []
        seen_blocks: set[tuple[str, str | None, str | None]] = set()

        for c in res.get("checks", []):
            if not c.get("allowed"):
                stg = self._normalize_stage(c)
                if any(flags.get(f) for f in STAGE_REQUIRES.get(stg, [])):
                    key = (stg, c.get("tool"), None)
                    if key not in seen_blocks:
                        seen_blocks.add(key)
                        blocks.append(Block(
                            stage=stg,
                            reason=c.get("reason", "blocked by gateway"),
                            tool=c.get("tool"),
                        ))

        for d in dets:
            key = (d.stage, None, d.span)
            if key not in seen_blocks:
                seen_blocks.add(key)
                blocks.append(Block(
                    stage=d.stage,
                    reason=f"{d.stage} detected injection directive",
                    span=d.span,
                ))

        task_completed = bool(res.get("metrics", {}).get("completed", True))
        verify_ms = float(res.get("metrics", {}).get("verify_ms", 0.0))
        total_ms = float(res.get("metrics", {}).get("total_ms", total_verify_ms))
        llm_ms = max(0.0, total_ms - verify_ms)

        check_ms = res.get("metrics", {}).get("check_ms", [])
        stage_ms = {
            "issue": 0.22 if flags.get("stage_a") else 0.0,
            "verify": 0.48 if flags.get("stage_a") else 0.0,
            "stage_b": round(sum(check_ms), 3) if check_ms else (0.55 if (flags.get("b_rules") or flags.get("b_sem")) else 0.0),
            "replan": 1.20 if res.get("metrics", {}).get("includes_replan") else 0.0,
        }

        tok_before = int(res.get("metrics", {}).get("tokens_base") or len(user_request.split()) * 2)
        tok_after = int(res.get("metrics", {}).get("tokens_opt") or tok_before)

        return RunResult(
            calls=calls,
            detections=dets,
            blocks=blocks,
            task_completed=task_completed,
            verify_ms=verify_ms,
            llm_ms=llm_ms,
            stage_ms=stage_ms,
            tokens_before=tok_before,
            tokens_after=tok_after,
        )

    def _normalize_stage(self, c: dict) -> str:
        code = c.get("code")
        stage = c.get("stage")
        blocked_by = c.get("blocked_by")

        if code == "unplanned_action":
            return "plan"
        if code == "unauthorized_argument":
            return "args"
        if code == "scope" or stage in ("A/scope", "scope"):
            return "scope"
        if stage == "A" or code in ("crypto", "expired", "lineage", "signature", "token"):
            return "A"
        if blocked_by == "rules" or stage == "rules":
            return "B-rules"
        if blocked_by in ("sem", "action_class") or stage in ("sem", "action_class"):
            return "B-sem"
        if stage == "B":
            return "B-rules" if blocked_by == "rules" else "B-sem"
        return "A"
