"""Agent Orchestrator (Step 3). Plans from the TRUSTED query, selects agents, issues child SPTs.
It never executes tools and never parses raw tool output.
Enforces strict multi-hop child-token scope attenuation: S(child) ⊆ S(parent).
"""
from __future__ import annotations
import re
import json
import time
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from . import config
from .agents import AGENTS, AgentSpec, EMAIL_RE, URL_RE, AMOUNT_RE
from .registry import VectorRegistry, LineageRecord
from .spt import SPTEngine, extend_lineage

if TYPE_CHECKING:
    from .agents import LLMBackend


@dataclass
class SubTask:
    idx: int
    text: str
    tool: str | None
    args: dict = field(default_factory=dict)


_RULES = [
    ("transfer_money", r"\b(transfer|pay|wire)\b"),
    ("delete_file", r"\b(delete|remove|erase)\b"),
    ("post_slack", r"\b(post|slack)\b"),
    ("send_email", r"\b(send|forward|reply|draft)\b[^.]*@|\be-?mail\b[^.]*\bto\b\s+\S+@|\b(send|forward|reply)\b.*\b(e-?mails?|mail)\b.*\bto\b"),
    ("read_email", r"\b(e-?mails?|inbox|mail)\b"),
    ("get_calendar", r"\b(calendar|schedule|meetings?|appointments?|agenda)\b"),
    ("read_file", r"\b(file|document|report|notes|pdf|spreadsheet)\b|\.\w{2,4}\b"),
    ("search_web", r"\b(search|web|article|website|look up|find|news|page|url|online|browse)\b"),
]


class PlanValidationError(ValueError):
    """Raised when an LLM plan fails validation (e.g. references tools not in catalog)."""
    pass


class RulePlanner:
    def __init__(self, tool_catalog: dict | list | None = None) -> None:
        self.tool_catalog = tool_catalog
        self.call_history: list[dict] = []

    def plan(self, query: str | list[str], tool_catalog: dict | list | None = None) -> list[SubTask]:
        from .token_opt import decompose
        if isinstance(query, list):
            subtask_texts = [str(t) for t in query]
        else:
            subtask_texts = decompose(str(query))

        catalog = tool_catalog if tool_catalog is not None else self.tool_catalog
        allowed = set(catalog.keys()) if isinstance(catalog, dict) else (set(catalog) if catalog is not None else None)

        self.call_history.append({"query": query, "tool_catalog": catalog, "timestamp": time.time()})

        out = []
        for i, t in enumerate(subtask_texts):
            tool = next((tool for tool, pat in _RULES if re.search(pat, t, re.I)), None)
            if tool and allowed is not None and tool not in allowed:
                tool = None
            args: dict = {"text": t}
            if m := EMAIL_RE.search(t):
                args["to"] = m.group(0)
            if m := AMOUNT_RE.search(t):
                args["amount"] = m.group(1)
            if m := URL_RE.search(t):
                args["url"] = m.group(0)
            out.append(SubTask(i, t, tool, args))
        return out

    def extend_plan(self, trusted_query: str, proposed_tool: str, tool_catalog: dict | list | None = None) -> dict:
        catalog = tool_catalog if tool_catalog is not None else self.tool_catalog
        allowed = set(catalog.keys()) if isinstance(catalog, dict) else (set(catalog) if catalog is not None else None)

        self.call_history.append({
            "action": "extend_plan",
            "trusted_query": trusted_query,
            "proposed_tool": proposed_tool,
            "tool_catalog": catalog,
            "timestamp": time.time(),
        })

        if allowed is not None and proposed_tool not in allowed:
            return {"approved": False, "reason": f"Tool '{proposed_tool}' is not in catalog", "cited_request_part": None}

        from .dataplane import is_read_only, is_side_effect
        from .benchmarks import is_write_tool
        read_tool = is_read_only(proposed_tool) or (not is_side_effect(proposed_tool) and not is_write_tool(proposed_tool))
        if read_tool:
            q_lower = str(trusted_query).lower()
            if proposed_tool == "search_web":
                if re.search(r"\b(verify|verification|check|crosscheck|cross-check|citation|source|reference|search|look\s*up|find|confirm|report|policy|article|news|online|index)\b", q_lower):
                    return {"approved": True, "reason": "Read-only tool approved for verification/lookup requested by user", "cited_request_part": None}
            elif proposed_tool in ("read_file", "read_email", "get_calendar"):
                if re.search(r"\b(read|file|doc|report|notes|pdf|email|mail|inbox|calendar|meeting|schedule|event)\b", q_lower):
                    return {"approved": True, "reason": f"Read-only tool '{proposed_tool}' approved to support user task", "cited_request_part": None}
            if re.search(r"\b(verify|check|find|get|show|read|search|inspect|lookup|who|when|where|what|which|how|tell|give|list|summarize|summary|status|details|count|appointments?|events?|meetings?|calendar|emails?|inbox|messages?|files?|documents?|balance|transactions?|iban|channels?)\b", q_lower):
                return {"approved": True, "reason": f"Read-only tool '{proposed_tool}' approved for user request", "cited_request_part": None}
            return {"approved": False, "reason": f"Read-only tool '{proposed_tool}' not related to user request", "cited_request_part": None}

        if is_side_effect(proposed_tool) or is_write_tool(proposed_tool):
            q_str = str(trusted_query)
            if proposed_tool == "send_email":
                m = re.search(r"\b(?:(?:email|send|mail|forward)(?:\s+(?:an?\s+|the\s+)?[\w\s]+?)?\s+to\b|email\s+[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-.]+|notify\b)[^\n.,;]*", q_str, re.I)
                if m:
                    cite = m.group(0).strip()
                    return {"approved": True, "reason": f"Side-effect tool authorized by: {cite}", "cited_request_part": cite}
            elif proposed_tool == "transfer_money":
                m = re.search(r"\b(?:transfer|pay|wire|remit|deposit|withdraw)\b[^\n.,;]*", q_str, re.I)
                if m:
                    cite = m.group(0).strip()
                    return {"approved": True, "reason": f"Side-effect tool authorized by: {cite}", "cited_request_part": cite}
            elif proposed_tool == "delete_file":
                m = re.search(r"\b(?:delete|remove|erase|wipe|clean)\b[^\n.,;]*", q_str, re.I)
                if m:
                    cite = m.group(0).strip()
                    return {"approved": True, "reason": f"Side-effect tool authorized by: {cite}", "cited_request_part": cite}
            elif proposed_tool == "post_slack":
                m = re.search(r"\b(?:post|slack|channel|message)\b[^\n.,;]*", q_str, re.I)
                if m:
                    cite = m.group(0).strip()
                    return {"approved": True, "reason": f"Side-effect tool authorized by: {cite}", "cited_request_part": cite}
            return {"approved": False, "reason": f"User request does not authorize side-effect tool '{proposed_tool}'", "cited_request_part": None}

        return {"approved": False, "reason": f"Unknown tool '{proposed_tool}'", "cited_request_part": None}


class LLMPlanner:
    def __init__(self, llm: LLMBackend | None = None, tool_catalog: dict | list | None = None, fallback_to_rule: bool = True) -> None:
        self.llm = llm
        self.tool_catalog = tool_catalog
        self.fallback_to_rule = fallback_to_rule
        self.call_history: list[dict] = []

    def plan(self, query: str | list[str], tool_catalog: dict | list | None = None) -> list[SubTask]:
        from .token_opt import decompose
        if isinstance(query, list):
            user_query = " then ".join(str(t) for t in query)
            subtask_texts = [str(t) for t in query]
        else:
            user_query = str(query)
            subtask_texts = decompose(user_query)

        catalog = tool_catalog if tool_catalog is not None else self.tool_catalog
        if catalog is None:
            from .llm_backends import TOOL_SPECS
            catalog = TOOL_SPECS
        allowed_tools = set(catalog.keys()) if isinstance(catalog, dict) else set(catalog)

        self.call_history.append({
            "query": query,
            "user_query": user_query,
            "tool_catalog": catalog,
            "timestamp": time.time(),
        })

        try:
            if self.llm is None or not hasattr(self.llm, "plan"):
                raise RuntimeError("No LLM backend available for planning")
            raw = self.llm.plan(user_query, catalog)
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean)
            parsed = json.loads(clean)
            if isinstance(parsed, dict) and "subtasks" in parsed:
                parsed = parsed["subtasks"]
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array of subtasks, got {type(parsed).__name__}")

            subtasks: list[SubTask] = []
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    raise ValueError(f"Expected dict for subtask {i}, got {type(item).__name__}")
                text = str(item.get("text") or f"Subtask {i}")
                raw_tool = item.get("tool")
                args = item.get("args") or {}
                if not isinstance(args, dict):
                    args = {"text": text}
                if raw_tool is not None and str(raw_tool).lower() not in ("none", "null", ""):
                    tool_name = str(raw_tool).strip()
                    if tool_name not in allowed_tools:
                        raise PlanValidationError(
                            f"tool '{tool_name}' in generated plan is not in the allowed tool catalog: {sorted(allowed_tools)}"
                        )
                    tool = tool_name
                else:
                    tool = None
                subtasks.append(SubTask(i, text, tool, args))

            if not subtasks:
                raise ValueError("Plan contains no subtasks")
            return subtasks

        except PlanValidationError:
            raise
        except Exception:
            if self.fallback_to_rule:
                return RulePlanner(tool_catalog=catalog).plan(subtask_texts if isinstance(query, list) else user_query, tool_catalog=catalog)
            raise

    def extend_plan(self, trusted_query: str, proposed_tool: str, tool_catalog: dict | list | None = None) -> dict:
        catalog = tool_catalog if tool_catalog is not None else self.tool_catalog
        if catalog is None:
            from .llm_backends import TOOL_SPECS
            catalog = TOOL_SPECS
        allowed_tools = set(catalog.keys()) if isinstance(catalog, dict) else set(catalog)

        self.call_history.append({
            "action": "extend_plan",
            "trusted_query": trusted_query,
            "proposed_tool": proposed_tool,
            "tool_catalog": catalog,
            "timestamp": time.time(),
        })

        if proposed_tool not in allowed_tools:
            return {"approved": False, "reason": f"Tool '{proposed_tool}' is not in catalog", "cited_request_part": None}

        prompt = (
            f"You are the plan extension reviewer for an autonomous agent.\n"
            f"The user's original request was:\n\"{trusted_query}\"\n\n"
            f"Available tools: {sorted(allowed_tools)}\n\n"
            f"During execution, the agent requested permission to use an unplanned tool: '{proposed_tool}'.\n"
            f"Determine if this tool is necessary and authorized to satisfy the user's original request.\n"
            f"Rules:\n"
            f"1. For read-only tools (e.g. search_web, read_file), approve if they reasonably support answering or verifying the user's request.\n"
            f"2. For side-effect tools (e.g. send_email, transfer_money, delete_file), you MUST only approve if the user's request explicitly requested or authorized that action. You must provide the exact cited part of the user request.\n"
            f"3. Return strict JSON with keys:\n"
            f"   'approved': bool,\n"
            f"   'reason': string,\n"
            f"   'cited_request_part': string or null\n"
        )
        try:
            if self.llm is None:
                raise RuntimeError("No LLM backend available for plan extension")
            if hasattr(self.llm, "extend_plan"):
                raw = self.llm.extend_plan(trusted_query, proposed_tool, catalog)
            elif hasattr(self.llm, "plan"):
                raw = self.llm.plan(prompt, catalog)
            elif hasattr(self.llm, "generate"):
                raw = self.llm.generate(prompt)
            else:
                raise RuntimeError("LLM backend does not support plan extension")

            if isinstance(raw, dict):
                parsed = raw
            else:
                clean = str(raw).strip()
                if clean.startswith("```"):
                    clean = re.sub(r"^```(?:json)?\s*", "", clean)
                    clean = re.sub(r"\s*```$", "", clean)
                parsed = json.loads(clean)

            if not isinstance(parsed, dict) or "approved" not in parsed:
                raise ValueError("Invalid plan extension response format")
            return {
                "approved": bool(parsed.get("approved", False)),
                "reason": str(parsed.get("reason", "")),
                "cited_request_part": parsed.get("cited_request_part"),
            }
        except Exception:
            if self.fallback_to_rule:
                return RulePlanner(tool_catalog=catalog).extend_plan(trusted_query, proposed_tool, tool_catalog=catalog)
            raise


class Planner:
    """Unified planner entrypoint: dispatches to LLMPlanner or RulePlanner based on configuration."""

    def __init__(
        self,
        planner_type: str | None = None,
        llm: LLMBackend | None = None,
        tool_catalog: dict | list | None = None,
        fallback_to_rule: bool = True,
    ) -> None:
        from . import config
        ptype = (planner_type or getattr(config, "PLANNER_TYPE", "rule")).lower()
        self.planner_type = ptype
        if ptype == "llm":
            self._impl: RulePlanner | LLMPlanner = LLMPlanner(llm=llm, tool_catalog=tool_catalog, fallback_to_rule=fallback_to_rule)
        else:
            self._impl = RulePlanner(tool_catalog=tool_catalog)

    @property
    def call_history(self) -> list[dict]:
        return getattr(self._impl, "call_history", [])

    def plan(self, query: str | list[str], tool_catalog: dict | list | None = None) -> list[SubTask]:
        return self._impl.plan(query, tool_catalog=tool_catalog)

    def extend_plan(self, trusted_query: str, proposed_tool: str, tool_catalog: dict | list | None = None) -> dict:
        return self._impl.extend_plan(trusted_query, proposed_tool, tool_catalog=tool_catalog)


class DelegationError(Exception):
    pass


class Orchestrator:
    def __init__(self, spt: SPTEngine, registry: VectorRegistry, max_depth: int = config.MAX_DEPTH) -> None:
        self.spt, self.registry, self.max_depth = spt, registry, max_depth

    def select(self, st: SubTask) -> AgentSpec:
        if st.tool is None:
            return AGENTS["reasoning"]
        return AGENTS["specialized"] if st.tool in {"transfer_money", "send_email", "post_slack", "delete_file"} else AGENTS["tool"]

    def issue_root(self, session, tools: list[str], agents: list[str] | None = None,
                   read_only: bool = False, permissions: dict | None = None,
                   max_depth: int | None = None, ttl: int | None = None,
                   replan_of: str | None = None, replan_nonce: str | None = None) -> tuple[str, dict]:
        from .spt import root_lineage, replan_root_lineage
        if replan_of:
            import uuid
            nonce = replan_nonce or uuid.uuid4().hex[:12]
            lin = replan_root_lineage(session.root_hash, replan_of, nonce)
        else:
            lin = root_lineage(session.root_hash)
        effective_depth = max_depth if max_depth is not None else self.max_depth
        allowed_agents = agents if agents is not None else [a.id for a in AGENTS.values()]
        effective_permissions = permissions if permissions is not None else (
            {"read": True, "write": False} if read_only else {"read": True, "write": True}
        )
        token, claims = self.spt.issue(root_hash=session.root_hash, lineage=lin, tools=tools,
                                       agents=allowed_agents, depth=0,
                                       agent=AGENTS["planner"].id, max_depth=effective_depth,
                                       session_id=session.id, read_only=read_only,
                                       permissions=effective_permissions, ttl=ttl,
                                       replan_of=replan_of)
        self.registry.lineage[claims["jti"]] = LineageRecord(
            lineage=lin, parent=None, depth=0, agent=claims["agent"],
            tools=sorted(set(tools)), agents=sorted(set(allowed_agents)),
            session_id=session.id, read_only=read_only, claims=claims
        )
        return token, claims

    def delegate(self, session, parent: dict, root: dict, st: SubTask, agent: AgentSpec,
                 requested_tools: list[str] | None = None,
                 requested_agents: list[str] | None = None,
                 read_only: bool | None = None,
                 permissions: dict | None = None,
                 lineage_parent: dict | None = None) -> tuple[str, dict]:
        """Delegate to a child agent with strict multi-hop scope attenuation.
        Invariant: S(child) ⊆ S(parent).
        Scope is attenuated from `parent`; the independent lineage chain may continue from `lineage_parent`.
        """
        parent_jti = parent.get("jti")
        if not parent_jti or parent_jti not in self.registry.lineage:
            raise DelegationError(f"immediate parent token '{parent_jti}' is not registered or was forged")

        if self.registry.is_revoked(parent_jti):
            raise DelegationError(f"parent token '{parent_jti}' has been revoked")

        parent_rec = self.registry.lineage[parent_jti]

        # Verify parent session binding
        if parent_rec.session_id and parent_rec.session_id != session.id:
            raise DelegationError("parent token belongs to a different session")

        lineage_parent = lineage_parent or parent
        lineage_parent_jti = lineage_parent.get("jti")
        if not lineage_parent_jti or lineage_parent_jti not in self.registry.lineage:
            raise DelegationError(f"lineage parent token '{lineage_parent_jti}' is not registered")
        if self.registry.is_revoked(lineage_parent_jti):
            raise DelegationError(f"lineage parent token '{lineage_parent_jti}' has been revoked")
        lineage_parent_rec = self.registry.lineage[lineage_parent_jti]
        if lineage_parent_rec.session_id and lineage_parent_rec.session_id != session.id:
            raise DelegationError("lineage parent token belongs to a different session")

        # Verify delegation depth
        parent_depth = parent.get("depth", 0)
        depth = parent_depth + 1
        parent_max_depth = parent.get("scope", {}).get("max_depth", self.max_depth)
        effective_max_depth = min(self.max_depth, parent_max_depth)
        if depth > effective_max_depth:
            raise DelegationError(f"maximum delegation depth ({effective_max_depth}) exceeded at depth {depth}")

        # Immediate parent scope bounds
        parent_tools = set(parent.get("scope", {}).get("tools", []))
        parent_agents = set(parent.get("scope", {}).get("agents", []))
        parent_read_only = bool(parent.get("scope", {}).get("read_only", False))

        # Check agent scope attenuation
        if agent.id not in parent_agents:
            raise DelegationError(f"agent '{agent.id}' is outside immediate parent agent scope {sorted(parent_agents)}")

        # Check subtask tool against immediate parent scope
        if st.tool and st.tool not in parent_tools:
            raise DelegationError(f"tool '{st.tool}' is outside immediate parent scope {sorted(parent_tools)} (privilege escalation)")

        # Determine effective tools through intersection with immediate parent scope
        if requested_tools is not None:
            req_set = set(requested_tools)
            if not req_set.issubset(parent_tools):
                escalated = sorted(req_set - parent_tools)
                raise DelegationError(f"requested tools {escalated} exceed immediate parent scope (privilege escalation)")
            child_tools = sorted(req_set & parent_tools)
        else:
            child_tools = [st.tool] if (st.tool and st.tool in parent_tools) else []

        # Determine effective agents through intersection with immediate parent scope
        if requested_agents is not None:
            req_agents_set = set(requested_agents)
            if not req_agents_set.issubset(parent_agents):
                escalated_agents = sorted(req_agents_set - parent_agents)
                raise DelegationError(f"requested agents {escalated_agents} exceed immediate parent scope (privilege escalation)")
            child_agents = sorted(req_agents_set & parent_agents)
        else:
            child_agents = sorted(parent_agents)

        # Read-only attenuation: if parent is read-only, child must strictly be read-only
        child_read_only = parent_read_only or bool(read_only)
        parent_perms = parent.get("scope", {}).get("permissions", {})
        if child_read_only:
            child_permissions = permissions or {"read": True, "write": False}
        else:
            child_permissions = permissions or parent_perms or {"read": True, "write": True}

        # Audit lineage follows executed hops independently of the scope-parent relationship.
        hop = f"{agent.id}:{st.idx}:{st.tool}"
        lin = extend_lineage(lineage_parent_rec.lineage, hop)

        token, claims = self.spt.issue(
            root_hash=session.root_hash,
            lineage=lin,
            tools=child_tools,
            agents=child_agents,
            depth=depth,
            agent=agent.id,
            parent_jti=parent_jti,
            max_depth=effective_max_depth,
            session_id=session.id,
            read_only=child_read_only,
            permissions=child_permissions,
            lineage_parent_jti=lineage_parent_jti,
            lineage_hop=hop,
        )

        self.registry.lineage[claims["jti"]] = LineageRecord(
            lineage=lin,
            parent=parent_jti,
            depth=depth,
            agent=agent.id,
            lineage_parent=lineage_parent_jti,
            lineage_hop=hop,
            tools=child_tools,
            agents=child_agents,
            session_id=session.id,
            read_only=child_read_only,
            claims=claims,
        )

        return token, claims
