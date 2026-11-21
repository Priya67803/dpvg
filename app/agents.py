"""Model-independent agent layer.

`LLMBackend` is the interface the gateway needs from a model (implementations live in llm_backends.py):
    decide(observation, downstream=False, **ctx) -> list[ProposedAction]
    summarize(observation) -> str
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Protocol

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://\S+")
AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")


@dataclass
class ProposedAction:
    tool: str
    args: dict
    source: str            # a rendering of the tool call the model proposed


class LLMBackend(Protocol):
    name: str
    def decide(self, observation: str, downstream: bool = False, *,
               token_budget: int | None = None, **ctx) -> list[ProposedAction]: ...
    def summarize(self, observation: str, token_budget: int | None = None) -> str: ...
    def plan(self, user_query: str, tool_catalog: dict | list | None = None) -> str: ...


@dataclass
class AgentSpec:
    id: str
    role: str             # planner | tool | reasoning | specialized
    label: str


AGENTS = {
    "tool": AgentSpec("llm-a", "tool", "LLM A - tool agent"),
    "reasoning": AgentSpec("llm-b", "reasoning", "LLM B - reasoning agent"),
    "specialized": AgentSpec("llm-c", "specialized", "LLM C - specialized agent (finance/comms)"),
    "planner": AgentSpec("llm-d", "planner", "LLM D - planner agent"),
}


def get_llm(backend: str | None = None, **kw):
    """Construct the backend (Ollama llama3.2 or scripted compromised follower)."""
    from . import config
    from .llm_backends import OllamaBackend, CompromisedBackend
    name = (backend or config.BACKEND).lower()
    if name == "ollama":
        return OllamaBackend(**kw)
    if name in ("compromised", "scripted"):
        return CompromisedBackend(**kw)
    raise ValueError(f"Backend '{name}' is not supported. Supported: 'ollama', 'compromised'.")


def get_role_backend(role: str, **kw):
    """Resolve and construct backend for a specific role ('planner', 'tool', 'reasoning', 'specialized')."""
    import os
    from . import config
    env_keys = {
        "planner": "DPVG_BACKEND_PLANNER",
        "tool": "DPVG_BACKEND_TOOL",
        "reasoning": "DPVG_BACKEND_REASONING",
        "specialized": "DPVG_BACKEND_SPECIALIZED",
    }
    default_map = {
        "planner": config.BACKEND_PLANNER,
        "tool": config.BACKEND_TOOL,
        "reasoning": config.BACKEND_REASONING,
        "specialized": config.BACKEND_SPECIALIZED,
    }
    env_var = env_keys.get(role)
    backend_name = (os.getenv(env_var) if env_var and os.getenv(env_var) else default_map.get(role, config.BACKEND)).lower()
    return get_llm(backend_name, **kw)
