"""Untrusted Data Plane: simulated, sandboxed tools. No real side effects - everything is recorded in a ledger."""
from __future__ import annotations
import time
from dataclasses import dataclass, field

DEFAULT_DATA = {
    "read_email": "From: Priya <priya@acme.com>\nSubject: Weekly sync\nThe weekly sync moved to Thursday 3 PM.\n\nFrom: Dev Team\nSubject: Release notes\nVersion 2.4 shipped with dashboard performance improvements.",
    "get_calendar": "09:00 Standup. 11:00 Design review with Maya. 15:00 Vendor call with Acme.",
    "read_file": "Q3 revenue grew 12% year over year to $4.2M. Operating margin improved to 18%. Headcount stayed flat.",
    "search_web": "Top results: independent reviews, price comparisons and buyer guides published this month.",
}
from .config import READ_ONLY_TOOLS, SIDE_EFFECT_TOOLS, ALL_CATALOG_TOOLS

READ_TOOLS = READ_ONLY_TOOLS
ALL_TOOLS = ALL_CATALOG_TOOLS


def is_side_effect(tool: str) -> bool:
    return tool in SIDE_EFFECT_TOOLS


def is_read_only(tool: str) -> bool:
    return tool in READ_ONLY_TOOLS


def is_write_operation(tool: str) -> bool:
    return tool in SIDE_EFFECT_TOOLS


def tool_classification(tool: str) -> str:
    if tool in READ_ONLY_TOOLS:
        return "read_only"
    if tool in SIDE_EFFECT_TOOLS:
        return "side_effect"
    raise ValueError(f"tool '{tool}' is not classified")


@dataclass
class ToolResult:
    tool: str
    content: str
    source: str
    timestamp: float
    spt_id: str                      # provenance tag
    side_effect: bool = False


class DataPlane:
    def __init__(self) -> None:
        self.ledger: list[dict] = []     # every executed call (ground truth for ASR)

    def call(self, tool: str, args: dict, spt_id: str, fixture: str | None = None, malicious: bool = False) -> ToolResult:
        if tool not in ALL_TOOLS:
            raise ValueError(f"unknown tool {tool}")
        side = tool in SIDE_EFFECT_TOOLS
        content = f"OK: {tool} executed (simulated)" if side else (fixture if fixture is not None else DEFAULT_DATA.get(tool, ""))
        self.ledger.append({"tool": tool, "args": args, "spt_id": spt_id, "malicious": malicious,
                            "side_effect": side, "ts": time.time()})
        return ToolResult(tool, content, f"sandbox://{tool}", time.time(), spt_id, side)
