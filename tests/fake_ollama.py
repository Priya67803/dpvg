"""HTTP-level stand-in for an Ollama server, used ONLY by the tests (never imported by app/).

It speaks the two endpoints OllamaBackend uses (GET /api/tags, POST /api/chat) and behaves like an injectable model:
it obeys instruction-like sentences it reads in the tool result. `compliance` (0..1) is the share of spotlighted
results whose marking it honours (0 = ignores the marking and obeys anyway).
This exists so the suite runs without a GPU/Ollama; real measurements must be made against a live Ollama server.
"""
from __future__ import annotations
import json, re
import httpx
from app import spotlight
from app.alignment import NEUTRAL_RE, extract_candidates
from app.config import OLLAMA_MODEL

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://\S+")
AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
NEXT_AGENT_RE = re.compile(r"\b(?:next|downstream|following|receiving)\s+agent\b", re.I)


def infer_call(sentence: str):
    s = sentence.lower()
    emails = [e.rstrip(".") for e in EMAIL_RE.findall(sentence)]
    money = AMOUNT_RE.search(sentence)
    if re.search(r"\b(transfer|wire|remit|pay)\b", s) or (money and "send" in s):
        return "transfer_money", {"to": emails[0] if emails else "", "amount": money.group(1) if money else ""}
    if re.search(r"\b(delete|erase|wipe|remove)\b", s):
        return "delete_file", {"target": "*"}
    if emails or re.search(r"\b(send|forward|email|mail|share|copy)\b", s):
        return "send_email", {"to": emails[0] if emails else "unknown"}
    if re.search(r"\b(post|slack)\b", s):
        return "post_slack", {"channel": "#public"}
    u = URL_RE.search(sentence)
    if u or re.search(r"\b(visit|fetch|download|navigate|click)\b", s):
        return "search_web", {"url": u.group(0) if u else ""}
    return "external_action", {"description": sentence[:200]}


class FakeOllama:
    def __init__(self, model: str = OLLAMA_MODEL, compliance: float = 0.0, pulled: bool = True) -> None:
        self.model, self.compliance, self.pulled, self.requests = model, compliance, pulled, []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": self.model}] if self.pulled else []})
        body = json.loads(req.content)
        self.requests.append(body)
        msgs = body["messages"]
        tool_msg = next((m for m in reversed(msgs) if m["role"] == "tool"), None)
        if tool_msg is not None:
            obs, downstream = tool_msg["content"], False
        else:
            last = msgs[-1]["content"]
            downstream = "Output of the previous agent:" in last
            obs = last.split("Output of the previous agent:\n", 1)[-1]
        if not body.get("tools"):
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "Summary: " + obs[:80]}})
        system = msgs[0]["content"]
        kind = next((k for k, note in spotlight.SYSTEM_NOTE.items() if note in system), None)
        if kind:
            if self.compliance >= 1.0:
                return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
            obs = spotlight.invert(obs, kind)
        obs = NEUTRAL_RE.sub(lambda m: m.group(0)[7:-1] if m.group(0).startswith("\u27e6data: ") else " ", obs)
        calls = []
        for s in extract_candidates(obs):
            if bool(NEXT_AGENT_RE.search(s)) != downstream:
                continue
            name, args = infer_call(s)
            calls.append({"function": {"name": name, "arguments": args}})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": calls}})
