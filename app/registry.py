"""Vector Registry (Step 2): SHA-256 hash + embedding of the ORIGINAL intent, kept in Control Plane memory."""
from __future__ import annotations
import hashlib, time, uuid
from dataclasses import dataclass, field
import numpy as np


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def spec_to_text(spec: dict) -> str:
    parts = [str(spec.get("goal", ""))]
    parts += [str(c) for c in spec.get("constraints", [])]
    parts += [str(p) for p in spec.get("policy", [])]
    return " ".join(p for p in parts if p).strip()


@dataclass
class Session:
    id: str
    spec: dict
    intent_text: str
    root_hash: str
    e_root: np.ndarray
    created: float = field(default_factory=time.time)


@dataclass
class LineageRecord:
    lineage: str
    parent: str | None
    depth: int
    agent: str
    lineage_parent: str | None = None
    lineage_hop: str | None = None
    tools: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    session_id: str | None = None
    read_only: bool = False
    claims: dict | None = None


class VectorRegistry:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.sessions: dict[str, Session] = {}
        self.lineage: dict[str, LineageRecord] = {}      # jti -> record (authoritative)
        self.revoked_jtis: set[str] = set()

    def register(self, spec: dict) -> Session:
        text = spec_to_text(spec)
        s = Session(id=uuid.uuid4().hex[:12], spec=spec, intent_text=text,
                    root_hash=sha256(text), e_root=self.embedder.encode(text))
        self.sessions[s.id] = s
        return s

    def revoke(self, jti: str) -> None:
        """Revoke a token and transitively revoke all of its descendants."""
        to_revoke = {jti}
        changed = True
        while changed:
            changed = False
            for child_jti, rec in self.lineage.items():
                if (rec.parent in to_revoke or rec.lineage_parent in to_revoke) and child_jti not in to_revoke:
                    to_revoke.add(child_jti)
                    changed = True
        self.revoked_jtis.update(to_revoke)

    def is_revoked(self, jti: str) -> bool:
        return jti in self.revoked_jtis
