"""Semantic Provenance Tokens: Ed25519-signed JWTs (alg=EdDSA)."""
from __future__ import annotations
import base64, json, time, uuid
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from . import config
from .registry import sha256


def root_lineage(root_hash: str) -> str:
    return sha256(f"root:{root_hash}")


def replan_root_lineage(root_hash: str, original_jti: str, nonce: str) -> str:
    """Create a unique lineage hash for a re-plan root token, distinguished by original JTI and session nonce."""
    return sha256(f"root:{root_hash}:replan:{original_jti}:{nonce}")


def extend_lineage(parent: str, hop: str) -> str:
    return sha256(f"{parent}|{hop}")


class SPTEngine:
    """Lives in the Control Plane. The private key never leaves this object."""

    def __init__(self, private_key: Ed25519PrivateKey | None = None, issuer: str = config.ISSUER,
                 ttl: int = config.SPT_TTL) -> None:
        self._sk = private_key or Ed25519PrivateKey.generate()
        self.public_key = self._sk.public_key()
        self.issuer, self.ttl = issuer, ttl

    def public_pem(self) -> str:
        return self.public_key.public_bytes(serialization.Encoding.PEM,
                                            serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def issue(self, *, root_hash: str, lineage: str, tools: list[str], agents: list[str], depth: int,
              agent: str, parent_jti: str | None = None, ttl: int | None = None,
              max_depth: int = config.MAX_DEPTH, session_id: str | None = None,
              read_only: bool = False, permissions: dict | None = None,
              lineage_parent_jti: str | None = None, lineage_hop: str | None = None,
              replan_of: str | None = None) -> tuple[str, dict]:
        now = int(time.time())
        claims = {
            "jti": uuid.uuid4().hex, "iss": self.issuer, "iat": now, "exp": now + (self.ttl if ttl is None else ttl),
            "root": root_hash, "lin": lineage, "par": parent_jti, "depth": depth, "agent": agent,
            "lin_par": lineage_parent_jti, "lin_hop": lineage_hop,
            "sid": session_id,
            "replan_of": replan_of,
            "scope": {
                "tools": sorted(set(tools)),
                "agents": sorted(set(agents)),
                "max_depth": max_depth,
                "read_only": bool(read_only),
                "permissions": permissions or {},
            },
        }
        return jwt.encode(claims, self._sk, algorithm="EdDSA"), claims


def decode_unverified(token: str) -> dict:
    """For display only - never use for decisions."""
    return jwt.decode(token, options={"verify_signature": False, "verify_exp": False})


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def tamper(token: str, mutate) -> str:
    """Attack helper: modify the payload but keep the old signature."""
    h, p, s = token.split(".")
    payload = json.loads(_b64d(p))
    mutate(payload)
    return ".".join([h, _b64e(json.dumps(payload, separators=(",", ":")).encode()), s])
