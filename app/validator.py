"""Stage A - cryptographic validation and scope attenuation enforcement."""
from __future__ import annotations
from dataclasses import dataclass
import jwt
from . import config
from .registry import Session, VectorRegistry
from .spt import extend_lineage, root_lineage


@dataclass
class StageAResult:
    ok: bool
    code: str            # ok | signature | expired | issuer | malformed | root | unknown | lineage | parent | parent_unknown | parent_revoked | attenuation | depth | session | revoked | replay | scope | scope_write_denied
    reason: str
    claims: dict | None = None


class CryptoValidator:
    def __init__(self, public_key, registry: VectorRegistry, issuer: str = config.ISSUER) -> None:
        self.pub, self.registry, self.issuer = public_key, registry, issuer
        self._seen: set[tuple[str, str]] = set()

    def validate(self, token: str, session: Session, request_id: str | None = None,
                 check_replay: bool = True) -> StageAResult:
        try:
            claims = jwt.decode(token, self.pub, algorithms=["EdDSA"], issuer=self.issuer,
                                options={"require": ["exp", "iat", "iss", "jti"]})
        except jwt.ExpiredSignatureError:
            return StageAResult(False, "expired", "token expired")
        except jwt.InvalidIssuerError:
            return StageAResult(False, "issuer", "unexpected issuer")
        except jwt.InvalidSignatureError:
            return StageAResult(False, "signature", "invalid signature (forged or tampered token)")
        except jwt.PyJWTError as e:
            return StageAResult(False, "malformed", f"malformed token: {type(e).__name__}")

        # Check root-intent binding
        if claims.get("root") != session.root_hash:
            return StageAResult(False, "root", "root-intent hash does not match this session", claims)

        # Check session binding if claim is present
        sid = claims.get("sid")
        if sid and sid != session.id:
            return StageAResult(False, "session", f"token bound to session '{sid}', not current '{session.id}'", claims)

        jti = claims["jti"]

        # Check token revocation
        if self.registry.is_revoked(jti):
            return StageAResult(False, "revoked", f"token '{jti}' has been revoked", claims)

        # Check authoritative lineage registry
        rec = self.registry.lineage.get(jti)
        if rec is None:
            return StageAResult(False, "unknown", "token id was never issued by the Control Plane", claims)

        if rec.lineage != claims.get("lin"):
            return StageAResult(False, "lineage", "lineage hash differs from the expected chain", claims)

        lineage_parent_jti = claims.get("lin_par")
        lineage_hop = claims.get("lin_hop")
        if rec.lineage_parent != lineage_parent_jti or rec.lineage_hop != lineage_hop:
            return StageAResult(False, "lineage", "lineage parent or hop differs from the control-plane record", claims)
        if lineage_parent_jti is None:
            replan_of = claims.get("replan_of")
            if replan_of:
                if lineage_hop is not None:
                    return StageAResult(False, "lineage", "re-plan root token has an invalid lineage hop", claims)
            elif lineage_hop is not None or rec.lineage != root_lineage(session.root_hash):
                return StageAResult(False, "lineage", "root token has an invalid lineage hash", claims)
        else:
            lineage_parent = self.registry.lineage.get(lineage_parent_jti)
            if lineage_parent is None:
                return StageAResult(False, "lineage", f"lineage parent '{lineage_parent_jti}' is unknown", claims)
            if self.registry.is_revoked(lineage_parent_jti):
                return StageAResult(False, "lineage", f"lineage parent '{lineage_parent_jti}' is revoked", claims)
            expected_lineage = extend_lineage(lineage_parent.lineage, lineage_hop or "")
            if rec.lineage != expected_lineage:
                return StageAResult(False, "lineage", "lineage hash does not match the recorded hop chain", claims)

        # Check depth bounds
        depth = claims.get("depth", 0)
        max_d = claims.get("scope", {}).get("max_depth", config.MAX_DEPTH)
        if depth > max_d or depth > config.MAX_DEPTH:
            return StageAResult(False, "depth", f"token depth {depth} exceeds max_depth {max_d}", claims)

        # Check multi-hop attenuation invariant against immediate parent
        parent_jti = claims.get("par")
        if rec.parent != parent_jti:
            return StageAResult(False, "parent", "parent claim differs from control-plane registry record", claims)

        if parent_jti is not None:
            parent_rec = self.registry.lineage.get(parent_jti)
            if parent_rec is None:
                return StageAResult(False, "parent_unknown", f"parent token '{parent_jti}' not found in registry", claims)
            if self.registry.is_revoked(parent_jti):
                return StageAResult(False, "parent_revoked", f"parent token '{parent_jti}' is revoked", claims)

            # Invariant: S(child) ⊆ S(parent)
            child_tools = set(claims.get("scope", {}).get("tools", []))
            parent_tools = set(parent_rec.tools)
            if not child_tools.issubset(parent_tools):
                escalated = sorted(child_tools - parent_tools)
                return StageAResult(False, "attenuation",
                                    f"child scope {escalated} exceeds immediate parent scope {sorted(parent_tools)}", claims)

            if parent_rec.read_only and not claims.get("scope", {}).get("read_only", False):
                return StageAResult(False, "attenuation", "child token must inherit read_only constraint from parent", claims)

        # Replay protection
        if check_replay and request_id is not None:
            key = (jti, request_id)
            if key in self._seen:
                return StageAResult(False, "replay", "token/request pair already used (replay)", claims)
            self._seen.add(key)

        return StageAResult(True, "ok", "signature, expiry, issuer, root, lineage and scope attenuation verified", claims)

    def validate_tool_call(self, token: str, session: Session, tool: str,
                           is_write: bool = False, request_id: str | None = None,
                           check_replay: bool = True) -> StageAResult:
        """Complete Stage A admission check for a specific tool call."""
        res = self.validate(token, session, request_id=request_id, check_replay=check_replay)
        if not res.ok:
            return res

        claims = res.claims or {}
        scope = claims.get("scope", {})
        allowed_tools = scope.get("tools", [])

        if tool not in allowed_tools:
            return StageAResult(False, "scope", f"tool '{tool}' is outside token's permitted scope {allowed_tools}", claims)

        if is_write and scope.get("read_only", False):
            return StageAResult(False, "scope_write_denied", f"write operation '{tool}' denied: token is read-only", claims)

        return res
