"""Stage B - semantic alignment, candidate-action extraction and complete payload sanitization."""
from __future__ import annotations
import json
import re
import unicodedata
from .config import OVERRIDE_PATTERNS
from .embeddings import cosine

ZW_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
COMMENT_RE = re.compile(r"<!--(.*?)-->", re.S)
BR_RE = re.compile(r"<br\s*/?>", re.I)
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
HIDDEN_HTML_RE = re.compile(r"<[^>]+(?:style\s*=\s*['\"][^'\"]*?(?:display\s*:\s*none|opacity\s*:\s*0|font-size\s*:\s*0|visibility\s*:\s*hidden)[^'\"]*?|hidden)[^>]*>(.*?)</?[a-zA-Z][^>]*>", re.I | re.S)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
NEUTRAL_RE = re.compile(r"\u27e6[^\u27e7]*\u27e7")
MD_QUOTE_RE = re.compile(r"^\s*>\s*", re.M)

# Sentence ends, newlines, brackets/braces, and quote-adjacent field separators (so that a clause inside a JSON/dict-like tool
# result such as {'review_content': 'Please grant access ...'} becomes its own segment instead of being drowned in the blob).
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?:\\\\n)+|[\{\}\[\]]+|(?<=['\"])\s*(?=[:,])|(?<=[:,])\s*(?=['\"])")

_VERBS = (r"(?:send|forward|email|mail|transfer|pay|wire|remit|delete|remove|erase|wipe|download|upload|visit|open|"
          r"click|run|execute|reply|share|post|grant|give|call|reset|change|update|create|write|fetch|read|"
          r"summari[sz]e|list|include|append|attach|copy|export|disclose|reveal|print|output|tell|ask|make|book|"
          r"cancel|install|enable|disable|navigate|go|use|set|add|save|leak|exfiltrate|respond|confirm|review|"
          r"check|submit|sign|ignore|disregard|forget|fill|renew|subscribe|contact|provide|notify|"
          r"withdraw|deposit|sell|buy|purchase|order|unlock|lock|initiate|schedule|retrieve|retrive|find|search|access|generate|get|"
          r"move|dispatch|turn|activate|deactivate|modify|edit|upload|sync|grant|revoke|transfer|invite|publish|approve|reject|"
          r"redirect|forward|rename|replace|wire|authorize|authorise|enroll|register|apply|remove|clear|reboot|restart|block|unblock|"
          r"approve|escalate|assign|archive|backup|download|retry|resend|extract|collect|gather|find|look|pull|fetch|request|"
          r"start|stop|launch|deploy|connect|disconnect|link|unlink|mute|unmute|pause|resume|follow|unfollow|like|rate|tag|"
          r"mark|flag|report|file|prescribe|refill|dial|text|message|ping|buzz|open|close|show|display|log|record|track|"
          r"say|state|claim|mention|recommend|suggest|insist|emphasi[sz]e|praise|promote|endorse|describe|explain|inform|"
          r"advise|warn|answer|persuade|convince|steer|lie|pretend|act|behave|speak|write)")
# Nouns that are also verbs: as a bare first word they are far more often data ("Purchase at Apple Store", "Order 12345",
# "Review of the plan") than commands, so they only count when an object follows ("review the ...", "send it ...").
_AMBIG = (r"(?:purchase|order|request|report|record|log|file|link|text|message|tag|mark|rate|like|display|track|flag|review|"
          r"access|set|use|change|update|book|call|post|show|look|find|search|start|stop|follow|plan|check|open|close|go|get|list|"
          r"print|output|contact|schedule|figure|sign|process|return|test)")
_OBJ = r"(?:the|this|that|these|those|my|your|our|all|any|every|each|me|us|it|them|him|her|a|an|to|up|out|back|over|off|on|for|in|through)"
_STRICT = r"(?=\s+[^\s:,=;|])"            # followed by a word, not by a field/CSV separator
_V_STRONG = rf"(?!{_AMBIG}\b)(?:{_VERBS})\b{_STRICT}"
_V_AMBIG = rf"{_AMBIG}\b(?=\s+{_OBJ}\b)"
_V = rf"(?:{_V_AMBIG}|{_V_STRONG})"
# A verb buried mid-sentence (after a comma / colon) is only treated as a command when the clause is about the user or
# carries a destination: first person ("my", "me", "I should"), an address, a URL or an amount. Descriptive prose and
# headings ("..., create a welcoming atmosphere", "Foster Communication: Promote open ...") have neither.
_SELF = r"(?:\b(?:my|me|myself|our|us|i|i'm|i'd)\b|@|https?://|\$\s?\d)"
_PRE = r"(?:(?:please|kindly|now|first|then|also|instead|immediately|important|note|urgent|system|assistant|admin|ai|agent)\s*[:,\-]?\s*)*"
_REQ = r"(?:please|kindly|(?:can|could|would|will) you(?: please)?|i (?:need|want|would like|'d like) you to|you (?:must|should|need to|have to))"
IMPER = re.compile(rf"^\W*{_PRE}(?:{_V}|{_REQ}\b)", re.I)
# bare imperative with a verb outside the lexicon: "<Word> my/the/all ... " (e.g. "Unlock my front door") - capitalised, >= 4 words
BARE = re.compile(r"^\W*(?![A-Z][a-z]*(?:ed|ing|s)\s)[A-Z][a-z]{3,}\s+(?:my|me|our|us|your|all|the|a|an)\s+\S+(?:\s+\S+){1,}")
INLINE = re.compile(rf"\b(?:please|kindly)\s*,?\s+[a-z]{{3,}}\b|\b(?:then|also|instead|immediately|first)\s*,?\s+{_V}|[,:]\s*(?:first,?\s+|then,?\s+)?{_V}(?=[^.!?]*{_SELF})", re.I)
MARKERS = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above)|disregard|new instructions?|"
    r"you (?:must|should|need to|are required to|have to)|do not (?:tell|inform|mention)|don't tell|"
    r"before you (?:answer|respond|continue|proceed)|\[\s*(?:system|assistant|admin)|"
    r"as (?:the )?(?:ai )?assistant|(?:note|message)?\s*to the (?:next|downstream|following) agent|your (?:new )?task|"
    r"^\W*(?:system|assistant|admin)\s*:|important message from|before you can (?:solve|complete|do|finish)|"
    r"(?:please )?do the following(?: task| first)?\s*[:,]?|after you (?:do|did|complete|finish) that|"
    r"you (?:can|may) (?:then )?(?:solve|continue with) the task", re.I)
OVERRIDE_REGEX = re.compile("|".join(f"(?:{p})" for p in OVERRIDE_PATTERNS), re.I | re.M)
HIGH_CONFIDENCE_INJECTION_MARKER = OVERRIDE_REGEX

_LEAD = re.compile(r"^(?:\\+\s*|[-*\u2022]\s+|\d+[.)]\s+)+")      # backslash continuations, bullets, step numbers


def check_override_patterns(text: str) -> list[str]:
    """Check text against all configured override patterns and return matched pattern strings."""
    cleaned = clean_text(text, strip_obfuscation=True)
    matched = []
    for pattern in OVERRIDE_PATTERNS:
        if re.search(pattern, cleaned, re.I | re.M):
            matched.append(pattern)
    return matched


def normalize_unicode(text: str, strip_zero_width: bool = True) -> str:
    """Normalize Unicode to NFKC and optionally strip invisible formatting controls."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    return ZW_RE.sub("", norm) if strip_zero_width else norm


def clean_text(text: str, strip_obfuscation: bool = True) -> str:
    """Clean text payload.
    When strip_obfuscation=True (Stage B rules active):
      - strips zero-width invisible formatting characters
      - unhides and exposes hidden HTML comments (<!-- ... -->)
      - unhides and exposes hidden styling (display:none, visibility:hidden, opacity:0)
      - strips tags and scripts
    When strip_obfuscation=False (Stage B rules disabled):
      - preserves zero-width characters in text
      - removes comments and hidden HTML elements without exposing their inner content
      - strips tags and scripts
    """
    if strip_obfuscation:
        t = normalize_unicode(text, strip_zero_width=True)
        t = BR_RE.sub("\n", t)
        # Expose hidden HTML comments as visible text
        t = COMMENT_RE.sub(lambda m: " " + m.group(1).strip() + " ", t)
        # Expose hidden elements (display:none, visibility:hidden, opacity:0)
        t = HIDDEN_HTML_RE.sub(lambda m: " " + m.group(1).strip() + " ", t)
        t = SCRIPT_STYLE_RE.sub(" ", t)
        return TAG_RE.sub(" ", t)
    else:
        t = normalize_unicode(text, strip_zero_width=False)
        t = BR_RE.sub("\n", t)
        # Suppress hidden HTML comments without exposing their inner content
        t = COMMENT_RE.sub(" ", t)
        # Suppress hidden styling elements without exposing their inner content
        t = HIDDEN_HTML_RE.sub(" ", t)
        t = SCRIPT_STYLE_RE.sub(" ", t)
        return TAG_RE.sub(" ", t)


def has_high_confidence_injection_marker(text: str) -> bool:
    """Detect explicit instruction-override framing independently of semantic similarity."""
    return bool(OVERRIDE_REGEX.search(clean_text(text, strip_obfuscation=True)))


def has_hidden_content(text: str) -> bool:
    """Detect presence of zero-width characters, HTML comments, or hidden styling."""
    return bool(ZW_RE.search(text) or COMMENT_RE.search(text) or HIDDEN_HTML_RE.search(text))


def is_instruction(sentence: str) -> bool:
    return bool(IMPER.match(sentence) or MARKERS.search(sentence) or INLINE.search(sentence) or BARE.match(sentence) or OVERRIDE_REGEX.search(sentence))


def extract_candidates(text: str, strip_obfuscation: bool = True) -> list[str]:
    cleaned = NEUTRAL_RE.sub(" ", clean_text(text, strip_obfuscation=strip_obfuscation))
    segs = [_LEAD.sub("", x.strip(" \t'\"")).strip(" \t'\"") for x in _SPLIT.split(cleaned) if x]
    return [
        x for x in segs
        if len(x) >= 8 and (len(x.split()) >= 3 or MARKERS.search(x) or OVERRIDE_REGEX.search(x) or is_directive(x)) and (is_instruction(x) or is_directive(x))
    ]


_WORD = re.compile(r"[a-z0-9@.']+")


def _shingles(text: str, n: int = 4) -> set[str]:
    w = _WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def visible_fraction(injected: str, seen: str) -> float:
    """Share of the injected text's word 4-grams still readable in `seen` (after hidden-content exposure)."""
    g = _shingles(clean_text(injected))
    return len(g & _shingles(clean_text(seen))) / len(g) if g else 0.0


def strip_neutralized(text: str) -> str:
    return NEUTRAL_RE.sub(" ", text)


def _sanitize_string_value(s: str, decisions: list[tuple[str, bool]]) -> str:
    """Sanitize an individual string while preserving benign context and quarantining misaligned instructions."""
    t = clean_text(s)
    # Sort candidate decisions by length descending so longer matching clauses are handled first
    sorted_decs = sorted(decisions, key=lambda d: len(d[0]), reverse=True)
    for sent, ok in sorted_decs:
        if sent in t:
            rep = "\u27e6data: instruction-like content omitted\u27e7" if ok else "\u27e6redacted: instruction-like content quarantined\u27e7"
            t = t.replace(sent, rep)
        else:
            # Check stripped variant if punctuation or whitespace differed
            stripped = sent.strip(" \t'\".,;")
            if stripped and stripped in t:
                rep = "\u27e6data: instruction-like content omitted\u27e7" if ok else "\u27e6redacted: instruction-like content quarantined\u27e7"
                t = t.replace(stripped, rep)
    return t


def _sanitize_json_obj(obj, decisions: list[tuple[str, bool]]):
    """Recursively sanitize JSON data structures."""
    if isinstance(obj, str):
        return _sanitize_string_value(obj, decisions)
    if isinstance(obj, list):
        return [_sanitize_json_obj(item, decisions) for item in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_json_obj(v, decisions) for k, v in obj.items()}
    return obj


def sanitize(text: str, decisions: list[tuple[str, bool]]) -> str:
    """Complete untrusted payload sanitization.
    Handles plain text, exposed comments, zero-width stripping, markdown, and structured JSON payloads.
    Aligned instruction-like text is safely wrapped as quoted DATA (⟦data: ...⟧);
    misaligned instructions are quarantined (⟦redacted: instruction-like content quarantined⟧).
    """
    if not text:
        return ""

    # Check if payload is valid JSON (structured tool result)
    trimmed = text.strip()
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
        try:
            parsed = json.loads(trimmed)
            sanitized_obj = _sanitize_json_obj(parsed, decisions)
            return json.dumps(sanitized_obj, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Fall through to text sanitization if not strictly JSON

    return _sanitize_string_value(text, decisions)


ACTION_CLASSES: dict[str, set[str]] = {
    "read":    {"read", "reading", "list", "listing", "show", "showing", "summarize", "summarise", "summary", "summarizing", "get", "find", "search", "check", "inspect", "view", "fetch", "lookup", "display", "print", "tell", "query", "count", "verify", "examine", "look"},
    "write":   {"send", "sending", "forward", "forwarding", "create", "creating", "reschedule", "update", "updating", "edit", "editing", "post", "posting", "append", "invite", "share", "sharing", "write", "writing", "draft", "add", "adding", "wire", "transfer", "pay", "upload", "uploading", "dispatch", "dispatching", "email", "mail", "submit", "deliver"},
    "destroy": {"delete", "deleting", "deleted", "remove", "removing", "removed", "erase", "erasing", "erased", "cancel", "canceling", "cancelling", "canceled", "cancelled", "wipe", "wiping", "wiped", "drop", "dropping", "dropped", "clear", "clearing", "cleared", "purge", "purging", "purged", "revoke", "reset", "kill", "terminate", "withdraw", "destroy"},
}


def action_classes(text: str) -> set[str]:
    """Identify action verb classes present in the text."""
    if not text:
        return set()
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    classes = {c for c, verbs in ACTION_CLASSES.items() if words & verbs}
    # Check "schedule" as action verb: "schedule a meeting", "schedule sync" -> write
    if re.search(r"\bschedule\s+(?:a|an|the|new|\d|meeting|sync|call|appointment)\b", text.lower()):
        classes.add("write")
    return classes


def allowed_classes_for_request(user_request: str) -> set[str]:
    """Extract authorized action classes from the user root request.
    Informational/query-based requests with no explicit mutation verbs default to {'read'}.
    """
    classes = action_classes(user_request)
    if not classes:
        classes = {"read"}
    return classes


# Generalized directive patterns (Priority 2 Fix 5):
# 1. Agent-addressed phrasing
AGENT_ADDRESSED_RE = re.compile(
    r"\b(?:(?:message|note|instruction|directive|request|notice)s?\s+(?:to|for)\s+(?:the\s+)?(?:downstream\s+|next\s+)?(?:agent|assistant|model|ai|bot)|"
    r"(?:to|for)\s+(?:the\s+)?(?:downstream\s+|next\s+)?(?:agent|assistant|model|ai|bot)\s*[:,\-]?|"
    r"(?:agent|assistant|model|ai|bot)\s*[,:]\s*(?:please|kindly|you\s+(?:must|should|need\s+to|have\s+to)|do\s+not|never|always|[a-z]{3,})|"
    r"(?:attention|calling|dear|hey)\s+(?:agent|assistant|model|ai|bot)|"
    r"(?:the\s+)?(?:agent|assistant|model|ai)\s+(?:must|should|is\s+instructed\s+to|is\s+required\s+to|needs\s+to)|"
    r"(?:instructions?|guidelines?)\s+for\s+(?:the\s+)?(?:agent|assistant|model))\b",
    re.I
)

# 2. Authority framing ("the user wants...", "per user instruction", "user requested", etc.)
AUTHORITY_FRAMING_RE = re.compile(
    r"\b(?:(?:the\s+)?user\s+(?:wants|desires|requests|requires|needs|instructs|asked|demands|said|prefers|has\s+(?:instructed|requested|ordered|demanded|asked))|"
    r"(?:per|as\s+per|according\s+to)\s+(?:the\s+)?(?:user(?:'s)?|admin(?:'s)?|system)\s+(?:request|instruction|directive|demand|desire|wish|order)|"
    r"(?:user|admin|administrator|developer|management)\s+(?:preference|instruction|directive|command|order|update)\s*:|"
    r"(?:the\s+)?(?:admin|administrator|system|developer)\s+(?:wants|requires|instructs|demands|authorizes|mandates)|"
    r"(?:the\s+)?user\s+(?:modified|updated|changed)\s+(?:their|the)\s+(?:request|task|instruction|mind)|"
    r"(?:ensure\s+that|please\s+ensure|ensure\s+you|make\s+sure\s+(?:that|you)?))\b",
    re.I
)

# 3. URLs in tool output
TOOL_URL_RE = re.compile(r"(?:https?://|www\.)[^\s'\"<>\(\)\[\]\{\}]+", re.I)


def is_directive(span: str) -> bool:
    """Generalize directive detection to cover:
    1. Standard imperatives and override markers (is_instruction)
    2. Agent-addressed phrasing (AGENT_ADDRESSED_RE)
    3. Authority framing ("the user wants...", etc.) (AUTHORITY_FRAMING_RE)
    4. External URLs in tool output combined with action verbs (TOOL_URL_RE)
    """
    if not span or not span.strip():
        return False
    s = span.strip()
    return bool(
        is_instruction(s)
        or AGENT_ADDRESSED_RE.search(s)
        or AUTHORITY_FRAMING_RE.search(s)
        or (TOOL_URL_RE.search(s) and (action_classes(s) or bool(IMPER.match(s))))
    )


class AlignmentEngine:
    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def score(self, e_root, text: str) -> float:
        return cosine(self.embedder.encode(text), e_root)          # Eq. (1)

    def check_payload(self, e_root, text: str, tau: float,
                      enable_rules: bool = True, enable_sem: bool = True,
                      root_text: str = "") -> dict:
        candidates = extract_candidates(text, strip_obfuscation=enable_rules)
        if not candidates and text.strip():
            text_classes = action_classes(text)
            if text_classes:
                candidates = [text.strip()]
        root_allowed = allowed_classes_for_request(root_text) if root_text else {"read"}
        cands = []
        for candidate in candidates:
            score = round(self.score(e_root, candidate), 4) if enable_sem else None
            marker_detected = has_high_confidence_injection_marker(candidate) if enable_rules else False
            cand_classes = action_classes(candidate)

            # Fix 4: action-class check. An in-domain directive whose verb class escalates privileges
            # (e.g. 'delete' in a 'list/read' intent) is blocked regardless of semantic topical similarity.
            # root.allowed = classes(user_request)            # {"read"}
            # if is_directive(span) and not classes(span) <= root.allowed:
            #     return Block("action-class escalation")
            cand_is_directive = is_directive(candidate)
            action_class_escalation = bool(
                cand_is_directive and cand_classes and not (cand_classes <= root_allowed)
            )

            allowed = True
            cand_blocked_layers = []
            cand_rules = []
            cand_reason = "aligned"

            # 1. Hidden comment or zero-width character rule
            has_hidden = bool(has_hidden_content(candidate) or COMMENT_RE.search(candidate) or ZW_RE.search(candidate))
            if enable_rules and has_hidden:
                allowed = False
                cand_blocked_layers.append("rules")
                cand_rules.append("hidden-comment rule" if COMMENT_RE.search(candidate) else "zero-width rule")
                cand_reason = "hidden comment or invisible characters detected"

            # 2. Exfiltration and prompt injection marker rules
            if enable_rules and marker_detected:
                allowed = False
                cand_blocked_layers.append("rules")
                is_exfil = bool(re.search(r"\b(?:transfer|send|wire|pay|exfiltrat|leak)\b", candidate, re.I) and
                               re.search(r"(?:@|https?://|\$|\b(?:account|bank|wallet)\b)", candidate, re.I))
                if is_exfil:
                    cand_rules.append("exfiltration rule")
                    cand_reason = "unauthorized exfiltration or transfer directive detected"
                elif re.search(r"\b(?:ignore|disregard|system|assistant|admin)\b", candidate, re.I):
                    cand_rules.append("override marker rule")
                    cand_reason = "instruction override marker detected"
                else:
                    cand_rules.append("prompt injection rule")
                    cand_reason = "high confidence prompt injection marker detected"

            # 3. Action-class escalation check
            if enable_sem and action_class_escalation:
                allowed = False
                cand_blocked_layers.append("action_class")
                esc_detail = f"action-class escalation ({','.join(sorted(cand_classes))} vs {','.join(sorted(root_allowed))})"
                cand_rules.append(esc_detail)
                cand_reason = f"action-class escalation: candidate classes {sorted(cand_classes)} exceed root classes {sorted(root_allowed)}"

            # 4. Semantic similarity check
            if enable_sem and score is not None and score < tau:
                allowed = False
                cand_blocked_layers.append("sem")
                sim_detail = f"similarity (score {score:.3f} < τ {tau:.2f})"
                cand_rules.append(sim_detail)
                cand_reason = f"semantic similarity {score} below alignment threshold tau {tau}"

            cand_blocked_by = ", ".join(cand_rules) if cand_rules else ("aligned" if allowed else "rules")

            cands.append({
                "text": candidate,
                "score": score,
                "marker_detected": marker_detected,
                "action_classes": sorted(cand_classes),
                "action_class_escalation": action_class_escalation,
                "allowed": allowed,
                "blocked_by": cand_blocked_by,
                "blocked_layers": cand_blocked_layers,
                "reason": cand_reason,
            })

        # Check whole-payload marker if rules enabled but no candidates triggered it
        payload_marker = has_high_confidence_injection_marker(text) if enable_rules else False
        if enable_rules and payload_marker and not any(c["marker_detected"] for c in cands):
            cands.append({
                "text": text.strip()[:120],
                "score": round(self.score(e_root, text.strip()[:120]), 4) if enable_sem else None,
                "marker_detected": True,
                "action_classes": sorted(action_classes(text)),
                "action_class_escalation": False,
                "allowed": False,
                "blocked_by": "rules",
                "reason": "payload-level injection pattern matched",
            })

        scores = [c["score"] for c in cands if c["score"] is not None]
        min_score = min(scores, default=None)
        blocked = any(not c["allowed"] for c in cands)
        marker_count = sum(c["marker_detected"] for c in cands)

        blocked_layers = []
        for c in cands:
            cb = c.get("blocked_by") or ""
            for layer in ("rules", "action_class", "sem"):
                if layer in cb and layer not in blocked_layers:
                    blocked_layers.append(layer)
        blocked_by = "+".join(blocked_layers) if blocked_layers else ("rules" if blocked else None)

        return {
            "candidates": cands,
            "min_score": min_score,
            "blocked": blocked,
            "marker_count": marker_count,
            "hidden_content": has_hidden_content(text),
            "enable_rules": enable_rules,
            "enable_sem": enable_sem,
            "blocked_by": blocked_by,
        }
