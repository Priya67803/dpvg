"""Token Optimization Module (Step 1).
Four operations: prompt compression, context pruning, task decomposition, token budgeting.
Intent, constraint, policy, entities, amounts, negations, and dependencies are PROTECTED.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from .embeddings import cosine

_ENC = None
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

_WORDPIECE_TOK = None
try:
    from tokenizers import Tokenizer
    from . import config
    import pathlib
    tok_path = pathlib.Path(config.MODEL_DIR) / "tokenizer.json"
    if tok_path.exists():
        _WORDPIECE_TOK = Tokenizer.from_file(str(tok_path))
        _WORDPIECE_TOK.no_padding()
        _WORDPIECE_TOK.no_truncation()
except Exception:
    _WORDPIECE_TOK = None


def count_tokens(text: str) -> int:
    """Exact token counting using tiktoken (cl100k_base) or WordPiece tokenizer."""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            pass
    if _WORDPIECE_TOK is not None:
        try:
            return len(_WORDPIECE_TOK.encode(text).ids)
        except Exception:
            pass
    # BPE-like approximation as last resort
    return sum(1 + len(w) // 8 for w in re.findall(r"\w+|[^\w\s]", text))


def get_token_counter_source() -> str:
    """Return the name of the token counting implementation used."""
    if _ENC is not None:
        return "tiktoken"
    if _WORDPIECE_TOK is not None:
        return "wordpiece"
    return "approximate"


def truncate_to_tokens(text: str, budget: int) -> str:
    """Return a prefix whose measured size does not exceed `budget`."""
    if budget <= 0 or not text:
        return ""
    if count_tokens(text) <= budget:
        return text
    if _ENC is not None:
        return _ENC.decode(_ENC.encode(text)[:budget])
    if _WORDPIECE_TOK is not None:
        enc = _WORDPIECE_TOK.encode(text)
        k = min(budget, len(enc.ids))
        res = _WORDPIECE_TOK.decode(enc.ids[:k], skip_special_tokens=True)
        while k > 0 and count_tokens(res) > budget:
            k -= 1
            res = _WORDPIECE_TOK.decode(enc.ids[:k], skip_special_tokens=True)
        return res

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip()


_FILLERS = [
    (re.compile(r"\b(?:hi|hello|hey)(?: there)?\b[,!]?\s*", re.I), ""),
    (re.compile(r"\b(?:could|can|would) you(?: please)?\s*", re.I), ""),
    (re.compile(r"\bI (?:would |'d )?(?:really )?(?:like|want) you to\s*", re.I), ""),
    (re.compile(r"\bI (?:would )?(?:really )?appreciate (?:it|that|your help)[,!.]?\s*", re.I), ""),
    (re.compile(r",?\s*\b(?:thanks|thank you)(?: a lot| so much| in advance)?[,!.]?", re.I), ""),
    (re.compile(r"\b(?:please|kindly|just|basically|actually|really|very|quite|simply)\b\s*", re.I), ""),
    (re.compile(r"\bin order to\b", re.I), "to"),
]

# Patterns for critical entities, constraints, numbers, negations that must never be corrupted
_CRITICAL_PATTERNS = [
    re.compile(r"\b(?:not|never|no|none|neither|nor|without|except|unless|only|do not|don't|must not|cannot|can't|should not|shouldn't|would not|wouldn't|will not|won't|refuse|deny|prohibit|forbidden)\b", re.I),
    re.compile(r"\b(?:only\s+read|do\s+not\s+(?:transfer|send|delete|modify|execute|wire|pay|write)|never\s+(?:transfer|send|delete|modify|execute|wire|pay|write)|must\s+not\s+(?:transfer|send|delete|modify|execute|wire|pay|write)|only\s+use|restricted\s+to|read-only|read\s+only)\b", re.I),
    re.compile(r"\b(?:transfer_money|send_email|delete_file|read_file|read_email|get_calendar|search_web|post_slack|external_action)\b", re.I),
    re.compile(r"\b(?:do\s+not\s+reveal|never\s+reveal|do\s+not\s+share|never\s+share|do\s+not\s+disclose|never\s+disclose|confidential|secret|private|sensitive|credentials?|passwords?|api_key|token)\b", re.I),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                                        # emails
    re.compile(r"https?://\S+"),                                                    # urls
    re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP|dollars?|cents?)\b", re.I), # currency amounts
    re.compile(r"\b(?:INV|REF|ID|TXN|ACC|PO|AUTH|KEY|SKU|PROD)[-_#]?[A-Z0-9_-]+\b", re.I),           # identifier codes
    re.compile(r"\b[A-Z0-9]{8,}(?:-[A-Z0-9]+)*\b"),                                # alphanumeric IDs (e.g. Amazon ASIN B08KFQ9HK5)
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", re.I),                           # times
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),                              # date formats
    re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I), # weekdays
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:\d{4})?\b", re.I), # full dates
    re.compile(r"\b(?:before|after|by|until|deadline|due\s+date|due\s+by)\b", re.I), # deadlines/conditionals
    re.compile(r"\b\d+(?:\.\d+)?%?\b"),                                             # numbers & percentages
]


PROTECTED_CLASSES = {"user_preference", "policy", "constraint", "intent"}

_USER_PREFERENCE_PATTERNS = [
    re.compile(r"\b(?:the\s+)?user\s+prefers?\b", re.I),
    re.compile(r"\b(?:user\s+)?preference[s]?\s*:", re.I),
    re.compile(r"\b(?:language|style|format|tone)\s+preference\b", re.I),
    re.compile(r"\bprefers?\s+(?:concise|detailed|brief|formal|informal|plain|bullet|markdown|json)\b", re.I),
    re.compile(r"\bwritten\s+in\s+(?:British|American|Australian|Canadian)?\s*English\b", re.I),
    re.compile(r"\b(?:always\s+respond\s+in|always\s+write\s+in|always\s+use)\b", re.I),
]


def is_user_preference(text: str | dict) -> bool:
    """Return True if text or chunk represents a user preference."""
    if isinstance(text, dict):
        if text.get("class") == "user_preference" or text.get("type") == "user_preference":
            return True
        raw = str(text.get("text") or text.get("content") or "")
    else:
        raw = str(text)
    return any(p.search(raw) for p in _USER_PREFERENCE_PATTERNS)


def is_protected_chunk(chunk: str | dict, protected: list[str] | tuple[str, ...] = (),
                       protected_classes: set[str] | None = None) -> bool:
    """Return True if chunk belongs to a protected class (user_preference, policy, constraint, intent)."""
    classes = protected_classes or PROTECTED_CLASSES
    if isinstance(chunk, dict):
        cls = chunk.get("class") or chunk.get("type") or chunk.get("category")
        if cls in classes:
            return True
        raw = str(chunk.get("text") or chunk.get("content") or "")
    else:
        raw = str(chunk)
    if is_user_preference(raw):
        return True
    for p in protected:
        if p and (p in raw or raw in p):
            return True
    return False


def extract_critical_spans(text: str) -> list[str]:
    """Identify numbers, amounts, negations, emails, and identifiers to protect from deletion."""
    spans = []
    for pat in _CRITICAL_PATTERNS:
        for m in pat.finditer(text):
            spans.append(m.group(0).strip())
    return [s for s in spans if s]


@dataclass
class OptResult:
    query: str
    context: list[str]
    subtasks: list[str]
    budget: dict
    dropped_context: list[str] = field(default_factory=list)
    context_by_subtask: dict[str, list[str]] = field(default_factory=dict)
    tokens_before: int = 0
    tokens_after: int = 0
    latency_ms: float = 0.0

    @property
    def r_tok(self) -> float:
        return 0.0 if self.tokens_before == 0 else 1.0 - self.tokens_after / self.tokens_before   # Eq. (3)


_SPLIT_TASK = re.compile(r",?\s+and then\s+|,?\s+then\s+|;\s*|\.\s+(?=[A-Z])", re.I)


def decompose(query: str) -> list[str]:
    parts = [p.strip(" .,") for p in _SPLIT_TASK.split(query) if p and p.strip(" .,")]
    return parts or [query.strip()]


class TokenOptimizer:
    def __init__(self, embedder, total_budget: int = 3000, min_sim: float = 0.30) -> None:
        if total_budget < 1:
            raise ValueError("total_budget must be positive")
        if not -1.0 <= min_sim <= 1.0:
            raise ValueError("min_sim must be between -1 and 1")
        self.embedder, self.total_budget, self.min_sim = embedder, total_budget, min_sim

    # 1) prompt compression with structured protection & semantic validation ---------
    def compress(self, text: str, protected: tuple[str, ...] | list[str] = ()) -> str:
        if not text:
            return ""

        if is_user_preference(text):
            return text

        # Extract structured critical spans from the text itself
        critical_spans = extract_critical_spans(text)

        # Process caller-supplied protected elements
        # Avoid treating the entire raw prompt as a monolithic uncompressible block
        clean_protected = []
        for p in protected:
            if not p:
                continue
            p_str = str(p).strip()
            # If the protected item is identical to the whole text, extract its critical clauses instead
            if p_str.lower() == text.strip().lower():
                clean_protected.extend(extract_critical_spans(p_str))
            else:
                clean_protected.append(p_str)

        all_protected = set(clean_protected + critical_spans)

        # Assign unique placeholders (§0§, §1§, ...) to protected spans
        holders: dict[str, str] = {}
        t_work = text
        for i, p in enumerate(sorted({p for p in all_protected if p and p in t_work}, key=len, reverse=True)):
            ph = f"\u00a7{i}\u00a7"
            t_work = t_work.replace(p, ph)
            holders[ph] = p

        for pat, rep in _FILLERS:
            t_work = pat.sub(rep, t_work)

        t_work = re.sub(r"\s+([,.!?;])", r"\1", t_work)
        t_work = re.sub(r"\s{2,}", " ", t_work).strip(" ,")

        seen, sents = set(), []                                  # drop duplicated sentences
        for s in re.split(r"(?<=[.!?])\s+", t_work):
            k = s.lower().strip()
            if k and k not in seen:
                seen.add(k)
                sents.append(s)
        t_work = " ".join(sents)

        # Restore protected fields exactly
        for ph, p in holders.items():
            t_work = t_work.replace(ph, p)

        compressed = t_work[:1].upper() + t_work[1:] if t_work else t_work

        # --- Security Invariant & Semantic Equivalence Verification ---
        # 1. Invariant Check: All critical spans must be present in compressed text
        c_low = compressed.lower()
        for span in critical_spans:
            if span.lower() not in c_low:
                # Critical entity or negation lost: fail safe to original text
                return text

        # 2. Semantic Equivalence Check: Verify cosine similarity >= 0.45 (PAPER_TAU)
        if self.embedder is not None and text.strip() and compressed.strip():
            try:
                from .embeddings import cosine
                sim = cosine(self.embedder.encode(text), self.embedder.encode(compressed))
                if sim < 0.45:
                    # Meaning drifted excessively: fail safe to original text
                    return text
            except Exception:
                pass

        return compressed

    # 2) context pruning ---------------------------------------------------------------
    def prune(self, chunks: list[str], query: str, budget: int,
              min_similarity: float | None = None,
              protected: list[str] | tuple[str, ...] | None = None) -> tuple[list[str], list[str]]:
        if not chunks:
            return [], []
        cutoff = self.min_sim if min_similarity is None else min_similarity
        if not -1.0 <= cutoff <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        prot_list = list(protected or [])
        keep_indices = set()
        used = 0
        # Protected chunks (user_preference, policy, constraint, intent) are NEVER dropped by context pruning
        for i, c in enumerate(chunks):
            if is_protected_chunk(c, prot_list):
                keep_indices.add(i)
                used += count_tokens(c if isinstance(c, str) else str(c))
        eq = self.embedder.encode(query) if self.embedder is not None else None
        ranked = sorted(
            ((cosine(self.embedder.encode(c if isinstance(c, str) else str(c)), eq) if eq is not None else 0.0, i, c)
             for i, c in enumerate(chunks)),
            reverse=True
        )
        for sim, i, c in ranked:
            if i in keep_indices:
                continue
            n = count_tokens(c if isinstance(c, str) else str(c))
            if sim >= cutoff and used + n <= budget:
                keep_indices.add(i)
                used += n
        keep = [(i, chunks[i]) for i in sorted(keep_indices)]
        kept = [c for _, c in keep]
        return kept, [c for i, c in enumerate(chunks) if i not in keep_indices]

    # 4) token budgeting ---------------------------------------------------------------
    def allocate(self, subtasks: list[str]) -> dict:
        if not subtasks:
            return {}
        w = [max(1, count_tokens(s)) for s in subtasks]
        allocations = []
        remaining_budget = self.total_budget
        remaining_weight = sum(w)
        for index, weight in enumerate(w):
            allocation = (
                remaining_budget if index == len(w) - 1
                else int(remaining_budget * weight / remaining_weight)
            )
            allocations.append(allocation)
            remaining_budget -= allocation
            remaining_weight -= weight
        return {f"subtask_{i}": value for i, value in enumerate(allocations)}

    def optimize(self, query: str, context: list[str], protected: list[str],
                 min_similarity: float | None = None) -> OptResult:
        import time
        t0 = time.perf_counter()
        before = count_tokens(query) + sum(count_tokens(c) for c in context)
        q = self.compress(query, protected)
        if count_tokens(q) > count_tokens(query):
            q = query  # Zero compression is strictly preferable to token expansion
        subtasks = decompose(q)
        cleaned_context = [self.compress(c, protected) for c in context]
        budgets = self.allocate(subtasks)
        context_by_subtask = {}
        for i, task in enumerate(subtasks):
            key = f"subtask_{i}"
            context_budget = max(0, budgets[key] - count_tokens(task))
            kept, _ = self.prune(cleaned_context, task, context_budget, min_similarity=min_similarity, protected=protected)
            context_by_subtask[key] = kept
        kept_set = {chunk for chunks in context_by_subtask.values() for chunk in chunks}
        kept = [chunk for chunk in cleaned_context if chunk in kept_set]
        dropped = [chunk for chunk in cleaned_context if chunk not in kept_set]
        after = count_tokens(q) + sum(count_tokens(c) for c in kept)
        lat = (time.perf_counter() - t0) * 1000
        return OptResult(q, kept, subtasks, budgets, dropped, context_by_subtask,
                         before, after, latency_ms=lat)
