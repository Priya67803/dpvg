"""
Script to validate Step 3 token optimization across all 10 required scenarios.
"""
import sys, os
from pathlib import Path
ROOT = Path(r"c:\Users\tthip\Downloads\dpvg\dpvg-main\dpvg")
sys.path.insert(0, str(ROOT))

from app.token_opt import TokenOptimizer, count_tokens
from app.embeddings import get_embedder

def validate():
    E = get_embedder()
    opt = TokenOptimizer(E)

    test_cases = [
        {
            "id": 1,
            "category": "Conversational filler removal",
            "prompt": "Hi there, could you please basically kindly read my latest emails, thanks so much!",
            "context": [],
            "constraints": ["read my latest emails"],
            "expected_survived": ["read", "emails"]
        },
        {
            "id": 2,
            "category": "Negation preservation",
            "prompt": "Could you please do not transfer money and never send this email to anyone?",
            "context": [],
            "constraints": ["do not transfer", "never send"],
            "expected_survived": ["do not", "never", "transfer", "send"]
        },
        {
            "id": 3,
            "category": "Financial amount preservation",
            "prompt": "Can you pay the pending invoice for $12,450.00 USD at your earliest convenience?",
            "context": [],
            "constraints": ["$12,450.00"],
            "expected_survived": ["$12,450.00", "invoice"]
        },
        {
            "id": 4,
            "category": "Email address preservation",
            "prompt": "I would really appreciate if you could forward the quarterly report to finance-team@securecorp.org please.",
            "context": [],
            "constraints": ["finance-team@securecorp.org"],
            "expected_survived": ["finance-team@securecorp.org", "quarterly report"]
        },
        {
            "id": 5,
            "category": "Date and deadline preservation",
            "prompt": "Please make sure to submit the audit files before Friday at 4:30 pm or by October 31, 2026.",
            "context": [],
            "constraints": ["Friday", "4:30", "October 31, 2026"],
            "expected_survived": ["Friday", "4:30", "October 31, 2026", "audit files"]
        },
        {
            "id": 6,
            "category": "Tool permission preservation",
            "prompt": "Kindly note you are restricted to read_file only, and must not call execute_command or post_slack.",
            "context": [],
            "constraints": ["restricted to", "read_file", "must not", "execute_command", "post_slack"],
            "expected_survived": ["read_file", "must not", "execute_command", "post_slack"]
        },
        {
            "id": 7,
            "category": "Conditional instruction preservation",
            "prompt": "Could you search the archives, but do not delete any files unless approved by the compliance officer?",
            "context": [],
            "constraints": ["do not delete", "unless approved"],
            "expected_survived": ["search", "archives", "do not delete", "unless"]
        },
        {
            "id": 8,
            "category": "Multi-clause prompt compression",
            "prompt": "Hi, could you please fetch the logs, and then analyze the error rates, and then summarize the findings?",
            "context": [],
            "constraints": ["fetch the logs", "analyze the error rates", "summarize the findings"],
            "expected_survived": ["fetch", "logs", "analyze", "error", "summarize"]
        },
        {
            "id": 9,
            "category": "Empty context handling",
            "prompt": "Can you fetch me the details and reviews of the Dell laptop with product ID B08KFQ9HK5 from Amazon?",
            "context": [],
            "constraints": ["B08KFQ9HK5", "Dell laptop", "Amazon"],
            "expected_survived": ["B08KFQ9HK5", "Dell", "laptop", "Amazon"]
        },
        {
            "id": 10,
            "category": "Original intent preservation",
            "prompt": "Please summarize the financial statements without disclosing any proprietary algorithms or passwords.",
            "context": [],
            "constraints": ["without disclosing", "passwords", "proprietary"],
            "expected_survived": ["summarize", "financial statements", "without", "proprietary", "passwords"]
        }
    ]

    print("=" * 105)
    print(f"{'ID':2} | {'Category':32} | {'Orig':5} | {'Opt':5} | {'Comp%':6} | {'All Constraints Survived':24}")
    print("=" * 105)

    results = []
    for tc in test_cases:
        p_orig = tc["prompt"]
        c_orig = tc["context"]
        prot = tc["constraints"]

        res = opt.optimize(p_orig, c_orig, prot)
        p_opt = res.query
        t_orig = res.tokens_before
        t_opt = res.tokens_after
        comp_pct = ((t_orig - t_opt) / t_orig) * 100 if t_orig > 0 else 0.0

        survived = True
        missing = []
        for term in tc["expected_survived"]:
            if term.lower() not in p_opt.lower():
                survived = False
                missing.append(term)

        print(f"{tc['id']:2} | {tc['category']:32} | {t_orig:5} | {t_opt:5} | {comp_pct:5.1f}% | {str(survived):24}")
        if not survived:
            print(f"   --> Missing terms: {missing}")

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "original_prompt": p_orig,
            "optimized_prompt": p_opt,
            "original_tokens": t_orig,
            "optimized_tokens": t_opt,
            "compression_percentage": round(comp_pct, 2),
            "survived": survived,
            "missing_terms": missing
        })

    print("=" * 105)
    import json
    out_path = ROOT / "results" / "step3_token_opt_validation.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Detailed Step 3 results written to {out_path}")

if __name__ == "__main__":
    validate()

