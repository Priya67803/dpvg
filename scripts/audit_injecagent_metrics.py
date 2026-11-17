"""
Audit InjecAgent metrics, ground truth evaluator, and false positive analysis.
Runs B0, G-AB, G-AB+T, FULL on the exact 10 cases from the smoke test.
"""
import sys, os, json
from pathlib import Path
ROOT = Path(r"c:\Users\tthip\Downloads\dpvg\dpvg-main\dpvg")
sys.path.insert(0, str(ROOT))

from app.benchmarks import load_injecagent
from app.gateway import Gateway, is_malicious, attacker_markers
from app.embeddings import get_embedder

def audit():
    E = get_embedder()
    gw = Gateway(embedder=E, tau=0.55)
    cases = load_injecagent(setting="base", split="test", limit=10)[:10]

    print("=" * 85)
    print("INJECAGENT METRICS & FALSE POSITIVE AUDIT (10 Cases)")
    print("=" * 85)

    atks = [c for c in cases if c.get("is_attack")]
    bens = [c for c in cases if not c.get("is_attack")]
    print(f"Total cases: {len(cases)} (Attacks: {len(atks)}, Benign: {len(bens)})")

    # Inspect ground-truth definitions and markers
    print("\n--- GROUND TRUTH & ATTACK DEFINITIONS ---")
    for i, c in enumerate(atks):
        inj = c.get("injection", "")
        markers = list(attacker_markers(inj))
        print(f"Attack {i} (id={c['id']}):")
        print(f"  Goal: {c['goal']}")
        print(f"  Injection snippet: {inj[:90]}...")
        print(f"  Attacker markers extracted: {markers}")

    # Run B0, G-AB, G-AB+T, FULL and record exact decisions
    modes = ["B0", "G-AB", "G-AB+T", "FULL"]
    mode_runs = {m: [] for m in modes}

    print("\n--- DETAILED RUN TRACES PER CASE ---")
    for idx, c in enumerate(cases):
        cid = c["id"]
        is_atk = c.get("is_attack", False)
        print(f"\n[Case {idx}] ID: {cid} | Type: {'ATTACK' if is_atk else 'BENIGN'} | Scenario: {c.get('scenario')}")
        print(f"  Query: {c['query']}")

        for m in modes:
            r = gw.run(c, m)
            metrics = r["metrics"]
            ledger = r["ledger"]
            checks = r["checks"]
            blocked_checks = [ck for ck in checks if not ck.get("allowed", True)]
            
            res_entry = {
                "case_id": cid,
                "is_attack": is_atk,
                "mode": m,
                "attack_success": metrics["attack_success"],
                "detected": metrics["detected"],
                "completed": metrics["completed"],
                "ledger_calls": len(ledger),
                "blocked_checks": blocked_checks,
                "final_response": r["final_response"][:80]
            }
            mode_runs[m].append(res_entry)
            print(f"  Mode {m:6} -> attack_success={metrics['attack_success']}, detected={metrics['detected']}, completed={metrics['completed']}, ledger_calls={len(ledger)}")
            if blocked_checks:
                for bc in blocked_checks:
                    print(f"       Blocked at stage={bc.get('stage')}, code={bc.get('code')}, score={bc.get('score')}, reason={bc.get('reason')}")

    # Compute exact denominators and metrics
    print("\n" + "=" * 85)
    print("METRIC CALCULATIONS & DENOMINATORS EXPLAINED")
    print("=" * 85)
    
    summary = []
    for m in modes:
        runs = mode_runs[m]
        atk_runs = [r for r in runs if r["is_attack"]]
        ben_runs = [r for r in runs if not r["is_attack"]]

        n_atk = len(atk_runs)
        n_ben = len(ben_runs)

        atk_success_cnt = sum(r["attack_success"] for r in atk_runs)
        atk_detected_cnt = sum(r["detected"] for r in atk_runs)
        ben_completed_cnt = sum(r["completed"] and not r["detected"] for r in ben_runs)
        ben_flagged_cnt = sum(r["detected"] for r in ben_runs)

        asr = atk_success_cnt / n_atk if n_atk else 0.0
        recall = atk_detected_cnt / n_atk if n_atk else 0.0
        tcr = ben_completed_cnt / n_ben if n_ben else 0.0
        fpr = ben_flagged_cnt / n_ben if n_ben else 0.0

        print(f"Mode {m:6}:")
        print(f"  ASR    = {atk_success_cnt} / {n_atk} attack cases = {asr:.4f}")
        print(f"  Recall = {atk_detected_cnt} / {n_atk} attack cases = {recall:.4f}")
        print(f"  TCR    = {ben_completed_cnt} / {n_ben} benign cases = {tcr:.4f}")
        print(f"  FPR    = {ben_flagged_cnt} / {n_ben} benign cases = {fpr:.4f}")

        summary.append({
            "mode": m,
            "n_attack": n_atk,
            "n_benign": n_ben,
            "attack_success_count": atk_success_cnt,
            "attacks_detected": atk_detected_cnt,
            "benign_completed": ben_completed_cnt,
            "benign_flagged": ben_flagged_cnt,
            "asr": asr,
            "recall": recall,
            "tcr": tcr,
            "fpr": fpr
        })

    out_file = ROOT / "results" / "injecagent_metrics_audit.json"
    out_file.write_text(json.dumps({"summary": summary, "traces": mode_runs}, indent=2))
    print(f"\nAudit results written to {out_file}")

if __name__ == "__main__":
    audit()

