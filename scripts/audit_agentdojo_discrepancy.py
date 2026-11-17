"""
Diagnostic audit script to investigate the AgentDojo discrepancy:
Why did G-AB block 6 unauthorized calls while FULL blocked 5?
Log every single call, arguments, and authorization decision.
"""
import sys, os, json
from pathlib import Path
ROOT = Path(r"c:\Users\tthip\Downloads\dpvg\dpvg-main\dpvg")
sys.path.insert(0, str(ROOT))

from app.benchmarks import run_agentdojo

def main():
    print("=" * 80)
    print("INVESTIGATING AGENTDOJO G-AB vs FULL TOOL CALL DISCREPANCY")
    print("=" * 80)

    # Re-run agentdojo for G-AB and FULL on banking suite (limit_user=2, limit_inj=2)
    res = run_agentdojo(modes=["G-AB", "FULL"], suites=["banking"], limit_user=2, limit_inj=2,
                        progress=lambda m, s, r: print(f"  [{m}] {s}: {r}", flush=True))

    rows = res["rows"]
    print("\nSummary Results:")
    for r in rows:
        print(f"Mode {r['mode']:6}: unauthorized_blocked={r['unauthorized_blocked']}, n_checks={r['n_checks']}")
        print(f"  Details: {r['by_suite']}")

    out_file = ROOT / "results" / "agentdojo_discrepancy_check.json"
    out_file.write_text(json.dumps(res, indent=2))
    print(f"\nWritten diagnostic output to {out_file}")

if __name__ == "__main__":
    main()

