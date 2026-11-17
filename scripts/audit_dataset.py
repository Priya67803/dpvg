"""InjecAgent Dataset Audit & Split Leakage Analysis.

Audits raw case counts across settings (base, enhanced, dh, ds), details benign case construction,
enumerates user/attacker tool distributions, checks for validation/test leakage, and validates
the grouped split.
"""
from __future__ import annotations

import collections
import difflib
import json
import pathlib
import re
import sys

# Ensure repository root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmarks import DATA_DIR, IA_FILES, BENIGN_FILL, _split_of, load_injecagent


def audit_raw_files() -> dict:
    """Audit raw InjecAgent JSON files."""
    results = {}
    for (kind, setting), fname in IA_FILES.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        rows = json.loads(fpath.read_text(encoding="utf-8"))
        results[(kind, setting)] = {
            "filename": fname,
            "count": len(rows),
            "rows": rows,
        }
    return results


def audit_tool_distributions(rows: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    """Count occurrences per user tool and attacker tool."""
    user_tools = collections.Counter()
    attacker_tools = collections.Counter()
    for r in rows:
        user_tools[r.get("User Tool", "unknown")] += 1
        at = tuple(sorted(r.get("Attacker Tools", [])))
        at_name = " + ".join(at) if at else "none"
        attacker_tools[at_name] += 1
    return dict(user_tools), dict(attacker_tools)


def audit_splits(rows: list[dict], split_fn) -> dict:
    """Analyze a split function for case counts and group/text leakage."""
    val_rows = [r for r in rows if split_fn(r) == "val"]
    test_rows = [r for r in rows if split_fn(r) == "test"]

    val_groups = {(r.get("User Tool"), tuple(sorted(r.get("Attacker Tools", [])))) for r in val_rows}
    test_groups = {(r.get("User Tool"), tuple(sorted(r.get("Attacker Tools", [])))) for r in test_rows}
    group_overlap = val_groups & test_groups

    val_at = {tuple(sorted(r.get("Attacker Tools", []))) for r in val_rows}
    test_at = {tuple(sorted(r.get("Attacker Tools", []))) for r in test_rows}
    at_overlap = val_at & test_at

    val_inst = {r.get("Attacker Instruction", "") for r in val_rows}
    test_inst = {r.get("Attacker Instruction", "") for r in test_rows}
    inst_overlap = val_inst & test_inst
    val_norm_inst = {_normalize_text(text) for text in val_inst if text}
    test_norm_inst = {_normalize_text(text) for text in test_inst if text}
    normalized_overlap = val_norm_inst & test_norm_inst
    near_duplicates = []
    for val_text in sorted(val_norm_inst):
        for test_text in sorted(test_norm_inst):
            similarity = difflib.SequenceMatcher(None, val_text, test_text).ratio()
            if similarity >= 0.90:
                near_duplicates.append({
                    "similarity": round(similarity, 4),
                    "validation": val_text,
                    "test": test_text,
                })

    val_user_tools = {r.get("User Tool") for r in val_rows}
    test_user_tools = {r.get("User Tool") for r in test_rows}
    ut_overlap = val_user_tools & test_user_tools

    return {
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "group_overlap_count": len(group_overlap),
        "val_attacker_tools": len(val_at),
        "test_attacker_tools": len(test_at),
        "attacker_tool_overlap_count": len(at_overlap),
        "val_attacker_instructions": len(val_inst),
        "test_attacker_instructions": len(test_inst),
        "attacker_instruction_overlap_count": len(inst_overlap),
        "normalized_attacker_instruction_overlap_count": len(normalized_overlap),
        "near_duplicate_attacker_instruction_count": len(near_duplicates),
        "near_duplicate_examples": near_duplicates[:10],
        "user_tool_overlap_count": len(ut_overlap),
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def main():
    print("=" * 80)
    print("INJECAGENT DATASET AUDIT & LEAKAGE REPORT")
    print("=" * 80)

    raw_files = audit_raw_files()
    print("\n--- 1. Raw Case Counts by File & Setting ---")
    rows_by_setting = {"base": [], "enhanced": []}

    for (kind, setting), data in sorted(raw_files.items()):
        print(f"  {data['filename']:<35} ({kind.upper()}, {setting:<8}): {data['count']:>5} rows")
        rows_by_setting[setting].extend(data["rows"])

    total_base = len(rows_by_setting["base"])
    total_enhanced = len(rows_by_setting["enhanced"])
    print(f"  {'Total InjecAgent Base:':<48} {total_base:>5} rows (510 DH + 544 DS)")
    print(f"  {'Total InjecAgent Enhanced:':<48} {total_enhanced:>5} rows (510 DH + 544 DS)")
    print(f"  {'Combined Raw Prompts:':<48} {total_base + total_enhanced:>5} rows")

    print("\n--- 2. Benign Case Construction & Harness Expansion ---")
    print("  InjecAgent provides ONLY attack prompts in raw JSON files.")
    print("  To measure false positive rates (FPR) and benign task completion (TCR),")
    print("  app.benchmarks.load_injecagent() derives matched benign cases from each row:")
    print("    1. 'static' (Attack Case): Uses unmodified 'Tool Response' with attacker injection.")
    print("    2. 'benign' (Slot Emptied): Replaces '<Attacker Instruction>' in 'Tool Response Template' with empty string ''.")
    print("    3. 'benign_hard' (Harmless Request): Replaces '<Attacker Instruction>' with benign imperative fill from BENIGN_FILL.")
    print(f"  Multiplier per attack row: 3 cases (1 attack + 2 benign).")
    for setting, rows in rows_by_setting.items():
        filled = sum("<Attacker Instruction>" in r.get("Tool Response Template", "") for r in rows)
        harness_total = filled * 3
        print(f"  {setting.title():<10} harness cases: {harness_total} ({filled} attacks, {filled * 2} benign)")

    print("\n--- 3. Tool Distributions (Base Setting: N=1,054) ---")
    user_tool_dist, atk_tool_dist = audit_tool_distributions(rows_by_setting["base"])
    print(f"  Unique User Tools:     {len(user_tool_dist)}")
    for ut, cnt in sorted(user_tool_dist.items()):
        print(f"    - {ut:<35}: {cnt:>3} cases")

    print(f"\n  Unique Attacker Tools: {len(atk_tool_dist)}")
    for at, cnt in sorted(atk_tool_dist.items()):
        print(f"    - {at:<50}: {cnt:>3} cases")

    print("\n--- 4. Split Leakage Analysis (Base Setting) ---")
    split_audit = audit_splits(rows_by_setting["base"], _split_of)
    print("  Current Split Function (_split_of in app.benchmarks):")
    print(f"    Validation attack cases: {split_audit['val_count']}")
    print(f"    Test attack cases:       {split_audit['test_count']}")
    print(f"    Attacker tool overlap:   {split_audit['attacker_tool_overlap_count']} (Target: 0)")
    print(f"    Attacker text overlap:   {split_audit['attacker_instruction_overlap_count']} (Target: 0)")
    print(f"    Normalized text overlap: {split_audit['normalized_attacker_instruction_overlap_count']} (Target: 0)")
    print(f"    Near-duplicate overlap:  {split_audit['near_duplicate_attacker_instruction_count']} (threshold: 0.90; Target: 0)")
    print(f"    Group (UT, AT) overlap:  {split_audit['group_overlap_count']} (Target: 0)")
    if split_audit["near_duplicate_examples"]:
        print("    Near-duplicate examples:")
        for pair in split_audit["near_duplicate_examples"]:
            print(f"      {pair['similarity']:.3f}: {pair['validation']!r} <> {pair['test']!r}")
    if (split_audit['attacker_tool_overlap_count'] == 0 and
            split_audit["group_overlap_count"] == 0 and
            split_audit["near_duplicate_attacker_instruction_count"] == 0):
        print("    STATUS: LEAK-FREE (Strictly disjoint attacker tools, text, and groups).")
    else:
        print("    STATUS: LEAKY (Attacker text/tools appear in both splits).")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
