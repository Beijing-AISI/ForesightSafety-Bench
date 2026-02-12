#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse

import pandas as pd


def norm_text(s: str) -> str:
    """Normalize goal text to improve join robustness."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    return s


def find_pair_key(jailbroken: dict, prefer: str | None = None) -> str | None:
    """Find the PAIR_* key in jailbroken dict."""
    if not isinstance(jailbroken, dict):
        return None
    keys = list(jailbroken.keys())
    pair_keys = [k for k in keys if "PAIR" in k]  # tolerate e.g. PAIR_xxx
    if not pair_keys:
        return None
    if prefer and prefer in jailbroken:
        return prefer
    pair_keys.sort()
    return pair_keys[0]


def iter_result_items(json_path: str):
    """Yield each item dict with goal and jailbroken from a results.json file."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    items = obj.get("results", obj)  # support either {"results":[...]} or direct list
    if not isinstance(items, list):
        return

    for item in items:
        if not isinstance(item, dict):
            continue
        goal = item.get("goal", "")
        jailbroken = item.get("jailbroken", {})
        yield goal, jailbroken


def infer_llm_name(fp: str, input_dir: str) -> str:
    """
    Infer LLM name from path assuming layout:
      {input_dir}/{llm_name}/.../results.json
    """
    rel = os.path.relpath(fp, input_dir)
    parts = rel.split(os.sep)
    return parts[0] if parts else "UNKNOWN_LLM"


def parse_pair_value(val) -> int | None:
    """
    Normalize judge output to 0/1 if possible.
    Returns:
      0 or 1 on success, None if unparseable.
    """
    # Common clean cases
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, int):
        if val in (0, 1):
            return val
        # sometimes returns other ints; interpret nonzero as 1
        return 1 if val != 0 else 0
    if isinstance(val, float):
        if val == 0.0:
            return 0
        if val == 1.0:
            return 1
        # interpret nonzero as 1
        return 1 if val != 0.0 else 0

    # String cases
    s = str(val).strip().lower()
    if s in ("0", "false", "no", "n", "fail", "failed"):
        return 0
    if s in ("1", "true", "yes", "y", "pass", "passed", "success", "successful"):
        return 1

    # last resort: try int conversion
    try:
        iv = int(str(val).strip())
        return 1 if iv != 0 else 0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing judged results.json files (recursive). e.g. ../../benchmarks/jbb_judged/",
    )
    ap.add_argument(
        "--type-csv",
        type=str,
        required=True,
        help="CSV/Excel-export with columns: Goal, Type. e.g. /mnt/data/embodied.csv",
    )
    ap.add_argument(
        "--pair-key",
        type=str,
        default=None,
        help="Optional exact PAIR key to use, e.g. PAIR_gpt-4o-2024-11-20. If omitted, auto-pick.",
    )
    ap.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save by-LLM by-type summary CSV.",
    )
    ap.add_argument(
        "--save-txt",
        type=str,
        default=None,
        help="Optional path to save all output results to a txt file.",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="pair_eq_1_rate",
        choices=["pair_eq_1_rate", "pair_eq_0_rate"],
        help="Which rate to compute. pair_eq_1_rate means PAIR==1 proportion.",
    )
    args = ap.parse_args()

    # Load type mapping
    df_type = pd.read_csv(args.type_csv)
    if "Goal" not in df_type.columns or "Type" not in df_type.columns:
        raise ValueError("type-csv must contain columns: Goal, Type")

    df_type = df_type.copy()
    df_type["Goal_norm"] = df_type["Goal"].apply(norm_text)
    df_type["Type"] = df_type["Type"].astype(str)
    goal2type = dict(zip(df_type["Goal_norm"], df_type["Type"]))

    # Find all results.json files
    pattern = os.path.join(args.input_dir, "**", "results.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No results.json found under: {args.input_dir}")

    rows = []
    missing_type = 0
    missing_pair = 0
    bad_pair_value = 0

    # Iterate files and items
    for fp in files:
        llm = infer_llm_name(fp, args.input_dir)
        for goal, jailbroken in iter_result_items(fp):
            goal_n = norm_text(goal)
            typ = goal2type.get(goal_n, "UNKNOWN")
            if typ == "UNKNOWN":
                missing_type += 1

            pair_key = find_pair_key(jailbroken, prefer=args.pair_key)
            if pair_key is None:
                missing_pair += 1
                continue

            raw_val = jailbroken.get(pair_key, None)
            val_int = parse_pair_value(raw_val)
            if val_int is None:
                bad_pair_value += 1
                continue

            rows.append(
                {
                    "llm": llm,
                    "file": fp,
                    "Goal": goal,
                    "Type": typ,
                    "pair_key": pair_key,
                    "pair_value": val_int,
                }
            )

    if not rows:
        raise RuntimeError("No usable PAIR entries found (pair_key missing or values unparseable).")

    df = pd.DataFrame(rows)

    # Metric column: whether pair == 1 (or 0)
    if args.metric == "pair_eq_1_rate":
        df["metric_hit"] = (df["pair_value"] == 1).astype(int)
        metric_name = "PAIR==1 rate"
    else:
        df["metric_hit"] = (df["pair_value"] == 0).astype(int)
        metric_name = "PAIR==0 rate"

    # Overall rate per LLM
    overall_by_llm = (
        df.groupby("llm")
        .agg(
            n=("metric_hit", "size"),
            hits=("metric_hit", "sum"),
            rate=("metric_hit", "mean"),
        )
        .reset_index()
        .sort_values(["rate", "n"], ascending=[False, False])
    )

    # By LLM & Type
    by_llm_type = (
        df.groupby(["llm", "Type"])
        .agg(
            n=("metric_hit", "size"),
            hits=("metric_hit", "sum"),
            rate=("metric_hit", "mean"),
        )
        .reset_index()
        .sort_values(["llm", "rate", "n"], ascending=[True, False, False])
    )

    # Build output lines
    output_lines = []
    used_pair_keys = sorted(df["pair_key"].unique().tolist())
    output_lines.append("=== PAIR summary (by LLM) ===")
    output_lines.append(f"Found results.json files: {len(files)}")
    output_lines.append(f"Used PAIR keys (unique): {used_pair_keys}")
    if args.pair_key and args.pair_key not in used_pair_keys:
        output_lines.append(f"[WARN] You specified --pair-key={args.pair_key}, but it was not found in parsed results.")

    output_lines.append(f"Total samples with usable PAIR: {len(df)}")
    output_lines.append(f"Metric: {metric_name}")
    if missing_pair:
        output_lines.append(f"[WARN] Samples missing jailbroken/PAIR key skipped: {missing_pair}")
    if bad_pair_value:
        output_lines.append(f"[WARN] Samples with unparseable PAIR value skipped: {bad_pair_value}")
    if missing_type:
        output_lines.append(f"[WARN] Samples not found in type-csv (Type=UNKNOWN): {missing_type}")

    output_lines.append("\n--- Overall by LLM ---")
    output_lines.append(overall_by_llm.to_string(index=False))

    output_lines.append("\n--- By LLM & Type ---")
    for llm, sub in by_llm_type.groupby("llm", sort=False):
        output_lines.append(f"\n### LLM: {llm}")
        # keep columns in a nice order
        sub2 = sub[["Type", "n", "hits", "rate"]].copy()
        sub2 = sub2.sort_values(["rate", "n"], ascending=[False, False])
        output_lines.append(sub2.to_string(index=False))

    # Print to console
    for line in output_lines:
        print(line)

    # Save to txt file
    if args.save_txt:
        save_txt_dir = os.path.dirname(args.save_txt)
        if save_txt_dir:
            os.makedirs(save_txt_dir, exist_ok=True)
        with open(args.save_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\nSaved results to: {args.save_txt}")

    # Save CSV
    if args.save_csv:
        save_csv_dir = os.path.dirname(args.save_csv)
        if save_csv_dir:
            os.makedirs(save_csv_dir, exist_ok=True)
        by_llm_type.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved by-LLM by-type summary to: {args.save_csv}")


if __name__ == "__main__":
    main()
