"""Regenerate WAHARP_ANALYSIS_APPENDIX.md and the paper's figure data.

Reads only committed files under ``artifacts/`` and writes:

    artifacts/WAHARP_ANALYSIS_APPENDIX.md
    artifacts/paper_figures/*.csv

Run ``python scripts/make_analysis_appendix.py --check`` to verify that the
committed outputs are up to date without rewriting them.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "paper_figures"
APPENDIX = ART / "WAHARP_ANALYSIS_APPENDIX.md"

DATASETS = ("arizona", "crimes", "twitter")
CAPACITIES = (128, 256, 512)
WAHARP = "NewNeural_075_025"
ANGLES = [f"{j * 180 / 16:g}" for j in range(16)]
FRACTIONS = ("0.25", "0.375", "0.5", "0.625", "0.75")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def geomean(values) -> float:
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def name(ds: str) -> str:
    return ds.capitalize()


def build() -> dict[Path, str]:
    outputs: dict[Path, str] = {}
    md: list[str] = []
    summary = [r for r in read(ART / "final_evaluation" / "ALL_PAIRWISE_SUMMARY.csv") if r["left_method"] == WAHARP]
    cell = {(r["dataset"], int(r["capacity"]), r["right_method"]): r for r in summary}

    md.append("# WAHARP analysis appendix\n")
    md.append(
        "Companion to the WAHARP paper. Every table below is regenerated from committed files under "
        "`artifacts/` by `scripts/make_analysis_appendix.py`; the CSV data behind each paper figure is "
        "written to `artifacts/paper_figures/`. The evaluated model is the frozen checkpoint "
        "`artifacts/model/selected.pt` (0.75/0.25 loss weights, member 2). Ratios are WAHARP / baseline "
        "logical node accesses over 6,000 paired final queries; values below 1 favor WAHARP. Confidence "
        "intervals are pointwise paired-bootstrap 95% intervals over queries (2,000 draws) and describe "
        "fixed trees and workloads, not training-seed variability.\n")

    # Figure 2: aggregate ratios.
    fig2 = []
    rows = []
    for ds in DATASETS:
        for b in CAPACITIES:
            line = [name(ds), b]
            for m in ("PLATON", "STR", "TGS"):
                r = cell[(ds, b, m)]
                ratio, lo, hi = float(r["ratio_left_over_right"]), float(r["bootstrap_ci_low"]), float(r["bootstrap_ci_high"])
                fig2.append({"dataset": ds, "capacity": b, "baseline": m, "waharp_over_baseline": round(ratio, 6),
                             "ci_low": round(lo, 6), "ci_high": round(hi, 6),
                             "waharp_accesses": int(r["left_total_access"]), "baseline_accesses": int(r["right_total_access"])})
                line.append(f"{ratio:.6f} [{lo:.4f}, {hi:.4f}]")
            rows.append(line)
    gm = {m: geomean(float(cell[(ds, b, m)]["ratio_left_over_right"]) for ds in DATASETS for b in CAPACITIES)
          for m in ("PLATON", "STR", "TGS")}
    rows.append(["**Geometric mean**", ""] + [f"**{gm[m]:.6f}**" for m in ("PLATON", "STR", "TGS")])
    outputs[FIG / "fig2_aggregate_access_ratios.csv"] = csv_text(fig2)
    md.append("## 1. Aggregate node-access ratios (Section 7.2, Figure 2)\n")
    md.append(md_table(["Dataset", "B", "vs. PLATON [95% CI]", "vs. STR [95% CI]", "vs. TGS [95% CI]"], rows) + "\n")
    md.append(f"Geometric-mean change: {100 * (gm['PLATON'] - 1):+.2f}% vs. PLATON, "
              f"{100 * (gm['STR'] - 1):+.2f}% vs. STR, {100 * (gm['TGS'] - 1):+.2f}% vs. TGS. "
              "A CI that contains 1 means that comparison is inconclusive at the query level.\n")
    base_rows = []
    for m in ("STR", "TGS"):
        vals = [float(cell[(ds, b, m)]["right_total_access"]) / float(cell[(ds, b, "PLATON")]["right_total_access"])
                for ds in DATASETS for b in CAPACITIES]
        base_rows.append([f"{m} / PLATON", f"{geomean(vals):.6f}"])
    md.append(md_table(["Baseline ratio", "Geometric mean over nine cells"], base_rows) + "\n")

    # Capacity transfer.
    rows = []
    for b in CAPACITIES:
        rows.append([b] + [f"{geomean(float(cell[(ds, b, m)]['ratio_left_over_right']) for ds in DATASETS):.6f}"
                           for m in ("PLATON", "STR", "TGS")])
    md.append("## 2. Capacity transfer (Section 7.7)\n")
    md.append("The model was trained at B = 128 only; B = 256 and 512 reuse the same weights.\n")
    md.append(md_table(["B", "WAHARP/PLATON", "WAHARP/STR", "WAHARP/TGS"], rows) + "\n")

    # Figure 3: construction time.
    times = read(ART / "final_evaluation" / "CONSTRUCTION_TIMES.csv")
    outputs[FIG / "fig3_construction_times.csv"] = csv_text(times)
    rows = [[name(t["dataset"]), t["capacity"], f"{float(t['waharp_seconds']):,.3f}", f"{float(t['str_seconds']):,.3f}",
             f"{float(t['tgs_seconds']):,.3f}", f"{float(t['platon_mcts_seconds']):,.3f}",
             f"{float(t['platon_native_build_seconds']):,.3f}", f"{float(t['platon_seconds']):,.3f}",
             f"{int(t['platon_mcts_decisions']):,}", f"{100 * float(t['platon_mcts_share']):.1f}%",
             f"{float(t['platon_over_waharp']):.2f}x"] for t in times]
    speed = [float(t["platon_over_waharp"]) for t in times]
    share = [float(t["platon_mcts_share"]) for t in times]
    md.append("## 3. Construction time (Section 7.3, Figure 3)\n")
    md.append("Seconds of method-specific construction only: WAHARP = policy inference plus Python tree "
              "assembly; PLATON = summed MCTS decision time plus native bulk loading. Dataset loading, final "
              "queries, and idle resumable-run time are excluded. Because the records do not establish "
              "identical hardware, timing ratios retain their method-specific definition.\n")
    md.append(md_table(["Dataset", "B", "WAHARP", "STR", "TGS", "PLATON MCTS", "PLATON native", "PLATON total",
                        "MCTS decisions", "MCTS share", "PLATON/WAHARP"], rows) + "\n")
    md.append(f"PLATON/WAHARP: {min(speed):.2f}-{max(speed):.2f}x (arithmetic mean {sum(speed) / len(speed):.2f}x). "
              f"MCTS share of PLATON time: {100 * min(share):.1f}-{100 * max(share):.1f}%.\n")

    # PLATON budget sensitivity.
    budget = read(ART / "platon_budget" / "RESULTS.csv")
    btimes = {(r["dataset"], r["method"]): float(r["construction_seconds"])
              for r in read(ART / "platon_budget" / "CONSTRUCTION_TIMES.csv")}
    rows, fig_budget = [], []
    for r in budget:
        ds, k = r["dataset"], int(r["platon_rollouts"])
        pt, wt = btimes[(ds, f"PLATON-{k}")], btimes[(ds, "WAHARP")]
        fig_budget.append({"dataset": ds, "platon_rollouts": k, "waharp_over_platon": round(float(r["waharp_over_platon_ratio"]), 6),
                           "waharp_seconds": round(wt, 3), "platon_seconds": round(pt, 3), "platon_over_waharp_time": round(pt / wt, 2)})
        rows.append([name(ds), k, f"{int(r['waharp_logical_node_accesses']):,}", f"{int(r['platon_logical_node_accesses']):,}",
                     f"{float(r['waharp_over_platon_ratio']):.6f}", f"{wt:,.1f}", f"{pt:,.1f}", f"{pt / wt:.2f}x"])
    outputs[FIG / "platon_budget_quality_vs_time.csv"] = csv_text(fig_budget)
    md.append("### PLATON search-budget sensitivity (Section 7.3)\n")
    md.append("B = 128, 100 simulation steps, frozen WAHARP, and one fresh 6,000-query workload per dataset "
              "(`artifacts/platon_budget/`). Zero correctness mismatches.\n")
    md.append(md_table(["Dataset", "Rollouts", "WAHARP acc.", "PLATON acc.", "WAHARP/PLATON", "WAHARP s",
                        "PLATON s", "PLATON/WAHARP time"], rows) + "\n")

    # Figure 4: query families.
    fam = defaultdict(lambda: [0, 0])
    for r in read(ART / "final_evaluation" / "ALL_PAIRWISE_BY_WORKLOAD.csv"):
        if r["right_method"] == "PLATON":
            tot = fam[(r["dataset"], int(r["capacity"]), r["query_type"])]
            tot[0] += int(r["left_total_access"])
            tot[1] += int(r["right_total_access"])
    fig4, rows = [], []
    for ds in DATASETS:
        for qt in ("range", "point", "knn"):
            line = [name(ds), qt]
            for b in CAPACITIES:
                w, p = fam[(ds, b, qt)]
                fig4.append({"dataset": ds, "capacity": b, "query_family": qt, "waharp_accesses": w,
                             "platon_accesses": p, "waharp_over_platon": round(w / p, 6)})
                line.append(f"{w / p:.4f}")
            rows.append(line)
    outputs[FIG / "fig4_query_family_ratios.csv"] = csv_text(fig4)
    md.append("## 4. Query families (Section 7.4, Figure 4)\n")
    md.append("WAHARP/PLATON access ratio by query family (range 4,200, point 300, kNN 1,500 queries per cell).\n")
    md.append(md_table(["Dataset", "Family", "B=128", "B=256", "B=512"], rows) + "\n")

    # Figure 5: paired outcomes.
    fig5, rows = [], []
    for ds in DATASETS:
        for b in CAPACITIES:
            r = cell[(ds, b, "PLATON")]
            win, tie, loss = (100 * float(r[k]) for k in ("strict_win_rate", "tie_rate", "loss_rate"))
            fig5.append({"dataset": ds, "capacity": b, "waharp_fewer_pct": round(win, 2), "tie_pct": round(tie, 2),
                         "waharp_more_pct": round(loss, 2)})
            rows.append([name(ds), b, f"{win:.2f}%", f"{tie:.2f}%", f"{loss:.2f}%", r["correctness_mismatches"]])
    outputs[FIG / "fig5_paired_outcomes.csv"] = csv_text(fig5)
    md.append("## 5. Paired query outcomes vs. PLATON (Section 7.5, Figure 5)\n")
    md.append(md_table(["Dataset", "B", "WAHARP fewer", "Tie", "WAHARP more", "Mismatches"], rows) + "\n")

    # Comparison with previous checkpoint.
    rows = []
    for ds in DATASETS:
        for b in CAPACITIES:
            r = cell[(ds, b, "OldNeural_060_040")]
            rows.append([name(ds), b, f"{float(r['ratio_left_over_right']):.6f}",
                         f"[{float(r['bootstrap_ci_low']):.6f}, {float(r['bootstrap_ci_high']):.6f}]",
                         f"{int(r['left_total_access']):,}", f"{int(r['right_total_access']):,}"])
    old = [float(cell[(ds, b, "OldNeural_060_040")]["ratio_left_over_right"]) for ds in DATASETS for b in CAPACITIES]
    pooled = (sum(int(cell[(ds, b, "OldNeural_060_040")]["left_total_access"]) for ds in DATASETS for b in CAPACITIES)
              / sum(int(cell[(ds, b, "OldNeural_060_040")]["right_total_access"]) for ds in DATASETS for b in CAPACITIES))
    md.append("## 6. Comparison with the previous checkpoint (Section 7.6)\n")
    md.append("Selected 0.75/0.25 checkpoint vs. the earlier 0.60/0.40 checkpoint on the same locked queries. "
              "Hyperparameters were fixed before the locked test, so the selected checkpoint was retained "
              "regardless of these outcomes.\n")
    md.append(md_table(["Dataset", "B", "New/old", "95% CI", "New accesses", "Old accesses"], rows) + "\n")
    md.append(f"Cell geometric mean {geomean(old):.6f}; pooled ratio {pooled:.6f}.\n")

    # Figure 6: action usage, and Table 9: tree structure.
    fig6, rows_ang, rows_tree = [], [], []
    for ds in DATASETS:
        builds = {int(c["capacity"]): c for c in json.loads((ART / "construction" / f"{ds}_CONSTRUCTION_ALL.json").read_text())}
        for b in CAPACITIES:
            counts = {int(a): int(n) for a, n in builds[b]["action_counts"].items()}
            total = sum(counts.values())
            ang = [0] * 16
            frac = [0] * 5
            for a, n in counts.items():
                ang[a // 5] += n
                frac[a % 5] += n
            row = {"dataset": ds, "capacity": b, "decisions": total}
            row.update({f"angle_{ANGLES[j]}_count": ang[j] for j in range(16)})
            row.update({f"fraction_{FRACTIONS[k]}_count": frac[k] for k in range(5)})
            fig6.append(row)
            axis = 100 * (ang[0] + ang[8]) / total
            rows_ang.append([{"arizona": "AZ", "crimes": "CR", "twitter": "TW"}[ds] + f" {b}", f"{axis:.2f}%", f"{100 - axis:.2f}%"]
                            + [f"{100 * frac[k] / total:.1f}%" for k in range(5)])
            s = json.loads((ART / "tree_structure" / f"{ds}_B{b}_TREE_STRUCTURE.json").read_text())
            rows_tree.append([name(ds), b, s["height"], f"{s['leaf_nodes']:,}", f"{s['internal_nodes']:,}",
                              f"{builds[b]['neural_decision_count']:,}", len(counts), f"{s['mean_fill']:.6f}",
                              f"{s['mean_sibling_overlap']:.6f}"])
    outputs[FIG / "fig6_action_usage.csv"] = csv_text(fig6)
    md.append("## 7. Neural action usage (Section 7.8, Figure 6)\n")
    md.append("Share of recorded split decisions at object and parent levels. Action id a maps to angle "
              "index j = a // 5 (j * 11.25 degrees) and page fraction index k = a mod 5. Decision counts for "
              "all 16 angles and 5 fractions are in `paper_figures/fig6_action_usage.csv`.\n")
    md.append(md_table(["Setting", "Axis-aligned (0/90 deg)", "Other angles"] + [f"f={f}" for f in FRACTIONS], rows_ang) + "\n")
    md.append("## 8. Tree structure (Section 7.9, Table 9)\n")
    md.append(md_table(["Dataset", "B", "Height", "Leaves", "Internal", "Decisions", "Distinct actions", "Mean fill",
                        "Sibling overlap burden"], rows_tree) + "\n")

    # Table 10: LODO.
    lodo = read(ART / "generalization" / "LODO_RESULTS.csv")
    by_family = read(ART / "generalization" / "LODO_BY_QUERY_FAMILY.csv")
    md.append("## 9. Leave-one-dataset-out transfer (Section 7.10, Table 10)\n")
    rows = []
    for ref in ("PLATON", "WAHARP_ALL_DOMAIN", "STR", "TGS"):
        sel = [r for r in lodo if r["right_method"] == ref and r["query_family"] == "all"]
        cell_gm = geomean(float(r["access_ratio"]) for r in sel)
        pool = sum(int(r["left_total_accesses"]) for r in sel) / sum(int(r["right_total_accesses"]) for r in sel)
        wins = sum(float(r["access_ratio"]) < 1 for r in sel)
        rows.append([f"LODO / {ref}", f"{cell_gm:.4f}", f"{pool:.4f}", f"{wins} of {len(sel)}"])
    for qt in ("range", "point", "knn"):
        sel = [r for r in by_family if r["right_method"] == "PLATON" and r["query_family"] == qt]
        if sel:
            rows.append([f"LODO / PLATON, {qt}", f"{geomean(float(r['access_ratio']) for r in sel):.4f}", "", ""])
    md.append(md_table(["Comparison", "Cell geometric mean", "Pooled", "Cells below 1"], rows) + "\n")
    md.append("Per-cell ratios, intervals, and correctness are in `artifacts/generalization/`.\n")

    md.append("## Reproducing one cell\n")
    md.append("`scripts/reproduce_locked_cell.py` rebuilds any dataset-capacity cell from the frozen checkpoint "
              "and the committed queries in `artifacts/workloads/`, then compares total and per-workload "
              "accesses, policy decisions, action counts, and tree structure with these records.\n")
    outputs[APPENDIX] = "\n".join(md)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Fail if committed outputs are stale")
    args = parser.parse_args()
    outputs = build()
    stale = [p for p, text in outputs.items()
             if not p.is_file() or p.read_text(encoding="utf-8").replace("\r\n", "\n") != text]
    if args.check:
        if stale:
            sys.exit("Stale analysis outputs: " + ", ".join(str(p.relative_to(ROOT)) for p in stale))
        print(f"Analysis appendix and {len(outputs) - 1} figure-data files are up to date")
        return
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {APPENDIX.relative_to(ROOT)} and {len(outputs) - 1} files under {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
