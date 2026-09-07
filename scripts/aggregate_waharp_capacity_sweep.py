from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results" / "waharp_final_capacity_sweep_b128_b256_b512"
CHECKPOINT_SHA256 = "8709a7f9025292636f0526ccf7b702eda8ce0bce953c02cc1c05245bb13d7e1e"
METHODS = ("Neural", "STR", "TGS", "PLATON")
ROOTS = {
    ("arizona", 128): REPO / "results/kaggle_waharp_final_fullscale_arizona_b128_v1/WAHARP_FINAL_FULLSCALE_ARIZONA",
    ("arizona", 256): REPO / "results/kaggle_waharp_final_arizona_b256_v1/WAHARP_FINAL_FULLSCALE_ARIZONA_B256",
    ("arizona", 512): REPO / "results/kaggle_waharp_final_arizona_b512_v1/WAHARP_FINAL_FULLSCALE_ARIZONA_B512",
    ("crimes", 128): REPO / "results/kaggle_waharp_final_fullscale_crimes_b128_v1/WAHARP_FINAL_FULLSCALE_CRIMES",
    ("crimes", 256): REPO / "results/kaggle_waharp_final_crimes_b256_v1/WAHARP_FINAL_FULLSCALE_CRIMES_B256",
    ("crimes", 512): REPO / "results/kaggle_waharp_final_crimes_b512_v1/WAHARP_FINAL_FULLSCALE_CRIMES_B512",
    ("twitter", 128): REPO / "results/kaggle_waharp_final_fullscale_twitter_b128_v1_complete/WAHARP_FINAL_FULLSCALE_TWITTER",
    ("twitter", 256): REPO / "results/kaggle_waharp_final_twitter_b256_v1/WAHARP_FINAL_FULLSCALE_TWITTER_B256",
    ("twitter", 512): REPO / "results/kaggle_waharp_final_twitter_b512_v1/WAHARP_FINAL_FULLSCALE_TWITTER_B512",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mismatch_count(row: dict[str, str]) -> int:
    if "correctness_mismatches" in row:
        return int(row["correctness_mismatches"])
    return sum(int(row.get(key, 0)) for key in (
        "range_mismatches", "point_mismatches", "knn_distance_mismatches"
    ))


def build_seconds(meta: dict) -> float:
    return float(meta.get("total_construction_seconds", meta.get("build_seconds")))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    workload_rows: list[dict] = []
    query_hashes: dict[str, dict] = {}

    for (dataset, capacity), root in sorted(ROOTS.items()):
        state = json.loads((root / "FULLSCALE_STATE.json").read_text(encoding="utf-8"))
        if not str(state["status"]).startswith("COMPLETE"):
            raise RuntimeError(f"incomplete run: {dataset} B={capacity}: {state['status']}")
        if int(state["capacity"]) != capacity:
            raise RuntimeError(f"capacity mismatch: {dataset} B={capacity}")
        if set(state["completed_methods"]) != set(METHODS):
            raise RuntimeError(f"method set mismatch: {dataset} B={capacity}")
        if state["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise RuntimeError(f"checkpoint drift: {dataset} B={capacity}")

        constructions = {
            method: json.loads((root / "methods" / method / "CONSTRUCTION.json").read_text(encoding="utf-8"))
            for method in METHODS
        }
        neural = constructions["Neural"]
        if neural.get("fallbacks_used") is not False or neural.get("training_performed") is not False:
            raise RuntimeError(f"neural protocol violation: {dataset} B={capacity}")
        if not neural["validation"]["passed"]:
            raise RuntimeError(f"invalid neural tree: {dataset} B={capacity}")
        platon_seconds = build_seconds(constructions["PLATON"])

        pairwise = read_csv(root / "PAIRWISE_VS_PLATON.csv")
        if {row["left_method"] for row in pairwise} != {"Neural", "STR", "TGS"}:
            raise RuntimeError(f"pairwise method mismatch: {dataset} B={capacity}")
        for row in pairwise:
            if int(row["queries"]) != 6000 or mismatch_count(row):
                raise RuntimeError(f"query gate failed: {dataset} B={capacity} {row['left_method']}")
            method = row["left_method"]
            seconds = build_seconds(constructions[method])
            summary.append({
                "dataset": dataset,
                "objects": int(neural["objects"]),
                "capacity": capacity,
                "method": method,
                "queries": 6000,
                "method_total_access": int(row["left_total_access"]),
                "platon_total_access": int(row["right_total_access"]),
                "ratio_method_over_platon": float(row["ratio_left_over_right"]),
                "strict_win_rate": float(row["strict_win_rate"]),
                "tie_rate": float(row["tie_rate"]),
                "loss_rate": float(row["loss_rate"]),
                "method_build_seconds": seconds,
                "platon_build_seconds": platon_seconds,
                "platon_over_method_build_speedup": platon_seconds / seconds,
                "correctness_mismatches": 0,
                "checkpoint_sha256": CHECKPOINT_SHA256 if method == "Neural" else "",
                "fallbacks_used": False if method == "Neural" else "",
                "training_capacity": 128 if method == "Neural" else "",
            })

        for row in read_csv(root / "PAIRWISE_BY_WORKLOAD.csv"):
            workload_rows.append({"capacity": capacity, **row})

        hashes = {
            path.name: sha256(path)
            for path in sorted((root / "workload").glob("final_*.npy"))
        }
        query_hashes.setdefault(dataset, {})[str(capacity)] = hashes

    for dataset, capacities in query_hashes.items():
        reference = capacities["128"]
        for capacity, hashes in capacities.items():
            if hashes != reference:
                raise RuntimeError(f"paired final-query hash mismatch: {dataset} B={capacity}")

    write_csv(OUTPUT / "CAPACITY_SWEEP_SUMMARY.csv", summary)
    write_csv(OUTPUT / "CAPACITY_SWEEP_BY_WORKLOAD.csv", workload_rows)
    (OUTPUT / "PAIRED_QUERY_HASHES.json").write_text(
        json.dumps(query_hashes, indent=2, sort_keys=True), encoding="utf-8"
    )

    neural_rows = [row for row in summary if row["method"] == "Neural"]
    ratios = [float(row["ratio_method_over_platon"]) for row in neural_rows]
    geom_ratio = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    str_rows = {(row["dataset"], row["capacity"]): row for row in summary if row["method"] == "STR"}
    tgs_rows = {(row["dataset"], row["capacity"]): row for row in summary if row["method"] == "TGS"}
    neural_beats_str = sum(
        row["method_total_access"] < str_rows[(row["dataset"], row["capacity"])]["method_total_access"]
        for row in neural_rows
    )
    neural_beats_tgs = sum(
        row["method_total_access"] < tgs_rows[(row["dataset"], row["capacity"])]["method_total_access"]
        for row in neural_rows
    )

    lines = [
        "# WAHARP Frozen Capacity Sweep: B=128/256/512",
        "",
        "The same B=128-trained frozen member-2 checkpoint is evaluated without retraining or deterministic fallback.",
        "For each dataset, all capacities use byte-identical final query arrays. Every run contains 6,000 queries and passed all correctness gates.",
        "",
        "| Dataset | B | Neural access | PLATON access | Neural/PLATON | Neural build (s) | PLATON build (s) | Build speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in neural_rows:
        lines.append(
            f"| {row['dataset'].title()} | {row['capacity']} | {row['method_total_access']:,} | "
            f"{row['platon_total_access']:,} | {row['ratio_method_over_platon']:.6f} | "
            f"{row['method_build_seconds']:.1f} | {row['platon_build_seconds']:.1f} | "
            f"{row['platon_over_method_build_speedup']:.1f}x |"
        )
    lines += [
        "",
        f"Geometric-mean Neural/PLATON ratio across nine settings: **{geom_ratio:.6f}**.",
        f"Neural beats STR in **{neural_beats_str}/9** settings and TGS in **{neural_beats_tgs}/9** settings.",
        "Neural does not beat PLATON in aggregate node access in any of the nine settings.",
        "The results support strong construction-time efficiency and broad baseline robustness, but not a universal PLATON node-access win.",
        "Cross-capacity degradation is expected because the policy was trained only at B=128; B=256/512 are frozen OOD capacity tests.",
    ]
    (OUTPUT / "SCIENTIFIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "runs": len(neural_rows),
        "geometric_mean_neural_over_platon": geom_ratio,
        "neural_beats_str": neural_beats_str,
        "neural_beats_tgs": neural_beats_tgs,
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
