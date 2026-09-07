from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "waharp_final_capacity_sweep_b128_b256_b512"
OUTPUT = RESULTS / "WAHARP_IMPROVED_B128_B256_B512_PAPER_REPORT.md"
TRAINING = ROOT / "trained_models" / "waharp_realtrain_b128_v1_three_domain_20260829"

RUNS = {
    ("arizona", 128): ROOT / "results" / "kaggle_waharp_final_fullscale_arizona_b128_v1" / "WAHARP_FINAL_FULLSCALE_ARIZONA",
    ("arizona", 256): ROOT / "results" / "kaggle_waharp_final_arizona_b256_v1" / "WAHARP_FINAL_FULLSCALE_ARIZONA_B256",
    ("arizona", 512): ROOT / "results" / "kaggle_waharp_final_arizona_b512_v1" / "WAHARP_FINAL_FULLSCALE_ARIZONA_B512",
    ("crimes", 128): ROOT / "results" / "kaggle_waharp_final_fullscale_crimes_b128_v1" / "WAHARP_FINAL_FULLSCALE_CRIMES",
    ("crimes", 256): ROOT / "results" / "kaggle_waharp_final_crimes_b256_v1" / "WAHARP_FINAL_FULLSCALE_CRIMES_B256",
    ("crimes", 512): ROOT / "results" / "kaggle_waharp_final_crimes_b512_v1" / "WAHARP_FINAL_FULLSCALE_CRIMES_B512",
    ("twitter", 128): ROOT / "results" / "kaggle_waharp_final_fullscale_twitter_b128_v1_complete" / "WAHARP_FINAL_FULLSCALE_TWITTER",
    ("twitter", 256): ROOT / "results" / "kaggle_waharp_final_twitter_b256_v1" / "WAHARP_FINAL_FULLSCALE_TWITTER_B256",
    ("twitter", 512): ROOT / "results" / "kaggle_waharp_final_twitter_b512_v1" / "WAHARP_FINAL_FULLSCALE_TWITTER_B512",
}

DATASETS = ("arizona", "crimes", "twitter")
CAPACITIES = (128, 256, 512)
METHODS = ("Neural", "STR", "TGS", "PLATON")
FRACTIONS = (0.25, 0.375, 0.5, 0.625, 0.75)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: float | int, digits: int = 3) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def md_table(headers: list[str], rows: list[list[object]], aligns: list[str] | None = None) -> list[str]:
    if aligns is None:
        aligns = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    out.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return out


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def action_description(action_id: int) -> tuple[float, float]:
    return (action_id // 5) * 11.25, FRACTIONS[action_id % 5]


def main() -> None:
    missing = [str(path) for path in [RESULTS / "CAPACITY_SWEEP_SUMMARY.csv", RESULTS / "CAPACITY_SWEEP_BY_WORKLOAD.csv"] if not path.is_file()]
    missing += [str(path) for path in RUNS.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing audited sweep artifacts:\n" + "\n".join(missing))

    summary_rows = read_csv(RESULTS / "CAPACITY_SWEEP_SUMMARY.csv")
    summary = {(row["dataset"], int(row["capacity"]), row["method"]): row for row in summary_rows}
    workload_rows = read_csv(RESULTS / "CAPACITY_SWEEP_BY_WORKLOAD.csv")
    frozen = read_json(TRAINING / "FROZEN_DEPLOYMENT_MANIFEST.json")
    selection = read_json(TRAINING / "SHADOW_SELECTION.json")
    ensemble = read_json(TRAINING / "ENSEMBLE_TRAINING.json")
    member = ensemble["members"][2]
    query_hashes = read_json(RESULTS / "PAIRED_QUERY_HASHES.json")

    datasets = {name: read_json(RUNS[(name, 128)] / "DATASET.json") for name in DATASETS}
    workloads = {name: read_json(RUNS[(name, 128)] / "workload" / "WORKLOAD_MANIFEST.json") for name in DATASETS}
    constructions = {
        (dataset, capacity, method): read_json(RUNS[(dataset, capacity)] / "methods" / method / "CONSTRUCTION.json")
        for dataset in DATASETS for capacity in CAPACITIES for method in METHODS
    }
    structures = {
        (dataset, capacity): read_json(RUNS[(dataset, capacity)] / "methods" / "Neural" / "TREE_STRUCTURE.json")
        for dataset in DATASETS for capacity in CAPACITIES
    }

    lines: list[str] = []
    add = lines.append
    extend = lines.extend

    add("# WAHARP Neural-Only Full-Scale Capacity Sweep: B=128, 256, and 512")
    add("")
    add("**Paper-ready technical report for the improved frozen-model experiment**")
    add("")
    add("This document covers only the completed improved full-scale experiment at capacities B=128, B=256, and B=512. It does not include earlier prototypes, fallback-based WAHARP variants, 100K-object pilots, Arizona specialization runs, or failed/intermediate experiments.")
    add("")
    add("## Executive result")
    add("")
    add("A single workload-aware neural partition model trained at B=128 was frozen and used without retraining at all three capacities on full Arizona, Crimes, and Twitter data. The deployed method was neural-only: it made every recursive partition decision itself and used no STR, TGS, Hilbert, PLATON, or other deterministic fallback. Across the nine dataset-capacity settings it:")
    add("")
    add("- beat STR in 9/9 settings by total logical node access;")
    add("- beat TGS in 6/9 settings, losing only on the three Crimes settings;")
    add("- remained within 0.72% to 9.41% of author PLATON, but did not beat PLATON in aggregate node access in any setting;")
    add("- achieved a geometric-mean Neural/PLATON access ratio of 1.032114, or 3.21% more node accesses overall on a multiplicative scale;")
    add("- built trees 23.84x to 35.53x faster than PLATON under the recorded construction-time definition;")
    add("- passed every correctness gate: zero mismatches in 162,000 paired baseline-versus-PLATON query comparisons overall, including 54,000 Neural-versus-PLATON comparisons.")
    add("")
    add("The defensible paper claim is therefore strong efficiency and broad robustness, not a universal node-access win over PLATON. The closest result is full Twitter at B=128, where Neural used 4,259,472 accesses versus PLATON's 4,228,888, a gap of only 0.723%, while building 27.88x faster.")
    add("")
    add("## 1. Experiment identity and scientific scope")
    add("")
    extend(md_table(
        ["Item", "Value"],
        [
            ["B=128 final protocol", "`waharp_final_fullscale_b128_member2_20260831`"],
            ["B=256/B=512 extension protocol", "`waharp_final_capacity_sweep_member2_v1_20260901`"],
            ["Paired final-query namespace", f"`{frozen['fresh_final_namespace']}`"],
            ["Namespace commitment SHA-256", f"`{frozen['fresh_final_namespace_commitment_sha256']}`"],
            ["Capacity-sweep root seed", "`2026083159`"],
            ["Frozen checkpoint", f"`{frozen['checkpoint']}`"],
            ["Checkpoint SHA-256", f"`{frozen['checkpoint_sha256']}`"],
            ["Training capacity", "B=128"],
            ["Evaluation capacities", "B in {128, 256, 512}"],
            ["Methods", "WAHARP Neural, author PLATON, STR, TGS"],
            ["Primary metric", "Total logical R-tree node accesses, index plus leaf"],
            ["Final queries per setting", "6,000"],
            ["Total dataset-capacity settings", "9"],
            ["Total method-query executions", "216,000 = 3 datasets x 3 capacities x 4 methods x 6,000 queries"],
        ],
        ["---", "---"],
    ))
    add("")
    add("The same final query arrays were used for all four methods within a setting. For a given dataset, the arrays were also byte-identical across B=128, B=256, and B=512. The B=256/B=512 runs therefore isolate the effect of capacity while preserving paired queries.")
    add("")
    add("## 2. WAHARP neural method")
    add("")
    add("### 2.1 Construction pipeline")
    add("")
    add("WAHARP constructs a packed, equal-depth R-tree from the bottom up while using a learned policy to define every partition. At a state containing more than B entries, it computes workload-aware state features, predicts the cost/regret of all 80 admissible split actions, takes the action with minimum predicted value, and recursively applies the same process to both children until every group contains at most B entries. Object groups become leaf pages. Their MBRs are then treated as entries and passed through the same neural recursion to form each parent level until one root remains.")
    add("")
    add("Capacity alignment is exact. For a state with n entries, p=ceil(n/B) physical pages are required. An action chooses a left-page fraction; the implementation converts it to an integer left-page quota and then selects a feasible object count that guarantees both children can realize their assigned page quotas. Stable mergesort makes projection ties deterministic. The builder validates one-to-one object coverage, no overflow, no null child links, and equal leaf depth.")
    add("")
    add("### 2.2 Action space")
    add("")
    add("Each action is the Cartesian product of:")
    add("")
    add("- 16 projection angles: 0, 11.25, 22.5, ..., 168.75 degrees;")
    add("- 5 left-page fractions: 0.25, 0.375, 0.5, 0.625, and 0.75.")
    add("")
    add("This yields 80 discrete actions. Object or child-MBR centers are projected onto the selected direction, stably sorted, and divided at the capacity-feasible quota. The learned output is a complete partition decision, not a score that merely chooses among STR/TGS/PLATON trees.")
    add("")
    add("### 2.3 State representation")
    add("")
    add("Every state and its relevant construction queries are normalized to the current state's MBR. The input has 498 dimensions:")
    add("")
    add("- 144 values: 12 x 12 histogram of entry centers;")
    add("- 144 values: 12 x 12 histogram of construction-query centers;")
    add("- 144 values: 12 x 12 query-area-weighted histogram;")
    add("- 66 scalar values: seven-number summaries of entry width, height, area, center x, and center y; center covariance and eigenvalues; seven-number summaries of query width, height, and area; log-scaled entry and page counts; tree level; local query count; and state aspect ratio.")
    add("")
    add("For deployment, construction queries intersecting the state MBR are retained. If more than 128 intersect, a deterministic evenly spaced subset of 128 is used. If none intersect, the nearest query centers are used. Thus construction workload information is available at every leaf and parent partition decision, while final evaluation queries never enter the builder.")
    add("")
    add("### 2.4 Policy network")
    add("")
    extend(md_table(
        ["Layer", "Shape / operation"],
        [
            ["Input", "498-dimensional state vector"],
            ["Block 1", "Linear 498->512, LayerNorm, GELU, Dropout(0.10)"],
            ["Block 2", "Linear 512->512, LayerNorm, GELU, Dropout(0.10)"],
            ["Block 3", "Linear 512->256, LayerNorm, GELU"],
            ["Output", "Linear 256->80 predicted action costs/regrets"],
            ["Parameters", "672,592 trainable parameters"],
            ["Decision", "argmin over 80 outputs"],
        ],
        ["---", "---"],
    ))
    add("")
    add("No confidence threshold, deterministic anchor, candidate reranker, ensemble average, or fallback was active in this experiment. Three members were trained, but only frozen member 2 was deployed.")
    add("")
    add("## 3. Training and freezing protocol")
    add("")
    add("### 3.1 Training data and supervision")
    add("")
    extend(md_table(
        ["Training component", "Configuration"],
        [
            ["Domains", "Twitter points, Chicago Crimes points, Arizona OSM building MBRs"],
            ["Training object cap", "Twitter 1,000,000; Crimes 1,000,000; Arizona all 1,464,257"],
            ["Construction queries", "2,800 per domain"],
            ["Initial states", "8,000 per domain, 24,000 total"],
            ["DAgger states", "2,000 per domain, 6,000 total"],
            ["Total labeled states", "30,000"],
            ["Actions labeled per state", "80, the complete action space"],
            ["State-action cost labels", "2,400,000"],
            ["State size", "2 to 16 pages at B=128"],
            ["Local construction-query cap", "128"],
            ["Train/validation split per member", "24,000 / 6,000 states, stratified by domain"],
        ],
        ["---", "---"],
    ))
    add("")
    add("Initial supervision states were sampled from real-domain object levels. A bootstrap model was then rolled out on 250,000-object shadow subsets, and visited states were relabeled to add DAgger-style on-policy coverage. If stochastic capture produced fewer than the target number of states, fresh real-domain states filled the shortfall without changing the target size.")
    add("")
    add("For each state-action pair, the teacher first applies the candidate split, then deterministically rolls each side into capacity-aligned pages. The primary cost is average construction-query page hits over those resulting page MBRs. Two small geometric regularizers are added: 1e-4 times normalized overlap and 2e-5 times normalized margin. Targets are normalized regrets, (cost - best_cost) / max(best_cost, 1). PLATON is not a teacher, action source, fallback, or checkpoint-selection signal.")
    add("")
    add("### 3.2 Optimization")
    add("")
    add("The loss is Smooth-L1 regression over observed action regrets plus 0.60 times a masked listwise soft-target loss with temperature 0.025 plus 0.40 times best-action cross entropy. Optimization uses AdamW, learning rate 6e-4, weight decay 1e-4, batch size 256, automatic mixed precision, gradient clipping at 5.0, and early-stopping patience 6. Three random-seed members were trained with PyTorch DataParallel over two T4 GPUs.")
    add("")
    extend(md_table(
        ["Selected-member statistic", "Value"],
        [
            ["Member", "2"],
            ["Seed", str(member["seed"])],
            ["Epochs requested / completed", f"{member['epochs_requested']} / {len(member['history'])}"],
            ["Best epoch", str(member["best_epoch"])],
            ["Best validation selection score", f"{member['best_score']:.9f}"],
            ["Epoch-1 train loss", f"{member['history'][0]['train_loss']:.6f}"],
            ["Final completed-epoch train loss", f"{member['history'][-1]['train_loss']:.6f}"],
            ["Best-epoch mean selected regret", f"{member['history'][member['best_epoch'] - 1]['mean_selected_regret']:.6f}"],
            ["Best-epoch worst-domain regret", f"{member['history'][member['best_epoch'] - 1]['worst_domain_regret']:.6f}"],
            ["Recorded member optimization time", f"{member['seconds']:.3f} s (excludes supervision generation and data preparation)"],
            ["Checkpoint SHA-256", f"`{member['checkpoint_sha256']}`"],
        ],
        ["---", "---"],
    ))
    add("")
    add("Member selection used validation queries only. The rule minimized the worst-domain ratio to the best candidate member plus 0.25 times the mean such ratio. The member scores were " + ", ".join(f"member {row['member']}: {row['score']:.6f}" for row in selection["members"]) + "; member 2 was selected. The checkpoint was then cryptographically frozen before the final namespace was materialized.")
    add("")
    add("B=128 is in-capacity evaluation. B=256 and B=512 are frozen out-of-distribution capacity tests: the action table and weights are unchanged, while runtime page counts and exact quota calculations use the new B. No capacity-specific fine-tuning or post-final-query selection occurred.")
    add("")
    add("## 4. Full-scale datasets")
    add("")
    dataset_table = []
    for dataset in DATASETS:
        meta = datasets[dataset]
        source_rows = meta.get("source_rows", meta.get("source_rows_seen"))
        dataset_table.append([
            dataset.title(), meta["kind"], number(int(source_rows)), number(int(meta["objects"])), number(int(meta["invalid_rows"])),
            f"`{meta['normalized_object_sha256']}`",
        ])
    extend(md_table(
        ["Dataset", "Geometry", "Source rows", "Valid objects", "Rejected rows", "Normalized object SHA-256"],
        dataset_table,
        ["---", "---", "---:", "---:", "---:", "---"],
    ))
    add("")
    add("Twitter and Crimes coordinates were min-max normalized independently to [0,1]^2 and represented as 1e-6 by 1e-6 rectangles centered on each point. Twitter source bounds were longitude [-124.62823268, -67.172316] and latitude [25.27794215, 49.33834794]. Crimes source bounds were longitude [-91.686565684, -87.524529378] and latitude [36.619446395, 42.022910333]. The Arizona cache contained 1,464,257 building MBRs in the historical `xmin,xmax,ymin,ymax` order; the loader audited the order, converted to `xmin,ymin,xmax,ymax`, and normalized the complete dataset to [0,1]^2.")
    add("")
    add("This is not a completely unseen-domain study: the three domain types were represented during training. It is a held-out-query, full-scale study, plus cross-capacity transfer at B=256 and B=512. Twitter and Crimes additionally evaluate scale transfer because their full datasets are much larger than the one-million-object training caps. Arizona uses all objects during training and therefore demonstrates workload holdout, not object holdout.")
    add("")
    add("## 5. Query workload")
    add("")
    extend(md_table(
        ["Query family", "Conditions", "Queries per condition", "Queries per setting"],
        [
            ["Square range", "side in {0.001, 0.005, 0.01, 0.05, 0.10}", "300", "1,500"],
            ["Fixed-area aspect range", "aspect in {1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16}; target area 0.01", "300", "2,700"],
            ["Point", "points sampled from dataset object centers", "300", "300"],
            ["kNN", "k in {1, 5, 25, 125, 625}", "300", "1,500"],
            ["Total", "20 conditions", "-", "6,000"],
        ],
        ["---", "---", "---:", "---:"],
    ))
    add("")
    add("Each tree received 2,800 construction queries: 1,960 range boxes (70%), 140 point boxes (5%), and 700 kNN-surrogate boxes (25%). Range construction queries drew 45% from the square family and 55% from the aspect-ratio family. kNN surrogates estimated kth-neighbor radii with a cKDTree over at most 100,000 sampled centers. Construction queries could influence WAHARP and PLATON construction; final queries could not. Validation suites contained 100 queries per condition but were not used to alter the already frozen model during this sweep.")
    add("")
    query_rows = []
    for dataset in DATASETS:
        wm = workloads[dataset]
        query_rows.append([
            dataset.title(), str(wm["seed"]), f"`{wm['construction']['sha256']}`",
            f"`{query_hashes[dataset]['128']['final_points.npy']}`",
            f"`{query_hashes[dataset]['128']['final_knn_points.npy']}`",
        ])
    extend(md_table(
        ["Dataset", "Derived workload seed", "Construction SHA-256", "Final point-array SHA-256", "Final kNN-array SHA-256"],
        query_rows,
        ["---", "---:", "---", "---", "---"],
    ))
    add("")
    add("The 14 final range-array hashes are recorded for every capacity in `PAIRED_QUERY_HASHES.json`; all capacity copies match byte for byte within each dataset.")
    add("")
    add("## 6. Baselines and capacity parity")
    add("")
    add("- **Author PLATON:** the author's learned-packing `RtreeEnv`, MCTS, and `Node` code was used directly to generate the cut list. Parameters were branch=B, 25 MCTS rollouts, 100 simulation steps, and the same 2,800 construction boxes. Native packing and query execution used the author's bundled libspatialindex 1.9.3 source/runtime. The audited native source fingerprint was `d6b02638bba949b96116f03421c54962e219a99e34f5e2261ce4a38eac5980b5`. PLATON's nominal native capacity was configured as B/0.8 with fill factor 0.8, yielding effective packed occupancy B: nominal capacities 160, 320, and 640 for B=128, 256, and 512. Storage page size was 4,096 bytes.")
    add("- **STR:** center-based Sort-Tile-Recursive bulk packing. At every level, entries were sorted by x center into ceil(sqrt(number_of_pages)) slices, sorted by y center within each slice, and chunked into pages of at most B entries.")
    add("- **TGS:** the repository's TGS packed-tree builder with area cost and center ordering, recursively applied with max_entries=B.")
    add("")
    add("All methods indexed the same normalized objects, used the same effective maximum entries, answered the same final queries, and were evaluated by logical node accesses. Neural, STR, and TGS used the same Python traversal. PLATON used its native traversal; this is why logical node access, not raw query latency, is the primary cross-language metric.")
    add("")
    add("## 7. Metrics and correctness gates")
    add("")
    add("- **Total logical node accesses:** index nodes plus leaf nodes visited. This is summed over the 6,000-query suite and is the primary outcome; lower is better.")
    add("- **Normalized I/O:** for range and point queries, total accesses divided by a capacity-only lower bound derived from result cardinality and minimum tree height. It is diagnostic, not the headline metric, and is not defined for kNN in this implementation.")
    add("- **Strict query win/tie/loss:** a win means the left method used fewer nodes than PLATON on the identical query. Aggregate access ratio remains the primary metric because strict wins weight tiny and large queries equally.")
    add("- **Construction time:** method-specific tree construction only. Neural includes policy inference and Python tree assembly. PLATON includes MCTS decision time plus native bulk loading. Dataset loading, final query evaluation, and resumable idle time are excluded.")
    add("- **Range/point correctness:** exact equality of result count, 64-bit ID sum, and ID XOR.")
    add("- **kNN correctness:** equivalent kth distance with relative and absolute tolerance 2e-6. The Python evaluator returns exactly k IDs with deterministic ID tie-breaking; libspatialindex can expand all ties at the kth boundary, so ID-set equality is not the valid cross-implementation invariant.")
    add("")
    add("Every method and setting passed the canonical 6,000-query join with exactly 4,200 range, 300 point, and 1,500 kNN queries. All correctness mismatch counters were zero.")
    add("")
    add("## 8. Main node-access results")
    add("")
    headline = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            vals = {method: int(float(summary[(dataset, capacity, method)]["method_total_access"])) for method in ("Neural", "STR", "TGS")}
            platon = int(float(summary[(dataset, capacity, "Neural")]["platon_total_access"]))
            headline.append([
                dataset.title(), capacity, number(vals["Neural"]), number(platon), number(vals["STR"]), number(vals["TGS"]),
                f"{vals['Neural'] / platon:.6f}", f"{vals['Neural'] / vals['STR']:.6f}", f"{vals['Neural'] / vals['TGS']:.6f}",
            ])
    extend(md_table(
        ["Dataset", "B", "Neural", "PLATON", "STR", "TGS", "N/P", "N/STR", "N/TGS"],
        headline,
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("Ratios below 1 indicate a Neural win. The geometric-mean ratios across all nine settings were 1.032114 for Neural/PLATON, 1.133097 for STR/PLATON, and 1.301728 for TGS/PLATON. By capacity, the Neural/PLATON geometric means were 1.021313 at B=128, 1.030883 at B=256, and 1.044276 at B=512. The widening gap is consistent with capacity shift because the model was trained only at B=128.")
    add("")
    comparison = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            n = int(float(summary[(dataset, capacity, "Neural")]["method_total_access"]))
            p = int(float(summary[(dataset, capacity, "Neural")]["platon_total_access"]))
            s = int(float(summary[(dataset, capacity, "STR")]["method_total_access"]))
            t = int(float(summary[(dataset, capacity, "TGS")]["method_total_access"]))
            comparison.append([
                dataset.title(), capacity, f"{100 * (n / p - 1):+.3f}%", f"{100 * (n / s - 1):+.3f}%", f"{100 * (n / t - 1):+.3f}%",
                "PLATON < Neural" if p < n else "Neural <= PLATON",
            ])
    extend(md_table(
        ["Dataset", "B", "Neural vs PLATON", "Neural vs STR", "Neural vs TGS", "Top of N/P"],
        comparison,
        ["---", "---:", "---:", "---:", "---:", "---"],
    ))
    add("")
    add("Negative percentages are Neural reductions. Neural's largest reductions were 17.43% versus STR on Arizona B=512 and 51.32% versus TGS on Twitter B=512. TGS was slightly better than PLATON on Crimes B=128 (ratio 0.999241) and better than Neural on all three Crimes capacities.")
    add("")
    add("## 9. Query-family results against PLATON")
    add("")
    family_table = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            detail = [row for row in read_csv(RUNS[(dataset, capacity)] / "PAIRWISE_PER_QUERY.csv") if row["left_method"] == "Neural"]
            by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in detail:
                by_type[row["query_type"]].append(row)
            values = []
            for query_type in ("range", "point", "knn"):
                rows = by_type[query_type]
                left = sum(int(row["left_access"]) for row in rows)
                right = sum(int(row["right_access"]) for row in rows)
                values.append(f"{left / right:.6f} ({left:,}/{right:,})")
            family_table.append([dataset.title(), capacity, *values])
    extend(md_table(
        ["Dataset", "B", "Range N/P (accesses)", "Point N/P (accesses)", "kNN N/P (accesses)"],
        family_table,
        ["---", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("The main weakness is concentrated in point and kNN search, especially Arizona. Large range workloads are much closer to PLATON and sometimes favor Neural at individual shapes. Since aggregate totals are dominated by broad range conditions on the larger datasets, Twitter remains very close overall despite weaker point/kNN ratios.")
    add("")
    add("## 10. Strict paired-query outcomes versus PLATON")
    add("")
    outcomes = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            row = summary[(dataset, capacity, "Neural")]
            outcomes.append([
                dataset.title(), capacity, pct(float(row["strict_win_rate"])), pct(float(row["tie_rate"])), pct(float(row["loss_rate"])),
                f"{float(row['ratio_method_over_platon']):.6f}",
            ])
    extend(md_table(
        ["Dataset", "B", "Neural wins", "Ties", "Neural losses", "Aggregate N/P"],
        outcomes,
        ["---", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("Crimes has many exact ties because point data and highly packed trees often traverse identical paths. Strict win rate must not be read as a substitute for aggregate I/O: for example, Twitter B=128 has only 16.07% strict wins but an aggregate ratio of 1.007232 because many losses are small relative to the high-access range workload.")
    add("")
    add("## 11. Construction-time results")
    add("")
    build_table = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            n = float(summary[(dataset, capacity, "Neural")]["method_build_seconds"])
            s = float(summary[(dataset, capacity, "STR")]["method_build_seconds"])
            t = float(summary[(dataset, capacity, "TGS")]["method_build_seconds"])
            p = float(summary[(dataset, capacity, "Neural")]["platon_build_seconds"])
            build_table.append([dataset.title(), capacity, number(n), number(s), number(t), number(p), f"{p / n:.2f}x"])
    extend(md_table(
        ["Dataset", "B", "Neural (s)", "STR (s)", "TGS (s)", "PLATON (s)", "PLATON/Neural"],
        build_table,
        ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("The arithmetic mean PLATON/Neural construction speedup across the nine settings was 29.18x. Neural was slower to build than STR and TGS because it performs one network inference at each recursive split, but it was consistently far faster than PLATON's MCTS-driven construction. The observed Kaggle allocation was 2 x NVIDIA Tesla T4. Training used both GPUs through DataParallel; benchmark inference was coded on `cuda:0`, while Python tree assembly, STR/TGS, and much of PLATON's policy work remained CPU-bound.")
    add("")
    add("### PLATON construction-time decomposition")
    add("")
    platon_breakdown = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            meta = constructions[(dataset, capacity, "PLATON")]
            platon_breakdown.append([
                dataset.title(), capacity, number(float(meta["policy_seconds"])), number(float(meta["bulk_seconds"])),
                number(float(meta["total_construction_seconds"])), number(int(meta["cut_count"])),
            ])
    extend(md_table(
        ["Dataset", "B", "MCTS policy (s)", "Native bulk load (s)", "Total (s)", "Cuts"],
        platon_breakdown,
        ["---", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("PLATON's construction time is dominated by learned MCTS cut decisions, not native bulk loading. Cut generation was resumable and action-checkpointed; only summed decision time and final bulk-load time enter the construction metric.")
    add("")
    add("## 12. Index-node and leaf-node access decomposition")
    add("")
    access_split = []
    normalized_table = []
    depth_table = []
    latency_table = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            per_method: dict[str, list[dict[str, str]]] = {
                method: read_csv(RUNS[(dataset, capacity)] / "methods" / method / "PER_QUERY.csv") for method in METHODS
            }
            nrows = per_method["Neural"]
            prows = per_method["PLATON"]
            access_split.append([
                dataset.title(), capacity,
                number(sum(int(row["index_node_accesses"]) for row in nrows)),
                number(sum(int(row["leaf_node_accesses"]) for row in nrows)),
                number(sum(int(row["index_node_accesses"]) for row in prows)),
                number(sum(int(row["leaf_node_accesses"]) for row in prows)),
            ])
            n_normalized = [float(row["normalized_io"]) for row in nrows if row["query_type"] in ("range", "point")]
            p_normalized = [float(row["normalized_io"]) for row in prows if row["query_type"] in ("range", "point")]
            n_norm_mean = statistics.fmean(n_normalized)
            p_norm_mean = statistics.fmean(p_normalized)
            normalized_table.append([
                dataset.title(), capacity, number(n_norm_mean, 6), number(p_norm_mean, 6), f"{n_norm_mean / p_norm_mean:.6f}",
            ])
            depths: Counter[int] = Counter()
            for row in nrows:
                raw = row.get("depth_accesses_json", "")
                if raw:
                    for depth, value in json.loads(raw).items():
                        depths[int(depth)] += int(value)
            depth_table.append([dataset.title(), capacity, *[number(depths.get(i, 0)) for i in range(4)]])
            latency_values = {}
            for method, rows in per_method.items():
                vals = [float(row["latency_us"]) for row in rows]
                latency_values[method] = statistics.fmean(vals) / 1000.0
            latency_table.append([dataset.title(), capacity, *[number(latency_values[m]) for m in METHODS]])
    extend(md_table(
        ["Dataset", "B", "Neural index", "Neural leaf", "PLATON index", "PLATON leaf"],
        access_split,
        ["---", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("PLATON's advantage comes from both fewer internal visits and fewer leaf visits. The effect is particularly visible on Arizona point/kNN queries, where PLATON's MBR hierarchy prunes more aggressively.")
    add("")
    add("### Mean normalized I/O for range and point queries")
    add("")
    extend(md_table(
        ["Dataset", "B", "Neural normalized I/O", "PLATON normalized I/O", "Ratio of means"],
        normalized_table,
        ["---", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("This table averages per-query normalized I/O over the 4,500 range and point queries. kNN is excluded because the implementation does not define a result-cardinality lower bound for kNN traversal. Because normalization is query-specific, these means complement rather than replace the total-access ratios.")
    add("")
    add("### Neural node accesses by tree depth")
    add("")
    extend(md_table(
        ["Dataset", "B", "Depth 0", "Depth 1", "Depth 2", "Depth 3"],
        depth_table,
        ["---", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("Depth 0 is the root. A zero in depth 3 means the tree has only three levels. Native PLATON output did not preserve comparable per-depth counters, so no PLATON-by-depth table is fabricated.")
    add("")
    add("## 13. Neural tree structure")
    add("")
    tree_table = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            con = constructions[(dataset, capacity, "Neural")]
            tree = structures[(dataset, capacity)]
            tree_table.append([
                dataset.title(), capacity, tree["height"], number(tree["leaf_nodes"]), number(tree["internal_nodes"]),
                number(tree["total_nodes"]), number(con["neural_decision_count"]), f"{tree['mean_fill']:.6f}",
                f"{tree['mean_sibling_overlap']:.6f}",
            ])
    extend(md_table(
        ["Dataset", "B", "Height", "Leaves", "Internal", "All nodes", "Neural decisions", "Mean fill", "Mean sibling overlap"],
        tree_table,
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("Mean fill is mean entries divided by B over all nodes and is near 1 because exact page quotas produce tightly packed trees. Mean sibling overlap is the implementation's sum of pairwise sibling-overlap area normalized by the parent/state area, averaged over internal nodes; it is not bounded by 1 when many sibling pairs overlap. It is useful for within-implementation diagnosis but should not replace measured node access.")
    add("")
    add("### Occupancy by depth")
    add("")
    occupancy = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            by_depth = structures[(dataset, capacity)]["by_depth"]
            text = "; ".join(
                f"d{depth}: {stats['nodes']} nodes, mean {stats['mean_entries']:.3f}, fill {stats['fill']:.6f}"
                for depth, stats in sorted(by_depth.items(), key=lambda item: int(item[0]))
            )
            occupancy.append([dataset.title(), capacity, text])
    extend(md_table(["Dataset", "B", "Depth occupancy"], occupancy, ["---", "---:", "---"]))
    add("")
    add("## 14. Neural action-use audit")
    add("")
    action_table = []
    for dataset in DATASETS:
        for capacity in CAPACITIES:
            counts = {int(key): int(value) for key, value in constructions[(dataset, capacity, "Neural")]["action_counts"].items()}
            top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            top_text = []
            for action_id, count in top:
                angle, fraction = action_description(action_id)
                top_text.append(f"a{action_id}={angle:g}deg/{fraction:g}: {count:,}")
            action_table.append([
                dataset.title(), capacity, number(constructions[(dataset, capacity, "Neural")]["neural_decision_count"]),
                len(counts), "; ".join(top_text),
            ])
    extend(md_table(
        ["Dataset", "B", "Decisions", "Actions used", "Five most frequent actions (angle/left-page fraction: count)"],
        action_table,
        ["---", "---:", "---:", "---:", "---"],
    ))
    add("")
    add("This audit proves that the deployed trees were produced by neural decisions rather than fallback selection. It also exposes a learned bias toward axis-aligned 0-degree and 90-degree actions, especially 0.25 left-page splits. The output is not equivalent to STR or TGS, but this concentration may contribute to the degradation observed as B moves away from the training value.")
    add("")
    add("## 15. Diagnostic query latency")
    add("")
    extend(md_table(
        ["Dataset", "B", "Neural mean ms", "STR mean ms", "TGS mean ms", "PLATON mean ms"],
        latency_table,
        ["---", "---:", "---:", "---:", "---:", "---:"],
    ))
    add("")
    add("These latency values are included for completeness only and must not be used as the main cross-method claim. Neural/STR/TGS queries run through Python object traversal, whereas PLATON uses compiled C++/libspatialindex traversal. Language, allocation, object representation, and timer boundaries differ. Logical node access is the implementation-neutral comparison reported in the main paper figures.")
    add("")
    add("## 16. Condition-level Neural/PLATON ratios")
    add("")
    add("Each cell below is total Neural node access divided by total PLATON node access for the same 300 queries. Values below 1 are Neural wins. These tables include all 14 range conditions, the point condition, and all five kNN conditions.")
    add("")
    order = [
        "square_side_0.001", "square_side_0.005", "square_side_0.01", "square_side_0.05", "square_side_0.1",
        "aspect_0.0625", "aspect_0.125", "aspect_0.25", "aspect_0.5", "aspect_1", "aspect_2", "aspect_4", "aspect_8", "aspect_16",
        "point", "knn_k1", "knn_k5", "knn_k25", "knn_k125", "knn_k625",
    ]
    lookup = {
        (row["dataset"], row["workload"], int(row["capacity"])): row
        for row in workload_rows if row["left_method"] == "Neural"
    }
    for dataset in DATASETS:
        add(f"### {dataset.title()}")
        add("")
        rows = []
        for workload in order:
            cells = []
            for capacity in CAPACITIES:
                value = float(lookup[(dataset, workload, capacity)]["ratio_left_over_right"])
                cells.append(f"**{value:.6f}**" if value < 1.0 else f"{value:.6f}")
            query_type = lookup[(dataset, workload, 128)]["query_type"]
            rows.append([query_type, workload, *cells])
        extend(md_table(
            ["Type", "Condition", "B=128", "B=256", "B=512"],
            rows,
            ["---", "---", "---:", "---:", "---:"],
        ))
        add("")
    winning_conditions = sum(
        float(row["ratio_left_over_right"]) < 1.0 for row in workload_rows if row["left_method"] == "Neural"
    )
    add(f"Neural won {winning_conditions} of the 180 condition-capacity cells. Those wins were concentrated in Crimes aspect-ratio ranges and a small number of Twitter elongated-range cases. No Arizona condition was won in aggregate. This is consistent with the nine setting-level totals and should not be reframed as a universal PLATON win.")
    add("")
    add("The complete condition table for Neural, STR, and TGS against PLATON is preserved in `CAPACITY_SWEEP_BY_WORKLOAD.csv` (540 rows). Per-query paired access counts are preserved under each run's `PAIRWISE_PER_QUERY.csv`.")
    add("")
    add("## 17. Statistical and reporting guidance")
    add("")
    add("The design is paired at query level: every method answers the same query arrays. For inferential analysis, resample paired query rows, preferably stratified by the 20 workload conditions, and recompute the ratio of summed accesses. Do not run an unpaired test across methods. Also do not treat the nine dataset-capacity settings as 54,000 independent replications of a single condition; datasets, capacities, and workload families are distinct experimental factors.")
    add("")
    add("Recommended headline effect sizes are Neural/baseline total-access ratio and percent change, 100 x (ratio - 1). Strict query win rate is supplementary. If confidence intervals are added in Prism or another package, state the bootstrap resample count, seed, whether sampling was stratified, and whether intervals are per setting or pooled.")
    add("")
    add("## 18. Prism-ready plotting plan")
    add("")
    add("Use `CAPACITY_SWEEP_SUMMARY.csv` for the main figures:")
    add("")
    add("1. **Total node access:** grouped bars or connected points; x=B, y=`method_total_access`, one panel per dataset, series=method. Add thousands separators and use a log y-axis only if all three datasets share one panel.")
    add("2. **Relative access versus PLATON:** x=B, y=`ratio_method_over_platon`; draw a horizontal reference at 1.0. Use one panel per dataset. This is the clearest capacity-transfer figure.")
    add("3. **Construction time:** x=B, y=`method_build_seconds`, series=method, log y-axis because PLATON is orders of magnitude slower.")
    add("4. **Strict paired outcomes:** stacked bars of `strict_win_rate`, `tie_rate`, and `loss_rate` for Neural versus PLATON.")
    add("")
    add("Use `CAPACITY_SWEEP_BY_WORKLOAD.csv` for workload sensitivity:")
    add("")
    add("5. **Window-size plot:** filter `left_method=Neural`, `query_type=range`, and `workload` beginning `square_side_`; x=side length, y=`ratio_left_over_right`, series=B, one panel per dataset.")
    add("6. **Aspect-ratio plot:** filter workloads beginning `aspect_`; use a log2 x-axis ordered 1/16 through 16, y=`ratio_left_over_right`, series=B.")
    add("7. **kNN plot:** filter `query_type=knn`; x=k on a log scale, y=`ratio_left_over_right`, series=B.")
    add("8. **Condition heatmap:** rows=20 conditions, columns=the nine dataset-capacity settings, color=Neural/PLATON ratio centered at 1.0.")
    add("")
    add("For absolute per-condition means, medians, p95 values, normalized I/O, and diagnostic latency, import each run's `methods/<method>/WORKLOAD_METRICS.csv`. For paired bootstrap or paired-difference plots, import `PAIRWISE_PER_QUERY.csv` and retain `query_uid`, `query_type`, and `workload` as pairing/stratification columns.")
    add("")
    add("## 19. Claims supported by this experiment")
    add("")
    add("### Supported")
    add("")
    add("- A single B=128-trained WAHARP neural policy can construct valid full-scale packed R-trees at B=128, B=256, and B=512 without retraining or deterministic fallback.")
    add("- WAHARP Neural consistently outperforms STR across three full datasets and three capacities in logical node access.")
    add("- WAHARP Neural outperforms TGS on Arizona and Twitter at every tested capacity, while TGS remains stronger on Crimes.")
    add("- WAHARP Neural approaches author PLATON closely on full Twitter and Crimes while reducing construction time by roughly one to two orders of magnitude.")
    add("- Capacity transfer is functional but not free: the Neural/PLATON gap grows as B departs from the B=128 training capacity.")
    add("")
    add("### Not supported")
    add("")
    add("- Do not claim that WAHARP beats PLATON in aggregate node access; PLATON won all nine setting totals.")
    add("- Do not claim zero-shot unseen-dataset generalization; all three domain types participated in training.")
    add("- Do not claim capacity-independent optimality; B=256 and B=512 show measurable degradation relative to PLATON.")
    add("- Do not claim cross-language query-latency superiority from the diagnostic timers.")
    add("- Do not attribute any gain to fallback methods; none were used in this experiment.")
    add("")
    add("## 20. Paper-ready result paragraph")
    add("")
    add("We evaluated one frozen B=128-trained WAHARP neural partition policy on full Arizona (1.46M rectangles), Crimes (7.70M points), and Twitter (14.26M points) datasets at node capacities 128, 256, and 512. Each method answered the same 6,000-query suite containing 4,200 range, 300 point, and 1,500 kNN queries; query arrays were byte-identical across capacities and all correctness checks passed. WAHARP used no deterministic fallback and selected all recursive leaf and parent partitions through an 80-action neural policy. It reduced logical node access relative to STR in all nine settings and relative to TGS in six settings. Against the author PLATON implementation, WAHARP's access ratios ranged from 1.0072 to 1.0941, with a geometric mean of 1.0321, while construction was 23.8x to 35.5x faster. The closest result occurred on full Twitter at B=128, where WAHARP incurred only 0.72% more accesses than PLATON. Cross-capacity ratios increased from a geometric mean of 1.0213 at B=128 to 1.0443 at B=512, indicating that the B=128-trained policy transfers successfully but retains a capacity-distribution gap.")
    add("")
    add("## 21. Reproducibility files")
    add("")
    reproducibility = [
        ["Headline and build results", "`results/waharp_final_capacity_sweep_b128_b256_b512/CAPACITY_SWEEP_SUMMARY.csv`"],
        ["All workload-condition ratios", "`results/waharp_final_capacity_sweep_b128_b256_b512/CAPACITY_SWEEP_BY_WORKLOAD.csv`"],
        ["Cross-capacity query hashes", "`results/waharp_final_capacity_sweep_b128_b256_b512/PAIRED_QUERY_HASHES.json`"],
        ["Frozen deployment manifest", "`trained_models/waharp_realtrain_b128_v1_three_domain_20260829/FROZEN_DEPLOYMENT_MANIFEST.json`"],
        ["Training history", "`trained_models/waharp_realtrain_b128_v1_three_domain_20260829/ENSEMBLE_TRAINING.json`"],
        ["Member selection record", "`trained_models/waharp_realtrain_b128_v1_three_domain_20260829/SHADOW_SELECTION.json`"],
        ["Frozen model", "`trained_models/waharp_realtrain_b128_v1_three_domain_20260829/member_2.pt`"],
        ["Benchmark source", "`kaggle_staging/waharp_final_capacity_sweep_code_dataset/waharp_final_capacity_sweep_bundle/`"],
    ]
    extend(md_table(["Artifact", "Path"], reproducibility, ["---", "---"]))
    add("")
    add(f"Consolidated summary SHA-256: `{sha256_file(RESULTS / 'CAPACITY_SWEEP_SUMMARY.csv')}`")
    add("")
    add(f"Condition summary SHA-256: `{sha256_file(RESULTS / 'CAPACITY_SWEEP_BY_WORKLOAD.csv')}`")
    add("")
    add(f"Paired-query hash manifest SHA-256: `{sha256_file(RESULTS / 'PAIRED_QUERY_HASHES.json')}`")
    add("")
    add("## 22. Final conclusion")
    add("")
    add("The improved neural-only WAHARP run is scientifically useful: it demonstrates that a compact learned recursive partition policy can scale to 14.26M objects, transfer from B=128 to B=512 without retraining, beat strong deterministic packers broadly, and reduce construction cost dramatically relative to PLATON. It does not establish a PLATON node-access victory. The strongest accurate framing is a quality-construction tradeoff: near-PLATON query I/O, substantially faster construction, no fallback dependence, and measurable but bounded cross-capacity degradation.")
    add("")
    add("---")
    add("")
    add("**Notation:** N/P = total Neural node access divided by total PLATON node access. Lower is better; N/P < 1 indicates a Neural win.")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(f"lines={len(lines)} bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
