"""Assemble the code bundles expected by the historical scripts in scripts/kaggle/.

Each historical entry point discovers its code by searching ``/kaggle/input``
for a bundle marker with a recorded protocol id. This script recreates those
bundles from the repository, using the markers committed under
``scripts/kaggle/bundle_markers/``. Upload each output directory as a Kaggle
dataset (or place it under the directory passed as the scripts' input root).

    python scripts/make_kaggle_bundles.py --output kaggle_bundles_out

Bundles:

    realtrain_v1          teacher states and DAgger (python -m realtrain.pipeline),
                          run_loss_weight_ablation.py, run_final_train.py
    final_fullscale_b128  run_fullscale_b128.py (B=128 baselines, locked queries)
    final_capacity_sweep  run_fullscale.py (B=256/512), run_locked_eval.py,
                          run_lodo_zero_shot.py, run_platon_budget_b128.py

The script also checks the repository's code against the hashes of the code
that executed the recorded experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ROOT / "scripts" / "kaggle" / "bundle_markers"
PREVIOUS_CHECKPOINT = ROOT / "artifacts" / "model" / "previous_060_040"
CODE_DIRS = ("realtrain", "rtreelib", "native")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")

# Bundle name -> (marker directory, runner placed at scripts/run_fullscale.py, needs previous checkpoint)
BUNDLES = {
    "realtrain_v1": ("realtrain_v1", None, False),
    "final_fullscale_b128": ("final_fullscale_b128", "run_fullscale_b128.py", True),
    "final_capacity_sweep": ("final_capacity_sweep", "run_fullscale.py", True),
}


def compare_executed_code() -> list[str]:
    """Return a report line for each code file that differs from the executed bundle."""
    lines = []
    with (MARKERS / "realtrain_v1" / "EXECUTED_CODE_MANIFEST.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            path = ROOT / row["path"]
            if row["path"] == "requirements.txt":
                continue
            if not path.is_file():
                lines.append(f"  missing  {row['path']}")
            # Git may rewrite line endings on checkout, so compare LF-normalized content.
            elif hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != row["lf_normalized_sha256"]:
                lines.append(f"  differs  {row['path']}")
    return lines


def build_bundle(name: str, output: Path) -> Path:
    marker_dir, runner, needs_checkpoint = BUNDLES[name]
    dest = output / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for directory in CODE_DIRS:
        shutil.copytree(ROOT / directory, dest / directory, ignore=IGNORE)
    shutil.copy2(ROOT / "requirements.txt", dest / "requirements.txt")
    shutil.copytree(ROOT / "scripts" / "kaggle", dest / "scripts", ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "bundle_markers"))
    if runner:
        shutil.copy2(ROOT / "scripts" / "kaggle" / runner, dest / "scripts" / "run_fullscale.py")
    if needs_checkpoint:
        shutil.copytree(PREVIOUS_CHECKPOINT, dest / "frozen_model")
    for marker in (MARKERS / marker_dir).iterdir():
        if marker.suffix == ".json":
            shutil.copy2(marker, dest / marker.name)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle_bundles_out")
    parser.add_argument("--bundle", choices=sorted(BUNDLES), action="append",
                        help="Bundle to build (repeatable; default: all)")
    parser.add_argument("--check", action="store_true",
                        help="Only compare with the executed code; fail if any file differs")
    args = parser.parse_args()

    if not args.check:
        for name in args.bundle or BUNDLES:
            print(f"built {build_bundle(name, args.output)}")
    differences = compare_executed_code()
    print("Repository code vs. the code that executed the recorded experiments:")
    print("\n".join(differences) if differences else "  identical")
    if args.check and differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
