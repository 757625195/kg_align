import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent


def parse_list(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def run_command(command: List[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_pipeline_summary(output_root: Path, variants: List[str]) -> None:
    raw_runs: List[Dict[str, object]] = []
    selected_runs: List[Dict[str, object]] = []
    selected_aggregate: List[Dict[str, object]] = []
    for variant in variants:
        variant_root = output_root / variant
        raw_path = variant_root / "raw_runs_summary.json"
        selection_root = variant_root / "selection"
        if raw_path.exists():
            raw_runs.extend(load_json(raw_path))
        if (selection_root / "test_runs.json").exists():
            selected_runs.extend(
                {"variant": variant, **row}
                for row in load_json(selection_root / "test_runs.json")
            )
        if (selection_root / "test_aggregate.json").exists():
            selected_aggregate.extend(
                {"variant": variant, **row}
                for row in load_json(selection_root / "test_aggregate.json")
            )
    (output_root / "pipeline_raw_runs.json").write_text(
        json.dumps(raw_runs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "pipeline_selected_runs.json").write_text(
        json.dumps(selected_runs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "pipeline_selected_aggregate.json").write_text(
        json.dumps(selected_aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--depths", default="1,2,3")
    parser.add_argument(
        "--semantic-weights",
        default="0.20,0.30,0.40,0.50,0.60,0.70,0.80",
    )
    parser.add_argument("--csls-k", default="3,5,7,10,15,20")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    variants = parse_list(args.variants)
    output_root = ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    for index, variant in enumerate(variants, start=1):
        variant_root = output_root / variant
        selection_root = variant_root / "selection"
        selection_complete = selection_root / "test_aggregate.json"
        if selection_complete.exists():
            print(f"[{index}/{len(variants)}] SKIP {variant}: selection already complete", flush=True)
            continue

        selected_depths = args.depths
        selected_semantic_weights = args.semantic_weights
        if variant == "structure_only":
            selected_semantic_weights = "0.00"
        elif variant == "semantic_only":
            selected_depths = "3"
            selected_semantic_weights = "1.00"

        run_command([
            sys.executable,
            "run_controlled_ablation_retraining.py",
            "--datasets", ",".join(datasets),
            "--variants", variant,
            "--seeds", args.seeds,
            "--output-dir", args.output_dir,
            "--jobs", str(max(1, args.jobs)),
            "--threads-per-job", str(max(1, args.threads_per_job)),
            "--batch-size", str(args.batch_size),
        ])

        runner_summary = output_root / "runs_summary.json"
        if not runner_summary.exists():
            raise FileNotFoundError(f"Training did not produce {runner_summary}")
        shutil.copy2(runner_summary, variant_root / "raw_runs_summary.json")

        run_command([
            sys.executable,
            "select_dataset_hyperparameters.py",
            "--datasets", ",".join(datasets),
            "--seeds", args.seeds,
            "--depths", selected_depths,
            "--semantic-weights", selected_semantic_weights,
            "--csls-k", args.csls_k,
            "--csls-blends", "1.00",
            "--checkpoint-root", str(variant_root / "checkpoints"),
            "--output-dir", str(selection_root),
        ])

        if not args.keep_checkpoints:
            shutil.rmtree(variant_root / "checkpoints", ignore_errors=True)
        write_pipeline_summary(output_root, variants)
        print(f"[{index}/{len(variants)}] COMPLETE {variant}", flush=True)

    write_pipeline_summary(output_root, variants)
    print(f"WROTE {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
