import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from run_two_dataset_3seed_ablation import (
    DatasetSpec,
    RunSpec,
    VariantSpec,
    parse_metrics,
    parse_seed_list,
    write_outputs,
)


def build_datasets() -> List[DatasetSpec]:
    return [
        DatasetSpec(
            name="dbp15k_zh_en",
            env={
                "KG_ALIGN_DATASET_FAMILY": "dbp15k_raw",
                "KG_ALIGN_ROOT": "data/dbp15k",
                "KG_ALIGN_PAIR": "zh_en",
                "KG_ALIGN_RAW_SPLIT": "0_3",
                "KG_ALIGN_PROTOCOL": "simplified_core",
            },
        ),
        DatasetSpec(
            name="openea_en_fr",
            env={
                "KG_ALIGN_DATASET_FAMILY": "openea",
                "KG_ALIGN_ROOT": "data/openea",
                "KG_ALIGN_PAIR": "EN_FR_15K_V2",
                "KG_ALIGN_OPENEA_SPLIT": "721_5fold/1",
                "KG_ALIGN_PROTOCOL": "simplified_core",
            },
        ),
    ]


def build_variants(profile: str) -> List[VariantSpec]:
    profiles: Dict[str, List[VariantSpec]] = {
        "core": [
            VariantSpec("base", {}),
            VariantSpec("w_o_mst", {"KG_ALIGN_USE_MST": "0"}),
            VariantSpec("w_o_ce", {"KG_ALIGN_USE_CE": "0"}),
        ],
        "semantic": [
            VariantSpec("base", {}),
            VariantSpec("w_o_token_view", {"KG_ALIGN_SEM_USE_TOKEN_VIEW": "0"}),
            VariantSpec("w_o_phrase_view", {"KG_ALIGN_SEM_USE_PHRASE_VIEW": "0"}),
            VariantSpec("w_o_global_view", {"KG_ALIGN_SEM_USE_GLOBAL_VIEW": "0"}),
            VariantSpec(
                "only_token_view",
                {
                    "KG_ALIGN_SEM_USE_TOKEN_VIEW": "1",
                    "KG_ALIGN_SEM_USE_PHRASE_VIEW": "0",
                    "KG_ALIGN_SEM_USE_GLOBAL_VIEW": "0",
                },
            ),
            VariantSpec(
                "only_phrase_view",
                {
                    "KG_ALIGN_SEM_USE_TOKEN_VIEW": "0",
                    "KG_ALIGN_SEM_USE_PHRASE_VIEW": "1",
                    "KG_ALIGN_SEM_USE_GLOBAL_VIEW": "0",
                },
            ),
            VariantSpec(
                "only_global_view",
                {
                    "KG_ALIGN_SEM_USE_TOKEN_VIEW": "0",
                    "KG_ALIGN_SEM_USE_PHRASE_VIEW": "0",
                    "KG_ALIGN_SEM_USE_GLOBAL_VIEW": "1",
                },
            ),
        ],
        "structural": [
            VariantSpec("base", {}),
            VariantSpec("w_o_relation_gnn", {"KG_ALIGN_USE_RELATION_GNN": "0"}),
            VariantSpec("w_o_layer_fusion", {"KG_ALIGN_RELATION_LAYER_FUSION": "0"}),
            VariantSpec("neighbors6", {"KG_ALIGN_NUM_NEIGHBORS": "6"}),
            VariantSpec("neighbors10", {"KG_ALIGN_NUM_NEIGHBORS": "10"}),
        ],
        "fusion": [
            VariantSpec("base", {}),
            VariantSpec("w_o_ce", {"KG_ALIGN_USE_CE": "0"}),
            VariantSpec(
                "late_concat_mlp",
                {"KG_ALIGN_FUSION_MODE": "late_concat_mlp"},
            ),
        ],
    }
    if profile == "all":
        merged: List[VariantSpec] = []
        seen = set()
        for key in ("core", "semantic", "structural", "fusion"):
            for variant in profiles[key]:
                if variant.name not in seen:
                    merged.append(variant)
                    seen.add(variant.name)
        return merged
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}")
    return profiles[profile]


def build_runs(seeds: List[int], profile: str) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for dataset in build_datasets():
        for variant in build_variants(profile):
            for seed in seeds:
                env = {
                    **dataset.env,
                    **variant.env,
                    "KG_ALIGN_SEED": str(seed),
                }
                runs.append(
                    RunSpec(
                        dataset=dataset.name,
                        variant=variant.name,
                        seed=seed,
                        env=env,
                    )
                )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run teacher-requested experiment profiles for KG alignment."
    )
    parser.add_argument(
        "--profile",
        choices=("core", "semantic", "structural", "fusion", "all"),
        required=True,
        help="Experiment group to run.",
    )
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds.")
    parser.add_argument(
        "--output-dir",
        default="outputs/teacher_required_experiments",
        help="Output directory for logs and summaries.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset names to run.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant names to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful runs already present in runs_summary.json.",
    )
    parser.add_argument("--list", action="store_true", help="Print planned runs and exit.")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    runs = build_runs(seeds=seeds, profile=args.profile)
    if args.datasets:
        selected_datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
        runs = [run for run in runs if run.dataset in selected_datasets]
    if args.variants:
        selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
        runs = [run for run in runs if run.variant in selected_variants]
    if not runs:
        raise ValueError("No runs remain after applying dataset/variant filters.")
    if args.list:
        print(json.dumps([{"name": run.name, "env": run.env} for run in runs], ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir) / args.profile
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(__file__).with_name("train.py")
    summary_path = output_dir / "runs_summary.json"
    results: List[Dict[str, object]] = []
    if args.resume and summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            results = loaded

    valid_profile_variants = {variant.name for variant in build_variants(args.profile)}
    results = [
        item for item in results
        if str(item.get("variant")) in valid_profile_variants
    ]

    def result_key(item: Dict[str, object]):
        return str(item["dataset"]), str(item["variant"]), int(item["seed"])

    results_by_key = {result_key(item): item for item in results}

    for index, run in enumerate(runs, start=1):
        log_path = log_dir / f"{run.name}.log"
        key = (run.dataset, run.variant, run.seed)
        existing = results_by_key.get(key)
        if args.resume and existing and existing.get("status") == "ok":
            print(f"\n[{index}/{len(runs)}] Reusing {run.name}")
            continue
        env = os.environ.copy()
        env.update(run.env)
        env["PYTHONUNBUFFERED"] = "1"

        print(f"\n[{index}/{len(runs)}] Running {run.name}")
        print(f"Log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                [sys.executable, str(train_path)],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        item: Dict[str, object] = {
            "dataset": run.dataset,
            "variant": run.variant,
            "seed": run.seed,
            "status": "failed",
            "log_path": str(log_path),
        }
        metrics = parse_metrics(log_path) if log_path.exists() else None
        if process.returncode == 0 and metrics:
            item["status"] = "ok"
            item.update(metrics)
            print(
                f"Finished {run.name}: "
                f"Hits@1={metrics['Hits@1']:.4f}, "
                f"Hits@10={metrics['Hits@10']:.4f}, "
                f"MRR={metrics['MRR']:.4f}"
            )
        else:
            print(f"Failed {run.name}; inspect {log_path}")
        results_by_key[key] = item
        results = sorted(
            results_by_key.values(),
            key=lambda value: (
                str(value["dataset"]),
                str(value["variant"]),
                int(value["seed"]),
            ),
        )
        write_outputs(output_dir, results)

    write_outputs(output_dir, results)

    print(f"\nRun summary: {output_dir / 'runs_summary.tsv'}")
    print(f"Aggregate summary: {output_dir / 'aggregate_summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
