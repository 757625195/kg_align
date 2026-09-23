import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from run_eight_dataset_retraining import (
    DatasetSpec,
    checkpoint_exists,
    dataset_specs,
    parse_list,
    parse_metrics,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    env: Dict[str, str]


@dataclass(frozen=True)
class RunSpec:
    dataset: DatasetSpec
    variant: VariantSpec
    seed: int

    @property
    def name(self) -> str:
        return f"{self.dataset.name}_{self.variant.name}_seed{self.seed}"


def variant_specs() -> List[VariantSpec]:
    new_main = {
        "KG_ALIGN_STRUCTURE_INITIALIZATION": "topology",
        "KG_ALIGN_TOPOLOGY_ENTITY_RESIDUAL_RATIO": "0.10",
        "KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "0.10",
        "KG_ALIGN_DECAY_STRUCTURE_LOSS": "1",
        "KG_ALIGN_NUM_NEIGHBORS": "0",
        "KG_ALIGN_NEIGHBOR_ATTENTION": "entmax15",
        "KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE": "0.25",
    }
    variants = [
        VariantSpec("main", {}),
        VariantSpec(
            "learned_structure_init",
            {"KG_ALIGN_STRUCTURE_INITIALIZATION": "learned"},
        ),
        VariantSpec(
            "no_structure_supervision",
            {
                "KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "0.0",
                "KG_ALIGN_DECAY_STRUCTURE_LOSS": "0",
            },
        ),
        VariantSpec(
            "fixed_k8_softmax",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "8",
                "KG_ALIGN_NEIGHBOR_ATTENTION": "softmax",
                "KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE": "1.0",
            },
        ),
        VariantSpec(
            "all_neighbor_softmax",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "0",
                "KG_ALIGN_NEIGHBOR_ATTENTION": "softmax",
                "KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE": "0.25",
            },
        ),
        VariantSpec(
            "structure_loss_010",
            {"KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "0.10"},
        ),
        VariantSpec(
            "seed_shared_structure_loss_010",
            {
                "KG_ALIGN_SHARE_SEED_STRUCTURE_EMBEDDINGS": "1",
                "KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "0.10",
            },
        ),
        VariantSpec(
            "topology_init_decay_structure_010",
            {
                "KG_ALIGN_STRUCTURE_INITIALIZATION": "topology",
                "KG_ALIGN_TOPOLOGY_ENTITY_RESIDUAL_RATIO": "0.10",
                "KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "0.10",
                "KG_ALIGN_DECAY_STRUCTURE_LOSS": "1",
            },
        ),
        VariantSpec(
            "topology_structure_all_neighbor_entmax_t025",
            {
                "KG_ALIGN_STRUCTURE_INITIALIZATION": "topology",
                "KG_ALIGN_TOPOLOGY_ENTITY_RESIDUAL_RATIO": "0.10",
                "KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "0.10",
                "KG_ALIGN_DECAY_STRUCTURE_LOSS": "1",
                "KG_ALIGN_NUM_NEIGHBORS": "0",
                "KG_ALIGN_NEIGHBOR_ATTENTION": "entmax15",
                "KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE": "0.25",
            },
        ),
        VariantSpec(
            "structure_loss_030",
            {"KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "0.30"},
        ),
        VariantSpec(
            "structure_loss_100",
            {"KG_ALIGN_STRUCTURE_LOSS_WEIGHT": "1.00"},
        ),
        VariantSpec("rrea_early", {"KG_ALIGN_STRUCTURE_ENCODER": "rrea"}),
        VariantSpec("rdgcn_early", {"KG_ALIGN_STRUCTURE_ENCODER": "rdgcn"}),
        VariantSpec(
            "rdgcn_name_early",
            {
                "KG_ALIGN_STRUCTURE_ENCODER": "rdgcn",
                "KG_ALIGN_STRUCTURE_INITIALIZATION": "name",
            },
        ),
        VariantSpec("reciprocal_add", {"KG_ALIGN_FUSION_MODE": "reciprocal_add"}),
        VariantSpec(
            "reciprocal_residual_add",
            {"KG_ALIGN_FUSION_MODE": "reciprocal_residual_add"},
        ),
        VariantSpec("no_relation_types", {"KG_ALIGN_USE_RELATION_TYPES": "0"}),
        VariantSpec(
            "disjoint_relation_vocab",
            {"KG_ALIGN_SHARE_CROSS_GRAPH_RELATIONS": "0"},
        ),
        VariantSpec(
            "bidirectional_same_relation",
            {"KG_ALIGN_ADD_REVERSE_EDGES": "1"},
        ),
        VariantSpec("no_layer_selector", {"KG_ALIGN_RELATION_LAYER_FUSION": "0"}),
        VariantSpec("no_token_view", {"KG_ALIGN_SEM_USE_TOKEN_VIEW": "0"}),
        VariantSpec("no_phrase_view", {"KG_ALIGN_SEM_USE_PHRASE_VIEW": "0"}),
        VariantSpec("no_global_view", {"KG_ALIGN_SEM_USE_GLOBAL_VIEW": "0"}),
        VariantSpec(
            "no_global_residual",
            {"KG_ALIGN_SEM_RESIDUAL_MODE": "gated_no_global"},
        ),
        VariantSpec(
            "variable_entmax_neighbors",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "0",
                "KG_ALIGN_NEIGHBOR_ATTENTION": "entmax15",
            },
        ),
        VariantSpec(
            "variable_entmax_t025",
            {
                "KG_ALIGN_NUM_NEIGHBORS": "0",
                "KG_ALIGN_NEIGHBOR_ATTENTION": "entmax15",
                "KG_ALIGN_NEIGHBOR_ATTENTION_TEMPERATURE": "0.25",
            },
        ),
        VariantSpec("late_concat_matched", {"KG_ALIGN_FUSION_MODE": "late_concat_matched"}),
        VariantSpec(
            "structural_neighbor_query",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "structural",
                "KG_ALIGN_NEIGHBOR_GATE_MODE": "semantic",
            },
        ),
        VariantSpec(
            "query_sem_gate_sem",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "semantic",
                "KG_ALIGN_NEIGHBOR_GATE_MODE": "semantic",
            },
        ),
        VariantSpec(
            "query_struct_gate_sem",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "structural",
                "KG_ALIGN_NEIGHBOR_GATE_MODE": "semantic",
            },
        ),
        VariantSpec(
            "query_sem_gate_struct",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "semantic",
                "KG_ALIGN_NEIGHBOR_GATE_MODE": "structural",
            },
        ),
        VariantSpec(
            "query_struct_gate_struct",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "structural",
                "KG_ALIGN_NEIGHBOR_GATE_MODE": "structural",
            },
        ),
        VariantSpec(
            "hybrid_neighbor_query",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "hybrid",
                "KG_ALIGN_NEIGHBOR_QUERY_SEMANTIC_WEIGHT": "0.50",
            },
        ),
        VariantSpec(
            "hybrid_neighbor_query_s025",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "hybrid",
                "KG_ALIGN_NEIGHBOR_QUERY_SEMANTIC_WEIGHT": "0.25",
            },
        ),
        VariantSpec(
            "hybrid_neighbor_query_s075",
            {
                "KG_ALIGN_NEIGHBOR_QUERY_MODE": "hybrid",
                "KG_ALIGN_NEIGHBOR_QUERY_SEMANTIC_WEIGHT": "0.75",
            },
        ),
        VariantSpec(
            "token_only",
            {
                "KG_ALIGN_SEM_USE_PHRASE_VIEW": "0",
                "KG_ALIGN_SEM_USE_GLOBAL_VIEW": "0",
            },
        ),
        VariantSpec("structure_only", {"KG_ALIGN_FUSION_MODE": "structure_only"}),
        VariantSpec("semantic_only", {"KG_ALIGN_FUSION_MODE": "semantic_only"}),
        VariantSpec("mean_fusion", {"KG_ALIGN_FUSION_MODE": "mean"}),
    ]
    return [
        VariantSpec(spec.name, {**new_main, **spec.env})
        for spec in variants
    ]


def selected_datasets() -> Dict[str, DatasetSpec]:
    retained = {
        "dbp15k_zh_en",
        "dbp15k_ja_en",
        "dbp15k_fr_en",
        "openea_en_fr",
        "eventea_en_en",
    }
    return {spec.name: spec for spec in dataset_specs() if spec.name in retained}


def run_one(
    run: RunSpec,
    root: Path,
    output_dir: Path,
    batch_size: int,
    epochs: Optional[int],
    threads_per_job: int,
    resume: bool,
    discard_checkpoints: bool,
) -> Dict[str, object]:
    variant_root = output_dir / run.variant.name
    log_dir = variant_root / "logs"
    run_dir = variant_root / "checkpoints" / run.name
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run.name}.log"

    existing = parse_metrics(log_path)
    if resume and existing is not None and (
        discard_checkpoints or checkpoint_exists(run_dir)
    ):
        return {
            "dataset": run.dataset.name,
            "variant": run.variant.name,
            "seed": run.seed,
            "status": "ok",
            "resumed": True,
            "checkpoint_retained": not discard_checkpoints,
            "log_path": str(log_path),
            **existing,
        }

    env = os.environ.copy()
    env.update(run.dataset.env)
    env.update(run.variant.env)
    env.update({
        "KG_ALIGN_SEED": str(run.seed),
        "KG_ALIGN_SAVE_DIR": str(run_dir),
        "KG_ALIGN_BATCH_SIZE": str(batch_size),
        "KG_ALIGN_EVAL_BATCH_SIZE": str(batch_size),
        "KG_ALIGN_USE_EARLY_STOPPING": "1",
        "KG_ALIGN_EARLY_STOP_PATIENCE": "4",
        "OMP_NUM_THREADS": str(threads_per_job),
        "MKL_NUM_THREADS": str(threads_per_job),
        "PYTHONUNBUFFERED": "1",
    })
    if epochs is not None:
        env["KG_ALIGN_JOINT_EPOCHS"] = str(epochs)

    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            [sys.executable, str(root / "train.py")],
            cwd=str(root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    metrics = parse_metrics(log_path)
    result: Dict[str, object] = {
        "dataset": run.dataset.name,
        "variant": run.variant.name,
        "seed": run.seed,
        "status": "ok" if completed.returncode == 0 and metrics else f"failed({completed.returncode})",
        "resumed": False,
        "log_path": str(log_path),
    }
    if metrics:
        result.update(metrics)
    if result["status"] == "ok" and discard_checkpoints:
        shutil.rmtree(run_dir)
        result["checkpoint_retained"] = False
    else:
        result["checkpoint_retained"] = checkpoint_exists(run_dir)
    return result


def write_results(output_dir: Path, results: List[Dict[str, object]]) -> None:
    summary_path = output_dir / "runs_summary.json"
    existing: List[Dict[str, object]] = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    merged = {
        (str(row["dataset"]), str(row["variant"]), int(row["seed"])): row
        for row in [*existing, *results]
    }
    ordered = sorted(
        merged.values(),
        key=lambda row: (str(row["variant"]), str(row["dataset"]), int(row["seed"])),
    )
    summary_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    variants = {spec.name: spec for spec in variant_specs()}
    datasets = selected_datasets()

    parser = argparse.ArgumentParser(
        description="Run controlled multi-seed ablations on representative datasets."
    )
    parser.add_argument("--datasets", default=",".join(datasets))
    parser.add_argument("--variants", default=",".join(variants))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", default="outputs/controlled_ablation_20260820")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads-per-job", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--discard-checkpoints", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    dataset_names = parse_list(args.datasets, str)
    variant_names = parse_list(args.variants, str)
    seeds = parse_list(args.seeds, int)
    unknown_datasets = sorted(set(dataset_names) - set(datasets))
    unknown_variants = sorted(set(variant_names) - set(variants))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")

    runs = [
        RunSpec(datasets[dataset_name], variants[variant_name], seed)
        for variant_name in variant_names
        for dataset_name in dataset_names
        for seed in seeds
    ]
    if args.list:
        print(json.dumps([
            {
                "name": run.name,
                "dataset_env": run.dataset.env,
                "variant_env": run.variant.env,
            }
            for run in runs
        ], ensure_ascii=False, indent=2))
        return 0

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                run_one,
                run,
                root,
                output_dir,
                args.batch_size,
                args.epochs,
                max(1, args.threads_per_job),
                not args.no_resume,
                args.discard_checkpoints,
            ): run
            for run in runs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run = futures[future]
            result = future.result()
            results.append(result)
            write_results(output_dir, results)
            if result["status"] == "ok":
                print(
                    f"[{index}/{len(runs)}] {run.name} | "
                    f"Hits@1={result['Hits@1']:.4f} MRR={result['MRR']:.4f}",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(runs)}] {run.name} | {result['status']}", flush=True)

    failed = [row for row in results if row["status"] != "ok"]
    print(f"WROTE {output_dir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
