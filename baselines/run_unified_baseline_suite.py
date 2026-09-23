import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "outputs/unified_existing_baselines_20260826"
OPENEA_METHODS = {"mtranse", "jape", "bootea", "rdgcn"}
ALL_METHODS = ("mtranse", "jape", "bootea", "rdgcn", "rrea_basic")
ALL_DATASETS = ("dbp15k_zh_en", "openea_en_fr")


def parse_csv(value, allowed, label):
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported {label}: {', '.join(unknown)}")
    return selected


def output_path(output_dir, method, dataset, seed):
    return ROOT / output_dir / method / f"{dataset}_seed{seed}.json"


def command_for(python, output_dir, method, dataset, seed):
    if method in OPENEA_METHODS:
        return [
            python,
            str(ROOT / "baselines" / "run_openea_baseline.py"),
            "--method",
            method,
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--output-dir",
            output_dir,
        ]
    return [
        python,
        str(ROOT / "baselines" / "run_rrea_baseline.py"),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run official-source entity-alignment baselines serially and resumably."
    )
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--datasets", default=",".join(ALL_DATASETS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = parse_csv(args.methods, set(ALL_METHODS), "methods")
    datasets = parse_csv(args.datasets, set(ALL_DATASETS), "datasets")
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    python = str(ROOT / ".venv-baselines" / "bin" / "python")
    log_dir = ROOT / args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    total = len(methods) * len(datasets) * len(seeds)
    completed = 0
    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                completed += 1
                result_path = output_path(args.output_dir, method, dataset, seed)
                if result_path.exists() and not args.force:
                    print(f"[{completed}/{total}] skip existing {result_path}", flush=True)
                    continue

                command = command_for(
                    python, args.output_dir, method, dataset, seed
                )
                print(
                    f"[{completed}/{total}] run {method} {dataset} seed={seed}",
                    flush=True,
                )
                if args.dry_run:
                    print(" ".join(command), flush=True)
                    continue

                log_path = log_dir / f"{method}_{dataset}_seed{seed}.log"
                with log_path.open("w", encoding="utf-8") as log_file:
                    process = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode != 0:
                    raise RuntimeError(
                        f"{method} {dataset} seed={seed} failed with code "
                        f"{process.returncode}; see {log_path}"
                    )
                if not result_path.exists():
                    raise RuntimeError(
                        f"Run succeeded but did not create expected result: {result_path}"
                    )

    print("All requested baseline runs are complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
