# KG Align

This repository contains the streamlined entity-alignment model used in the
current experiments.

Current scope:

- datasets: DBP15K raw splits and OpenEA official splits
- default protocol: `DBP15K zh_en / raw_split / 0_3`
- final scoring: `joint-only`

## Main Components

- structural encoder: [models/gnn_encoder.py](models/gnn_encoder.py)
  - relation-aware lightweight message passing
  - 3 GNN layers
  - node-wise layer fusion across hop depths
  - deterministic fixed-budget neighbor collection
- semantic encoder: [models/text_encoder.py](models/text_encoder.py)
  - token, phrase, and Transformer global views with gated aggregation
- fusion module: [models/fusion.py](models/fusion.py)
  - cross-modal structure/semantics fusion into `z_joint`
- alignment scorer: [models/alignment_head.py](models/alignment_head.py)
  - cosine retrieval with CSLS correction
- training objective: [models/losses.py](models/losses.py)
  - bidirectional InfoNCE
  - non-matching entities in the current batch act as negatives

## Entry Point

Run the formal baseline with:

```bash
source .venv/bin/activate
python train.py
```

Single-seed sweep:

```bash
source .venv/bin/activate
python sweep_raw_split.py --seeds 42
```

Multi-seed sweep:

```bash
source .venv/bin/activate
python sweep_raw_split.py --seeds 42,43,44
```

## Teacher-requested Experiments

The teacher-facing experiment matrix is managed by
`run_teacher_required_experiments.py`. The fusion profile distinguishes three
actual forward paths:

- `early_interaction`: semantics score and gate structural neighbors before aggregation;
- `late_concat_mlp`: independently encoded structure and semantics are concatenated only at the output and projected by a two-layer MLP;
- `mean`: independently encoded branches are averaged at the output (`w_o_ce`).

Run the missing late-fusion comparison on both datasets with three seeds:

```bash
source .venv/bin/activate
python run_teacher_required_experiments.py \
  --profile fusion \
  --variants late_concat_mlp \
  --seeds 42,43,44 \
  --resume
```

The runner also accepts `--datasets` and `--variants` filters. `--resume`
preserves completed records in `runs_summary.json` and reruns only missing or
failed configurations.

When running multiple seeds of the same dataset concurrently, set a distinct
`KG_ALIGN_SAVE_DIR` for each job so validation checkpoints cannot overwrite one
another.

## Default Configuration

The training entrypoint is fixed to the verified baseline path in
[train.py](train.py):

- `data_source = raw_split`
- `raw_split = 0_3`
- `joint_epochs = 36`
- `gnn_layers = 3`
- `relation_layer_fusion = True`
- `temperature = 0.07`
- `joint objective = bidirectional InfoNCE`
- `alignment_csls_k = 10`
- `alignment_csls_blend = 1.0`
- `use_final_weight_averaging = True`
- `weight_average_last_k = 10`
- semantic views: token, phrase, and global

The entrypoint supports dataset, optimization, semantic-view, fusion, depth,
and CSLS overrides through the `KG_ALIGN_*` environment variables defined in
`apply_runtime_overrides`.

Example:

```bash
KG_ALIGN_SEED=42 python train.py
```

Dataset-specific fusion weight, effective propagation depth, and CSLS
parameters can be selected from validation data with:

```bash
python select_dataset_hyperparameters.py
```

## Best Verified Reference

Formal reference log:

- `outputs/sweeps/baseline_seed42.log` (local experiment artifact)

Reference metrics:

- `Hits@1 = 0.7041`
- `Hits@10 = 0.8155`
- `MRR = 0.7449`

Reference checkpoint:

- `outputs/best_model_zh_en_best_baseline.pt` (local experiment artifact)

## Local Data and Reference Implementations

Datasets, generated outputs, migration archives, and third-party checkouts are
intentionally excluded from Git. Restore datasets separately under `data/`.
The baseline wrappers use these reference implementations when present:

```bash
mkdir -p third_party
git clone https://github.com/nju-websoft/OpenEA.git third_party/OpenEA
git clone https://github.com/MaoXinn/RREA.git third_party/RREA
```

The revisions used for the current experiments were OpenEA
`b59e014153c27c7166d78475e3474c7e86a10be9` and RREA
`2271ac33dae0baf53dfa5b7ca1955090a1567a0a`.

## Repository Layout

```text
kg_align/
├── data_utils.py
├── dataset.py
├── evaluate.py
├── graph_utils.py
├── sweep_raw_split.py
├── test_model.py
├── train.py
├── models/
│   ├── alignment_head.py
│   ├── full_model.py
│   ├── fusion.py
│   ├── gnn_encoder.py
│   ├── losses.py
│   └── text_encoder.py
└── outputs/
```
