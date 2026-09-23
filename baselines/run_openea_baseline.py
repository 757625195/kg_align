import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENEA_ROOT = ROOT / "third_party" / "OpenEA"
OPENEA_SRC = OPENEA_ROOT / "src"


def install_tensorflow_v1_compat(seed):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.set_random_seed(seed)
    tf.contrib = types.SimpleNamespace(
        layers=types.SimpleNamespace(
            xavier_initializer=lambda uniform=False: (
                tf.glorot_uniform_initializer()
                if uniform
                else tf.glorot_normal_initializer()
            )
        )
    )
    sys.modules["tensorflow"] = tf
    return tf


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def csls_corrections(left, right, k, batch_size):
    k = max(1, min(int(k), left.shape[0], right.shape[0]))
    row_parts = []
    column_top = np.full((k, right.shape[0]), -np.inf, dtype=np.float32)
    for start in range(0, left.shape[0], batch_size):
        similarity = left[start : start + batch_size] @ right.T
        row_k = min(k, similarity.shape[1])
        row_parts.append(
            np.partition(similarity, similarity.shape[1] - row_k, axis=1)[:, -row_k:].mean(axis=1)
        )
        col_k = min(k, similarity.shape[0])
        chunk_top = np.partition(similarity, similarity.shape[0] - col_k, axis=0)[-col_k:, :]
        merged = np.concatenate([column_top, chunk_top], axis=0)
        column_top = np.partition(merged, merged.shape[0] - k, axis=0)[-k:, :]
    return np.concatenate(row_parts), column_top.mean(axis=0)


def ranking_metrics(left, right, gt_indices, csls_k, batch_size=256):
    left = normalize_rows(left)
    right = normalize_rows(right)
    row_correction = None
    column_correction = None
    if csls_k > 0:
        row_correction, column_correction = csls_corrections(
            left, right, csls_k, batch_size
        )

    ranks = []
    for start in range(0, left.shape[0], batch_size):
        end = min(start + batch_size, left.shape[0])
        similarity = left[start:end] @ right.T
        if row_correction is not None:
            similarity = (
                2.0 * similarity
                - row_correction[start:end, None]
                - column_correction[None, :]
            )
        local_gt = gt_indices[start:end]
        gold = similarity[np.arange(end - start), local_gt]
        greater = (similarity > gold[:, None]).sum(axis=1)
        tied = (similarity == gold[:, None]).sum(axis=1)
        ranks.append(greater + 1.0 + 0.5 * np.maximum(tied - 1, 0))
    ranks = np.concatenate(ranks)
    return {
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@5": float(np.mean(ranks <= 5)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
    }


def evaluate_pairs(embeddings, mapping, pairs, candidate_right_ids, csls_k):
    left_ids = [left for left, _ in pairs]
    right_index = {entity_id: index for index, entity_id in enumerate(candidate_right_ids)}
    gt_indices = np.asarray([right_index[right] for _, right in pairs], dtype=np.int64)
    left = embeddings[left_ids]
    if mapping is not None:
        left = left @ mapping
    right = embeddings[candidate_right_ids]
    return ranking_metrics(left, right, gt_indices, csls_k)


def select_csls(embeddings, mapping, validation_pairs, candidate_right_ids, candidates):
    rows = []
    for k in candidates:
        metrics = evaluate_pairs(
            embeddings,
            mapping,
            validation_pairs,
            candidate_right_ids,
            k,
        )
        rows.append({"k": k, **metrics})
    best = max(rows, key=lambda row: (row["MRR"], row["Hits@1"], -row["k"]))
    return best, rows


def dataset_spec(name, seed):
    if name == "dbp15k_zh_en":
        return ROOT / "data" / "baseline_unified" / "zh_en", f"seed{seed}/"
    if name == "openea_en_fr":
        return ROOT / "data" / "openea" / "EN_FR_15K_V2", "721_5fold/1/"
    raise ValueError(f"Unsupported dataset: {name}")


def build_model(method):
    from openea.approaches import BootEA, JAPE, MTransE, RDGCN

    classes = {
        "mtranse": MTransE,
        "jape": JAPE,
        "bootea": BootEA,
        "rdgcn": RDGCN,
    }
    return classes[method]()


def install_rdgcn_random_initializer(model, seed):
    """Use RDGCN's trainable input without unavailable anonymized names."""

    def _get_random_input(self):
        entity_count = self.kgs.entities_num
        limit = np.sqrt(6.0 / (entity_count + self.args.dim))
        rng = np.random.RandomState(seed)
        embeddings = rng.uniform(
            -limit,
            limit,
            size=(entity_count, self.args.dim),
        ).astype(np.float32)
        print(
            "using identity-independent Glorot entity initialization: "
            f"{entity_count}x{self.args.dim}"
        )
        return None, None, embeddings

    model._get_desc_input = types.MethodType(_get_random_input, model)


def load_method_args(method):
    from openea.modules.args.args_hander import load_args

    path = OPENEA_ROOT / "run" / "args" / f"{method}_args_15K.json"
    return load_args(str(path))


def extract_embeddings(model, method):
    if method == "rdgcn":
        return np.asarray(model.sess.run(model.output), dtype=np.float32), None
    embeddings = np.asarray(model.ent_embeds.eval(session=model.session), dtype=np.float32)
    mapping = None
    if getattr(model, "mapping_mat", None) is not None:
        mapping = np.asarray(model.mapping_mat.eval(session=model.session), dtype=np.float32)
    return embeddings, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mtranse", "jape", "bootea", "rdgcn"], required=True)
    parser.add_argument("--dataset", choices=["dbp15k_zh_en", "openea_en_fr"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/unified_existing_baselines_20260826")
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--attr-max-epoch", type=int)
    parser.add_argument("--start-valid", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--csls-k", default="3,5,7,10,15,20")
    args_cli = parser.parse_args()

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    tf = install_tensorflow_v1_compat(args_cli.seed)
    sys.path.insert(0, str(OPENEA_SRC))

    from openea.modules.load.kgs import read_kgs_from_folder

    method_args = load_method_args(args_cli.method)
    data_dir, division = dataset_spec(args_cli.dataset, args_cli.seed)
    method_args.training_data = str(data_dir) + "/"
    method_args.dataset_division = division
    method_args.output = str(ROOT / args_cli.output_dir / "official_outputs") + "/"
    method_args.is_save = False
    method_args.batch_threads_num = 1
    method_args.test_threads_num = 1
    if args_cli.max_epoch is not None:
        method_args.max_epoch = args_cli.max_epoch
    if args_cli.attr_max_epoch is not None:
        method_args.attr_max_epoch = args_cli.attr_max_epoch
    if args_cli.start_valid is not None:
        method_args.start_valid = args_cli.start_valid
    if args_cli.eval_freq is not None:
        method_args.eval_freq = args_cli.eval_freq

    kgs = read_kgs_from_folder(
        method_args.training_data,
        method_args.dataset_division,
        method_args.alignment_module,
        method_args.ordered,
    )
    model = build_model(args_cli.method)
    model.set_args(method_args)
    model.set_kgs(kgs)
    method_variant = args_cli.method
    initialization = None
    if args_cli.method == "rdgcn":
        if args_cli.dataset == "openea_en_fr":
            install_rdgcn_random_initializer(model, args_cli.seed)
            method_variant = "rdgcn_random_init"
            initialization = "identity-independent Glorot random entity vectors"
        else:
            model.word_embed = str(ROOT / "data" / "dbp15k" / "raw" / "sub.glove.300d")
            initialization = "GloVe name vectors with zero vectors for uncovered names"

    started = time.time()
    model.init()
    model.run()
    elapsed = time.time() - started

    embeddings, mapping = extract_embeddings(model, args_cli.method)
    candidate_right_ids = sorted(kgs.kg2.entities_list)
    csls_candidates = sorted(
        {int(value) for value in args_cli.csls_k.split(",") if value.strip()}
    )
    selected, validation_grid = select_csls(
        embeddings,
        mapping,
        kgs.valid_links,
        candidate_right_ids,
        csls_candidates,
    )
    raw_test = evaluate_pairs(
        embeddings,
        mapping,
        kgs.test_links,
        candidate_right_ids,
        0,
    )
    test = evaluate_pairs(
        embeddings,
        mapping,
        kgs.test_links,
        candidate_right_ids,
        selected["k"],
    )

    result = {
        "method": args_cli.method,
        "method_variant": method_variant,
        "dataset": args_cli.dataset,
        "seed": args_cli.seed,
        "implementation": "OpenEA official source with TensorFlow 1 compatibility",
        "openea_commit": "b59e014153c27c7166d78475e3474c7e86a10be9",
        "candidate_protocol": "all target-KG entities",
        "tie_handling": "average rank for exact score ties",
        "initialization": initialization,
        "train_pairs": len(kgs.train_links),
        "validation_pairs": len(kgs.valid_links),
        "test_pairs": len(kgs.test_links),
        "candidate_right_entities": len(candidate_right_ids),
        "max_epoch": method_args.max_epoch,
        "attr_max_epoch": getattr(method_args, "attr_max_epoch", None),
        "selected_csls_k": selected["k"],
        "validation_grid": validation_grid,
        "raw_test": raw_test,
        "test": test,
        "elapsed_seconds": elapsed,
    }
    output_dir = ROOT / args_cli.output_dir / args_cli.method
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args_cli.dataset}_seed{args_cli.seed}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BASELINE_RESULT_JSON=" + json.dumps(result, ensure_ascii=False))
    print(output_path)

    for session_name in ("session", "sess"):
        session = getattr(model, session_name, None)
        if session is not None:
            session.close()
    tf.reset_default_graph()


if __name__ == "__main__":
    main()
