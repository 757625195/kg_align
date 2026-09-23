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
RREA_ROOT = ROOT / "third_party" / "RREA" / "CIKM"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.run_openea_baseline import evaluate_pairs, select_csls


def install_tensorflow_v1_compat(seed):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.set_random_seed(seed)
    sys.modules["tensorflow"] = tf
    return tf


def load_ids(path):
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(int(line.split("\t", 1)[0]))
    return ids


def load_pairs(path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            left, right = line.split("\t")
            pairs.append((int(left), int(right)))
    return pairs


def dataset_spec(name, seed):
    if name == "dbp15k_zh_en":
        root = ROOT / "data" / "baseline_unified" / "rrea" / "dbp15k_zh_en"
        return root, root / f"seed{seed}"
    if name == "openea_en_fr":
        root = ROOT / "data" / "baseline_unified" / "rrea" / "openea_en_fr"
        return root, root / "721_5fold_1"
    raise ValueError(f"Unsupported dataset: {name}")


def load_data(data_root, split_root):
    from utils import get_matrix, load_triples

    entity1, rel1, triples1 = load_triples(str(data_root / "triples_1"))
    entity2, rel2, triples2 = load_triples(str(data_root / "triples_2"))
    train_pairs = load_pairs(split_root / "train_links")
    valid_pairs = load_pairs(split_root / "valid_links")
    test_pairs = load_pairs(split_root / "test_links")
    matrices = get_matrix(
        triples1 + triples2,
        entity1.union(entity2),
        rel1.union(rel2),
    )
    return train_pairs, valid_pairs, test_pairs, matrices


def build_models(
    keras,
    tf,
    node_size,
    rel_size,
    triple_size,
    batch_size,
    node_hidden=100,
    depth=2,
    dropout_rate=0.30,
    gamma=3.0,
    learning_rate=0.005,
):
    from keras import backend as K
    from keras.layers import Concatenate, Dropout, Input, Lambda
    from layer import NR_GraphAttention

    class TokenEmbedding(keras.layers.Embedding):
        def compute_output_shape(self, input_shape):
            return self.input_dim, self.output_dim

        def compute_mask(self, inputs, mask=None):
            return None

        def call(self, inputs):
            return self.embeddings

    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype="int64")
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def average_features(tensors, size):
        adjacency = K.cast(K.squeeze(tensors[0], axis=0), dtype="int64")
        sparse = tf.SparseTensor(
            indices=adjacency,
            values=tf.ones_like(adjacency[:, 0], dtype="float32"),
            dense_shape=(node_size, size),
        )
        sparse = tf.sparse_softmax(sparse)
        return tf.sparse_tensor_dense_matmul(sparse, tensors[1])

    encoder_inputs = [rel_emb, adj_input, index_input, val_input]
    ent_feature = Lambda(average_features, arguments={"size": node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(average_features, arguments={"size": rel_size})([rel_adj, rel_emb])
    encoder = NR_GraphAttention(
        node_size,
        activation="relu",
        rel_size=rel_size,
        depth=depth,
        attn_heads=1,
        triple_size=triple_size,
        attn_heads_reduction="average",
        dropout_rate=dropout_rate,
    )
    output = Concatenate(-1)(
        [encoder([ent_feature] + encoder_inputs), encoder([rel_feature] + encoder_inputs)]
    )
    output = Dropout(dropout_rate)(output)

    alignment_input = Input(shape=(None, 4))
    selected = Lambda(
        lambda tensors: K.gather(
            reference=tensors[0],
            indices=K.cast(K.squeeze(tensors[1], axis=0), "int32"),
        )
    )([output, alignment_input])

    def alignment_loss(tensor):
        left, right, false_left, false_right = [
            tensor[:, 0, :],
            tensor[:, 1, :],
            tensor[:, 2, :],
            tensor[:, 3, :],
        ]
        l1 = lambda first, second: K.sum(
            K.abs(first - second), axis=-1, keepdims=True
        )
        loss = K.relu(gamma + l1(left, right) - l1(left, false_right))
        loss += K.relu(gamma + l1(left, right) - l1(false_left, right))
        return tf.reduce_sum(loss, keepdims=True) / batch_size

    loss = Lambda(alignment_loss)(selected)
    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    optimizer = keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer)
    feature_model = keras.Model(inputs=inputs, outputs=output)
    return train_model, feature_model


def training_batch(train_pairs, node_size, batch_size):
    negative_ratio = batch_size // len(train_pairs) + 1
    repeated = np.reshape(
        np.repeat(np.expand_dims(train_pairs, axis=0), axis=0, repeats=negative_ratio),
        newshape=(-1, 2),
    )
    np.random.shuffle(repeated)
    repeated = repeated[:batch_size]
    negatives = np.random.randint(0, node_size, repeated.shape)
    return np.concatenate([repeated, negatives], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dbp15k_zh_en", "openea_en_fr"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/unified_existing_baselines_20260826")
    parser.add_argument("--csls-k", default="3,5,7,10,15,20")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf = install_tensorflow_v1_compat(args.seed)
    sys.path.insert(0, str(RREA_ROOT))
    import keras

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session = tf.Session(config=session_config)
    tf.keras.backend.set_session(session)

    data_root, split_root = dataset_spec(args.dataset, args.seed)
    train_pairs, valid_pairs, test_pairs, matrices = load_data(data_root, split_root)
    adj_matrix, relation_index, relation_values, adjacency_features, relation_features = matrices
    adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
    relation_matrix = np.stack(relation_features.nonzero(), axis=1)
    relation_matrix_values = relation_features.data
    entity_matrix = np.stack(adjacency_features.nonzero(), axis=1)
    entity_matrix_values = adjacency_features.data
    node_size = adjacency_features.shape[0]
    relation_size = relation_features.shape[1]
    triple_size = len(adj_matrix)
    batch_size = node_size

    train_model, feature_model = build_models(
        keras,
        tf,
        node_size=node_size,
        rel_size=relation_size,
        triple_size=triple_size,
        batch_size=batch_size,
    )
    static_inputs = [
        adj_matrix,
        np.asarray(relation_index),
        np.asarray(relation_values),
        relation_matrix,
        entity_matrix,
    ]
    static_inputs = [np.expand_dims(value, axis=0) for value in static_inputs]

    def embeddings():
        return np.asarray(feature_model.predict_on_batch(static_inputs), dtype=np.float32)

    candidate_right_ids = sorted(load_ids(data_root / "ent_ids_2"))
    best_mrr = -1.0
    best_weights = None
    validation_history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        batch = training_batch(np.asarray(train_pairs), node_size, batch_size)
        inputs = static_inputs + [np.expand_dims(batch, axis=0)]
        loss = float(train_model.train_on_batch(inputs, np.zeros((1, 1))))
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            current_embeddings = embeddings()
            metrics = evaluate_pairs(
                current_embeddings,
                None,
                valid_pairs,
                candidate_right_ids,
                0,
            )
            validation_history.append({"epoch": epoch, "loss": loss, **metrics})
            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                best_weights = train_model.get_weights()
            print(
                f"epoch={epoch} loss={loss:.6f} "
                f"val_hits1={metrics['Hits@1']:.6f} val_mrr={metrics['MRR']:.6f}",
                flush=True,
            )

    if best_weights is not None:
        train_model.set_weights(best_weights)
    final_embeddings = embeddings()
    csls_candidates = sorted(
        {int(value) for value in args.csls_k.split(",") if value.strip()}
    )
    selected, validation_grid = select_csls(
        final_embeddings,
        None,
        valid_pairs,
        candidate_right_ids,
        csls_candidates,
    )
    raw_test = evaluate_pairs(
        final_embeddings, None, test_pairs, candidate_right_ids, 0
    )
    test = evaluate_pairs(
        final_embeddings, None, test_pairs, candidate_right_ids, selected["k"]
    )
    result = {
        "method": "rrea_basic",
        "dataset": args.dataset,
        "seed": args.seed,
        "implementation": "RREA official layer and training objective with explicit validation checkpointing",
        "rrea_commit": "2271ac33dae0baf53dfa5b7ca1955090a1567a0a",
        "candidate_protocol": "all target-KG entities",
        "train_pairs": len(train_pairs),
        "validation_pairs": len(valid_pairs),
        "test_pairs": len(test_pairs),
        "candidate_right_entities": len(candidate_right_ids),
        "epochs": args.epochs,
        "selected_csls_k": selected["k"],
        "validation_history": validation_history,
        "validation_grid": validation_grid,
        "raw_test": raw_test,
        "test": test,
        "elapsed_seconds": time.time() - started,
    }
    output_dir = ROOT / args.output_dir / "rrea_basic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_seed{args.seed}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BASELINE_RESULT_JSON=" + json.dumps(result, ensure_ascii=False))
    print(output_path)
    session.close()


if __name__ == "__main__":
    main()
