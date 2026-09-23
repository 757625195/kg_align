import argparse
import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_pairs(path: Path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            left, right = line.split("\t")
            pairs.append((left, right))
    return pairs


def write_pairs(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{left}\t{right}\n" for left, right in pairs),
        encoding="utf-8",
    )


def read_uri_to_id(path: Path):
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entity_id, uri = line.split("\t", 1)
            mapping[uri] = entity_id
    return mapping


def convert_attribute_presence(source: Path, target: Path, uri_to_id):
    triples = []
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or fields[0] not in uri_to_id:
            continue
        entity_id = uri_to_id[fields[0]]
        for attribute in fields[1:]:
            if attribute:
                triples.append((entity_id, attribute, "1"))
    target.write_text(
        "".join(f"{entity}\t{attribute}\t{value}\n" for entity, attribute, value in triples),
        encoding="utf-8",
    )
    return len(triples)


def prepare_dbp15k_zh_en(seeds, validation_ratio):
    source = ROOT / "data" / "dbp15k" / "zh_en"
    source_split = source / "0_3"
    target = ROOT / "data" / "baseline_unified" / "zh_en"
    target.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(source_split / "triples_1", target / "rel_triples_1")
    shutil.copyfile(source_split / "triples_2", target / "rel_triples_2")

    left_uri_to_id = read_uri_to_id(source_split / "ent_ids_1")
    right_uri_to_id = read_uri_to_id(source_split / "ent_ids_2")
    left_attributes = convert_attribute_presence(
        source / "training_attrs_1",
        target / "attr_triples_1",
        left_uri_to_id,
    )
    right_attributes = convert_attribute_presence(
        source / "training_attrs_2",
        target / "attr_triples_2",
        right_uri_to_id,
    )

    supervised_pairs = read_pairs(source_split / "sup_ent_ids")
    test_pairs = read_pairs(source_split / "ref_ent_ids")
    split_sizes = {}
    for seed in seeds:
        shuffled = list(supervised_pairs)
        random.Random(seed).shuffle(shuffled)
        validation_size = max(1, int(round(len(shuffled) * validation_ratio)))
        validation_size = min(validation_size, len(shuffled) - 1)
        valid_pairs = shuffled[:validation_size]
        train_pairs = shuffled[validation_size:]
        split_dir = target / f"seed{seed}"
        write_pairs(split_dir / "train_links", train_pairs)
        write_pairs(split_dir / "valid_links", valid_pairs)
        write_pairs(split_dir / "test_links", test_pairs)
        split_sizes[str(seed)] = {
            "train": len(train_pairs),
            "valid": len(valid_pairs),
            "test": len(test_pairs),
        }

    metadata = {
        "source": str(source_split.relative_to(ROOT)),
        "validation_ratio": validation_ratio,
        "splits": split_sizes,
        "attribute_presence_triples": {
            "left": left_attributes,
            "right": right_attributes,
        },
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target, metadata


def copy_dbp15k_for_rrea(seeds):
    source = ROOT / "data" / "dbp15k" / "zh_en" / "0_3"
    unified = ROOT / "data" / "baseline_unified" / "zh_en"
    target = ROOT / "data" / "baseline_unified" / "rrea" / "dbp15k_zh_en"
    target.mkdir(parents=True, exist_ok=True)

    def read_id_records(path):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_id, value = line.split("\t", 1)
                records.append((int(raw_id), value))
        return records

    left_records = read_id_records(source / "ent_ids_1")
    right_records = read_id_records(source / "ent_ids_2")
    left_entities = {raw_id: index for index, (raw_id, _) in enumerate(left_records)}
    right_offset = len(left_records)
    right_entities = {
        raw_id: right_offset + index
        for index, (raw_id, _) in enumerate(right_records)
    }
    left_relations = {
        raw_id: index
        for index, (raw_id, _) in enumerate(read_id_records(source / "rel_ids_1"))
    }
    right_rel_offset = len(left_relations)
    right_relations = {
        raw_id: right_rel_offset + index
        for index, (raw_id, _) in enumerate(read_id_records(source / "rel_ids_2"))
    }

    (target / "ent_ids_1").write_text(
        "".join(
            f"{left_entities[raw_id]}\t{uri}\n"
            for raw_id, uri in left_records
        ),
        encoding="utf-8",
    )
    (target / "ent_ids_2").write_text(
        "".join(
            f"{right_entities[raw_id]}\t{uri}\n"
            for raw_id, uri in right_records
        ),
        encoding="utf-8",
    )

    for side, entity_map, relation_map in (
        (1, left_entities, left_relations),
        (2, right_entities, right_relations),
    ):
        triples = []
        for line in (source / f"triples_{side}").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            head, relation, tail = [int(value) for value in line.split("\t")]
            triples.append(
                (entity_map[head], relation_map[relation], entity_map[tail])
            )
        (target / f"triples_{side}").write_text(
            "".join(
                f"{head}\t{relation}\t{tail}\n"
                for head, relation, tail in triples
            ),
            encoding="utf-8",
        )

    for seed in seeds:
        split_target = target / f"seed{seed}"
        split_target.mkdir(parents=True, exist_ok=True)
        for filename in ("train_links", "valid_links", "test_links"):
            pairs = read_pairs(unified / f"seed{seed}" / filename)
            mapped = [
                (left_entities[int(left)], right_entities[int(right)])
                for left, right in pairs
            ]
            write_pairs(split_target / filename, mapped)
    return target


def read_string_triples(path: Path):
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            triples.append(tuple(line.split("\t", 2)))
    return triples


def prepare_openea_for_rrea():
    source = ROOT / "data" / "openea" / "EN_FR_15K_V2"
    target = ROOT / "data" / "baseline_unified" / "rrea" / "openea_en_fr"
    target.mkdir(parents=True, exist_ok=True)

    left_triples = read_string_triples(source / "rel_triples_1")
    right_triples = read_string_triples(source / "rel_triples_2")
    train_links = read_pairs(source / "721_5fold" / "1" / "train_links")
    valid_links = read_pairs(source / "721_5fold" / "1" / "valid_links")
    test_links = read_pairs(source / "721_5fold" / "1" / "test_links")

    left_entities = {}
    right_entities = {}

    def add(mapping, value):
        if value not in mapping:
            mapping[value] = len(mapping)

    for head, _, tail in left_triples:
        add(left_entities, head)
        add(left_entities, tail)
    for head, _, tail in right_triples:
        add(right_entities, head)
        add(right_entities, tail)
    for left, right in train_links + valid_links + test_links:
        add(left_entities, left)
        add(right_entities, right)

    right_offset = len(left_entities)
    relation_ids = {}

    def relation_id(side, relation):
        key = (side, relation)
        if key not in relation_ids:
            relation_ids[key] = len(relation_ids)
        return relation_ids[key]

    (target / "ent_ids_1").write_text(
        "".join(f"{entity_id}\t{uri}\n" for uri, entity_id in left_entities.items()),
        encoding="utf-8",
    )
    (target / "ent_ids_2").write_text(
        "".join(
            f"{right_offset + entity_id}\t{uri}\n"
            for uri, entity_id in right_entities.items()
        ),
        encoding="utf-8",
    )
    (target / "triples_1").write_text(
        "".join(
            f"{left_entities[head]}\t{relation_id('left', relation)}\t{left_entities[tail]}\n"
            for head, relation, tail in left_triples
        ),
        encoding="utf-8",
    )
    (target / "triples_2").write_text(
        "".join(
            f"{right_offset + right_entities[head]}\t{relation_id('right', relation)}\t{right_offset + right_entities[tail]}\n"
            for head, relation, tail in right_triples
        ),
        encoding="utf-8",
    )

    split_target = target / "721_5fold_1"
    mapped_sets = {
        "train_links": train_links,
        "valid_links": valid_links,
        "test_links": test_links,
    }
    for filename, pairs in mapped_sets.items():
        mapped = [
            (left_entities[left], right_offset + right_entities[right])
            for left, right in pairs
        ]
        write_pairs(split_target / filename, mapped)

    metadata = {
        "left_entities": len(left_entities),
        "right_entities": len(right_entities),
        "relations": len(relation_ids),
        "train": len(train_links),
        "valid": len(valid_links),
        "test": len(test_links),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    target, metadata = prepare_dbp15k_zh_en(seeds, args.validation_ratio)
    print(target)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(copy_dbp15k_for_rrea(seeds))
    rrea_target, rrea_metadata = prepare_openea_for_rrea()
    print(rrea_target)
    print(json.dumps(rrea_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
