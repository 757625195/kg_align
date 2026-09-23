from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1] / "outputs"

JOBS = [
    (
        ROOT / "知识图谱实体对齐中的关系感知邻居上下文与语义引导选择_任务范围重写版_20260830.docx",
        ROOT / "知识图谱实体对齐中的关系感知邻居上下文与语义引导选择_删除轻量级表述版_20260830.docx",
        {
            "提出轻量的关系感知结构编码方法。": "提出关系感知结构编码方法。",
            "本文采用轻量的关系感知消息传递": "本文采用关系感知消息传递",
        },
    ),
    (
        ROOT / "Relation_Aware_Neighbor_Context_Knowledge_Graph_Entity_Alignment_Reframed_20260830.docx",
        ROOT / "Relation_Aware_Neighbor_Context_Knowledge_Graph_Entity_Alignment_No_Lightweight_Claim_20260830.docx",
        {
            "The paper presents a lightweight relation-aware structural encoder.": "The paper presents a relation-aware structural encoder.",
            "uses lightweight relation-aware message passing": "uses relation-aware message passing",
        },
    ),
]


def edit_document(source: Path, target: Path, replacements: dict[str, str]) -> None:
    temp = target.with_suffix(".tmp.docx")
    with ZipFile(source, "r") as src, ZipFile(temp, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "word/document.xml":
                for old, new in replacements.items():
                    old_bytes = old.encode("utf-8")
                    count = data.count(old_bytes)
                    if count != 1:
                        raise RuntimeError(
                            f"Expected one occurrence of {old!r} in {source.name}; found {count}"
                        )
                    data = data.replace(old_bytes, new.encode("utf-8"), 1)
            dst.writestr(info, data)

    temp.replace(target)

    with ZipFile(target, "r") as saved:
        xml = saved.read("word/document.xml")
        if "轻量".encode("utf-8") in xml or b"lightweight" in xml.lower():
            raise RuntimeError(f"Lightweight wording remains in {target.name}")
    print(f"saved: {target}")


for source, target, replacements in JOBS:
    edit_document(source, target, replacements)
