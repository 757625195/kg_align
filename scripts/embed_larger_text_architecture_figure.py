from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE = (
    ROOT
    / "outputs"
    / "paper_assets"
    / "figure_1_publication_architecture_padded_600dpi.png"
)

JOBS = (
    (
        ROOT
        / "outputs"
        / "知识图谱实体对齐中的关系感知邻居上下文与语义引导选择_删除轻量级表述版_20260830.docx",
        ROOT
        / "outputs"
        / "知识图谱实体对齐中的关系感知邻居上下文与语义引导选择_删除轻量级表述版_图1留白协调版_20260830.docx",
    ),
    (
        ROOT
        / "outputs"
        / "Relation_Aware_Neighbor_Context_Knowledge_Graph_Entity_Alignment_No_Lightweight_Claim_20260830.docx",
        ROOT
        / "outputs"
        / "Relation_Aware_Neighbor_Context_Knowledge_Graph_Entity_Alignment_No_Lightweight_Claim_Figure1_Padded_Balanced_20260830.docx",
    ),
)


def replace_figure(source: Path, output: Path) -> None:
    with zipfile.ZipFile(source, "r") as archive:
        media = sorted(name for name in archive.namelist() if name.startswith("word/media/"))
    if media != ["word/media/image1.png"]:
        raise RuntimeError(f"Unexpected media members in {source.name}: {media}")

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as handle:
        temporary = Path(handle.name)

    try:
        with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(temporary, "w") as dst:
            for item in src.infolist():
                data = FIGURE.read_bytes() if item.filename == "word/media/image1.png" else src.read(item.filename)
                dst.writestr(item, data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    with zipfile.ZipFile(output, "r") as archive:
        embedded = archive.read("word/media/image1.png")
    if hashlib.sha256(embedded).digest() != hashlib.sha256(FIGURE.read_bytes()).digest():
        raise RuntimeError(f"Figure replacement failed for {output.name}")


for source, output in JOBS:
    replace_figure(source, output)
    print(output)
