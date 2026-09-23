from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

JOBS = (
    (
        ROOT / "outputs" / "Relation_Aware_Neighbor_Context_Springer_Anti_Defensive_English_20260830.docx",
        ROOT / "outputs" / "paper_assets" / "figure_1_springer_submission_600dpi.png",
        ROOT / "outputs" / "Relation_Aware_Neighbor_Context_Springer_Anti_Defensive_English_20260830_Figure_ReRendered.docx",
    ),
    (
        ROOT / "outputs" / "跨语言实体对齐中的关系感知邻居上下文与语义引导选择_Springer去防御性中文译稿_20260830.docx",
        ROOT / "outputs" / "paper_assets" / "figure_1_springer_submission_zh_600dpi.png",
        ROOT / "outputs" / "跨语言实体对齐中的关系感知邻居上下文与语义引导选择_Springer去防御性中文译稿_20260830_图片重绘版.docx",
    ),
)


def replace_member(source: Path, figure: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    with zipfile.ZipFile(output, "r") as archive:
        media_members = [
            name for name in archive.namelist() if name.startswith("word/media/")
        ]
    if media_members != ["word/media/image1.png"]:
        raise RuntimeError(f"Unexpected media members in {source}: {media_members}")

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as handle:
        temporary = Path(handle.name)

    try:
        with zipfile.ZipFile(output, "r") as source_archive, zipfile.ZipFile(
            temporary, "w"
        ) as target_archive:
            for item in source_archive.infolist():
                data = (
                    figure.read_bytes()
                    if item.filename == "word/media/image1.png"
                    else source_archive.read(item.filename)
                )
                target_archive.writestr(item, data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    with zipfile.ZipFile(output, "r") as archive:
        embedded = archive.read("word/media/image1.png")
    if hashlib.sha256(embedded).digest() != hashlib.sha256(figure.read_bytes()).digest():
        raise RuntimeError(f"Figure replacement failed for {output}")


def main() -> None:
    for source, figure, output in JOBS:
        if not source.exists() or not figure.exists():
            raise FileNotFoundError(source if not source.exists() else figure)
        replace_member(source, figure, output)
        print(output)


if __name__ == "__main__":
    main()
