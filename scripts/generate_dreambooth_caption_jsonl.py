from __future__ import annotations

import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "datasets" / "dreambooth"
    n1_manifest_path = repo_root / "datasets" / "dreambooth_n1.json"
    class_manifest_path = repo_root / "datasets" / "dreambooth.json"

    n1_manifest = json.loads(n1_manifest_path.read_text())
    class_manifest = json.loads(class_manifest_path.read_text())

    concept_dirs = sorted(
        p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    created = 0
    for concept_dir in concept_dirs:
        concept = concept_dir.name
        images = sorted([p for p in concept_dir.iterdir() if is_image(p)])
        if not images:
            continue

        n1_entry = n1_manifest.get(concept, {})
        train_path_rel = n1_entry.get("path", "")
        train_path_abs = (
            (repo_root / train_path_rel).resolve() if train_path_rel else None
        )

        train_idx = None
        if train_path_abs is not None:
            for idx, img in enumerate(images):
                if img.resolve() == train_path_abs:
                    train_idx = idx
                    break

        if train_idx is None:
            train_idx = 0

        class_name = n1_entry.get("class") or class_manifest.get(concept, {}).get(
            "class", ""
        )

        records = []
        rest = [i for i in range(len(images)) if i != train_idx]
        val_idx = rest[0] if rest else None

        for idx, img in enumerate(images):
            if idx == train_idx:
                split = "train"
            elif idx == val_idx:
                split = "val"
            else:
                split = "test"

            rel_path = img.relative_to(repo_root).as_posix()
            records.append(
                {
                    "instance": concept,
                    "image_path": rel_path,
                    "file_name": img.name,
                    "split": split,
                    "class_name": class_name,
                    "template": "IMAGENET_SMALL",
                    "captions": [
                        {"source": "human", "text": ""},
                        {"source": "model_blip2", "text": ""},
                        {"source": "model_llava", "text": ""},
                    ],
                }
            )

        out_path = concept_dir / "captions.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        created += 1

    print(f"Created {created} captions.jsonl files.")


if __name__ == "__main__":
    main()
