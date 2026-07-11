#!/usr/bin/env python3
import argparse
import json
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path


VOC_YEARS = ("2007", "2012")
VOC_SPLITS = ("train", "val", "trainval", "test")


def normalize_min(value: int, size: int) -> int:
    return max(0, min(1000, round((value - 1) * 1000 / size)))


def normalize_max(value: int, size: int) -> int:
    return max(0, min(1000, round(value * 1000 / size)))


def parse_annotation(annotation_path: Path) -> tuple[int, int, list[dict]]:
    root = ElementTree.parse(annotation_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing image size: {annotation_path}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {annotation_path}: {width}x{height}")

    objects = []
    for object_element in root.findall("object"):
        label = object_element.findtext("name", "").strip()
        box = object_element.find("bndbox")
        if not label or box is None:
            continue

        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))
        if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
            raise ValueError(
                f"invalid box in {annotation_path}: {(xmin, ymin, xmax, ymax)} for {width}x{height}"
            )

        objects.append(
            {
                "label": label,
                "box": (
                    normalize_min(xmin, width),
                    normalize_min(ymin, height),
                    normalize_max(xmax, width),
                    normalize_max(ymax, height),
                ),
            }
        )

    return width, height, objects


def format_answer(objects: list[dict]) -> str:
    return "".join(
        f"<ref>{item['label']}</ref><box><{item['box'][0]}><{item['box'][1]}><{item['box'][2]}><{item['box'][3]}></box>"
        for item in objects
    )


def make_sample(image_path: str, prompt: str, objects: list[dict]) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": format_answer(objects)},
        ],
        "image": image_path,
    }


def convert(voc_root: Path, output_path: Path, split: str) -> dict:
    statistics = Counter()
    class_counts = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as output_handle:
        for year in VOC_YEARS:
            dataset_root = voc_root / "VOCdevkit" / f"VOC{year}"
            split_path = dataset_root / "ImageSets" / "Main" / f"{split}.txt"
            if not split_path.is_file():
                if split == "test":
                    continue
                raise FileNotFoundError(f"missing VOC{year} {split} split: {split_path}")

            image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for image_id in image_ids:
                annotation_path = dataset_root / "Annotations" / f"{image_id}.xml"
                image_absolute_path = dataset_root / "JPEGImages" / f"{image_id}.jpg"
                if not image_absolute_path.is_file():
                    raise FileNotFoundError(f"missing image: {image_absolute_path}")

                _, _, objects = parse_annotation(annotation_path)
                if not objects:
                    statistics["empty_images"] += 1
                    continue

                image_relative_path = image_absolute_path.relative_to(voc_root).as_posix()
                detection_sample = make_sample(
                    image_relative_path,
                    "Detect all objects in <image-1>.",
                    objects,
                )
                output_handle.write(json.dumps(detection_sample, ensure_ascii=False) + "\n")
                statistics["detection_samples"] += 1

                labels_in_order = list(dict.fromkeys(item["label"] for item in objects))
                for label in labels_in_order:
                    label_objects = [item for item in objects if item["label"] == label]
                    grounding_sample = make_sample(
                        image_relative_path,
                        f"Locate all the instances that match the following description: {label}.",
                        label_objects,
                    )
                    output_handle.write(json.dumps(grounding_sample, ensure_ascii=False) + "\n")
                    statistics["grounding_samples"] += 1
                    class_counts[label] += 1

                statistics[f"voc{year}_images"] += 1
                statistics["objects"] += len(objects)

    temporary_path.replace(output_path)
    statistics["split"] = split
    statistics["total_samples"] = statistics["detection_samples"] + statistics["grounding_samples"]
    return {"statistics": dict(statistics), "class_grounding_samples": dict(sorted(class_counts.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert one Pascal VOC split to LocateAnything JSONL.")
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=VOC_SPLITS, required=True)
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()

    voc_root = args.voc_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    result = convert(voc_root, output_path, args.split)

    if args.stats:
        stats_path = args.stats.expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result["statistics"], ensure_ascii=False, sort_keys=True))
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
