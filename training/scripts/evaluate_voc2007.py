#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import time
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
BOX_PATTERN = re.compile(
    r"<ref>([^<]+)</ref>((?:<box>.*?</box>)+)", re.IGNORECASE | re.DOTALL
)
COORD_PATTERN = re.compile(
    r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-ids-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_ground_truth(annotation_path):
    root = ElementTree.parse(annotation_path).getroot()
    objects = []
    for element in root.findall("object"):
        label = element.findtext("name", "").strip().lower()
        box = element.find("bndbox")
        if label not in VOC_CLASSES or box is None:
            continue
        objects.append({
            "label": label,
            "bbox": [
                float(box.findtext("xmin")), float(box.findtext("ymin")),
                float(box.findtext("xmax")), float(box.findtext("ymax")),
            ],
            "difficult": int(element.findtext("difficult", "0")) == 1,
        })
    return objects


def normalize_label(label):
    aliases = {"airplane": "aeroplane", "motorcycle": "motorbike", "tv": "tvmonitor",
               "dining table": "diningtable", "potted plant": "pottedplant"}
    value = label.strip().lower().replace("_", " ")
    return aliases.get(value, value.replace(" ", "") if value.replace(" ", "") in VOC_CLASSES else value)


def parse_predictions(text, width, height):
    predictions = []
    for label, boxes_text in BOX_PATTERN.findall(text):
        label = normalize_label(label)
        if label not in VOC_CLASSES:
            continue
        for values in COORD_PATTERN.findall(boxes_text):
            x1, y1, x2, y2 = (float(value) for value in values)
            x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
            y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
            box = [x1 * width / 1000, y1 * height / 1000,
                   x2 * width / 1000, y2 * height / 1000]
            if box[2] > box[0] and box[3] > box[1]:
                predictions.append({"label": label, "bbox": box})
    return predictions


def apply_chat_template(processor, messages):
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def process_vision_info(processor, messages):
    return processor.process_vision_info(messages)


def decode_output(output, input_ids, processor):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output and isinstance(output[0], str):
        return output[0]
    if torch.is_tensor(output):
        generated = output[:, input_ids.shape[1]:].detach().cpu()
        if hasattr(processor, "post_process_image_text_to_text"):
            decoded = processor.post_process_image_text_to_text(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        else:
            decoded = processor.tokenizer.batch_decode(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        return decoded[0] if isinstance(decoded, list) else decoded
    return str(output)


def load_completed(output_path):
    completed = {}
    if not output_path.is_file():
        return completed
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed[row["image_id"]] = row
    return completed


def box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def voc_ap(recall, precision):
    recall = [0.0] + recall + [1.0]
    precision = [0.0] + precision + [0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return sum(
        (recall[index] - recall[index - 1]) * precision[index]
        for index in range(1, len(recall)) if recall[index] != recall[index - 1]
    )


def evaluate_at_iou(rows, threshold):
    class_results = {}
    total_tp = total_fp = total_gt = 0
    for label in VOC_CLASSES:
        gt_by_image = {}
        positive_count = 0
        detections = []
        sequence = 0
        for row in rows:
            ground_truth = [item for item in row["ground_truth"] if item["label"] == label]
            gt_by_image[row["image_id"]] = {
                "items": ground_truth,
                "matched": [False] * len(ground_truth),
            }
            positive_count += sum(not item["difficult"] for item in ground_truth)
            for prediction in row["predictions"]:
                if prediction["label"] == label:
                    detections.append((sequence, row["image_id"], prediction["bbox"]))
                    sequence += 1
        tp, fp = [], []
        for _, image_id, predicted_box in detections:
            record = gt_by_image[image_id]
            best_iou, best_index = 0.0, -1
            for index, item in enumerate(record["items"]):
                overlap = box_iou(predicted_box, item["bbox"])
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index >= 0 and best_iou >= threshold:
                item = record["items"][best_index]
                if item["difficult"]:
                    continue
                if not record["matched"][best_index]:
                    record["matched"][best_index] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)
        cumulative_tp, cumulative_fp = [], []
        running_tp = running_fp = 0
        for tp_value, fp_value in zip(tp, fp):
            running_tp += tp_value
            running_fp += fp_value
            cumulative_tp.append(running_tp)
            cumulative_fp.append(running_fp)
        recall = [value / positive_count for value in cumulative_tp] if positive_count else []
        precision = [
            tp_value / max(1, tp_value + fp_value)
            for tp_value, fp_value in zip(cumulative_tp, cumulative_fp)
        ]
        final_tp = cumulative_tp[-1] if cumulative_tp else 0
        final_fp = cumulative_fp[-1] if cumulative_fp else 0
        class_results[label] = {
            "ap": voc_ap(recall, precision) if positive_count else None,
            "precision": final_tp / max(1, final_tp + final_fp),
            "recall": final_tp / max(1, positive_count),
            "gt": positive_count,
            "predictions": len(detections),
            "tp": final_tp,
            "fp": final_fp,
        }
        total_tp += final_tp
        total_fp += final_fp
        total_gt += positive_count
    micro_precision = total_tp / max(1, total_tp + total_fp)
    micro_recall = total_tp / max(1, total_gt)
    valid_aps = [item["ap"] for item in class_results.values() if item["ap"] is not None]
    return {
        "map": sum(valid_aps) / len(valid_aps) if valid_aps else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": 2 * micro_precision * micro_recall / max(1e-12, micro_precision + micro_recall),
        "classes": class_results,
    }


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    rank = max(1, math.ceil(len(values) * fraction))
    return values[min(len(values) - 1, rank - 1)]


def main():
    args = parse_args()
    dataset_root = args.voc_root.expanduser().resolve() / "VOCdevkit" / "VOC2007"
    image_ids = [line.strip() for line in (dataset_root / "ImageSets/Main/test.txt").read_text().splitlines() if line.strip()]
    if args.image_ids_file:
        selected_ids = [
            line.strip() for line in args.image_ids_file.expanduser().read_text().splitlines()
            if line.strip()
        ]
        unknown_ids = sorted(set(selected_ids) - set(image_ids))
        if unknown_ids:
            raise ValueError(f"image IDs are not in VOC2007 test: {unknown_ids[:10]}")
        image_ids = selected_ids
    if args.limit:
        image_ids = image_ids[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    completed = load_completed(args.output)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModel.from_pretrained(
        str(args.model_path), trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True, use_fast=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    output_mode = "a" if args.output.exists() else "w"
    with args.output.open(output_mode, encoding="utf-8", buffering=1) as output_handle:
        for position, image_id in enumerate(image_ids, 1):
            if image_id in completed:
                continue
            image_path = dataset_root / "JPEGImages" / f"{image_id}.jpg"
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            ground_truth = parse_ground_truth(dataset_root / "Annotations" / f"{image_id}.xml")
            categories = list(dict.fromkeys(item["label"] for item in ground_truth))
            started = time.perf_counter()
            predictions = []
            raw_responses = {}
            for category_index, category in enumerate(categories):
                prompt = f"Locate all the instances that match the following description: {category}."
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt},
                ]}]
                torch.manual_seed(args.seed + int(image_id) * 100 + category_index)
                text = apply_chat_template(processor, messages)
                images, videos = process_vision_info(processor, messages)
                inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True)
                inputs = inputs.to("cuda")
                input_ids = inputs["input_ids"]
                generate_kwargs = {
                    "pixel_values": inputs["pixel_values"].to("cuda", dtype=torch.bfloat16),
                    "input_ids": input_ids,
                    "attention_mask": inputs.get("attention_mask").to("cuda"),
                    "tokenizer": processor.tokenizer,
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "generation_mode": args.generation_mode,
                }
                if inputs.get("image_grid_hws") is not None:
                    image_grid_hws = inputs["image_grid_hws"]
                    if not torch.is_tensor(image_grid_hws):
                        image_grid_hws = torch.as_tensor(image_grid_hws, device="cuda")
                    generate_kwargs["image_grid_hws"] = image_grid_hws
                if args.generation_mode in ("fast", "hybrid"):
                    generate_kwargs["n_future_tokens"] = 6
                with torch.inference_mode():
                    output = model.generate(**generate_kwargs)
                raw_response = decode_output(output, input_ids, processor)
                raw_responses[category] = raw_response
                predictions.extend(
                    item for item in parse_predictions(raw_response, width, height)
                    if item["label"] == category
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {
                "image_id": image_id,
                "image": str(image_path),
                "width": width,
                "height": height,
                "ground_truth": ground_truth,
                "predictions": predictions,
                "query_count": len(categories),
                "latency_seconds": elapsed,
                "raw_responses": raw_responses,
            }
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed[image_id] = row
            print(f"[{position}/{len(image_ids)}] {image_id} {elapsed:.3f}s boxes={len(row['predictions'])}", flush=True)

    rows = [completed[image_id] for image_id in image_ids]
    thresholds = [round(0.5 + index * 0.05, 2) for index in range(10)]
    evaluations = {str(threshold): evaluate_at_iou(rows, threshold) for threshold in thresholds}
    latencies = [row["latency_seconds"] for row in rows]
    measured = latencies[5:] if len(latencies) > 5 else latencies
    total_seconds = sum(measured)
    measured_rows = rows[5:] if len(rows) > 5 else rows
    measured_queries = sum(row.get("query_count", 1) for row in measured_rows)
    metrics = {
        "model_path": str(args.model_path.resolve()),
        "images": len(rows),
        "protocol": {
            "dataset": "VOC2007 test",
            "classes": list(VOC_CLASSES),
            "prompting": "one single-category grounding query for each GT-positive class per image",
            "scope": "positive-category conditional localization; absent classes are not queried",
            "generation_mode": args.generation_mode,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "official sampling (do_sample=True, temperature=0.7, top_p=0.9)",
            "confidence_note": "Model emits no confidence scores; detections use stable output order for AP ranking.",
        },
        "ap": {
            "map_50_95": sum(item["map"] for item in evaluations.values()) / len(evaluations),
            "ap50": evaluations["0.5"]["map"],
            "ap75": evaluations["0.75"]["map"],
        },
        "accuracy_iou50": {
            key: evaluations["0.5"][key]
            for key in ("micro_precision", "micro_recall", "micro_f1")
        },
        "per_class_ap50": {label: evaluations["0.5"]["classes"][label]["ap"] for label in VOC_CLASSES},
        "performance": {
            "warmup_images_excluded": min(5, len(latencies)),
            "measured_images": len(measured),
            "total_seconds": total_seconds,
            "images_per_second": len(measured) / total_seconds if total_seconds else 0.0,
            "queries_per_second": measured_queries / total_seconds if total_seconds else 0.0,
            "measured_queries": measured_queries,
            "mean_latency_seconds": total_seconds / len(measured) if measured else 0.0,
            "p50_latency_seconds": percentile(measured, 0.50),
            "p95_latency_seconds": percentile(measured, 0.95),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
        },
    }
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
