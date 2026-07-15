#!/usr/bin/env python3
"""Test ground_multi mode hallucination on absent classes."""
import argparse
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
BOX_RE = re.compile(r"<box>.*?</box>", re.DOTALL)
NONE_RE = re.compile(r"<box>\s*(?:None|none)\s*</box>", re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--voc-root", type=Path, required=True)
    p.add_argument("--image-ids", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_gt(xml_path):
    root = ET.parse(xml_path).getroot()
    labels = set()
    for obj in root.findall("object"):
        labels.add(obj.findtext("name", "").strip().lower())
    return labels


def main():
    args = parse_args()
    dataset = args.voc_root.expanduser().resolve() / "VOCdevkit" / "VOC2007"
    image_ids = [l.strip() for l in args.image_ids.read_text().splitlines() if l.strip()][:args.limit]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading model: {args.model_path}")
    model = AutoModel.from_pretrained(
        str(args.model_path), trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True, use_fast=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    results = {"model": str(args.model_path.name), "total": 0, "hallucinated": 0, "correct_none": 0, "errors": 0, "details": []}
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, image_id in enumerate(image_ids, 1):
        xml_path = dataset / "Annotations" / f"{image_id}.xml"
        img_path = dataset / "JPEGImages" / f"{image_id}.jpg"
        gt_classes = parse_gt(xml_path)
        absent = [c for c in VOC_CLASSES if c not in gt_classes]
        if not absent:
            continue
        query_class = random.choice(absent)

        with Image.open(img_path) as source:
            image = source.convert("RGB")
        prompt = f"Locate all the instances that match the following description: {query_class}."

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages) if hasattr(processor, "process_vision_info") else (None, None)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True).to("cuda")

        kwargs = dict(
            pixel_values=inputs["pixel_values"].to("cuda", dtype=torch.bfloat16),
            input_ids=inputs["input_ids"],
            tokenizer=processor.tokenizer,
            max_new_tokens=512,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            generation_mode="hybrid",
        )
        if inputs.get("attention_mask") is not None:
            kwargs["attention_mask"] = inputs["attention_mask"].to("cuda")
        if inputs.get("image_grid_hws") is not None:
            ig = inputs["image_grid_hws"]
            kwargs["image_grid_hws"] = ig if torch.is_tensor(ig) else torch.as_tensor(ig, device="cuda")
        if "hybrid" in kwargs.get("generation_mode", ""):
            kwargs["n_future_tokens"] = 6

        with torch.inference_mode():
            out = model.generate(**kwargs)
        raw = out[0] if isinstance(out, tuple) else out
        if isinstance(raw, list):
            raw = raw[0]
        if not isinstance(raw, str):
            raw = str(raw)

        has_box = bool(BOX_RE.search(raw))
        has_none = bool(NONE_RE.search(raw))
        is_hallucination = has_box and not has_none

        results["total"] += 1
        if is_hallucination:
            results["hallucinated"] += 1
        else:
            results["correct_none"] += 1

        if i <= 20:
            results["details"].append({
                "image": image_id, "query": query_class,
                "gt": sorted(gt_classes),
                "raw": raw[:300],
                "hallucination": is_hallucination,
            })

        if i % 50 == 0:
            print(f"[{i}/{len(image_ids)}] hallucinated={results['hallucinated']} none={results['correct_none']}", flush=True)
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    rate = results["hallucinated"] / max(1, results["total"]) * 100
    print(f"\nDone: {results['total']} images, hallucinated={results['hallucinated']} ({rate:.1f}%), correct_none={results['correct_none']}")


if __name__ == "__main__":
    main()
