from __future__ import annotations

from typing import Literal

PromptMode = Literal[
    "raw",
    "detect",
    "ground_single",
    "ground_multi",
    "ground_text",
    "detect_text",
    "gui_box",
    "gui_point",
    "point",
]

def build_prompt(query: str, mode: PromptMode = "ground_single") -> str:
    cleaned = query.strip()
    if mode == "detect_text":
        return "Detect all the text in box format."
    if not cleaned:
        raise ValueError(f"query is required for mode={mode}")
    if mode == "raw":
        return cleaned
    if mode == "detect":
        categories = "</c>".join(
            category.strip() for category in cleaned.split(",") if category.strip()
        )
        return (
            "Locate all the instances that matches the following description: "
            f"{categories}."
        )
    if mode == "ground_single":
        return (
            "Locate a single instance that matches the following description: "
            f"{cleaned}."
        )
    if mode == "ground_multi":
        return (
            "Locate all the instances that match the following description: "
            f"{cleaned}."
        )
    if mode == "ground_text":
        return f"Please locate the text referred as {cleaned}."
    if mode == "gui_box":
        return f"Locate the region that matches the following description: {cleaned}."
    if mode in {"gui_point", "point"}:
        return f"Point to: {cleaned}."
    raise ValueError(f"unsupported mode: {mode}")
