import pytest

from locate_anything_service.prompts import build_prompt


def test_grounding_prompt_matches_official_template() -> None:
    assert build_prompt("red car", "ground_single") == (
        "Locate a single instance that matches the following description: red car."
    )


def test_detection_categories_use_separator() -> None:
    assert build_prompt("person, car", "detect") == (
        "Locate all the instances that matches the following description: "
        "person</c>car."
    )


def test_text_detection_does_not_require_query() -> None:
    assert build_prompt("", "detect_text") == "Detect all the text in box format."


def test_empty_grounding_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_prompt("", "ground_single")
