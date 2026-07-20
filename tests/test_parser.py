import math

import pytest

from locate_anything_service.parser import (
    parse_grasp_output,
    parse_grasp_rect_output,
    parse_output,
)


def test_parse_box_coordinates() -> None:
    output = "<ref>red cup</ref><box><100><200><700><800></box>"
    parsed = parse_output(output, image_width=1920, image_height=1080)

    assert len(parsed.boxes) == 1
    assert parsed.boxes[0].label == "red cup"
    assert parsed.boxes[0].normalized == (0.1, 0.2, 0.7, 0.8)
    assert parsed.boxes[0].pixels == (192, 216, 1344, 864)


def test_parse_point_encoded_with_box_tokens() -> None:
    output = "<ref>button</ref><box><500><250></box>"
    parsed = parse_output(output, image_width=1000, image_height=800)

    assert len(parsed.points) == 1
    assert parsed.points[0].label == "button"
    assert parsed.points[0].pixels == (500, 200)


def test_parser_clamps_coordinates() -> None:
    parsed = parse_output("<box><-10><0><1100><1200></box>", 100, 100)

    assert parsed.boxes[0].normalized == (0.0, 0.0, 1.0, 1.0)


def test_coordinate_one_uses_thousand_point_scale() -> None:
    parsed = parse_output("<box><1><1><500><500></box>", 1000, 1000)

    assert parsed.boxes[0].normalized == (0.001, 0.001, 0.5, 0.5)
    assert parsed.boxes[0].pixels == (1, 1, 500, 500)


def test_parse_grasp_uses_x1_y1_x2_y2_slot_order() -> None:
    output = "<ref>grasp</ref><grasp><100><250><700><900></grasp>"
    parsed = parse_grasp_output(output, image_width=2000, image_height=1000)

    assert parsed.status == "ok"
    assert parsed.error is None
    assert len(parsed.grasps) == 1
    grasp = parsed.grasps[0]
    assert grasp.contacts_pixels == (200, 250, 1400, 900)
    assert grasp.center_pixels == (800, 575)
    assert math.isclose(
        grasp.opening_width_pixels, math.hypot(1200, 650), rel_tol=1e-9
    )


def test_parse_grasp_none_is_distinct_from_invalid() -> None:
    none_result = parse_grasp_output("<grasp>none</grasp>", 640, 480)
    invalid_result = parse_grasp_output("no geometry", 640, 480)

    assert none_result.status == "none"
    assert none_result.grasps == []
    assert invalid_result.status == "invalid"
    assert invalid_result.error is not None


def test_parse_grasp_rejects_multiple_or_mixed_blocks() -> None:
    multiple = parse_grasp_output(
        "<grasp><1><2><3><4></grasp><grasp><5><6><7><8></grasp>",
        100,
        100,
    )
    mixed = parse_grasp_output(
        "<grasp>none</grasp><grasp><5><6><7><8></grasp>", 100, 100
    )

    assert multiple.status == "invalid"
    assert mixed.status == "invalid"


def test_parse_grasp_rejects_legacy_box_and_trailing_coordinates() -> None:
    legacy = parse_grasp_output("<box><1><2><3><4></box>", 100, 100)
    trailing = parse_grasp_output(
        "<ref>grasp</ref><grasp><1><2><3><4></grasp><500>", 100, 100
    )

    assert legacy.status == "invalid"
    assert trailing.status == "invalid"


def test_parse_grasp_rejects_out_of_range_and_coincident_contacts() -> None:
    out_of_range = parse_grasp_output(
        "<grasp><-1><2><3><4></grasp>", 100, 100
    )
    coincident = parse_grasp_output(
        "<grasp><10><20><10><20></grasp>", 100, 100
    )

    assert out_of_range.status == "invalid"
    assert coincident.status == "invalid"


def test_parse_grasp_rect_decodes_center_angle_width_and_points() -> None:
    parsed = parse_grasp_rect_output(
        "<ref>grasp pose</ref>"
        "<grasp_rect><500><250><0><400></grasp_rect>",
        200,
        100,
    )

    assert parsed.status == "ok"
    assert parsed.error is None
    assert len(parsed.rectangles) == 1
    rectangle = parsed.rectangles[0]
    assert rectangle.label == "grasp pose"
    assert rectangle.center_pixels == (100, 25)
    assert rectangle.angle_degrees_image == 0
    assert rectangle.opening_width_pixels == pytest.approx(0.4 * math.hypot(200, 100))
    assert len(rectangle.rectangle_points_pixels) == 8


def test_parse_grasp_rect_none_is_distinct_from_invalid() -> None:
    none_result = parse_grasp_rect_output(
        "<grasp_rect>none</grasp_rect>", 100, 100
    )
    invalid_result = parse_grasp_rect_output(
        "<grasp_rect><500><500><0><0></grasp_rect>", 100, 100
    )

    assert none_result.status == "none"
    assert invalid_result.status == "invalid_geometry"
    assert "exceed" in (invalid_result.error or "")


def test_parse_grasp_rect_rejects_contact_or_trailing_coordinates() -> None:
    legacy = parse_grasp_rect_output(
        "<grasp><500><500><0><400></grasp>", 100, 100
    )
    trailing = parse_grasp_rect_output(
        "<grasp_rect><500><500><0><400></grasp_rect><9>", 100, 100
    )

    assert legacy.status == "invalid_structure"
    assert trailing.status == "invalid_structure"
