import math

from locate_anything_service.parser import parse_grasp_output, parse_output


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
    output = "<ref>grasp</ref><box><100><250><700><900></box>"
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
    none_result = parse_grasp_output("<box>none</box>", 640, 480)
    invalid_result = parse_grasp_output("no geometry", 640, 480)

    assert none_result.status == "none"
    assert none_result.grasps == []
    assert invalid_result.status == "invalid"
    assert invalid_result.error is not None


def test_parse_grasp_rejects_multiple_or_mixed_blocks() -> None:
    multiple = parse_grasp_output(
        "<box><1><2><3><4></box><box><5><6><7><8></box>", 100, 100
    )
    mixed = parse_grasp_output(
        "<box>none</box><box><5><6><7><8></box>", 100, 100
    )

    assert multiple.status == "invalid"
    assert mixed.status == "invalid"


def test_parse_grasp_rejects_out_of_range_and_coincident_contacts() -> None:
    out_of_range = parse_grasp_output(
        "<box><-1><2><3><4></box>", 100, 100
    )
    coincident = parse_grasp_output(
        "<box><10><20><10><20></box>", 100, 100
    )

    assert out_of_range.status == "invalid"
    assert coincident.status == "invalid"
