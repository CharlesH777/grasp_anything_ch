from locate_anything_service.parser import parse_output


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
