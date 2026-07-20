import math

import pytest

from locate_anything_service.grasp_geometry import polygon_area
from locate_anything_service.grasp_rect_geometry import (
    canonical_angle_degrees,
    decode_angle_token,
    derive_grasp_rectangle_geometry,
    encode_angle_token,
    encode_grasp_rectangle_pixels,
    points8_to_rect,
    rect_to_points8,
)


def test_rect_to_points8_matches_realvlg_corner_order() -> None:
    points = rect_to_points8(50, 50, 0, 40, 20)

    assert points == pytest.approx((30, 60, 70, 60, 70, 40, 30, 40))
    polygon = tuple((points[index], points[index + 1]) for index in range(0, 8, 2))
    assert polygon_area(polygon) == pytest.approx(800.0)


def test_points8_round_trip_is_pi_periodic() -> None:
    points = rect_to_points8(80, 60, 217.5, 50, 40)
    center_x, center_y, theta, width = points8_to_rect(points)

    assert center_x == pytest.approx(80.0)
    assert center_y == pytest.approx(60.0)
    assert theta == pytest.approx(37.5)
    assert width == pytest.approx(50.0)
    assert rect_to_points8(80, 60, 37.5, 50, 40) == pytest.approx(points)


def test_angle_tokens_form_a_1001_bin_circle() -> None:
    assert encode_angle_token(0) == 0
    assert encode_angle_token(180) == 0
    assert encode_angle_token(-180) == 0
    assert canonical_angle_degrees(-0.1) == pytest.approx(179.9)
    assert decode_angle_token(1000) == pytest.approx(180000 / 1001)
    assert encode_angle_token(decode_angle_token(1000)) == 1000


def test_token_geometry_uses_image_diagonal_width() -> None:
    geometry = derive_grasp_rectangle_geometry((500, 500, 0, 400), 100, 100)

    assert geometry.center_pixels == (50, 50)
    assert geometry.angle_degrees_image == 0
    assert geometry.opening_width_diagonal_normalized == pytest.approx(0.4)
    assert geometry.opening_width_pixels == pytest.approx(0.4 * math.sqrt(20000))
    assert len(geometry.rectangle_points_pixels) == 8


def test_width_must_strictly_exceed_minimum() -> None:
    with pytest.raises(ValueError, match="exceed"):
        derive_grasp_rectangle_geometry(
            (500, 500, 0, 1),
            100,
            100,
            minimum_width_diagonal=0.001,
        )

    geometry = derive_grasp_rectangle_geometry(
        (500, 500, 0, 1),
        100,
        100,
        minimum_width_diagonal=0.0009,
    )
    assert geometry.opening_width_diagonal_normalized == pytest.approx(0.001)


def test_pixel_encoding_rejects_unrepresentable_width_without_clipping() -> None:
    with pytest.raises(ValueError, match="outside the token representation"):
        encode_grasp_rectangle_pixels(50, 50, 0, 150, 100, 100)


def test_rectangle_points_are_not_clipped_to_the_image() -> None:
    geometry = derive_grasp_rectangle_geometry((0, 0, 0, 500), 100, 100)

    assert min(geometry.rectangle_points_pixels_float) < 0
