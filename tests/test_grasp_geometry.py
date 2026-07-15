import math

from locate_anything_service.grasp_geometry import (
    angular_error_degrees,
    derive_grasp_geometry,
    grasp_rectangle,
    polygon_area,
    polygon_iou,
)


def test_angle_is_computed_in_original_pixel_aspect_ratio() -> None:
    geometry = derive_grasp_geometry((0, 0, 500, 500), 1600, 900)

    assert math.isclose(
        geometry.angle_radians_image,
        math.atan2(450, 800),
        rel_tol=1e-9,
    )
    assert not math.isclose(geometry.angle_radians_image, math.pi / 4)


def test_corrected_angle_error_does_not_guess_units() -> None:
    error = angular_error_degrees(math.radians(1), math.radians(2))

    assert math.isclose(error, 1.0, abs_tol=1e-9)


def test_angle_error_is_endpoint_exchange_invariant() -> None:
    assert math.isclose(
        angular_error_degrees(math.radians(10), math.radians(190)),
        0.0,
        abs_tol=1e-9,
    )


def test_grasp_rectangle_area_and_polygon_iou() -> None:
    rectangle = grasp_rectangle((10, 20, 110, 20), thickness_pixels=40)

    assert math.isclose(polygon_area(rectangle), 4000.0, rel_tol=1e-9)
    assert math.isclose(polygon_iou(rectangle, rectangle), 1.0, rel_tol=1e-9)
