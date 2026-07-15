import pytest
from PIL import Image, ImageDraw

from locate_anything_service.collision_2d import evaluate_collision_2d
from locate_anything_service.grasp_geometry import derive_grasp_geometry


def _mask(size=(100, 100), box=None):
    mask = Image.new("L", size, 0)
    if box is not None:
        ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def test_collision_detects_obstacle_in_grasp_rectangle() -> None:
    geometry = derive_grasp_geometry((100, 500, 900, 500), 100, 100)
    result = evaluate_collision_2d(
        geometry,
        _mask(box=(45, 45, 55, 55)),
        100,
        100,
        thickness_pixels=20,
    )

    assert result.status == "collision"
    assert result.collision_ratio is not None
    assert result.collision_ratio > 0
    assert result.outside_ratio == 0


def test_collision_ratio_uses_rasterized_polygon_as_denominator() -> None:
    geometry = derive_grasp_geometry((100, 500, 900, 500), 100, 100)
    result = evaluate_collision_2d(
        geometry,
        _mask(box=(0, 0, 49, 99)),
        100,
        100,
        thickness_pixels=20,
    )

    # PIL rasterization includes both polygon boundaries: 40*21 / (81*21).
    assert result.collision_ratio == pytest.approx(840 / 1701)


def test_collision_returns_unknown_without_reliable_mask() -> None:
    geometry = derive_grasp_geometry((100, 500, 900, 500), 100, 100)
    result = evaluate_collision_2d(
        geometry, None, 100, 100, valid=False, detail="missing instances"
    )

    assert result.status == "unknown"
    assert result.detail == "missing instances"


def test_collision_counts_outside_image_as_collision() -> None:
    geometry = derive_grasp_geometry((0, 50, 1000, 50), 100, 100)
    result = evaluate_collision_2d(
        geometry,
        _mask(),
        100,
        100,
        thickness_pixels=40,
    )

    assert result.status == "collision"
    assert result.outside_ratio is not None
    assert result.outside_ratio > 0


def test_collision_rejects_mask_with_wrong_image_size() -> None:
    geometry = derive_grasp_geometry((100, 500, 900, 500), 100, 100)

    try:
        evaluate_collision_2d(geometry, _mask((50, 50)), 100, 100)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("expected mask size error")
