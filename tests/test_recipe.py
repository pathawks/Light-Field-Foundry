from __future__ import annotations

import copy
from typing import Any

import pytest

from lytro_avif.errors import LytroError
from lytro_avif.recipe import GEOMETRY, resolve_recipe


def source_recipe() -> dict:
    return {
        **copy.deepcopy(GEOMETRY),
        "viewFocus": 7,
        "viewAperture": 1,
        "viewPerspectiveU": 0.2,
        "viewPerspectiveV": 0.1,
        "viewPivot": -20,
        "viewStereoPivot": 3,
        "viewStereoBaseline": 0.2,
        "viewExposure": 0,
        "viewPriorities": ["viewFocus"],
        "viewFocusAnimation": {"values": [], "times": [], "handlePairs": []},
    }


def test_cli_explicit_zero_wins_and_stereo_pivot_is_used_for_all_stills() -> None:
    source = source_recipe()
    resolved = resolve_recipe(
        source,
        {"viewFocus": 2, "viewStereoPivot": 5},
        perspective_u=0,
        perspective_v=0,
        baseline=0.2,
        focus=0,
        stereo_pivot=0,
        aperture=0,
    )
    assert resolved.focus == 0
    assert resolved.stereo_pivot == 0
    assert resolved.center["viewAperture"] == 0
    assert resolved.center["viewPerspectiveU"] == 0
    assert resolved.left["viewPerspectiveU"] < 0 < resolved.right["viewPerspectiveU"]
    for recipe in resolved.views.values():
        assert recipe["viewPivot"] == recipe["viewStereoPivot"] == 0
        assert recipe["viewFocus"] == 0
        assert recipe["viewStereoBaseline"] == 0
    assert source == source_recipe()


def test_omitted_controls_use_saved_recipe_then_source() -> None:
    resolved = resolve_recipe(
        source_recipe(),
        {"viewFocus": 2, "viewStereoPivot": 5},
        perspective_u=0,
        perspective_v=0,
        baseline=0.1,
    )
    assert resolved.focus == 2
    assert resolved.stereo_pivot == 5
    assert resolved.center["viewPivot"] == 5


def test_edof_center_is_default_without_changing_eye_controls() -> None:
    resolved = resolve_recipe(source_recipe(), perspective_u=0, perspective_v=0, baseline=0.2)
    assert resolved.edof_center
    assert resolved.provenance["center_render"]["mode"] == "EDOF"
    assert not resolved.provenance["center_render"]["focus_and_aperture_applied"]
    assert resolved.left["viewFocus"] == resolved.right["viewFocus"] == 7
    assert resolved.left["viewAperture"] == resolved.right["viewAperture"] == 1


def test_refocused_center_is_explicit() -> None:
    resolved = resolve_recipe(
        source_recipe(), perspective_u=0, perspective_v=0, baseline=0.2, edof_center=False
    )
    assert not resolved.edof_center
    assert resolved.provenance["center_render"]["mode"] == "refocused"
    assert resolved.provenance["center_render"]["focus_and_aperture_applied"]


@pytest.mark.parametrize(
    "edits",
    [
        {"viewCrop": {"angle": 0, "left": 0.2}},
        {"viewOrientation": 6},
        {"viewZoom": 1.1},
        {"viewFocusSpread": 1},
        {"viewTiltY": 1},
        {"viewFocusAnimation": {"values": [1]}},
    ],
)
def test_unverified_saved_geometry_or_animation_fails(edits: dict) -> None:
    with pytest.raises(LytroError):
        resolve_recipe(source_recipe(), edits, perspective_u=0, perspective_v=0, baseline=0.1)


@pytest.mark.parametrize(
    "options",
    [
        {"perspective_u": 0.1},
        {"perspective_v": 0.1},
        {"baseline": 0},
        {"baseline": -0.1},
        {"baseline": 0.3},
        {"focus": float("nan")},
        {"stereo_pivot": float("inf")},
    ],
)
def test_invalid_or_unverified_controls_fail(options: dict) -> None:
    kwargs: dict[str, Any] = {"perspective_u": 0, "perspective_v": 0, "baseline": 0.1, **options}
    with pytest.raises(LytroError):
        resolve_recipe(source_recipe(), **kwargs)


def test_missing_convergence_is_rejected() -> None:
    recipe = source_recipe()
    del recipe["viewStereoPivot"]
    with pytest.raises(LytroError, match="stereo pivot"):
        resolve_recipe(recipe, perspective_u=0, perspective_v=0, baseline=0.1)


def test_saved_stack_preserves_crop_spread_and_static_values_without_mutating_source() -> None:
    source = source_recipe()
    source["viewCrop"] = {
        "angle": -2.045,
        "top": 0.03,
        "left": -0.001,
        "bottom": 0.97,
        "right": 1.001,
    }
    source["viewFocusSpread"] = 25.321
    source["viewAperture"] = 2
    source["viewFocusAnimation"]["values"] = [99]
    original = copy.deepcopy(source)
    resolved = resolve_recipe(
        source, perspective_u=0, perspective_v=0, baseline=0.2, saved_stack=True
    )
    for view in resolved.views.values():
        assert view["viewCrop"] == source["viewCrop"]
        assert view["viewFocusSpread"] == 25.321
        assert view["viewAperture"] == 2
        assert view["viewFocus"] == 7
        assert "viewFocusAnimation" not in view
    assert source == original
    with pytest.raises(LytroError):
        resolve_recipe(source, perspective_u=0, perspective_v=0, baseline=0.2)
