from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from light_field_foundry.errors import LytroError
from light_field_foundry.lfp import FloatDepth, validate_alignment
from light_field_foundry.pipeline import _check_cached_edof
from light_field_foundry.recipe import GEOMETRY


def recipe():
    return {
        **deepcopy(GEOMETRY),
        "viewPerspectiveU": 0,
        "viewPerspectiveV": 0,
        "viewFocus": 0,
        "viewAperture": 1,
        "viewPivot": 0,
        "viewStereoPivot": 0,
        "viewStereoBaseline": 0,
        "viewContrast": 0,
    }


def test_cached_edof_is_independent_of_defocus_but_finishing_must_match():
    cached = recipe()
    target = {
        **cached,
        "viewFocus": 12,
        "viewFocusSpread": 25,
        "viewAperture": 2,
        "viewPivot": 8,
        "viewStereoPivot": 8,
    }
    _check_cached_edof({"recipe": cached}, target, {})
    target["viewContrast"] = 50
    with pytest.raises(LytroError, match="recompute-edof"):
        _check_cached_edof({"recipe": cached}, target, {})
    with pytest.raises(LytroError, match="exactly one"):
        _check_cached_edof({"recipe": cached, "recipe4": target}, cached, {})


def depth(width, height, mapping=None):
    return FloatDepth(
        np.zeros((height, width), np.float32),
        width,
        height,
        -20,
        31,
        mapping,
        {"recipe": recipe()},
        {},
        b"",
        (0, 0),
        (1134, 1134),
        True,
    )


def test_depth_only_requires_aspect_consistency_or_explicit_supported_mapping():
    resolved = SimpleNamespace(center=recipe())
    validate_alignment(depth(4, 4), resolved, (1134, 1134))
    with pytest.raises(LytroError, match="aspect"):
        validate_alignment(depth(2, 4), resolved, (1134, 1134))
    mapping = {
        "origin": {"x": 0, "y": 0},
        "xStep": {"dx": 0.5, "dy": 0},
        "yStep": {"dx": 0, "dy": 0.25},
    }
    validate_alignment(depth(2, 4, mapping), resolved, (1134, 1134))
    ambiguous = depth(4, 4)
    ambiguous.view["recipe4"] = {**recipe(), "viewCrop": {}}
    with pytest.raises(LytroError, match="exactly one"):
        validate_alignment(ambiguous, resolved, (1134, 1134))
