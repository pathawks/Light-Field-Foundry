"""Resolve common stereo controls before invoking TNT.

The measured TNT 2.0 ordinary-still path uses viewPivot and ignores
viewStereoPivot. Each independent still therefore receives the saved/requested
stereo pivot in BOTH fields. No native CLI/recipe precedence is assumed.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import LytroError
from .lfp import read_json_bytes

FULL_CROP = {"angle": 0, "top": 0, "left": 0, "bottom": 1, "right": 1}
GEOMETRY = {
    "viewCrop": FULL_CROP,
    "viewOrientation": 1,
    "viewPanX": 0.5,
    "viewPanY": 0.5,
    "viewZoom": 1,
    "viewTiltX": 0,
    "viewTiltY": 0,
    "viewFocusSpread": 0,
}


@dataclass(frozen=True)
class ResolvedRecipe:
    center: dict[str, Any]
    left: dict[str, Any]
    right: dict[str, Any]
    focus: float
    stereo_pivot: float
    baseline: float
    edof_center: bool = True

    @property
    def views(self) -> dict[str, dict[str, Any]]:
        return {"center": self.center, "left": self.left, "right": self.right}

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "effective_views": self.views,
            "baseline": self.baseline,
            "baseline_units": "TNT normalized viewpoint separation; not millimeters",
            "convergence": "ordinary still viewPivot explicitly equals viewStereoPivot",
            "stereo_pivot": self.stereo_pivot,
            "focus_policy": "matched resolved focus/aperture for stereo eyes and optional refocused center",
            "center_render": {
                "mode": "EDOF" if self.edof_center else "refocused",
                "source": "TNT edofWarp center TIFF" if self.edof_center else "ordinary TNT still",
                "focus_and_aperture_applied": not self.edof_center,
                "meaning": "EDOF uses the native all-in-focus stack center; focus/aperture remain stereo-eye controls",
            },
            "geometry_scope": {
                "center": "F01 canonical center",
                "crop": self.center["viewCrop"],
                "orientation": self.center["viewOrientation"],
            },
            "eye_order": "left=U-B/2; right=U+B/2; manual scene acceptance pending",
        }

    def write(self, directory: Path) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for name, recipe in self.views.items():
            path = directory / f"{name}-recipe.json"
            path.write_text(json.dumps(recipe, allow_nan=False), encoding="utf-8")
            result[name] = path
        return result


def load_recipe(path: Path) -> dict[str, Any]:
    return read_json_bytes(Path(path).read_bytes(), "recipe")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LytroError(f"{name} must be a finite number")
    result = float(np.float32(value))
    if not math.isfinite(result):
        raise LytroError(f"{name} must fit TNT's Float32 representation")
    return result


def _check_tree(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _check_tree(child)
    elif isinstance(value, list):
        for child in value:
            _check_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise LytroError("Recipe contains a nonfinite number")


def resolve_recipe(
    source: dict[str, Any],
    requested: dict[str, Any] | None = None,
    *,
    perspective_u: float,
    perspective_v: float,
    baseline: float,
    focus: float | None = None,
    stereo_pivot: float | None = None,
    aperture: float | None = None,
    edof_center: bool = True,
    saved_stack: bool = False,
) -> ResolvedRecipe:
    """CLI controls override requested recipe, then TNT's exported source recipe.

    Missing and explicit zero values are distinct. The exported source defaults
    are frozen before rendering; TNT automatic focus is never invoked per eye.
    """
    _check_tree(source)
    requested = requested or {}
    _check_tree(requested)
    unknown = set(requested) - set(source)
    if unknown:
        raise LytroError(
            f"Recipe fields are not advertised by the installed TNT: {', '.join(sorted(unknown))}"
        )
    recipe = copy.deepcopy(source)
    recipe.update(copy.deepcopy(requested))
    if saved_stack:
        recipe = {key: value for key, value in recipe.items() if not key.endswith("Animation")}
    for key, value in recipe.items():
        if key.endswith("Animation") and (
            not isinstance(value, dict)
            or any(value.get(part) for part in ("values", "times", "handlePairs"))
        ):
            raise LytroError(f"Animated recipes are not admitted: {key}")
    for key, expected in GEOMETRY.items():
        if saved_stack and key in {"viewCrop", "viewFocusSpread"}:
            continue
        if recipe.get(key, expected) != expected:
            raise LytroError(
                f"Unsupported saved geometry {key}; only upright full-frame center is validated"
            )
        recipe[key] = copy.deepcopy(expected)
    if saved_stack:
        recipe.setdefault("viewFocusSpread", 0)
        crop = recipe.get("viewCrop")
        if not isinstance(crop, dict) or set(crop) != set(FULL_CROP):
            raise LytroError("Saved crop must declare angle/top/left/bottom/right")
        values = {key: _finite(value, "crop " + key) for key, value in crop.items()}
        if not (values["top"] < values["bottom"] and values["left"] < values["right"]):
            raise LytroError("Saved crop must have ordered, nonempty extents")
        spread = _finite(recipe.get("viewFocusSpread", 0), "focus spread")
        if spread < 0:
            raise LytroError("Focus spread must be nonnegative")
    u = _finite(perspective_u, "perspective U")
    v = _finite(perspective_v, "perspective V")
    separation = _finite(baseline, "baseline")
    if u != 0 or v != 0:
        raise LytroError(
            "Center depth alignment is validated only for --perspective-u 0 --perspective-v 0"
        )
    if not 0 < separation <= float(np.float32(0.2)):
        raise LytroError("Initial F01 stereo scope requires 0 < baseline <= 0.2")
    resolved_focus = _finite(focus if focus is not None else recipe.get("viewFocus"), "focus")
    resolved_pivot = _finite(
        stereo_pivot if stereo_pivot is not None else recipe.get("viewStereoPivot"), "stereo pivot"
    )
    resolved_aperture = _finite(
        aperture if aperture is not None else recipe.get("viewAperture"), "aperture"
    )
    aperture_limit = 2 if saved_stack else 1
    if not 0 <= resolved_aperture <= aperture_limit:
        raise LytroError(
            f"Aperture must be within the admitted normalized range 0..{aperture_limit}"
        )
    recipe.update(
        {
            "viewFocus": resolved_focus,
            "viewAperture": resolved_aperture,
            "viewPriorities": ["viewFocus"],
            "viewPivot": resolved_pivot,
            "viewStereoPivot": resolved_pivot,
            # Separate ordinary still calls provide the stereo baseline. Asking TNT
            # to produce stereo itself could select an unverified processing path.
            "viewStereoBaseline": 0,
            "viewPerspectiveV": v,
        }
    )
    views: list[dict[str, Any]] = []
    for coordinate in (u, u - separation / 2, u + separation / 2):
        view = copy.deepcopy(recipe)
        view["viewPerspectiveU"] = _finite(coordinate, "eye perspective U")
        views.append(view)
    return ResolvedRecipe(*views, resolved_focus, resolved_pivot, separation, edof_center)


def validate_render_log(output: str, recipe: dict[str, Any]) -> None:
    """Confirm TNT actually used the requested still controls, not just exit 0."""
    fields = {
        "Focus lambda": "viewFocus",
        "Focus spread": "viewFocusSpread",
        "Focus tilt x": "viewTiltX",
        "Focus tilt y": "viewTiltY",
        "Aperture ratio": "viewAperture",
        "Perspective center x": "viewPerspectiveU",
        "Perspective center y": "viewPerspectiveV",
        "Perspective zero lambda": "viewPivot",
        "Pan X": "viewPanX",
        "Pan Y": "viewPanY",
        "Zoom": "viewZoom",
    }
    for label, key in fields.items():
        matches = re.findall(re.escape(label) + r"\s*:\s*([-+0-9.eE]+)", output)
        if len(matches) != 1:
            raise LytroError(f"TNT did not report one effective {label}")
        actual = float(matches[0])
        expected = float(recipe[key])
        # TNT prints six decimal places; compare within one printed last digit.
        if not math.isfinite(actual) or abs(actual - expected) > 1e-6:
            raise LytroError(f"TNT effective {label} {actual} differs from resolved {expected}")
