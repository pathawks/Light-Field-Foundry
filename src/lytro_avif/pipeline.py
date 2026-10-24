"""One capture, three independent TNT renders, one atomically published AVIF."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__, bmff
from .codec import (
    encode_color,
    encode_depth,
    read_y4m,
    synthetic_probes,
    verify_color,
    verify_depth,
)
from .errors import LytroError
from .images import RGB16, matching_views, read_png, read_tiff
from .lfp import (
    MAX_JSON_BYTES,
    extract_depth,
    quantize_depth,
    read_json_bytes,
    read_lfp,
    validate_alignment,
)
from .metadata import build_metadata
from .native import verify_native
from .recipe import resolve_recipe, validate_render_log
from .tools import Tools, require_output, workspace

DEFAULT_DISPARITY_MIN = 0.0
# Exact metadata endpoint of the gain-only candidate accepted in Photos.
DEFAULT_DISPARITY_MAX = 4.316406011581421


def validate_disparity_range(minimum: float, maximum: float) -> None:
    """Reject render ranges that cannot retain the native 12-bit depth codes."""
    if not math.isfinite(minimum) or not math.isfinite(maximum) or not 0 <= minimum < maximum:
        raise LytroError(
            "Disparity endpoints must be finite and satisfy 0 <= --disparity-min < --disparity-max."
        )
    if maximum > float(np.finfo(np.float32).max):
        raise LytroError("Disparity endpoints must fit Apple's Float32 representation.")
    a, b = np.float32(minimum), np.float32(maximum)
    span = np.float32(b - a)
    if span == 0:
        raise LytroError(
            "Disparity endpoints must remain distinct in Apple's Float32 representation."
        )
    # Mirror ImageIO's L016 expansion and Float32 reconstruction for every
    # possible stored code, so an unusably small span/large offset fails early.
    codes = np.arange(4096, dtype=np.uint16)
    expanded = (codes << 4) | (codes >> 8)
    scale = np.float32(span / np.float32(65535))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        samples = (float(a) + expanded.astype(np.float64) * float(scale)).astype(np.float32)
        recovered = np.rint((samples - a) / span * np.float32(4095))
    if not np.array_equal(recovered, codes):
        raise LytroError(
            "Disparity range cannot preserve all 4096 depth codes in Apple's Float32 representation; "
            "use a wider span or a smaller minimum."
        )


@dataclass(frozen=True)
class Conversion:
    capture: Path
    calibration: Path
    output: Path
    perspective_u: float
    perspective_v: float
    baseline: float
    focus: float | None = None
    stereo_pivot: float | None = None
    aperture: float | None = None
    recipe: Path | None = None
    include_private_metadata: bool = False
    edof_center: bool = True
    disparity_min: float = DEFAULT_DISPARITY_MIN
    disparity_max: float = DEFAULT_DISPARITY_MAX


def _render_engine(log: str) -> str:
    match = re.search(r"TNT ([\d.]+) powered by LFE Lytro Lightfield Engine ([\d.]+)", log)
    if match is None or match[1] != "2.0" or float(match[2]) != 21:
        raise LytroError("The admitted rendering path requires TNT 2.0 / LFE 21.")
    return f"TNT {match[1]} / LFE {match[2]}"


def verify_final(
    tools: Tools,
    work: Path,
    final: Path,
    result: bmff.AssemblyResult,
    images: list[RGB16],
    codes: np.ndarray,
) -> None:
    data = final.read_bytes()
    for name, role, image in zip(
        ("center", "left", "right"),
        (result.center, result.left, result.right),
        images,
        strict=True,
    ):
        selected = work / f"final-{name}.avif"
        selected.write_bytes(bmff.select_primary(data, role.base))
        verify_color(tools, work, selected, image, f"final-{name}")
        # Verify the four-bit suffix's actual component values, not just the expression.
        suffix = work / f"final-{name}-suffix.avif"
        suffix.write_bytes(bmff.select_primary(data, role.suffix))
        for codec in ("aom", "dav1d"):
            native = work / f"{name}-{codec}-suffix.y4m"
            tools.run(
                [tools.avifdec, "--jobs", str(tools.threads), "--codec", codec, suffix, native],
                work,
            )
            depth, _, planes = read_y4m(native)
            if depth != 8 or len(planes) != 3:
                raise LytroError("Color suffix must be an ordinary RGB8 coded component.")
            rgb = np.stack((planes[2], planes[0], planes[1]), axis=2)
            if not np.array_equal(rgb, image.pixels & 15):
                raise LytroError(f"{name}: suffix samples are not exact low4 values.")
    selected_depth = work / "final-depth.avif"
    selected_depth.write_bytes(bmff.select_primary(data, result.depth.item))
    verify_depth(tools, work, selected_depth, codes)


def _publish(stage: Path, output: Path) -> None:
    """A hard-link commit is atomic and never replaces an existing path."""
    with stage.open("rb") as file:
        os.fsync(file.fileno())
    try:
        try:
            os.link(stage, output)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # A destination on another filesystem needs one publication copy.
            # The verified AVIF stays private until the final atomic link.
            with tempfile.TemporaryDirectory(
                prefix=f".{output.name}.lytro-avif-", dir=output.parent
            ) as directory:
                candidate = Path(directory) / "completed.avif"
                shutil.copyfile(stage, candidate)
                candidate.chmod(0o600)
                with candidate.open("rb") as file:
                    os.fsync(file.fileno())
                os.link(candidate, output)
    except FileExistsError as exc:
        raise LytroError(f"Output already exists: {output}") from exc
    except OSError as exc:
        raise LytroError(f"Could not atomically publish the AVIF: {exc}") from exc


def convert(
    options: Conversion, tools: Tools, report: Callable[[str], None] = print
) -> dict[str, object]:
    validate_disparity_range(options.disparity_min, options.disparity_max)
    if sys.platform != "darwin":
        raise LytroError("Only macOS TNT and native depth readback are validated in this release.")
    capture, calibration, output = (
        options.capture.expanduser().resolve(),
        options.calibration.expanduser().resolve(),
        options.output.expanduser().absolute(),
    )
    if capture.suffix.lower() not in (".lfr", ".lfp") or not capture.is_file():
        raise LytroError("Capture must be an existing raw .lfr or raw-containing .lfp file.")
    if not calibration.is_dir():
        raise LytroError("Calibration must be an existing matching camera calibration root.")
    if output.suffix.lower() != ".avif" or not output.parent.is_dir():
        raise LytroError("Output must end in .avif and its parent directory must exist.")
    if output.exists() or output.is_symlink():
        raise LytroError(f"Output already exists: {output}")
    source = read_lfp(capture)
    info = source.admit_capture()
    if options.recipe and options.recipe.stat().st_size > MAX_JSON_BYTES:
        raise LytroError("Recipe exceeds the 4 MiB JSON limit.")
    requested_bytes = options.recipe.read_bytes() if options.recipe else None
    requested = read_json_bytes(requested_bytes, "recipe") if requested_bytes is not None else None
    report("Checking installed codecs and the native Apple depth control…")
    control = doctor(tools, report)
    with workspace() as work:
        versions = {key: str(control[key]) for key in ("tnt", "avifenc", "avifdec")}
        # TNT --version echoes its own path. Paths are not part of embedded provenance.
        versions["tnt"] = versions["tnt"].replace(str(tools.tnt), "TNT")
        base = [
            tools.tnt,
            "--threads",
            str(tools.threads),
            "--lfp-in",
            capture,
            "--calibration-in",
            calibration,
        ]
        source_recipe_path = work / "source-recipe.json"
        report("Reading source recipe and resolving common focus and stereo convergence…")
        tools.run([*base, "--recipe-out", source_recipe_path], work)
        require_output(source_recipe_path)
        recipe_bytes = source_recipe_path.read_bytes()
        resolved = resolve_recipe(
            read_json_bytes(recipe_bytes, "TNT source recipe"),
            requested,
            perspective_u=options.perspective_u,
            perspective_v=options.perspective_v,
            baseline=options.baseline,
            focus=options.focus,
            stereo_pivot=options.stereo_pivot,
            aperture=options.aperture,
            edof_center=options.edof_center,
        )
        recipes: dict[str, Path] = {}
        for name, recipe in zip(
            ("center", "left", "right"),
            (resolved.center, resolved.left, resolved.right),
            strict=True,
        ):
            recipe_path = work / f"{name}-recipe.json"
            recipe_path.write_text(json.dumps(recipe, allow_nan=False), encoding="utf-8")
            recipes[name] = recipe_path
        depth_path = work / "depth-source.lfp"
        report("Building TNT's EDOF stack and floating depth…")
        depth_args = [
            *base,
            "--lfp-out",
            depth_path,
            "--imagerep",
            "tiff",
            "--depthrep",
            "float",
            "--recipe-in",
            recipes["center"],
        ]
        if options.edof_center:
            depth_args.append("--unpack")
        depth_log = tools.run(depth_args, work)
        versions["render_engine"] = _render_engine(depth_log)
        require_output(depth_path)
        depth = extract_depth(depth_path, asset_root=work if options.edof_center else None)
        images: list[RGB16] = []
        for name, recipe in zip(
            ("center", "left", "right"),
            (resolved.center, resolved.left, resolved.right),
            strict=True,
        ):
            recipe_path = recipes[name]
            target = work / f"{name}.tiff"
            if name == "center" and options.edof_center:
                report("Using TNT's native Extended Depth of Field center view…")
                target.write_bytes(depth.witness_image)
                images.append(read_tiff(target))
                continue
            report(f"Rendering independent {name} RGB16 view…")
            log = tools.run(
                [
                    *base,
                    "--image-out",
                    target,
                    "--imagerep",
                    "tiff",
                    "--recipe-in",
                    recipe_path,
                    "--perspective-u",
                    str(recipe["viewPerspectiveU"]),
                    "--perspective-v",
                    str(recipe["viewPerspectiveV"]),
                    "--focus",
                    str(recipe["viewFocus"]),
                ],
                work,
            )
            require_output(target)
            validate_render_log(log, recipe)
            engine = _render_engine(log)
            if "render_engine" in versions and versions["render_engine"] != engine:
                raise LytroError("TNT engine version changed between views.")
            versions["render_engine"] = engine
            images.append(read_tiff(target))
        matching_views(images)
        report("Checking canonical-center depth alignment…")
        validate_alignment(depth, resolved, (images[0].width, images[0].height))
        codes = quantize_depth(depth)
        report("Encoding three exact RGB16 sample transforms and one monochrome12 depth plane…")
        color_paths = [
            encode_color(tools, work, name, image)
            for name, image in zip(("center", "left", "right"), images, strict=True)
        ]
        encoded_depth = encode_depth(tools, work, codes)
        provenance = {
            "application": f"lytro-avif {__version__}",
            "tools": versions,
            "depth": depth.mapping_provenance,
            "relativeDisparity": {
                "atCode0": options.disparity_min,
                "atCode4095": options.disparity_max,
                "interpretation": "linear interpolation between rendering endpoints after L016 expansion; increases toward near; presentation scale, not metric depth",
            },
            "tiffColorTags": [image.tags for image in images],
            "color": {
                "CICP": images[0].cicp,
                "ICC": "original byte-exact TNT ICC; standardized primaries/transfer unspecified",
                "extraChannel": "discarded only after all samples equal 65535",
            },
            "recipePolicy": resolved.provenance,
        }
        metadata = build_metadata(
            info.public_json,
            recipe_bytes,
            {"center": resolved.center, "left": resolved.left, "right": resolved.right},
            provenance,
            images[0].width,
            images[0].height,
            options.disparity_min,
            options.disparity_max,
            srgb=images[0].srgb,
            private_json=info.private_json if options.include_private_metadata else (),
            master_json=info.master_json,
            source_depth_mapping=depth.mapping_provenance,
        )
        stage = work / "completed.avif"
        stage.touch(mode=0o600, exist_ok=False)
        report("Assembling Portrait + Spatial metadata and verifying every final image item…")
        result = bmff.assemble(
            color_paths[0], color_paths[1], color_paths[2], encoded_depth, metadata, stage
        )
        verify_final(tools, work, stage, result, images, codes)
        report("Checking ImageIO / AVDepthData storage expansion and exact depth-code recovery…")
        native = verify_native(
            tools, work, stage, codes, options.disparity_min, options.disparity_max
        )
        digest = hashlib.sha256(stage.read_bytes()).hexdigest()
        size = stage.stat().st_size
        _publish(stage, output)
    return {"output": str(output), "bytes": size, "sha256": digest, "native": native}


def doctor(tools: Tools, report: Callable[[str], None] = print) -> dict[str, object]:
    with workspace() as work:
        versions = tools.capabilities(work)
        report("Testing synthetic RGB16 and all 4096 depth codes through AOM and dav1d…")
        results: dict[str, object] = {**versions, **synthetic_probes(tools, work)}
        pixels, icc = read_png(work / "probe-color.png")
        if icc is None:
            raise LytroError("Synthetic bridge lost its ICC profile.")
        images = [
            RGB16(pixels, icc, {}),
            RGB16(np.flipud(pixels).copy(), icc, {}),
            RGB16(np.fliplr(pixels).copy(), icc, {}),
        ]
        color_paths = [
            work / "probe-color.avif",
            encode_color(tools, work, "probe-left", images[1]),
            encode_color(tools, work, "probe-right", images[2]),
        ]
        codes = np.arange(4096, dtype=np.uint16).reshape(64, 64)
        metadata = build_metadata(
            [b"{}"],
            b"{}",
            {},
            {"purpose": "generated contract probe"},
            64,
            64,
            DEFAULT_DISPARITY_MIN,
            DEFAULT_DISPARITY_MAX,
        )
        final = work / "probe-final.avif"
        result = bmff.assemble(
            color_paths[0], color_paths[1], color_paths[2], work / "depth.avif", metadata, final
        )
        report("Checking assembled image identities, RGB16 alternatives, and native Apple depth…")
        verify_final(tools, work, final, result, images, codes)
        results["native_Apple_depth"] = verify_native(
            tools, work, final, codes, DEFAULT_DISPARITY_MIN, DEFAULT_DISPARITY_MAX
        )
        results["final_container"] = (
            "three RGB16 alternatives + ordered RGB12 stereo + one depth12; exact"
        )
        return results
