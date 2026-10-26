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
    extract_edof,
    extract_recipe,
    quantize_depth,
    read_json_bytes,
    validate_alignment,
)
from .metadata import build_metadata
from .native import verify_native
from .recipe import resolve_recipe, validate_render_log
from .sources import select_sources, stage_raw
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
    capture: Path | None
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
    library: Path | None = None
    lfp_raw: Path | None = None
    lfp_recipe: Path | None = None
    lfp_depth: Path | None = None
    lfp_edof: Path | None = None
    recompute_depth: bool = False
    recompute_edof: bool = False
    no_recompute_depth: bool = False
    no_recompute_edof: bool = False


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
                prefix=f".{output.name}.lff-", dir=output.parent
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


def _check_cached_edof(
    view: dict[str, object], center: dict[str, object], defaults: dict[str, object]
) -> None:
    candidates = [value for key, value in view.items() if re.fullmatch(r"recipe\d*", key)]
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise LytroError("Cached EDOF requires exactly one effective recipe; use --recompute-edof")
    saved = candidates[0]
    ignored = {
        "viewFocus",
        "viewFocusSpread",
        "viewAperture",
        "viewPivot",
        "viewStereoPivot",
        "viewStereoBaseline",
        "viewPriorities",
    }
    for key, value in center.items():
        if not key.startswith("view") or key.endswith("Animation") or key in ignored:
            continue
        actual = saved.get(key, defaults.get(key))
        if isinstance(actual, (int, float)) and isinstance(value, (int, float)):
            if float(np.float32(actual)) == float(np.float32(value)):
                continue
        elif actual == value:
            continue
        raise LytroError(
            f"Cached EDOF {key} differs from the selected recipe; use --recompute-edof"
        )


def _validate_recompute_policy(options: Conversion) -> None:
    for role, force, forbid in (
        ("depth", options.recompute_depth, options.no_recompute_depth),
        ("edof", options.recompute_edof, options.no_recompute_edof),
    ):
        if force and forbid:
            raise LytroError(f"Cannot combine --recompute-{role} with --no-recompute-{role}")
    if options.no_recompute_edof and not options.edof_center:
        raise LytroError(
            "--no-recompute-edof requires an EDOF primary and cannot be used with --refocus"
        )


def convert(
    options: Conversion, tools: Tools, report: Callable[[str], None] = print
) -> dict[str, object]:
    _validate_recompute_policy(options)
    validate_disparity_range(options.disparity_min, options.disparity_max)
    if sys.platform != "darwin":
        raise LytroError("Only macOS TNT and native depth readback are validated in this release.")
    calibration = options.calibration.expanduser().resolve()
    output = options.output.expanduser().absolute()
    if not calibration.is_dir():
        raise LytroError("Calibration must be an existing matching camera calibration root.")
    if output.suffix.lower() != ".avif" or not output.parent.is_dir():
        raise LytroError("Output must end in .avif and its parent directory must exist.")
    output_parent = output.parent.resolve()
    inside_library = any(
        path.name.lower().endswith(".lytrolibrary")
        for path in (output_parent, *output_parent.parents)
    )
    if inside_library or (
        options.library and output_parent.is_relative_to(options.library.expanduser().resolve())
    ):
        raise LytroError("Output cannot be inside the read-only Lytro library")
    if output.exists() or output.is_symlink():
        raise LytroError(f"Output already exists: {output}")
    sources = select_sources(
        options.capture,
        raw=options.lfp_raw,
        recipe=options.lfp_recipe,
        depth=options.lfp_depth,
        edof=options.lfp_edof,
        library=options.library,
        recompute_depth=options.recompute_depth,
        recompute_edof=options.recompute_edof,
    )
    for role, required, source in (
        ("depth", options.no_recompute_depth, sources.depth),
        ("edof", options.no_recompute_edof, sources.edof),
    ):
        if required and source is None:
            raise LytroError(
                f"--no-recompute-{role} requires an existing matching lossless {role} asset; "
                f"supply --lfp-{role} or a positional native stack"
            )
    if options.recipe and options.recipe.stat().st_size > MAX_JSON_BYTES:
        raise LytroError("Recipe exceeds the 4 MiB JSON limit.")
    requested_bytes = options.recipe.read_bytes() if options.recipe else None
    requested = read_json_bytes(requested_bytes, "recipe") if requested_bytes is not None else None
    # Check that selected cache payloads exist and use admitted lossless
    # encodings before any native processing or generated codec controls.
    cached_depth = (
        extract_depth(
            sources.depth.path,
            asset_root=sources.depth.path.parent,
            exported_stack=True,
            require_witness=False,
        )
        if sources.depth is not None
        else None
    )
    cached_edof = (
        extract_edof(sources.edof.path, asset_root=sources.edof.path.parent)
        if options.edof_center and sources.edof is not None
        else None
    )
    report("Checking installed codecs and the native Apple depth control…")
    control = doctor(tools, report)
    with workspace() as work:
        versions = {key: str(control[key]) for key in ("tnt", "avifenc", "avifdec", "imagecodecs")}
        versions["tnt"] = versions["tnt"].replace(str(tools.tnt), "TNT")
        # No native command receives a source RAW. Strip all processed cache
        # references, retaining exact raw/metadata chunks in a private input.
        native_capture = work / "raw-input.lfp"
        stage_raw(sources.raw, native_capture)
        native_calibration = work / "calibration"
        shutil.copytree(calibration, native_calibration)
        base = [
            tools.tnt,
            "--threads",
            str(tools.threads),
            "--lfp-in",
            native_capture,
            "--calibration-in",
            native_calibration,
        ]
        source_recipe_path = work / "source-recipe.json"
        report("Resolving the selected recipe and common stereo controls…")
        tools.run([*base, "--recipe-out", source_recipe_path], work)
        require_output(source_recipe_path)
        recipe_bytes = source_recipe_path.read_bytes()
        defaults = read_json_bytes(recipe_bytes, "TNT default recipe")
        selected = dict(defaults)
        if sources.recipe is not None:
            saved = extract_recipe(sources.recipe)
            recipe_bytes = json.dumps(saved, allow_nan=False).encode()
            selected.update(saved)
        flexible_geometry = bool(sources.recipe or sources.depth or sources.edof or requested)
        resolved = resolve_recipe(
            selected,
            requested,
            perspective_u=options.perspective_u,
            perspective_v=options.perspective_v,
            baseline=options.baseline,
            focus=options.focus,
            stereo_pivot=options.stereo_pivot,
            aperture=options.aperture,
            edof_center=options.edof_center,
            saved_stack=flexible_geometry,
        )
        recipes: dict[str, Path] = {}
        for name, recipe in zip(
            ("center", "left", "right"),
            (resolved.center, resolved.left, resolved.right),
            strict=True,
        ):
            path = work / f"{name}-recipe.json"
            path.write_text(json.dumps(recipe, allow_nan=False), encoding="utf-8")
            recipes[name] = path
        depth = cached_depth
        if depth is not None:
            report("Using the selected native depth asset…")
            try:
                validate_alignment(depth, resolved, depth.image_size)
            except LytroError as exc:
                raise LytroError(
                    f"Cached depth geometry differs; use --recompute-depth: {exc}"
                ) from exc
        edof = cached_edof
        if edof is not None:
            report("Using the selected native EDOF center…")
            _check_cached_edof(edof.view, resolved.center, defaults)
            cached_center = work / "cached-center.tiff"
            cached_center.write_bytes(edof.witness_image)
            # Missing, unsupported, lossy, or incompatible EDOF input must fail
            # before any cache generation, even if the other role is automatic.
            read_tiff(cached_center, allow_black_coverage=True)
        generated_depth = depth is None
        generated_edof = options.edof_center and edof is None
        if generated_depth or generated_edof:
            generated = work / "generated-stack.lfp"
            report("Reconstructing native EDOF and floating depth from cache-free raw with TNT…")
            log = tools.run(
                [
                    *base,
                    "--lfp-out",
                    generated,
                    "--imagerep",
                    "tiff",
                    "--depthrep",
                    "float",
                    "--recipe-in",
                    recipes["center"],
                    "--unpack",
                ],
                work,
            )
            versions["render_engine"] = _render_engine(log)
            require_output(generated)
            if generated_depth:
                depth = extract_depth(
                    generated,
                    asset_root=work,
                    exported_stack=flexible_geometry,
                    measure_witness=True,
                )
            if generated_edof:
                edof = extract_edof(generated, asset_root=work)
        if depth is None:
            raise LytroError("No verified depth role was produced")
        images: list[RGB16] = []
        for name, recipe in zip(
            ("center", "left", "right"),
            (resolved.center, resolved.left, resolved.right),
            strict=True,
        ):
            target = work / f"{name}.tiff"
            if name == "center" and options.edof_center:
                if edof is None:
                    raise LytroError("No verified EDOF role was produced")
                report("Using the native Extended Depth of Field primary…")
                target.write_bytes(edof.witness_image)
            else:
                report(f"Rendering independent {name} RGB16 view…")
                log = tools.run(
                    [
                        *base,
                        "--image-out",
                        target,
                        "--imagerep",
                        "tiff",
                        "--recipe-in",
                        recipes[name],
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
            images.append(
                read_tiff(target, allow_black_coverage=True)
                if flexible_geometry
                else read_tiff(target)
            )
        separate_profile = sources.edof is not None and options.edof_center
        if separate_profile:
            matching_views(images, separate_center_profile=True)
        else:
            matching_views(images)
        report("Checking selected center/depth geometry…")
        validate_alignment(depth, resolved, (images[0].width, images[0].height))
        codes = quantize_depth(depth)
        report("Encoding three exact RGB16 sample transforms and one monochrome12 depth plane…")
        color_paths = [
            encode_color(tools, work, name, image)
            for name, image in zip(("center", "left", "right"), images, strict=True)
        ]
        encoded_depth = encode_depth(tools, work, codes)
        provenance = {
            "application": f"Light Field Foundry {__version__}",
            "inputRoles": sources.provenance,
            "recomputePolicy": {
                "depth": "never"
                if options.no_recompute_depth
                else "always"
                if options.recompute_depth
                else "auto",
                "edof": "never"
                if options.no_recompute_edof
                else "always"
                if options.recompute_edof
                else "auto",
                "cachedEncodings": "provably lossless TIFF compression; PNG or packedFloat depth",
            },
            "rawInputPolicy": "exact raw/metadata chunks; processed views and acceleration caches removed before TNT",
            "depthSource": "TNT recomputed Float32"
            if generated_depth
            else "selected LFP native depth",
            "primarySource": (
                "TNT recomputed native EDOF" if generated_edof else "selected LFP native EDOF"
            )
            if options.edof_center
            else "explicit --refocus ordinary still",
            "animationPolicy": "saved static fields used; animation tracks retained only in source recipe metadata",
            "tools": versions,
            "depth": depth.mapping_provenance,
            "relativeDisparity": {
                "atCode0": options.disparity_min,
                "atCode4095": options.disparity_max,
                "interpretation": "linear interpolation after L016 expansion; increases toward near; presentation scale, not metric depth",
            },
            "tiffColorTags": [image.tags for image in images],
            "color": {
                "CICP": images[0].cicp,
                "ICC": "original per-view byte-exact ICC; standardized primaries/transfer unspecified",
                "extraChannel": "opaque or binary undeclared black coverage baked onto black",
            },
            "recipePolicy": resolved.provenance,
        }
        source = sources.metadata
        metadata = build_metadata(
            sources.raw.public_json,
            recipe_bytes,
            {"center": resolved.center, "left": resolved.left, "right": resolved.right},
            provenance,
            images[0].width,
            images[0].height,
            options.disparity_min,
            options.disparity_max,
            srgb=images[0].srgb,
            private_json=sources.raw.private_json if options.include_private_metadata else (),
            master_json=source.master_json,
            source_depth_mapping=depth.mapping_provenance,
        )
        stage = work / "completed.avif"
        stage.touch(mode=0o600, exist_ok=False)
        report("Assembling Portrait + Spatial metadata and verifying every final image item…")
        result = bmff.assemble(
            color_paths[0],
            color_paths[1],
            color_paths[2],
            encoded_depth,
            metadata,
            stage,
            **({"separate_center_profile": True} if separate_profile else {}),
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
