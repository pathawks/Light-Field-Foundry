"""The user-facing single-capture CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .errors import LytroError
from .pipeline import (
    DEFAULT_DISPARITY_MAX,
    DEFAULT_DISPARITY_MIN,
    Conversion,
    convert,
    validate_disparity_range,
)
from .pipeline import doctor as run_doctor
from .tools import Tools

app = typer.Typer(
    no_args_is_help=True, help="Tools for working with Lytro captures and image exports."
)
console = Console(highlight=False)


def _show_version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


def _doctor(tools: Tools) -> None:
    result = run_doctor(tools, console.print)
    for name, value in result.items():
        if name == "native_Apple_depth" and isinstance(value, dict):
            console.print("Apple depth: exact storage expansion and depth-code recovery.")
            behavior = value.get("coreImageBehavior", {})
            if isinstance(behavior, dict) and behavior.get("available"):
                console.print(
                    "Core Image: focus/aperture parameters, finite renders and controls passed."
                )
            else:
                console.print("Core Image depth blur is unavailable on this system.")
            continue
        console.print(f"{name}: {value}", markup=False)


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show application version."
        ),
    ] = False,
) -> None:
    """Lytro capture and image tools."""


@app.command("doctor", no_args_is_help=False)
def doctor_command(
    avifenc: Annotated[Path | None, typer.Option(help="Explicit avifenc executable.")] = None,
    avifdec: Annotated[Path | None, typer.Option(help="Explicit avifdec executable.")] = None,
    threads: Annotated[int, typer.Option(min=1, max=32)] = 4,
    timeout: Annotated[float, typer.Option(min=1, max=86400, help="Per-process seconds.")] = 600,
) -> None:
    """Check tools and codec/native contracts without processing a capture."""
    try:
        _doctor(Tools.discover(avifenc, avifdec, threads, timeout))
    except (LytroError, OSError, ValueError) as exc:
        console.print(f"Error: {exc}", style="red", markup=False)
        raise typer.Exit(1) from None
    except (KeyboardInterrupt, InterruptedError):
        console.print("Interrupted; current-run temporary files removed.", markup=False)
        raise typer.Exit(130) from None


@app.command("image-out", no_args_is_help=True)
def image_out(
    capture: Annotated[
        Path | None,
        typer.Argument(
            help="Optional shorthand LFP input; explicit --lfp-* roles take precedence."
        ),
    ] = None,
    lfp_raw: Annotated[
        Path | None,
        typer.Option(help="Raw-containing .lfp/.lfr; used directly without a library database."),
    ] = None,
    lfp_recipe: Annotated[
        Path | None,
        typer.Option(
            "--lfp-recipe", "--lfp-receipe", help="LFP containing the saved static recipe."
        ),
    ] = None,
    lfp_depth: Annotated[
        Path | None,
        typer.Option(help="LFP containing existing depth; --recompute-depth ignores this cache."),
    ] = None,
    lfp_edof: Annotated[
        Path | None,
        typer.Option(help="LFP supplying an EDOF center; --recompute-edof ignores this cache."),
    ] = None,
    recompute_depth: Annotated[
        bool,
        typer.Option(
            "--recompute-depth",
            help="Ignore pre-existing depth and compute fresh depth with TNT.",
            show_default=False,
        ),
    ] = False,
    recompute_edof: Annotated[
        bool,
        typer.Option(
            "--recompute-edof",
            help="Ignore pre-existing EDOF images and compute a fresh EDOF center with TNT.",
            show_default=False,
        ),
    ] = False,
    recompute: Annotated[
        bool,
        typer.Option(
            "--recompute",
            help="Recompute both depth and EDOF with TNT, ignoring their existing caches.",
            show_default=False,
        ),
    ] = False,
    no_recompute_depth: Annotated[
        bool,
        typer.Option(
            "--no-recompute-depth",
            help="Require existing matching lossless depth; fail instead of generating it.",
            show_default=False,
        ),
    ] = False,
    no_recompute_edof: Annotated[
        bool,
        typer.Option(
            "--no-recompute-edof",
            help="Require an existing matching lossless EDOF center; fail instead of generating it.",
            show_default=False,
        ),
    ] = False,
    no_recompute: Annotated[
        bool,
        typer.Option(
            "--no-recompute",
            help="Require matching lossless depth and EDOF caches; stereo eyes are still rendered.",
            show_default=False,
        ),
    ] = False,
    calibration: Annotated[
        Path | None,
        typer.Option(
            help="Calibration root (parent of camera serial folders); required without --library."
        ),
    ] = None,
    library: Annotated[
        Path | None,
        typer.Option(help="Read-only .lytrolibrary fallback for raw lookup and calibration."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Completed .avif; existing files are never overwritten; required for export."
        ),
    ] = None,
    perspective_u: Annotated[
        float | None,
        typer.Option(
            help="Center horizontal TNT viewpoint; required for export (initial route: 0)."
        ),
    ] = None,
    perspective_v: Annotated[
        float | None,
        typer.Option(help="Center vertical TNT viewpoint; required for export (initial route: 0)."),
    ] = None,
    baseline: Annotated[
        float | None,
        typer.Option(help="Normalized eye separation, not millimeters; required for export."),
    ] = None,
    focus: Annotated[
        float | None,
        typer.Option(help="Stereo/refocused-center focus lambda; EDOF center is unaffected."),
    ] = None,
    stereo_pivot: Annotated[
        float | None, typer.Option(help="Common convergence lambda; overrides recipe.")
    ] = None,
    aperture: Annotated[
        float | None, typer.Option(help="Stereo/refocused-center aperture ratio; overrides recipe.")
    ] = None,
    disparity_min: Annotated[
        float | None,
        typer.Option(
            help="Far-end relative disparity for Portrait rendering; defaults to 0.",
            show_default=False,
        ),
    ] = None,
    disparity_max: Annotated[
        float | None,
        typer.Option(
            help="Near-end relative disparity for Portrait rendering; defaults to 4.316406011581421.",
            show_default=False,
        ),
    ] = None,
    recipe: Annotated[
        Path | None,
        typer.Option(
            help="JSON recipe overrides; applied after --lfp-recipe, before explicit CLI controls."
        ),
    ] = None,
    include_private_metadata: Annotated[
        bool, typer.Option(help="Explicitly embed original private JSON, including identifiers.")
    ] = False,
    edof_center: Annotated[
        bool,
        typer.Option(
            "--edof/--refocus",
            help="Use the native EDOF center by default; --refocus bakes focus/aperture into the center.",
        ),
    ] = True,
    avifenc: Annotated[Path | None, typer.Option(help="Explicit avifenc executable.")] = None,
    avifdec: Annotated[Path | None, typer.Option(help="Explicit avifdec executable.")] = None,
    threads: Annotated[int, typer.Option(min=1, max=32)] = 4,
    timeout: Annotated[float, typer.Option(min=1, max=86400, help="Per-process seconds.")] = 600,
) -> None:
    """Render and verify selected LFP sources; retain only the completed AVIF."""
    force_depth = recompute_depth or recompute
    force_edof = recompute_edof or recompute
    require_depth = no_recompute_depth or no_recompute
    require_edof = no_recompute_edof or no_recompute
    if force_depth and require_depth:
        raise typer.BadParameter(
            "Depth recomputation options cannot be combined: --recompute-depth/--recompute "
            "and --no-recompute-depth/--no-recompute."
        )
    if force_edof and require_edof:
        raise typer.BadParameter(
            "EDOF recomputation options cannot be combined: --recompute-edof/--recompute "
            "and --no-recompute-edof/--no-recompute."
        )
    if require_edof and not edof_center:
        raise typer.BadParameter(
            "--no-recompute-edof/--no-recompute requires an EDOF primary and cannot be "
            "combined with --refocus."
        )
    if calibration is None and library is not None:
        calibration = library.expanduser() / "cameras"
    required = {
        "CAPTURE or an --lfp-* input": capture
        or lfp_raw
        or lfp_recipe
        or lfp_depth
        or lfp_edof,
        "--calibration": calibration,
        "--output": output,
        "--perspective-u": perspective_u,
        "--perspective-v": perspective_v,
        "--baseline": baseline,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise typer.BadParameter("Required for export: " + ", ".join(missing))
    try:
        minimum = DEFAULT_DISPARITY_MIN if disparity_min is None else disparity_min
        maximum = DEFAULT_DISPARITY_MAX if disparity_max is None else disparity_max
        validate_disparity_range(minimum, maximum)
        tools = Tools.discover(avifenc, avifdec, threads, timeout)
        assert calibration is not None and output is not None
        assert perspective_u is not None and perspective_v is not None and baseline is not None
        options = Conversion(
            capture,
            calibration,
            output,
            perspective_u,
            perspective_v,
            baseline,
            focus,
            stereo_pivot,
            aperture,
            recipe,
            include_private_metadata,
            edof_center,
            library=library,
            lfp_raw=lfp_raw,
            lfp_recipe=lfp_recipe,
            lfp_depth=lfp_depth,
            lfp_edof=lfp_edof,
            recompute_depth=force_depth,
            recompute_edof=force_edof,
            no_recompute_depth=require_depth,
            no_recompute_edof=require_edof,
            disparity_min=minimum,
            disparity_max=maximum,
        )
        result = convert(options, tools, console.print)
        console.print(f"Created {result['output']} ({result['bytes']:,} bytes)", markup=False)
        console.print(f"SHA-256: {result['sha256']}", markup=False)
        console.print(
            "Verified: three exact RGB16 views, RGB12 fallbacks, depth12, native AVDepthData."
        )
        console.print(
            "Portrait / Spatial recognition and physical eye order require manual Photos acceptance."
        )
    except (LytroError, OSError, ValueError) as exc:
        console.print(f"Error: {exc}", style="red", markup=False)
        raise typer.Exit(1) from None
    except (KeyboardInterrupt, InterruptedError):
        console.print("Interrupted; current-run temporary files removed.", markup=False)
        raise typer.Exit(130) from None
