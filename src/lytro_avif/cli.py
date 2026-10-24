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
    no_args_is_help=True, help="Export TNT-rendered RGB16 stereo + depth to one AVIF."
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


@app.command(no_args_is_help=True)
def main(
    capture: Annotated[
        Path | None, typer.Argument(help="Raw .lfr or raw-containing .lfp capture.")
    ] = None,
    calibration: Annotated[
        Path | None,
        typer.Option(
            help="Matching calibration root (parent of camera serial folders); required for export."
        ),
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
        Path | None, typer.Option(help="Saved static recipe; CLI controls take precedence.")
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
    doctor: Annotated[
        bool,
        typer.Option(
            "--doctor", help="Check tools and codec/native contracts without exporting a capture."
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show application version."
        ),
    ] = False,
) -> None:
    """Render and verify CAPTURE; retain only the completed AVIF."""
    if doctor:
        if (
            include_private_metadata
            or not edof_center
            or any(
                value is not None
                for value in (
                    capture,
                    calibration,
                    output,
                    perspective_u,
                    perspective_v,
                    baseline,
                    focus,
                    stereo_pivot,
                    aperture,
                    disparity_min,
                    disparity_max,
                    recipe,
                )
            )
        ):
            raise typer.BadParameter(
                "--doctor cannot be combined with a capture or export options."
            )
    else:
        required = {
            "CAPTURE": capture,
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
        if doctor:
            _doctor(tools)
            return
        assert capture is not None and calibration is not None and output is not None
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
