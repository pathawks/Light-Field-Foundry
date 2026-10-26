from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from light_field_foundry import cli
from light_field_foundry.cli import app
from light_field_foundry.pipeline import DEFAULT_DISPARITY_MAX, DEFAULT_DISPARITY_MIN, Conversion


def test_version_does_not_require_a_capture() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "0.1.0"


def test_cli_help_explains_contract_and_required_inputs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    assert "Usage: lff" in result.output
    assert "Light Field Foundry" in result.output
    assert "image-out" in result.output
    assert "doctor" in result.output
    result = runner.invoke(app, ["image-out", "--help"], color=False)
    assert result.exit_code == 0
    for flag in (
        "--lfp-raw",
        "--lfp-recipe",
        "--lfp-depth",
        "--lfp-edof",
        "--recompute",
        "--no-recompute",
        "--calibration",
        "--output",
        "--perspective-u",
        "--baseline",
        "--stereo-pivot",
        "--disparity-min",
        "--disparity-max",
    ):
        assert flag in result.output
    assert "--tnt" not in result.output
    assert "--doctor" not in result.output


def test_missing_tool_fails_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNT_PATH", "/not/a/tnt")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "not an executable" in result.output
    assert "Traceback" not in result.output


def test_image_out_exports_a_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Conversion] = []
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: object())

    def export(options: Conversion, _tools: object, _report: object) -> dict[str, object]:
        calls.append(options)
        return {"output": str(options.output), "bytes": 123, "sha256": "verified"}

    monkeypatch.setattr(cli, "convert", export)
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            "raw.lfp",
            "--calibration",
            "cameras",
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            "0.2",
            "--focus",
            "0",
            "--stereo-pivot",
            "0",
            "--aperture",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    options = calls[0]
    assert options.capture == Path("raw.lfp")
    assert options.output == Path("photo.avif")
    assert options.perspective_u == 0
    assert options.perspective_v == 0
    assert options.focus == 0
    assert options.stereo_pivot == 0
    assert options.aperture == 0
    assert options.edof_center
    assert options.disparity_min == DEFAULT_DISPARITY_MIN == 0
    assert options.disparity_max == DEFAULT_DISPARITY_MAX == 4.316406011581421


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["--disparity-min", "1.5", "--disparity-max", "6"], (1.5, 6)),
        (["--disparity-min", "0"], (0, DEFAULT_DISPARITY_MAX)),
        (["--disparity-min", "1.5"], (1.5, DEFAULT_DISPARITY_MAX)),
        (["--disparity-max", "1"], (DEFAULT_DISPARITY_MIN, 1)),
    ],
)
def test_disparity_endpoints_are_independent_cli_overrides(
    monkeypatch: pytest.MonkeyPatch, overrides: list[str], expected: tuple[float, float]
) -> None:
    calls: list[Conversion] = []
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: object())

    def export(options: Conversion, _tools: object, _report: object) -> dict[str, object]:
        calls.append(options)
        return {"output": str(options.output), "bytes": 123, "sha256": "verified"}

    monkeypatch.setattr(cli, "convert", export)
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            "raw.lfp",
            "--calibration",
            "cameras",
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            "0.2",
            *overrides,
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert (calls[0].disparity_min, calls[0].disparity_max) == expected


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [("-1", "2"), ("0", "0"), ("2", "1"), ("nan", "2"), ("0", "inf")],
)
def test_invalid_disparity_range_fails_before_tool_discovery(
    monkeypatch: pytest.MonkeyPatch, minimum: str, maximum: str
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            "raw.lfp",
            "--calibration",
            "cameras",
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            "0.2",
            "--disparity-min",
            minimum,
            "--disparity-max",
            maximum,
        ],
    )
    assert result.exit_code != 0
    assert "disparity" in result.output.lower()
    assert "Traceback" not in result.output


def test_refocus_option_selects_a_focused_center(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Conversion] = []
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: object())

    def export(options: Conversion, _tools: object, _report: object) -> dict[str, object]:
        calls.append(options)
        return {"output": str(options.output), "bytes": 123, "sha256": "verified"}

    monkeypatch.setattr(cli, "convert", export)
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            "raw.lfp",
            "--calibration",
            "cameras",
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            "0.2",
            "--refocus",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert not calls[0].edof_center


def test_export_reports_missing_options_before_discovering_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["image-out", "raw.lfp"])
    assert result.exit_code == 2
    assert "Required for export" in result.output
    assert "--calibration" in result.output
    assert "--baseline" in result.output


def test_doctor_runs_without_an_image_input(monkeypatch: pytest.MonkeyPatch) -> None:
    discovered = []
    reports = []
    tools = object()
    monkeypatch.setattr(cli.Tools, "discover", lambda *args: discovered.append(args) or tools)
    monkeypatch.setattr(cli, "_doctor", reports.append)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert len(discovered) == 1
    assert reports == [tools]


@pytest.mark.parametrize(
    "overrides", [["--disparity-min", "0"], ["--disparity-max", str(DEFAULT_DISPARITY_MAX)]]
)
def test_doctor_rejects_explicit_disparity_options_even_at_defaults(
    monkeypatch: pytest.MonkeyPatch, overrides: list[str]
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["doctor", *overrides])
    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_no_arguments_shows_help() -> None:
    result = CliRunner().invoke(app, [])
    assert "Usage:" in result.output
    assert "image-out" in result.output
    assert "doctor" in result.output


def test_library_option_supplies_calibration_and_is_rejected_by_doctor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: object())

    def export(options: Conversion, *_: object) -> dict[str, object]:
        calls.append(options)
        return {"output": str(options.output), "bytes": 1, "sha256": "verified"}

    monkeypatch.setattr(cli, "convert", export)
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            "stack.lfp",
            "--library",
            "sample.lytrolibrary",
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            ".2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0].library == Path("sample.lytrolibrary")
    assert calls[0].calibration == Path("sample.lytrolibrary/cameras")
    result = CliRunner().invoke(app, ["doctor", "--library", "sample.lytrolibrary"])
    assert result.exit_code != 0
    assert "Usage:" in result.output


def _role_export(
    monkeypatch: pytest.MonkeyPatch, *arguments: str
) -> tuple[Result, list[Conversion]]:
    calls: list[Conversion] = []
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: object())

    def export(options: Conversion, *_: object) -> dict[str, object]:
        calls.append(options)
        return {"output": str(options.output), "bytes": 1, "sha256": "verified"}

    monkeypatch.setattr(cli, "convert", export)
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            *arguments,
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            ".2",
        ],
    )
    return result, calls


def test_explicit_roles_allow_split_sources_without_a_positional_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _role_export(
        monkeypatch,
        "--lfp-raw",
        "fresh-export/img00581.lfp",
        "--lfp-recipe",
        "saved/stack.view.lfp",
        "--lfp-depth",
        "depth/stack.lfp",
        "--lfp-edof",
        "edof/stack.lfp",
        "--recipe",
        "overrides.json",
        "--calibration",
        "cameras",
    )
    assert result.exit_code == 0, result.output
    options = calls[0]
    assert options.capture is None
    assert options.library is None
    assert options.lfp_raw == Path("fresh-export/img00581.lfp")
    assert options.lfp_recipe == Path("saved/stack.view.lfp")
    assert options.lfp_depth == Path("depth/stack.lfp")
    assert options.lfp_edof == Path("edof/stack.lfp")
    assert options.recipe == Path("overrides.json")
    assert not options.recompute_depth
    assert not options.recompute_edof
    assert not options.no_recompute_depth
    assert not options.no_recompute_edof
    assert options.edof_center


@pytest.mark.parametrize("flag", ["--lfp-recipe", "--lfp-depth", "--lfp-edof"])
def test_any_processed_role_can_establish_identity_for_library_fallback(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    result, calls = _role_export(
        monkeypatch, flag, "saved/stack.lfp", "--library", "library.lytrolibrary"
    )
    assert result.exit_code == 0, result.output
    assert calls[0].capture is None
    assert calls[0].lfp_raw is None
    assert calls[0].calibration == Path("library.lytrolibrary/cameras")


def test_lfp_receipe_alias_is_distinct_from_json_recipe_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _role_export(
        monkeypatch,
        "--lfp-raw",
        "raw.lfp",
        "--lfp-receipe",
        "stack.view.lfp",
        "--recipe",
        "overrides.json",
        "--calibration",
        "cameras",
    )
    assert result.exit_code == 0, result.output
    assert calls[0].lfp_recipe == Path("stack.view.lfp")
    assert calls[0].recipe == Path("overrides.json")


@pytest.mark.parametrize(
    ("flag", "depth", "edof"),
    [
        ("--recompute-depth", True, False),
        ("--recompute-edof", False, True),
        ("--recompute", True, True),
    ],
)
def test_recompute_flags_override_requested_cache_roles(
    monkeypatch: pytest.MonkeyPatch, flag: str, depth: bool, edof: bool
) -> None:
    result, calls = _role_export(
        monkeypatch,
        "--lfp-raw",
        "raw.lfp",
        "--lfp-depth",
        "cached-depth.lfp",
        "--lfp-edof",
        "cached-edof.lfp",
        "--calibration",
        "cameras",
        flag,
    )
    assert result.exit_code == 0, result.output
    assert calls[0].recompute_depth is depth
    assert calls[0].recompute_edof is edof
    # Retain the explicit paths in the request so provenance can record ignored inputs.
    assert calls[0].lfp_depth == Path("cached-depth.lfp")
    assert calls[0].lfp_edof == Path("cached-edof.lfp")


def test_raw_role_without_library_requires_calibration_before_discovering_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            "--lfp-raw",
            "raw.lfp",
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            ".2",
        ],
    )
    assert result.exit_code == 2
    assert "--calibration" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["--lfp-raw", "raw.lfp"],
        ["--lfp-recipe", "recipe.lfp"],
        ["--lfp-receipe", "recipe.lfp"],
        ["--lfp-depth", "depth.lfp"],
        ["--lfp-edof", "edof.lfp"],
        ["--recompute-depth"],
        ["--recompute-edof"],
        ["--recompute"],
    ],
)
def test_doctor_rejects_each_role_and_recompute_option(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["doctor", *arguments])
    assert result.exit_code == 2
    assert "Usage:" in result.output


@pytest.mark.parametrize(
    ("flag", "depth", "edof"),
    [
        ("--no-recompute-depth", True, False),
        ("--no-recompute-edof", False, True),
        ("--no-recompute", True, True),
    ],
)
def test_no_recompute_flags_require_existing_cache_roles(
    monkeypatch: pytest.MonkeyPatch, flag: str, depth: bool, edof: bool
) -> None:
    result, calls = _role_export(
        monkeypatch,
        "--lfp-raw",
        "raw.lfp",
        "--lfp-depth",
        "cached-depth.lfp",
        "--lfp-edof",
        "cached-edof.lfp",
        "--calibration",
        "cameras",
        flag,
    )
    assert result.exit_code == 0, result.output
    assert calls[0].no_recompute_depth is depth
    assert calls[0].no_recompute_edof is edof
    assert not calls[0].recompute_depth
    assert not calls[0].recompute_edof
    assert calls[0].lfp_depth == Path("cached-depth.lfp")
    assert calls[0].lfp_edof == Path("cached-edof.lfp")


@pytest.mark.parametrize(
    "flags",
    [
        ["--recompute-depth", "--no-recompute-depth"],
        ["--recompute-edof", "--no-recompute-edof"],
        ["--recompute", "--no-recompute-depth"],
        ["--recompute", "--no-recompute-edof"],
        ["--recompute-depth", "--no-recompute"],
        ["--recompute-edof", "--no-recompute"],
        ["--recompute", "--no-recompute"],
    ],
)
def test_conflicting_recompute_policies_fail_before_tools(
    monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(
        app,
        [
            "image-out",
            "raw.lfp",
            "--calibration",
            "cameras",
            "--output",
            "photo.avif",
            "--perspective-u",
            "0",
            "--perspective-v",
            "0",
            "--baseline",
            ".2",
            *flags,
        ],
    )
    assert result.exit_code == 2
    assert "cannot be combined" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("flags", "force_depth", "force_edof", "require_depth", "require_edof"),
    [
        (["--no-recompute-depth", "--recompute-edof"], False, True, True, False),
        (["--recompute-depth", "--no-recompute-edof"], True, False, False, True),
        (["--no-recompute-depth", "--no-recompute-edof"], False, False, True, True),
        (["--no-recompute", "--no-recompute-depth"], False, False, True, True),
    ],
)
def test_independent_recompute_policies_can_be_combined(
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    force_depth: bool,
    force_edof: bool,
    require_depth: bool,
    require_edof: bool,
) -> None:
    result, calls = _role_export(
        monkeypatch,
        "--lfp-raw",
        "raw.lfp",
        "--calibration",
        "cameras",
        *flags,
    )
    assert result.exit_code == 0, result.output
    request = calls[0]
    assert request.recompute_depth is force_depth
    assert request.recompute_edof is force_edof
    assert request.no_recompute_depth is require_depth
    assert request.no_recompute_edof is require_edof


@pytest.mark.parametrize("flag", ["--no-recompute-edof", "--no-recompute"])
def test_no_recompute_edof_rejects_a_refocused_primary_before_tools(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["image-out", "raw.lfp", "--refocus", flag])
    assert result.exit_code == 2
    assert "requires an EDOF primary" in result.output
    assert "--refocus" in result.output


def test_no_recompute_depth_can_be_used_with_refocus(monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _role_export(
        monkeypatch,
        "--lfp-raw",
        "raw.lfp",
        "--calibration",
        "cameras",
        "--refocus",
        "--no-recompute-depth",
    )
    assert result.exit_code == 0, result.output
    assert calls[0].no_recompute_depth
    assert not calls[0].edof_center
    assert not calls[0].no_recompute_edof


@pytest.mark.parametrize("flag", ["--no-recompute-depth", "--no-recompute-edof", "--no-recompute"])
def test_doctor_rejects_no_recompute_options_before_tools(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["doctor", flag])
    assert result.exit_code == 2
    assert "Usage:" in result.output
