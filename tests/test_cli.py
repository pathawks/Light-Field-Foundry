from pathlib import Path

import pytest
from typer.testing import CliRunner

from lytro_avif import cli
from lytro_avif.cli import app
from lytro_avif.pipeline import DEFAULT_DISPARITY_MAX, DEFAULT_DISPARITY_MIN, Conversion


def test_version_does_not_require_a_capture() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.strip() == "0.1.0"


def test_cli_help_explains_contract_and_required_inputs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    for flag in (
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
    assert "--doctor" in result.output
    assert "Commands" not in result.output


def test_missing_tool_fails_without_traceback(monkeypatch) -> None:
    monkeypatch.setenv("TNT_PATH", "/not/a/tnt")
    result = CliRunner().invoke(app, ["--doctor"])
    assert result.exit_code == 1
    assert "not an executable" in result.output and "Traceback" not in result.output


def test_capture_is_exported_without_a_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Conversion] = []
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: object())

    def export(options: Conversion, _tools: object, _report: object) -> dict[str, object]:
        calls.append(options)
        return {"output": str(options.output), "bytes": 123, "sha256": "verified"}

    monkeypatch.setattr(cli, "convert", export)
    result = CliRunner().invoke(
        app,
        [
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
    assert options.capture == Path("raw.lfp") and options.output == Path("photo.avif")
    assert options.perspective_u == 0 and options.perspective_v == 0
    assert options.focus == 0 and options.stereo_pivot == 0 and options.aperture == 0
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
    assert "disparity" in result.output.lower() and "Traceback" not in result.output


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
    assert len(calls) == 1 and not calls[0].edof_center


def test_export_reports_missing_options_before_discovering_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["raw.lfp"])
    assert result.exit_code == 2
    assert "Required for export" in result.output
    assert "--calibration" in result.output and "--baseline" in result.output


def test_doctor_rejects_export_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["--doctor", "raw.lfp"])
    assert result.exit_code == 2
    assert "cannot be combined" in result.output


@pytest.mark.parametrize(
    "overrides", [["--disparity-min", "0"], ["--disparity-max", str(DEFAULT_DISPARITY_MAX)]]
)
def test_doctor_rejects_explicit_disparity_options_even_at_defaults(
    monkeypatch: pytest.MonkeyPatch, overrides: list[str]
) -> None:
    monkeypatch.setattr(cli.Tools, "discover", lambda *_: pytest.fail("tools should not run"))
    result = CliRunner().invoke(app, ["--doctor", *overrides])
    assert result.exit_code == 2
    assert "cannot be combined" in result.output


def test_no_arguments_shows_help() -> None:
    result = CliRunner().invoke(app, [])
    assert "Usage:" in result.output and "--doctor" in result.output
