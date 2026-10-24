"""Failure/publication lifecycle checks with generated inputs and mocked tools."""

from __future__ import annotations

import errno
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lytro_avif import pipeline
from lytro_avif.errors import LytroError
from lytro_avif.images import RGB16
from lytro_avif.tools import Tools


def test_atomic_publish_never_replaces_existing_file(tmp_path: Path) -> None:
    stage = tmp_path / ".stage"
    target = tmp_path / "photo.avif"
    stage.write_bytes(b"verified candidate")
    target.write_bytes(b"original output")
    with pytest.raises(LytroError, match="already exists"):
        pipeline._publish(stage, target)
    assert target.read_bytes() == b"original output"
    assert stage.read_bytes() == b"verified candidate"


@pytest.mark.parametrize("collision", [False, True])
def test_cross_filesystem_publication_cleans_its_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: bool
) -> None:
    stage = tmp_path / "verified.avif"
    target = tmp_path / "photo.avif"
    stage.write_bytes(b"verified complete AVIF")
    if collision:
        target.write_bytes(b"existing output")
    original_link = pipeline.os.link
    copies: list[Path] = []

    def link(source: Path, destination: Path) -> None:
        if source == stage:
            raise OSError(errno.EXDEV, "simulated different filesystem")
        copies.append(source)
        original_link(source, destination)

    monkeypatch.setattr(pipeline.os, "link", link)
    if collision:
        with pytest.raises(LytroError, match="already exists"):
            pipeline._publish(stage, target)
    else:
        pipeline._publish(stage, target)
    assert target.read_bytes() == (b"existing output" if collision else stage.read_bytes())
    assert copies and all(not copy.parent.exists() for copy in copies)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["photo.avif", "verified.avif"]


def _mock_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pipeline.Conversion, Tools, list[Path], dict[str, Any]]:
    monkeypatch.setattr(sys, "platform", "darwin")
    capture = tmp_path / "source.lfr"
    capture.write_bytes(b"original capture")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "calibration.json").write_bytes(b"original calibration")
    executable = tmp_path / "tnt"
    executable.write_bytes(b"external executable fixture remains unchanged")
    output = tmp_path / "photo.avif"
    options = pipeline.Conversion(capture, calibration, output, 0, 0, 0.1)
    tools = Tools(executable, executable, executable)
    workdirs: list[Path] = []
    state: dict[str, Any] = {}
    info = SimpleNamespace(public_json=(b"{}",), private_json=(), master_json=b"{}")
    monkeypatch.setattr(pipeline, "read_lfp", lambda _: SimpleNamespace(admit_capture=lambda: info))
    monkeypatch.setattr(Tools, "capabilities", lambda *_: {"tnt": "TNT fixture"})
    monkeypatch.setattr(
        pipeline,
        "doctor",
        lambda *_: {"tnt": "TNT fixture", "avifenc": "fixture", "avifdec": "fixture"},
    )

    def run(_tools: Tools, args: list[Any], cwd: Path) -> str:
        state.setdefault("commands", []).append(args)
        if cwd not in workdirs:
            workdirs.append(cwd)
        for flag in ("--recipe-out", "--image-out", "--lfp-out"):
            if flag in args:
                Path(args[args.index(flag) + 1]).write_bytes(b"{}")
        return "TNT 2.0 powered by LFE Lytro Lightfield Engine 21.000000"

    monkeypatch.setattr(Tools, "run", run)
    recipe = {"viewPerspectiveU": 0, "viewPerspectiveV": 0, "viewFocus": 0}
    resolved = SimpleNamespace(center=recipe, left=recipe, right=recipe, provenance={})
    monkeypatch.setattr(pipeline, "resolve_recipe", lambda *_, **__: resolved)
    monkeypatch.setattr(pipeline, "validate_render_log", lambda *_: None)
    image = RGB16(np.zeros((16, 16, 3), dtype=np.uint16), b"original ICC", {})

    def read_tiff(path: Path) -> RGB16:
        if path.name == "center.tiff":
            state["center_tiff"] = path.read_bytes()
        return image

    monkeypatch.setattr(pipeline, "read_tiff", read_tiff)
    depth = SimpleNamespace(mapping_provenance={}, witness_image=b"native EDOF center TIFF")
    monkeypatch.setattr(pipeline, "extract_depth", lambda _, **__: depth)
    monkeypatch.setattr(pipeline, "validate_alignment", lambda *_: None)
    monkeypatch.setattr(pipeline, "quantize_depth", lambda _: np.zeros((16, 16), np.uint16))

    def encode(_tools: Tools, work: Path, name: str, _image: RGB16) -> Path:
        path = work / f"{name}.avif"
        path.write_bytes(b"coded image")
        return path

    monkeypatch.setattr(pipeline, "encode_color", encode)
    monkeypatch.setattr(
        pipeline, "encode_depth", lambda _tools, work, _codes: encode(_tools, work, "depth", image)
    )
    monkeypatch.setattr(pipeline, "build_metadata", lambda *_, **__: object())

    def assemble(*args: Any) -> object:
        stage = Path(args[-1])
        state["stage"] = stage
        stage.write_bytes(b"complete assembled candidate")
        return object()

    monkeypatch.setattr(pipeline.bmff, "assemble", assemble)
    monkeypatch.setattr(pipeline, "verify_final", lambda *_: None)
    monkeypatch.setattr(pipeline, "verify_native", lambda *_: {"verified": True})
    return options, tools, workdirs, state


@pytest.mark.parametrize("failure", ["codec", "assembly", "final-decode", "native", "interruption"])
def test_failure_removes_all_owned_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    options, tools, workdirs, state = _mock_conversion(tmp_path, monkeypatch)
    original_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def fail(*_args: Any, **_kwargs: Any) -> None:
        state["injected_failure_reached"] = True
        if failure == "interruption":
            raise KeyboardInterrupt
        raise LytroError("injected contract failure")

    if failure == "codec":
        monkeypatch.setattr(pipeline, "encode_color", fail)
    elif failure == "assembly":
        monkeypatch.setattr(pipeline.bmff, "assemble", fail)
    elif failure == "final-decode":
        monkeypatch.setattr(pipeline, "verify_final", fail)
    else:
        monkeypatch.setattr(pipeline, "verify_native", fail)
    with pytest.raises(KeyboardInterrupt if failure == "interruption" else LytroError):
        pipeline.convert(options, tools, lambda _: None)
    assert state.get("injected_failure_reached") is True
    assert not options.output.exists()
    assert workdirs and all(not path.exists() for path in workdirs)
    assert "stage" not in state or not state["stage"].exists()
    assert not list(tmp_path.glob(".photo.avif.lytro-avif-*"))
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == original_files


def test_success_publishes_one_file_and_preserves_external_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, tools, workdirs, state = _mock_conversion(tmp_path, monkeypatch)
    original_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = pipeline.convert(options, tools, lambda _: None)
    assert options.output.read_bytes() == b"complete assembled candidate"
    assert result["bytes"] == options.output.stat().st_size
    assert len(str(result["sha256"])) == 64
    assert workdirs and all(not path.exists() for path in workdirs)
    assert not state["stage"].exists()
    assert not list(tmp_path.glob(".photo.avif.lytro-avif-*"))
    remaining = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and path != options.output
    }
    assert remaining == original_files


def test_output_collision_after_validation_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, tools, workdirs, _ = _mock_conversion(tmp_path, monkeypatch)

    def race(*_: Any) -> dict[str, bool]:
        options.output.write_bytes(b"another writer won")
        return {"verified": True}

    monkeypatch.setattr(pipeline, "verify_native", race)
    with pytest.raises(LytroError, match="already exists"):
        pipeline.convert(options, tools, lambda _: None)
    assert options.output.read_bytes() == b"another writer won"
    assert all(not path.exists() for path in workdirs)
    assert not list(tmp_path.glob(".photo.avif.lytro-avif-*"))


def test_existing_output_fails_before_any_native_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, tools, workdirs, _ = _mock_conversion(tmp_path, monkeypatch)
    options.output.write_bytes(b"existing output")
    with pytest.raises(LytroError, match="already exists"):
        pipeline.convert(options, tools, lambda _: None)
    assert options.output.read_bytes() == b"existing output"
    assert workdirs == []


def test_default_center_uses_native_edof_tiff_and_two_independent_eyes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, tools, _, state = _mock_conversion(tmp_path, monkeypatch)
    pipeline.convert(options, tools, lambda _: None)
    commands = state["commands"]
    depth_command = next(args for args in commands if "--lfp-out" in args)
    assert "--unpack" in depth_command
    stills = [args for args in commands if "--image-out" in args]
    assert len(stills) == 2
    assert {Path(args[args.index("--image-out") + 1]).name for args in stills} == {
        "left.tiff",
        "right.tiff",
    }
    assert state["center_tiff"] == b"native EDOF center TIFF"


def test_refocus_retains_three_raw_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, tools, _, state = _mock_conversion(tmp_path, monkeypatch)
    pipeline.convert(replace(options, edof_center=False), tools, lambda _: None)
    commands = state["commands"]
    depth_command = next(args for args in commands if "--lfp-out" in args)
    assert "--unpack" not in depth_command
    stills = [args for args in commands if "--image-out" in args]
    assert len(stills) == 3
    assert state["center_tiff"] == b"{}"


@pytest.mark.parametrize("endpoints", [None, (1.5, 6.0)])
def test_disparity_range_reaches_metadata_and_native_verification_without_changing_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoints: tuple[float, float] | None
) -> None:
    options, tools, _, _ = _mock_conversion(tmp_path, monkeypatch)
    if endpoints is not None:
        options = replace(options, disparity_min=endpoints[0], disparity_max=endpoints[1])
    expected = endpoints or (0.0, 4.316406011581421)
    codes = np.arange(256, dtype=np.uint16).reshape(16, 16)
    source_mapping = {"lambda_at_code_0": 30.0, "lambda_at_code_4095": -14.0}
    depth = SimpleNamespace(
        mapping_provenance=source_mapping, witness_image=b"native EDOF center TIFF"
    )
    monkeypatch.setattr(pipeline, "extract_depth", lambda _, **__: depth)
    monkeypatch.setattr(pipeline, "quantize_depth", lambda _: codes)
    encoded: list[np.ndarray] = []
    original_encode = pipeline.encode_depth

    def encode_depth(tools: Tools, work: Path, prepared: np.ndarray) -> Path:
        encoded.append(prepared.copy())
        return original_encode(tools, work, prepared)

    monkeypatch.setattr(pipeline, "encode_depth", encode_depth)
    metadata_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def metadata(*args: Any, **kwargs: Any) -> object:
        metadata_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(pipeline, "build_metadata", metadata)
    native_calls: list[tuple[float, float]] = []

    def native(
        _tools: Tools,
        _work: Path,
        _stage: Path,
        prepared: np.ndarray,
        a: float = 0,
        b: float = 1,
    ) -> dict[str, bool]:
        np.testing.assert_array_equal(prepared, codes)
        native_calls.append((a, b))
        return {"verified": True}

    monkeypatch.setattr(pipeline, "verify_native", native)
    pipeline.convert(options, tools, lambda _: None)
    assert len(metadata_calls) == 1
    args, kwargs = metadata_calls[0]
    assert args[6:8] == expected
    assert args[3]["relativeDisparity"]["atCode0"] == expected[0]
    assert args[3]["relativeDisparity"]["atCode4095"] == expected[1]
    assert args[3]["depth"] == source_mapping
    assert kwargs["source_depth_mapping"] == source_mapping
    assert native_calls == [expected]
    assert len(encoded) == 1
    np.testing.assert_array_equal(encoded[0], codes)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (-1.0, 2.0),
        (0.0, 0.0),
        (2.0, 1.0),
        (float("nan"), 2.0),
        (0.0, float("inf")),
        (0.0, 3.5e38),
        (0.0, float(np.finfo(np.float32).max)),
        (1e8, 1e8 + 1),
        (1.0, 1.000001),
        (0.0, 1e-40),
        (0.0, 1e-50),
    ],
)
def test_invalid_disparity_range_fails_before_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimum: float, maximum: float
) -> None:
    options, tools, workdirs, _ = _mock_conversion(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "doctor", lambda *_: pytest.fail("native tools should not run"))
    with pytest.raises(LytroError, match="[Dd]isparity"):
        pipeline.convert(
            replace(options, disparity_min=minimum, disparity_max=maximum), tools, lambda _: None
        )
    assert workdirs == []
    assert not options.output.exists()
