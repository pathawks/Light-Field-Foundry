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
    source = SimpleNamespace(path=capture, public_json=(b"{}",), private_json=(), master_json=b"{}")
    selected = SimpleNamespace(
        raw=source, metadata=source, recipe=None, depth=None, edof=None, provenance={}
    )
    state["sources"] = selected
    monkeypatch.setattr(pipeline, "select_sources", lambda *_, **__: selected)
    monkeypatch.setattr(
        pipeline, "stage_raw", lambda source, target: target.write_bytes(source.path.read_bytes())
    )
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
    monkeypatch.setattr(
        pipeline,
        "extract_edof",
        lambda _, **__: SimpleNamespace(
            witness_image=depth.witness_image, view={"recipe": recipe}, image_size=(16, 16)
        ),
    )
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
    assert "--unpack" in depth_command  # Native TIFF witnesses measure cropped depth geometry.
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


@pytest.mark.parametrize("provide_library", [True, False])
def test_conversion_refuses_publication_inside_read_only_library(
    tmp_path, monkeypatch, provide_library
) -> None:
    options, tools, _, _ = _mock_conversion(tmp_path, monkeypatch)
    root = tmp_path / "library.lytrolibrary"
    root.mkdir()
    alias = tmp_path / "library-alias"
    alias.symlink_to(root, target_is_directory=True)
    options = replace(options, library=root if provide_library else None, output=alias / "new.avif")
    with pytest.raises(LytroError, match="read-only Lytro library"):
        pipeline.convert(options, tools)
    assert not list(root.iterdir())


@pytest.mark.parametrize("require_existing", [False, True])
def test_exported_stack_uses_original_center_depth_and_verified_raw_for_eyes(
    tmp_path, monkeypatch, require_existing
) -> None:
    options, tools, _, state = _mock_conversion(tmp_path, monkeypatch)
    options = replace(
        options, no_recompute_depth=require_existing, no_recompute_edof=require_existing
    )
    selected = state["sources"]
    stack = SimpleNamespace(
        path=options.capture,
        public_json=(b'{"exported":true}',),
        private_json=(),
        master_json=b'{"stack":true}',
    )
    selected.recipe = selected.depth = selected.edof = stack
    monkeypatch.setattr(pipeline, "extract_recipe", lambda _: {"viewFocusSpread": 25})
    monkeypatch.setattr(pipeline, "_check_cached_edof", lambda *_: None)
    depth = SimpleNamespace(
        mapping_provenance={"source": "original 8-bit depth"},
        witness_image=b"",
        image_size=(16, 16),
    )
    monkeypatch.setattr(pipeline, "extract_depth", lambda *_, **__: depth)
    edof = SimpleNamespace(witness_image=b"original exported center TIFF", view={})
    monkeypatch.setattr(pipeline, "extract_edof", lambda *_, **__: edof)
    original_tiff = pipeline.read_tiff
    monkeypatch.setattr(pipeline, "read_tiff", lambda p, **_: original_tiff(p))
    original_assemble = pipeline.bmff.assemble
    flags = []

    def assemble(*args, **kwargs):
        flags.append(kwargs)
        return original_assemble(*args)

    monkeypatch.setattr(pipeline.bmff, "assemble", assemble)
    metadata = []
    monkeypatch.setattr(
        pipeline, "build_metadata", lambda *args, **kwargs: metadata.append((args, kwargs))
    )
    pipeline.convert(options, tools, lambda _: None)
    assert state["center_tiff"] == b"original exported center TIFF"
    commands = state["commands"]
    assert not any("--lfp-out" in command for command in commands)
    eyes = [command for command in commands if "--image-out" in command]
    assert len(eyes) == 2
    assert all(
        Path(command[command.index("--lfp-in") + 1]).name == "raw-input.lfp" for command in eyes
    )
    assert flags == [{"separate_center_profile": True}]
    assert metadata[0][0][0] == selected.raw.public_json
    assert metadata[0][1]["master_json"] == selected.metadata.master_json


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recompute_depth": True, "no_recompute_depth": True},
        {"recompute_edof": True, "no_recompute_edof": True},
        {"edof_center": False, "no_recompute_edof": True},
    ],
)
def test_invalid_recompute_policy_fails_before_any_processing(tmp_path, monkeypatch, kwargs):
    options, tools, workdirs, state = _mock_conversion(tmp_path, monkeypatch)

    def unexpected(*_, **__):
        pytest.fail("Invalid policy reached source discovery or native processing")

    monkeypatch.setattr(pipeline, "select_sources", unexpected)
    monkeypatch.setattr(pipeline, "doctor", unexpected)
    with pytest.raises(LytroError, match="no-recompute"):
        pipeline.convert(replace(options, **kwargs), tools)
    assert not workdirs and not state.get("commands")
    assert not options.output.exists()


@pytest.mark.parametrize(
    "kwargs,flag",
    [
        ({"no_recompute_depth": True}, "--no-recompute-depth"),
        ({"no_recompute_edof": True}, "--no-recompute-edof"),
        ({"no_recompute_depth": True, "no_recompute_edof": True}, "--no-recompute-depth"),
    ],
)
def test_required_missing_cache_fails_before_doctor_or_native(tmp_path, monkeypatch, kwargs, flag):
    options, tools, workdirs, state = _mock_conversion(tmp_path, monkeypatch)

    def unexpected(*_, **__):
        pytest.fail("Missing required cache reached native processing")

    monkeypatch.setattr(pipeline, "doctor", unexpected)
    with pytest.raises(LytroError, match=flag):
        pipeline.convert(replace(options, **kwargs), tools)
    assert not workdirs and not state.get("commands")
    assert not options.output.exists()


@pytest.mark.parametrize("role", ["depth", "edof"])
def test_unusable_required_cache_never_falls_back_to_recomputation(tmp_path, monkeypatch, role):
    options, tools, workdirs, state = _mock_conversion(tmp_path, monkeypatch)
    selected = state["sources"]
    setattr(selected, role, SimpleNamespace(path=options.capture))

    def unusable(*_, **__):
        raise LytroError("Selected cache is not a lossless supported encoding")

    monkeypatch.setattr(pipeline, "extract_" + role, unusable)

    def unexpected(*_, **__):
        pytest.fail("Unusable required cache reached native processing")

    monkeypatch.setattr(pipeline, "doctor", unexpected)
    with pytest.raises(LytroError, match="lossless"):
        pipeline.convert(replace(options, **{"no_recompute_" + role: True}), tools)
    assert not workdirs and not state.get("commands")
    assert not options.output.exists()


@pytest.mark.parametrize("role", ["depth", "edof"])
def test_required_cache_can_be_combined_with_other_automatic_role(tmp_path, monkeypatch, role):
    options, tools, _, state = _mock_conversion(tmp_path, monkeypatch)
    selected = state["sources"]
    setattr(selected, role, SimpleNamespace(path=options.capture))
    if role == "depth":
        cached_depth = SimpleNamespace(
            mapping_provenance={}, witness_image=b"", image_size=(16, 16)
        )
        monkeypatch.setattr(pipeline, "extract_depth", lambda *_, **__: cached_depth)
    original_reader = pipeline.read_tiff
    monkeypatch.setattr(pipeline, "read_tiff", lambda path, **_: original_reader(path))
    original_assemble = pipeline.bmff.assemble
    monkeypatch.setattr(pipeline.bmff, "assemble", lambda *args, **_: original_assemble(*args))
    pipeline.convert(replace(options, **{"no_recompute_" + role: True}), tools, lambda _: None)
    commands = state["commands"]
    assert sum("--lfp-out" in command for command in commands) == 1
    assert sum("--image-out" in command for command in commands) == 2
    assert options.output.exists()


@pytest.mark.parametrize("failure", ["depth-geometry", "edof-finishing", "edof-color"])
def test_required_cache_late_validation_failure_prevents_other_role_generation(
    tmp_path, monkeypatch, failure
):
    options, tools, workdirs, state = _mock_conversion(tmp_path, monkeypatch)
    selected = state["sources"]
    role = "depth" if failure == "depth-geometry" else "edof"
    setattr(selected, role, SimpleNamespace(path=options.capture))

    def mismatch(*_, **__):
        raise LytroError("Required cached asset has incompatible data")

    if failure == "depth-geometry":
        cached = SimpleNamespace(mapping_provenance={}, witness_image=b"", image_size=(16, 16))
        monkeypatch.setattr(pipeline, "extract_depth", lambda *_, **__: cached)
        monkeypatch.setattr(pipeline, "validate_alignment", mismatch)
    elif failure == "edof-finishing":
        monkeypatch.setattr(pipeline, "_check_cached_edof", mismatch)
    else:
        monkeypatch.setattr(pipeline, "read_tiff", mismatch)
    with pytest.raises(LytroError, match="incompatible data"):
        pipeline.convert(replace(options, **{"no_recompute_" + role: True}), tools, lambda _: None)
    assert state["commands"]  # Recipe resolution occurred; no photo was generated.
    assert all(
        "--lfp-out" not in command and "--image-out" not in command for command in state["commands"]
    )
    assert all(not directory.exists() for directory in workdirs)
    assert not options.output.exists()
