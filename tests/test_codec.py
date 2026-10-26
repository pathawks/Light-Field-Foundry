import sys
from pathlib import Path
from typing import Never

import imagecodecs
import numpy as np
import pytest

from light_field_foundry import bmff
from light_field_foundry.codec import encode_depth, read_y4m
from light_field_foundry.errors import LytroError
from light_field_foundry.tools import Tools


@pytest.mark.parametrize("strided", [False, True])
def test_depth_encoder_preserves_all_codes_without_external_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strided: bool
) -> None:
    samples = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    if strided:
        samples = samples.T
    original = samples.copy()
    executable = Path(sys.executable)
    tools = Tools(executable, executable, executable, threads=2)

    def external_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Depth encoding must not launch an external tool")

    monkeypatch.setattr(Tools, "run", external_call)
    path = encode_depth(tools, tmp_path, samples)
    encoded = path.read_bytes()
    bmff.validate_depth(bmff.parse(encoded))
    np.testing.assert_array_equal(imagecodecs.avif_decode(encoded), samples)
    np.testing.assert_array_equal(samples, original)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(4, dtype=np.uint16),
        np.zeros((2, 2, 1), dtype=np.uint16),
        np.zeros((0, 2), dtype=np.uint16),
        np.array([[4096]], dtype=np.uint16),
    ],
)
def test_invalid_depth_is_rejected_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, samples: np.ndarray
) -> None:
    executable = Path(sys.executable)
    tools = Tools(executable, executable, executable)

    def encode(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Invalid depth reached the encoder")

    monkeypatch.setattr(imagecodecs, "avif_encode", encode)
    with pytest.raises(LytroError, match=r"uint16 plane with codes 0\.\.4095"):
        encode_depth(tools, tmp_path, samples)
    assert not list(tmp_path.iterdir())


def test_depth_encoder_failure_is_reported_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = Path(sys.executable)
    tools = Tools(executable, executable, executable)

    def encode(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("No codec available")

    monkeypatch.setattr(imagecodecs, "avif_encode", encode)
    with pytest.raises(LytroError, match="imagecodecs depth encoding failed: No codec available"):
        encode_depth(tools, tmp_path, np.zeros((2, 2), dtype=np.uint16))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "body",
    [
        b"invalid",
        b"YUV4MPEG2 W2 H2 Cmono12\nFRAME\n\0",
        b"YUV4MPEG2 W2 H2 C420\nFRAME\n",
        b"YUV4MPEG2 W1 H1 Cmono12\nFRAME\n\xff\xff",
    ],
)
def test_malformed_native_planes_fail(tmp_path: Path, body: bytes) -> None:
    path = tmp_path / "bad.y4m"
    path.write_bytes(body)
    with pytest.raises(LytroError):
        read_y4m(path)
