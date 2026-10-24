from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from lytro_avif.errors import LytroError
from lytro_avif.lfp import (
    CHUNK_MAGIC,
    FILE_MAGIC,
    MASTER_MAGIC,
    MAX_SECTION_BYTES,
    FloatDepth,
    extract_depth,
    quantize_depth,
    read_json_bytes,
    read_lfp,
)


def _reference(data: bytes) -> str:
    return "sha1-" + hashlib.sha1(data).hexdigest()


def _container(path: Path, master: bytes, *payloads: bytes) -> Path:
    output = bytearray(struct.pack(">8sII", FILE_MAGIC, 1, 0))
    for index, data in enumerate((master, *payloads)):
        output.extend(
            struct.pack(
                ">8sq80s",
                MASTER_MAGIC if index == 0 else CHUNK_MAGIC,
                len(data),
                _reference(data).encode(),
            )
        )
        output.extend(data)
        output.extend(b"\0" * (-len(output) % 16))
    path.write_bytes(output)
    return path


def test_original_json_tokens_and_bytes_are_preserved(tmp_path: Path) -> None:
    public = b'{ "signedZero": -0, "value": -0.0 }\n'
    master = (
        json.dumps({"frames": [{"frame": {"metadataRef": _reference(public)}}]}).encode() + b"\n"
    )
    lfp = read_lfp(_container(tmp_path / "source.lfp", master, public))
    assert lfp.master_json == master
    assert lfp.public_json == (public,)
    assert lfp.capture_metadata()["signedZero"] == 0


@pytest.mark.parametrize(
    "data",
    [
        b"",
        FILE_MAGIC,
        struct.pack(">8sII", FILE_MAGIC, 2, 0),
        struct.pack(">8sII", FILE_MAGIC, 1, 0),
    ],
)
def test_short_or_unsupported_container_fails(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "bad.lfp"
    path.write_bytes(data)
    with pytest.raises(LytroError):
        read_lfp(path)


def test_digest_mismatch_fails_before_parsing(tmp_path: Path) -> None:
    path = _container(tmp_path / "bad.lfp", b"{}")
    data = bytearray(path.read_bytes())
    data[112] ^= 1
    path.write_bytes(data)
    with pytest.raises(LytroError, match="digest mismatch"):
        read_lfp(path)


def test_truncated_or_negative_extent_fails(tmp_path: Path) -> None:
    path = _container(tmp_path / "bad.lfp", b"{}")
    for size in (-1, 1 << 35, 1000):
        data = bytearray(path.read_bytes())
        data[24:32] = struct.pack(">q", size)
        path.write_bytes(data)
        with pytest.raises(LytroError):
            read_lfp(path)


@pytest.mark.parametrize(
    "data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b"[]", b"{"]
)
def test_strict_json_reader(data: bytes) -> None:
    with pytest.raises(LytroError):
        read_json_bytes(data)


def test_processed_only_capture_rejected(tmp_path: Path) -> None:
    lfp = read_lfp(_container(tmp_path / "processed.lfp", b'{"frames":[{"frame":{}}]}'))
    with pytest.raises(LytroError, match="processed-only"):
        lfp.admit_capture()


def _depth_container(path: Path, values: np.ndarray | None = None) -> Path:
    values = np.array([[0, 1], [2, 3]], dtype="<f4") if values is None else values
    payload = values.tobytes()
    witness = b"synthetic tiff witness bytes"
    descriptor = {
        "width": 2,
        "height": 2,
        "representation": "packedFloat",
        "minLambda": 0,
        "maxLambda": 3,
        "imageRef": _reference(payload),
    }
    master = {
        "views": [
            {
                "accelerations": [
                    {
                        "kind": "edofWarp",
                        "width": 2,
                        "height": 2,
                        "depthMap": descriptor,
                        "representation": "tiff",
                        "perImage": [
                            {"perspective": {"u": 0, "v": 0}, "imageRef": _reference(witness)}
                        ],
                    }
                ]
            }
        ]
    }
    return _container(path, json.dumps(master).encode(), payload, witness)


def test_float_depth_descriptor_exact_extent_and_little_endian(tmp_path: Path) -> None:
    depth = extract_depth(_depth_container(tmp_path / "depth.lfp"))
    np.testing.assert_array_equal(depth.samples, [[0, 1], [2, 3]])
    assert (depth.min_lambda, depth.max_lambda) == (0, 3)
    assert depth.witness_image == b"synthetic tiff witness bytes"


def _unpacked_depth_container(
    path: Path, depth_path: object = "depth.dat", witness_path: object = "center.tiff"
) -> Path:
    master = {
        "views": [
            {
                "accelerations": [
                    {
                        "kind": "edofWarp",
                        "width": 2,
                        "height": 2,
                        "representation": "tiff",
                        "depthMap": {
                            "width": 2,
                            "height": 2,
                            "representation": "packedFloat",
                            "minLambda": 0,
                            "maxLambda": 3,
                            "imagePath": depth_path,
                        },
                        "perImage": [{"perspective": {"u": 0, "v": 0}, "imagePath": witness_path}],
                    }
                ]
            }
        ],
    }
    return _container(path, json.dumps(master).encode())


@pytest.mark.parametrize("absolute", [False, True])
def test_owned_unpacked_depth_and_tiff_witness_are_read(tmp_path: Path, absolute: bool) -> None:
    root = tmp_path / "run"
    root.mkdir()
    assets = root / "assets"
    assets.mkdir()
    values = np.array([[0, 1], [2, 3]], dtype="<f4")
    (assets / "depth.dat").write_bytes(values.tobytes())
    (assets / "center.tiff").write_bytes(b"original generated tiff with ICC")
    prefix = str(assets) + "/" if absolute else "assets/"
    path = _unpacked_depth_container(
        root / "depth.lfp", prefix + "depth.dat", prefix + "center.tiff"
    )
    depth = extract_depth(path, asset_root=root)
    np.testing.assert_array_equal(depth.samples, values)
    assert depth.witness_image == b"original generated tiff with ICC"
    assert depth.witness_perspective == (0, 0) and depth.image_size == (2, 2)


def test_unpacked_paths_are_disabled_without_explicit_root(tmp_path: Path) -> None:
    path = _unpacked_depth_container(tmp_path / "depth.lfp")
    (tmp_path / "depth.dat").write_bytes(np.zeros((2, 2), dtype="<f4").tobytes())
    with pytest.raises(LytroError, match="must be embedded"):
        extract_depth(path)


@pytest.mark.parametrize("field", ["depth", "witness"])
@pytest.mark.parametrize(
    "kind", ["absolute-outside", "traversal", "symlink-file", "symlink-directory"]
)
def test_unpacked_assets_cannot_escape_or_follow_symlinks(
    tmp_path: Path, field: str, kind: str
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    values = np.array([[0, 1], [2, 3]], dtype="<f4").tobytes()
    (root / "depth.dat").write_bytes(values)
    (root / "center.tiff").write_bytes(b"generated witness")
    outside = tmp_path / "outside.dat"
    outside.write_bytes(values)
    if kind == "absolute-outside":
        value = str(outside)
    elif kind == "traversal":
        value = "../outside.dat"
    elif kind == "symlink-file":
        (root / "link").symlink_to(outside)
        value = "link"
    else:
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
        value = "link/outside.dat"
    path = _unpacked_depth_container(
        root / "depth.lfp",
        value if field == "depth" else "depth.dat",
        value if field == "witness" else "center.tiff",
    )
    with pytest.raises(LytroError):
        extract_depth(path, asset_root=root)


@pytest.mark.parametrize("kind", ["directory", "fifo", "oversized", "empty"])
def test_unpacked_depth_requires_a_bounded_regular_file(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "depth.dat"
    if kind == "directory":
        target.mkdir()
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(b"\0" * (17 if kind == "oversized" else 0))
    path = _unpacked_depth_container(tmp_path / "depth.lfp")
    with pytest.raises(LytroError, match="bounded nonempty regular file"):
        extract_depth(path, asset_root=tmp_path)


def test_oversized_unpacked_witness_is_rejected_before_reading(tmp_path: Path) -> None:
    (tmp_path / "depth.dat").write_bytes(np.array([[0, 1], [2, 3]], dtype="<f4").tobytes())
    with (tmp_path / "center.tiff").open("wb") as file:
        file.truncate(MAX_SECTION_BYTES + 1)
    path = _unpacked_depth_container(tmp_path / "depth.lfp")
    with pytest.raises(LytroError, match="bounded nonempty regular file"):
        extract_depth(path, asset_root=tmp_path)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0, np.nan], [2, 3]], dtype="<f4"),
        np.array([[0, np.inf], [2, 3]], dtype="<f4"),
        np.array([[-1, 1], [2, 3]], dtype="<f4"),
        np.zeros((2, 3), dtype="<f4"),
    ],
)
def test_bad_float_depth_rejected(tmp_path: Path, values: np.ndarray) -> None:
    with pytest.raises(LytroError):
        extract_depth(_depth_container(tmp_path / "depth.lfp", values))


def test_quantization_is_ties_even_and_uses_declared_range() -> None:
    depth = FloatDepth(
        np.array([[0, 0.5, 1.5, 2.5, 4095]], dtype=np.float32),
        5,
        1,
        0,
        4095,
        None,
        {},
        {},
        b"",
        (0, 0),
        (5, 1),
    )
    np.testing.assert_array_equal(quantize_depth(depth), [[4095, 4094, 4094, 4092, 0]])


def test_no_observed_minmax_stretch() -> None:
    depth = FloatDepth(
        np.array([[1, 2]], dtype=np.float32), 2, 1, 0, 3, None, {}, {}, b"", (0, 0), (2, 1)
    )
    np.testing.assert_array_equal(quantize_depth(depth), [[2730, 1365]])
