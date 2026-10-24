from pathlib import Path

import numpy as np
import pytest

from lytro_avif.codec import read_y4m, write_depth
from lytro_avif.errors import LytroError


def test_native_mono12_bridge_preserves_codes_without_rescaling(tmp_path: Path) -> None:
    samples = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    path = tmp_path / "depth.y4m"
    write_depth(path, samples)
    depth, chroma, planes = read_y4m(path)
    assert depth == 12 and chroma == "mono12" and len(planes) == 1
    assert np.array_equal(planes[0], samples)


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
