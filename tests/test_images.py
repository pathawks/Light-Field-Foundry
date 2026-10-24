from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import ImageCms

from lytro_avif.errors import LytroError
from lytro_avif.images import RGB16, matching_views, read_png, read_tiff, write_png


def profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def test_rgb16_tiff_png_bridge_preserves_every_code_and_icc(tmp_path: Path) -> None:
    ramp = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    rgb = np.stack((ramp, np.fliplr(ramp), np.flipud(ramp)), axis=2)
    rgba = np.concatenate((rgb, np.full((256, 256, 1), 65535, np.uint16)), axis=2)
    path, icc = tmp_path / "source.tiff", profile()
    tifffile.imwrite(path, rgba, photometric="rgb", iccprofile=icc, metadata=None)
    image = read_tiff(path)
    assert np.array_equal(image.pixels, rgb)
    assert image.icc == icc
    target = tmp_path / "bridge.png"
    write_png(target, image)
    pixels, restored = read_png(target)
    assert np.array_equal(pixels, rgb)
    assert restored == icc
    assert image.cicp == "2/2/0" and not image.srgb


@pytest.mark.parametrize(
    "issue", ["rgb8", "missing-icc", "bad-icc", "alpha", "orientation", "conflicting-color"]
)
def test_invalid_tiff_contract_is_rejected(tmp_path: Path, issue: str) -> None:
    dtype = np.uint8 if issue == "rgb8" else np.uint16
    pixels = np.zeros((8, 8, 4 if issue == "alpha" else 3), dtype=dtype)
    tags = []
    if issue == "orientation":
        tags = [(274, "H", 1, 6, False)]
    elif issue == "conflicting-color":
        tags = [(318, "2I", 2, (3127, 10000, 3290, 10000), False)]
    icc = None if issue == "missing-icc" else b"broken" if issue == "bad-icc" else profile()
    path = tmp_path / "bad.tiff"
    tifffile.imwrite(path, pixels, photometric="rgb", iccprofile=icc, extratags=tags, metadata=None)
    with pytest.raises(LytroError):
        read_tiff(path)


def test_inconsistent_view_icc_rejected() -> None:
    pixels = np.zeros((8, 8, 3), np.uint16)
    with pytest.raises(LytroError, match="mismatched"):
        matching_views([RGB16(pixels, b"one", {}), RGB16(pixels, b"two", {})])
