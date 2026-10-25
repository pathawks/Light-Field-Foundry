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


@pytest.mark.parametrize("invalid", [False, True])
def test_exported_tnt_black_coverage_keeps_rgb_without_trimming(
    tmp_path: Path, invalid: bool
) -> None:
    import struct

    pixels = np.full((8, 8, 4), 65535, np.uint16)
    pixels[0, 0] = [1 if invalid else 0, 0, 0, 0]
    path = tmp_path / "coverage.tiff"
    tifffile.imwrite(path, pixels, photometric="rgb", iccprofile=profile(), metadata=None)
    # TNT's fourth channel has no ExtraSamples declaration; reproduce that layout.
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        assert isinstance(page, tifffile.TiffPage)
        offset = page.tags[338].offset
    data = bytearray(path.read_bytes())
    struct.pack_into("<H", data, offset, 65000)
    path.write_bytes(data)
    with pytest.raises(LytroError, match="Nonopaque"):
        read_tiff(path)
    if invalid:
        with pytest.raises(LytroError, match="Nonopaque"):
            read_tiff(path, allow_black_coverage=True)
    else:
        image = read_tiff(path, allow_black_coverage=True)
        np.testing.assert_array_equal(image.pixels, pixels[:, :, :3])
        assert image.tags["coverage"]["black_pixels"] == 1


@pytest.mark.parametrize("compression", [1, 5, 8, 32773, 32946, 34925, 34926, 50000])
def test_lossless_tiff_codecs_keep_exact_rgb16_and_icc(tmp_path: Path, compression: int) -> None:
    pixels = np.arange(192, dtype=np.uint16).reshape(8, 8, 3) * 337
    icc = profile()
    path = tmp_path / "lossless.tiff"
    tifffile.imwrite(
        path, pixels, photometric="rgb", compression=compression, iccprofile=icc, metadata=None
    )
    before = path.read_bytes()
    image = read_tiff(path)
    np.testing.assert_array_equal(image.pixels, pixels)
    assert image.icc == icc
    assert image.tags["259"]["value"] == compression
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "compression", [6, 7, 33004, 34712, 34892, 34927, 34934, 50001, 50002, 52546, 65534]
)
def test_ambiguous_or_lossy_tiff_codec_is_rejected_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: int
) -> None:
    import struct

    path = tmp_path / "not-lossless.tiff"
    tifffile.imwrite(
        path,
        np.zeros((8, 8, 3), dtype=np.uint16),
        photometric="rgb",
        iccprofile=profile(),
        metadata=None,
    )
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        assert isinstance(page, tifffile.TiffPage)
        offset = page.tags[259].valueoffset
    data = bytearray(path.read_bytes())
    # Keep the uncompressed payload intact: if a decoder is attempted, the
    # compression declaration would no longer describe the samples.
    struct.pack_into("<H", data, offset, compression)
    path.write_bytes(data)

    def decode(*_args, **_kwargs):
        raise AssertionError("Rejected TIFF compression reached a pixel decoder")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", decode)
    with pytest.raises(LytroError, match="lossless encoding"):
        read_tiff(path)
    assert path.read_bytes() == data
