"""Explicit RGB16 TIFF interpretation and independently checked PNG bridges."""

from __future__ import annotations

import base64
import binascii
import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import png
import tifffile
from PIL import ImageCms

from .errors import LytroError

type U16 = npt.NDArray[np.uint16]
MAX_PIXELS = 64_000_000
MAX_ICC = 16 * 1024 * 1024
COLOR_TAGS = (259, 262, 274, 301, 318, 319, 338, 339, 342, 529, 532, 34675)
# Only codecs that are inherently lossless. TIFF JPEG-family, WebP and JPEG-XL
# tags can describe lossy data; decoding them cannot establish source precision.
LOSSLESS_TIFF_COMPRESSION = frozenset({1, 5, 8, 32773, 32946, 34925, 34926, 50000})


@dataclass(frozen=True)
class RGB16:
    pixels: U16
    icc: bytes
    tags: dict[str, Any]
    # ICC remains authoritative. No profile-name heuristic implies standardized sRGB.
    cicp: str = "2/2/0"
    srgb: bool = False

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def validate_icc(icc: bytes) -> None:
    if not 132 <= len(icc) <= MAX_ICC:
        raise LytroError("Missing or invalid TIFF ICC profile length.")
    if struct.unpack_from(">I", icc)[0] != len(icc) or icc[36:40] != b"acsp":
        raise LytroError("Malformed ICC header or declared size.")
    if icc[16:20] != b"RGB " or icc[20:24] not in (b"XYZ ", b"Lab "):
        raise LytroError("The TIFF ICC does not describe RGB samples with a supported PCS.")
    count = struct.unpack_from(">I", icc, 128)[0]
    if count > 4096 or 132 + count * 12 > len(icc):
        raise LytroError("Invalid ICC tag table.")
    for index in range(count):
        offset, size = struct.unpack_from(">II", icc, 136 + 12 * index)
        if size < 8 or offset < 128 or offset + size > len(icc):
            raise LytroError("ICC tag extends outside its profile.")
    try:
        ImageCms.ImageCmsProfile(io.BytesIO(icc))
    except (OSError, ValueError, TypeError) as exc:
        raise LytroError(f"ICC profile failed LittleCMS validation: {exc}") from exc


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"encoding": "base64", "bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(part) for part in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def require_lossless_tiff(page: tifffile.TiffPage) -> None:
    """Reject ambiguous/lossy compression before decoding any TIFF samples."""
    compression = int(page.compression)
    if compression not in LOSSLESS_TIFF_COMPRESSION:
        name = getattr(page.compression, "name", str(compression))
        raise LytroError(
            f"TIFF compression {name} ({compression}) is not an admitted lossless encoding."
        )


def read_tiff(path: Path, *, allow_black_coverage: bool = False) -> RGB16:
    try:
        with tifffile.TiffFile(path) as tiff:
            if len(tiff.pages) != 1:
                raise LytroError("TNT TIFF must have exactly one image.")
            page = tiff.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                raise LytroError("TIFF must contain a complete image page.")
            require_lossless_tiff(page)
            if page.imagewidth * page.imagelength > MAX_PIXELS:
                raise LytroError("TIFF exceeds the supported pixel limit.")
            if page.dtype != np.dtype("uint16") or page.bitspersample != 16:
                raise LytroError("TNT must deliver unsigned 16-bit samples.")
            if page.photometric != 2 or page.planarconfig != 1:
                raise LytroError("TNT TIFF must contain interleaved RGB samples.")
            orientation = page.tags.get(274)
            if orientation is not None and orientation.value != 1:
                raise LytroError("Only upright TIFF orientation 1 is admitted.")
            icc_tag = page.tags.get(34675)
            if icc_tag is None:
                raise LytroError("TNT TIFF has no embedded ICC; no fallback profile is invented.")
            icc = bytes(icc_tag.value)
            validate_icc(icc)
            pixels = page.asarray()
            if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
                raise LytroError("TNT TIFF must have three RGB channels or opaque RGBA.")
            if pixels.shape[2] == 4:
                if len(page.extrasamples) > 1 or (
                    page.extrasamples and page.extrasamples[0] not in (0, 1, 2)
                ):
                    raise LytroError("Unspecified TIFF extra-channel interpretation.")
                if not np.all(pixels[:, :, 3] == 65535):
                    mask = pixels[:, :, 3]
                    if not (
                        allow_black_coverage
                        and not page.extrasamples
                        and np.all((mask == 0) | (mask == 65535))
                        and np.all(pixels[:, :, :3][mask == 0] == 0)
                    ):
                        raise LytroError("Nonopaque TIFF alpha is not supported by this route.")
                pixels = pixels[:, :, :3]
            elif page.extrasamples:
                raise LytroError("Unexpected TIFF extra samples.")
            tags = {
                str(code): {
                    "name": tag.name,
                    "type": int(tag.dtype),
                    "count": tag.count,
                    "value": _json_value(tag.value),
                }
                for code in COLOR_TAGS
                if (tag := page.tags.get(code)) is not None
            }
            if allow_black_coverage and page.samplesperpixel == 4:
                tags["coverage"] = {
                    "policy": "binary undeclared TNT coverage baked onto black; RGB samples unchanged",
                    "black_pixels": int(np.count_nonzero(page.asarray()[:, :, 3] == 0)),
                }
            # A nonidentity TransferFunction contradicts already ICC-described rendered codes.
            # Keep all original color tags in provenance; reject conflicts we cannot resolve.
            conflicting = {code for code in (301, 318, 319, 342, 529, 532) if code in page.tags}
            if conflicting:
                raise LytroError(
                    f"TIFF color tags {sorted(conflicting)} alongside ICC need a verified interpretation."
                )
            return RGB16(np.ascontiguousarray(pixels, dtype=np.uint16), icc, tags)
    except LytroError:
        raise
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise LytroError(f"Invalid TIFF {path.name}: {exc}") from exc


def chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data))
    )


def write_png(path: Path, image: RGB16) -> None:
    buffer = io.BytesIO()
    writer = png.Writer(image.width, image.height, greyscale=False, bitdepth=16)
    writer.write(buffer, image.pixels.reshape(image.height, image.width * 3).tolist())
    data = buffer.getvalue()
    profile = chunk(b"iCCP", b"TNT ICC\0\0" + zlib.compress(image.icc))
    path.write_bytes(data[:33] + profile + data[33:])
    decoded, icc = read_png(path)
    if not np.array_equal(decoded, image.pixels) or icc != image.icc:
        raise LytroError("TIFF-to-PNG bridge changed samples or ICC bytes.")


def read_png(path: Path) -> tuple[U16, bytes | None]:
    try:
        width, height, rows, info = png.Reader(filename=str(path)).asDirect()
        if width * height > MAX_PIXELS or info["bitdepth"] != 16:
            raise LytroError("Decoded PNG is not a bounded 16-bit image.")
        planes = info["planes"]
        if planes not in (3, 4) or info["greyscale"]:
            raise LytroError("Decoded PNG is not RGB16.")
        array = np.array(list(rows), dtype=np.uint16).reshape(height, width, planes)
        if planes == 4:
            if not np.all(array[:, :, 3] == 65535):
                raise LytroError("Decoded PNG contains nonopaque alpha.")
            array = array[:, :, :3]
        data = path.read_bytes()
        offset, icc = 8, None
        while offset + 12 <= len(data):
            size = struct.unpack_from(">I", data, offset)[0]
            kind = data[offset + 4 : offset + 8]
            if offset + size + 12 > len(data):
                raise LytroError("Truncated PNG chunk.")
            payload = data[offset + 8 : offset + 8 + size]
            if kind == b"iCCP":
                if icc is not None:
                    raise LytroError("Duplicate PNG ICC profile.")
                _name, compressed = payload.split(b"\0", 1)
                if not compressed or compressed[0] != 0:
                    raise LytroError("Unsupported PNG ICC compression.")
                inflater = zlib.decompressobj()
                icc = inflater.decompress(compressed[1:], MAX_ICC + 1)
                if len(icc) > MAX_ICC or not inflater.eof:
                    raise LytroError("Oversized or invalid PNG ICC profile.")
            offset += size + 12
        return array, icc
    except LytroError:
        raise
    except (OSError, ValueError, png.Error) as exc:
        raise LytroError(f"Invalid PNG: {exc}") from exc


def matching_views(views: list[RGB16], *, separate_center_profile: bool = False) -> None:
    first = views[0]
    for view in views[1:]:
        if view.pixels.shape != first.pixels.shape or (
            not separate_center_profile and view.icc != first.icc
        ):
            raise LytroError("TNT views have mismatched geometry or ICC profiles.")

    if separate_center_profile and views[1].icc != views[2].icc:
        raise LytroError("Stereo eyes have mismatched ICC profiles.")
