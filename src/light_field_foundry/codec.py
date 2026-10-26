"""Exact RGB16 sample transforms and native monochrome12 codec checks."""

from __future__ import annotations

import re
from pathlib import Path

import imagecodecs
import numpy as np
import png
from PIL import ImageCms

from . import bmff
from .errors import LytroError
from .images import RGB16, U16, read_png, write_png
from .tools import Tools, require_output


def read_y4m(path: Path) -> tuple[int, str, list[U16]]:
    """Read a single bounded native frame without RGB conversion or depth scaling."""
    with path.open("rb") as file:
        header = file.readline(4096)
        if not header.startswith(b"YUV4MPEG2 ") or not header.endswith(b"\n"):
            raise LytroError("Malformed native Y4M header.")
        fields = dict((token[:1], token[1:]) for token in header.strip().split()[1:])
        try:
            width, height = int(fields[b"W"]), int(fields[b"H"])
            chroma = fields[b"C"].decode("ascii")
        except (KeyError, ValueError, UnicodeError) as exc:
            raise LytroError("Missing native Y4M dimensions/chroma.") from exc
        if width < 1 or height < 1 or width * height > 64_000_000:
            raise LytroError("Native Y4M dimensions exceed supported limits.")
        if chroma.startswith("mono"):
            planes = 1
            bits = chroma[4:] or "8"
        elif chroma == "444":
            planes, bits = 3, "8"
        elif re.fullmatch(r"444p(?:10|12|16)", chroma):
            planes, bits = 3, chroma[4:]
        else:
            raise LytroError(f"Unsupported native Y4M chroma {chroma}.")
        depth = int(bits)
        if depth not in (8, 10, 12, 16):
            raise LytroError(f"Unsupported native bit depth {depth}.")
        frame = file.readline(4096)
        if not frame.startswith(b"FRAME") or not frame.endswith(b"\n"):
            raise LytroError("Malformed native Y4M frame.")
        count = width * height
        extent = count * planes * (2 if depth > 8 else 1)
        data = file.read(extent + 1)
        if len(data) != extent:
            raise LytroError("Y4M has a truncated plane or more than one frame.")
        values = np.frombuffer(data, dtype="<u2" if depth > 8 else "u1").astype(np.uint16)
        if np.max(values) >= 1 << depth:
            raise LytroError("Y4M sample exceeds declared precision.")
        return depth, chroma, [p.reshape(height, width) for p in np.split(values, planes)]


def encode_color(tools: Tools, work: Path, name: str, image: RGB16) -> Path:
    # TODO: Replace avifenc when imagecodecs exposes sato sample-transform
    # encoding (RGB16 as high12 + low4) and ICC profile embedding.
    source, icc, target = work / f"{name}.png", work / f"{name}.icc", work / f"{name}.avif"
    write_png(source, image)
    icc.write_bytes(image.icc)
    tools.run(
        [
            tools.avifenc,
            "--jobs",
            str(tools.threads),
            "--lossless",
            "--depth",
            "12,4",
            "--yuv",
            "444",
            "--range",
            "full",
            "--cicp",
            image.cicp,
            "--icc",
            icc,
            "--ignore-exif",
            "--ignore-xmp",
            source,
            target,
        ],
        work,
    )
    require_output(target)
    # libavif's built-in recipe codes low4 << 4. Normalize with a separately
    # losslessly encoded RGB8 component so the stored suffix codes are 0..15.
    low4_source, low4_target = work / f"{name}-low4.png", work / f"{name}-low4.avif"
    with low4_source.open("wb") as file:
        writer = png.Writer(image.width, image.height, greyscale=False, bitdepth=8)
        writer.write(file, (image.pixels & 15).reshape(image.height, image.width * 3).tolist())
    tools.run(
        [
            tools.avifenc,
            "--jobs",
            str(tools.threads),
            "--lossless",
            "--depth",
            "8",
            "--yuv",
            "444",
            "--range",
            "full",
            "--cicp",
            image.cicp,
            "--icc",
            icc,
            low4_source,
            low4_target,
        ],
        work,
    )
    require_output(low4_target)
    bmff.normalize_suffix(target, low4_target)
    return target


def encode_depth(tools: Tools, work: Path, codes: U16) -> Path:
    if codes.dtype != np.uint16 or codes.ndim != 2 or not codes.size or np.max(codes) > 4095:
        raise LytroError("Depth input must be one uint16 plane with codes 0..4095.")
    target = work / "depth.avif"
    try:
        encoded = bytes(
            imagecodecs.avif_encode(
                np.ascontiguousarray(codes),
                level=imagecodecs.AVIF.QUALITY.LOSSLESS,
                bitspersample=12,
                codec=imagecodecs.AVIF.CODEC_CHOICE.AOM,
                speed=6,
                numthreads=tools.threads,
                primaries=2,
                transfer=2,
                matrix=2,
            )
        )
    except (imagecodecs.AvifError, ValueError, RuntimeError) as exc:
        raise LytroError(f"imagecodecs depth encoding failed: {exc}") from exc
    bmff.validate_depth(bmff.parse(encoded))
    target.write_bytes(encoded)
    require_output(target)
    return target


def verify_color(
    tools: Tools,
    work: Path,
    source: Path,
    expected: RGB16,
    name: str,
    *,
    codecs: tuple[str, ...] = ("dav1d", "aom"),
) -> None:
    for codec in codecs:
        decoded = work / f"{name}-{codec}-rgb16.png"
        tools.run(
            [
                tools.avifdec,
                "--jobs",
                str(tools.threads),
                "--codec",
                codec,
                "--sato",
                "--depth",
                "16",
                source,
                decoded,
            ],
            work,
        )
        require_output(decoded)
        samples, icc = read_png(decoded)
        if not np.array_equal(samples, expected.pixels) or icc != expected.icc:
            raise LytroError(f"{name}: {codec} SATO reconstruction changed RGB16 samples or ICC.")
        native = work / f"{name}-{codec}-fallback.y4m"
        tools.run(
            [
                tools.avifdec,
                "--jobs",
                str(tools.threads),
                "--codec",
                codec,
                source,
                native,
            ],
            work,
        )
        depth, _chroma, planes = read_y4m(native)
        if depth != 12 or len(planes) != 3:
            raise LytroError(f"{name}: ordinary fallback must be native RGB12 4:4:4.")
        # Identity-matrix coding stores G,B,R as Y,U,V.
        rgb = np.stack((planes[2], planes[0], planes[1]), axis=2)
        if not np.array_equal(rgb, expected.pixels >> 4):
            raise LytroError(f"{name}: native RGB12 fallback samples differ from high12.")


def verify_depth(
    tools: Tools,
    work: Path,
    source: Path,
    codes: U16,
    *,
    codecs: tuple[str, ...] = ("dav1d", "aom"),
) -> None:
    for codec in codecs:
        decoded = work / f"depth-{codec}-decoded.y4m"
        tools.run(
            [
                tools.avifdec,
                "--jobs",
                str(tools.threads),
                "--codec",
                codec,
                source,
                decoded,
            ],
            work,
        )
        depth, _chroma, planes = read_y4m(decoded)
        if depth != 12 or len(planes) != 1 or not np.array_equal(planes[0], codes):
            raise LytroError(f"{codec} final native depth12 samples differ from prepared codes.")


def synthetic_probes(tools: Tools, work: Path) -> dict[str, str]:
    """All 4096 depth levels and color edge/low-bit patterns through both decoders."""
    ramp = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    codes = ((ramp.astype(np.uint32) * 65535) // 4095).astype(np.uint16)
    pixels = np.stack((codes, np.flipud(codes), np.bitwise_xor(codes, 0xFFFF)), axis=2)
    pixels[0, :8] = np.array(
        [
            [0, 0, 0],
            [65535, 0, 0],
            [0, 65535, 0],
            [0, 0, 65535],
            [65535, 65535, 65535],
            [1, 15, 16],
            [4095, 4096, 4097],
            [65534, 32767, 32768],
        ],
        dtype=np.uint16,
    )
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image = RGB16(pixels, profile, {})
    encoded = encode_color(tools, work, "probe-color", image)
    verify_color(tools, work, encoded, image, "probe-color")
    depth = encode_depth(tools, work, ramp)
    verify_depth(tools, work, depth, ramp)
    return {
        "imagecodecs": f"{imagecodecs.__version__} ({imagecodecs.avif_version()})",
        "RGB16_SATO": "exact, AOM + dav1d",
        "monochrome12": "4096/4096 levels exact, AOM + dav1d",
    }
