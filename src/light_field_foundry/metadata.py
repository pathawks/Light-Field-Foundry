"""Source-faithful capture metadata and the demonstrated Apple compatibility recipe.

Original JSON is base64 in the documented ``https://light-field-foundry.org/ns/1.0/``
XMP namespace: JSON numbers, signed zero, whitespace, and Unicode remain exact.
Generated compatibility declarations are separate from source capture metadata.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from importlib.resources import files
from typing import Any
from xml.etree import ElementTree as ET

from .errors import LytroError

REND_SHA256 = "c74bf7b16f10415053cd032902d0b0feb78b91acb046eb6d8e3a438cdff24622"
NAMESPACE = "https://light-field-foundry.org/ns/1.0/"
MAX_JSON_BYTES = 16 * 1024 * 1024
NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "apdi": "http://ns.apple.com/pixeldatainfo/1.0/",
    "depthData": "http://ns.apple.com/depthData/1.0/",
    "depthBlurEffect": "http://ns.apple.com/depthBlurEffect/1.0/",
    "portraitLightingEffect": "http://ns.apple.com/portraitLightingEffect/2.0/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stDim": "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "lff": NAMESPACE,
}
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)


@dataclass(frozen=True)
class SpatialProperties:
    """Complete nonessential BMFF property boxes for fallback stereo views."""

    cmin: bytes
    cmex_left: bytes
    cmex_right: bytes
    dadj: bytes


@dataclass(frozen=True)
class MetadataBundle:
    center_exif: bytes
    eye_exif: bytes
    primary_xmp: bytes
    depth_xmp: bytes
    spatial: SpatialProperties


def _tag(prefix: str, name: str) -> str:
    return f"{{{NS[prefix]}}}{name}"


def _packet() -> tuple[ET.Element, ET.Element]:
    root = ET.Element(_tag("x", "xmpmeta"))
    rdf = ET.SubElement(root, _tag("rdf", "RDF"))
    description = ET.SubElement(rdf, _tag("rdf", "Description"), {_tag("rdf", "about"): ""})
    return root, description


def _field(parent: ET.Element, prefix: str, name: str, value: object) -> ET.Element:
    child = ET.SubElement(parent, _tag(prefix, name))
    child.text = str(value)
    return child


def _sequence(parent: ET.Element, name: str, values: Sequence[object]) -> None:
    sequence = ET.SubElement(ET.SubElement(parent, _tag("depthData", name)), _tag("rdf", "Seq"))
    for value in values:
        _field(sequence, "rdf", "li", value)


def _xml(root: ET.Element) -> bytes:
    packet = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    if not isinstance(packet, bytes):
        raise LytroError("XML serializer did not return UTF-8 bytes")
    return packet


def rendering_parameters() -> bytes:
    """Read and verify the exact reference packet before every use."""
    try:
        encoded = files("light_field_foundry").joinpath("resources/portrait_rend.b64").read_bytes()
        packet = base64.b64decode(encoded.strip(), validate=True)
    except (OSError, ValueError) as error:
        raise LytroError(f"Cannot read Portrait rendering compatibility packet: {error}") from error
    if len(packet) != 1352 or hashlib.sha256(packet).hexdigest() != REND_SHA256:
        raise LytroError("Portrait rendering compatibility packet failed its identity check")
    return packet


def apple_maker_note() -> bytes:
    """Generate only PhotosAppFeatureFlags=1; no phone MakerNote is copied."""
    return b"Apple iOS\0" + struct.pack(">H2sHHHII", 1, b"MM", 1, 0x001F, 9, 1, 1) + b"\0" * 4


def _full_box(kind: bytes, flags: int, payload: bytes) -> bytes:
    return struct.pack(">I4sI", len(payload) + 12, kind, flags) + payload


def spatial_properties(width: int, height: int) -> SpatialProperties:
    if not (0 < width <= 0xFFFFFFFF and 0 < height <= 0xFFFFFFFF):
        raise LytroError("Spatial compatibility properties require valid output dimensions")
    denominator = 1 << 16
    # Deliberately normalize by the 640-pixel reference width, as demonstrated.
    focal = math.floor((1.34 * 1000 / 2.2 / 640) * denominator + 0.5)
    cmin = _full_box(
        b"cmin", 16 << 8, struct.pack(">iii", focal, denominator // 2, denominator // 2)
    )

    def cmex(position: int) -> bytes:
        return _full_box(b"cmex", 0x0F, struct.pack(">iiihhh", position, 0, 0, 0, 0, 0))

    return SpatialProperties(
        cmin, cmex(0), cmex(35000), _full_box(b"dadj", 0, struct.pack(">i", 0))
    )


def depth_xmp(
    width: int,
    height: int,
    a: float,
    b: float,
    source_mapping: Mapping[str, object] | None = None,
) -> bytes:
    if not all(math.isfinite(value) for value in (a, b)) or a == b:
        raise LytroError("Depth XMP requires finite distinct code-to-disparity endpoints")
    if width <= 0 or height <= 0:
        raise LytroError("Depth calibration compatibility fields require output dimensions")
    root, description = _packet()
    fields = {
        "NativeFormat": int.from_bytes(b"fdis", "big"),
        "AuxiliaryImageType": "disparity",
        "StoredFormat": int.from_bytes(b"L016", "big"),
        "FloatMinValue": repr(a),
        "FloatMaxValue": repr(b),
        "IntMinValue": 0,
        "IntMaxValue": 65535,
    }
    for name, value in fields.items():
        _field(description, "apdi", name, value)
    for name, value in {
        "DepthDataVersion": 65541,
        "Quality": "high",
        "Filtered": "True",
        "Accuracy": "relative",
    }.items():
        _field(description, "depthData", name, value)
    _field(description, "depthBlurEffect", "SimulatedAperture", "2.800000")
    _field(description, "portraitLightingEffect", "EffectStrength", "0.000000")
    _field(description, "depthData", "IntrinsicMatrixReferenceWidth", width)
    _field(description, "depthData", "IntrinsicMatrixReferenceHeight", height)
    _sequence(
        description,
        "IntrinsicMatrix",
        [826.67 / 1134 * width, 0, 0, 0, 826.67 / 1134 * height, 0, width / 2, height / 2, 1],
    )
    _sequence(description, "ExtrinsicMatrix", [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    _field(description, "depthData", "PixelSize", "0.001000")
    _sequence(description, "LensDistortionCoefficients", [0] * 8)
    _sequence(description, "InverseLensDistortionCoefficients", [0] * 8)
    _field(description, "depthData", "LensDistortionCenterOffsetX", 0)
    _field(description, "depthData", "LensDistortionCenterOffsetY", 0)
    _field(
        description,
        "depthBlurEffect",
        "RenderingParameters",
        base64.b64encode(rendering_parameters()).decode("ascii"),
    )
    _field(
        description,
        "lff",
        "DepthInterpretation",
        json.dumps(
            {
                "codedBits": 12,
                "storedFormat": "L016",
                "native12To16Expansion": "(q << 4) | (q >> 8)",
                "codeToRelativeDisparityEndpoints": [a, b],
                "rounding": "nearest, ties to even",
                "polarity": "ascending" if b > a else "descending",
                "meaning": "relative disparity; not metric distance",
                "sourceLambdaMapping": dict(source_mapping)
                if source_mapping is not None and len(source_mapping) > 0
                else None,
                "quantization": "q = roundTiesToEven(4095 * (lambda - sourceLambdaAtCode0) / (sourceLambdaAtCode4095 - sourceLambdaAtCode0)); original endpoint labels and polarity are in sourceLambdaMapping",
                "calibration": "synthetic compatibility values, scaled from the demonstrated 1134x1134 reference",
                "qualityAndFiltered": "compatibility declarations; no new quality measurement or filtering",
                "renderingParameters": {
                    "sha256": REND_SHA256,
                    "source": "iPhone IMG_3918.HEIC depth XMP; opaque REND version 7; compatibility only",
                },
            },
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
    return _xml(root)


def _get(value: object, *path: str) -> object:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _ascii(value: object) -> bytes | None:
    if not isinstance(value, str) or not value or len(value) > 1024 or "\0" in value:
        return None
    try:
        return value.encode("ascii") + b"\0"
    except UnicodeEncodeError:
        return None


def _positive_number(value: object, lower: float, upper: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) and lower <= number <= upper else None


def _rational(value: float) -> bytes | None:
    rational = Fraction(value).limit_denominator(
        min(1_000_000_000, max(1, math.floor(0xFFFFFFFF / value)))
    )
    if rational.numerator < 1 or rational.numerator > 0xFFFFFFFF:
        return None
    return struct.pack("<II", rational.numerator, rational.denominator)


def _timestamp(value: object) -> tuple[bytes, bytes, bytes | None] | None:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:\d\d)", value
    ):
        return None
    offset_text = "+00:00" if value.endswith("Z") else value[-6:]
    if int(offset_text[1:3]) > 23 or int(offset_text[4:6]) > 59:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    fraction = re.search(r"\.(\d+)", value)
    return (
        parsed.strftime("%Y:%m:%d %H:%M:%S").encode("ascii") + b"\0",
        offset_text.encode("ascii") + b"\0",
        value[fraction.start(1) : fraction.end(1)].encode("ascii") + b"\0" if fraction else None,
    )


def exif_payload(public: object, width: int, height: int, *, srgb: bool, portrait: bool) -> bytes:
    """Build bounded little-endian TIFF Exif, prefixed by AVIF's TIFF offset."""
    Entry = tuple[int, int, int, bytes]
    ifd0: list[Entry] = [
        (0x0100, 4, 1, struct.pack("<I", width)),
        (0x0101, 4, 1, struct.pack("<I", height)),
        (0x0112, 3, 1, struct.pack("<H", 1)),
    ]
    for tag, name in [(0x010F, "make"), (0x0110, "model")]:
        text = _ascii(_get(public, "camera", name))
        if text is not None:
            ifd0.append((tag, 2, len(text), text))
    exif: list[Entry] = [
        (0x9000, 7, 4, b"0232"),
        (0xA001, 3, 1, struct.pack("<H", 1 if srgb else 65535)),
        (0xA002, 4, 1, struct.pack("<I", width)),
        (0xA003, 4, 1, struct.pack("<I", height)),
    ]
    for tag, path, low, high, scale in [
        (0x829A, ("devices", "shutter", "pixelExposureDuration"), 1e-9, 86400, 1),
        (0x829D, ("devices", "lens", "fNumber"), 0.1, 1024, 1),
        (0x920A, ("devices", "lens", "focalLength"), 1e-6, 100, 1000),
    ]:
        number = _positive_number(_get(public, *path), low, high)
        data = _rational(number * scale) if number is not None else None
        if data is not None:
            exif.append((tag, 5, 1, data))
    iso = _positive_number(_get(public, "devices", "sensor", "iso"), 1, 65535)
    if iso is not None and iso.is_integer():
        exif.append((0x8827, 3, 1, struct.pack("<H", int(iso))))
    timestamp = _timestamp(_get(public, "devices", "clock", "zuluTime"))
    if timestamp is not None:
        date, offset, subseconds = timestamp
        exif.extend([(0x9003, 2, len(date), date), (0x9011, 2, len(offset), offset)])
        if subseconds is not None:
            exif.append((0x9291, 2, len(subseconds), subseconds))
    if portrait:
        note = apple_maker_note()
        exif.append((0x927C, 7, len(note), note))
    exif_offset = 8 + 2 + 12 * (len(ifd0) + 1) + 4
    ifd0.append((0x8769, 4, 1, struct.pack("<I", exif_offset)))
    data_start = exif_offset + 2 + 12 * len(exif) + 4
    extra = bytearray()

    def directory(entries: list[Entry]) -> bytes:
        result = bytearray(struct.pack("<H", len(entries)))
        for tag, kind, count, data in sorted(entries):
            result.extend(struct.pack("<HHI", tag, kind, count))
            if len(data) <= 4:
                result.extend(data.ljust(4, b"\0"))
            else:
                if len(extra) % 2:
                    extra.append(0)
                result.extend(struct.pack("<I", data_start + len(extra)))
                extra.extend(data)
        result.extend(b"\0" * 4)
        return bytes(result)

    directories = directory(ifd0) + directory(exif)
    return b"\0" * 4 + b"II*\0\x08\0\0\0" + directories + bytes(extra)


def _raw_json(packet: bytes) -> object:
    if len(packet) > MAX_JSON_BYTES:
        raise LytroError("Source metadata exceeds the 16 MiB embedding limit")
    try:
        return json.loads(
            packet, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
        )
    except (ValueError, UnicodeError) as error:
        raise LytroError(f"Cannot embed malformed source JSON: {error}") from error


def build_metadata(
    public_json: Sequence[bytes],
    source_recipe: bytes,
    effective_settings: Mapping[str, object],
    provenance: Mapping[str, object],
    width: int,
    height: int,
    depth_a: float,
    depth_b: float,
    srgb: bool = False,
    private_json: Sequence[bytes] = (),
    master_json: bytes | None = None,
    source_depth_mapping: Mapping[str, object] | None = None,
) -> MetadataBundle:
    """Prepare metadata; callers must explicitly supply private JSON to include it."""
    if not (0 < width <= 0xFFFFFFFF and 0 < height <= 0xFFFFFFFF):
        raise LytroError("Metadata requires valid output dimensions")
    public: Any = {}
    for packet in public_json:
        parsed = _raw_json(packet)
        if isinstance(parsed, dict) and isinstance(parsed.get("camera"), dict):
            public = parsed
            break
    _raw_json(source_recipe)
    root, description = _packet()
    regions = ET.SubElement(
        description, _tag("mwg-rs", "Regions"), {_tag("rdf", "parseType"): "Resource"}
    )
    dimensions = ET.SubElement(
        regions, _tag("mwg-rs", "AppliedToDimensions"), {_tag("rdf", "parseType"): "Resource"}
    )
    for name, value in [("w", width), ("h", height), ("unit", "pixel")]:
        _field(dimensions, "stDim", name, value)
    bag = ET.SubElement(ET.SubElement(regions, _tag("mwg-rs", "RegionList")), _tag("rdf", "Bag"))
    region = ET.SubElement(bag, _tag("rdf", "li"), {_tag("rdf", "parseType"): "Resource"})
    _field(region, "mwg-rs", "Type", "Focus")
    area = ET.SubElement(region, _tag("mwg-rs", "Area"), {_tag("rdf", "parseType"): "Resource"})
    for name, value in [("x", "0.5"), ("y", "0.5"), ("w", "1"), ("h", "1"), ("unit", "normalized")]:
        _field(area, "stArea", name, value)
    _field(
        description,
        "lff",
        "FocusRegionProvenance",
        "Full-frame presentation compatibility region; no detected-subject or source-focus-region claim",
    )
    if master_json is not None:
        _raw_json(master_json)
        _field(
            description,
            "lff",
            "SourceMasterJSONBase64",
            base64.b64encode(master_json).decode("ascii"),
        )
    for name, packets in [
        ("SourcePublicJSONBase64", public_json),
        ("SourcePrivateJSONBase64", private_json),
    ]:
        if not packets:
            continue
        sequence = ET.SubElement(ET.SubElement(description, _tag("lff", name)), _tag("rdf", "Seq"))
        for packet in packets:
            _raw_json(packet)
            _field(sequence, "rdf", "li", base64.b64encode(packet).decode("ascii"))
    _field(
        description,
        "lff",
        "SourceRecipeJSONBase64",
        base64.b64encode(source_recipe).decode("ascii"),
    )
    _field(
        description,
        "lff",
        "EffectiveRenderSettingsJSON",
        json.dumps(dict(effective_settings), separators=(",", ":"), allow_nan=False),
    )
    _field(
        description,
        "lff",
        "RenderProvenanceJSON",
        json.dumps(dict(provenance), separators=(",", ":"), allow_nan=False),
    )
    _field(
        description,
        "lff",
        "CompatibilityProvenanceJSON",
        json.dumps(
            {
                "portrait": "demonstrated Apple Portrait reference recipe; generated minimal feature MakerNote",
                "renderingParameters": {
                    "sha256": REND_SHA256,
                    "source": "iPhone IMG_3918.HEIC depth XMP; opaque REND version 7",
                },
                "spatial": {
                    "profile": "mpo-heif-default",
                    "baselineMillimeters": 35,
                    "focalLengthMillimeters": 1.34,
                    "pixelPitchMicrometers": 2.2,
                    "referenceWidth": 640,
                    "referenceHeight": 480,
                    "focalNormalizationWidth": 640,
                    "principalPoint": "centered",
                    "disparityAdjustment": 0,
                    "meaning": "presentation compatibility defaults; not measured Lytro eye geometry or TNT normalized baseline",
                },
                "colorPrecision": {"full": "exact RGB16 through SATO", "fallback": "RGB12"},
                "depthPrecision": "one native monochrome12 AV1 auxiliary; intentionally quantized from TNT Float32",
            },
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
    return MetadataBundle(
        exif_payload(public, width, height, srgb=srgb, portrait=True),
        exif_payload(public, width, height, srgb=srgb, portrait=False),
        _xml(root),
        depth_xmp(width, height, depth_a, depth_b, source_depth_mapping),
        spatial_properties(width, height),
    )
