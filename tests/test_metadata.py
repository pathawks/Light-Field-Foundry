from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
from xml.etree import ElementTree as ET

import pytest
import tifffile

from light_field_foundry.errors import LytroError
from light_field_foundry.metadata import (
    NS,
    REND_SHA256,
    apple_maker_note,
    build_metadata,
    depth_xmp,
    exif_payload,
    rendering_parameters,
    spatial_properties,
)


def source() -> dict[str, object]:
    return {
        "camera": {"make": "Lytro, Inc.", "model": "F01"},
        "devices": {
            "clock": {"zuluTime": "2013-01-07T20:51:31.012300Z"},
            "shutter": {"pixelExposureDuration": 0.004, "frameExposureDuration": 100},
            "lens": {"fNumber": 1.95, "focalLength": 0.00795},
            "sensor": {"iso": 125},
        },
    }


def exif_entries(payload: bytes) -> tuple[dict[int, object], dict[str, object]]:
    with tifffile.TiffFile(io.BytesIO(payload[4:])) as image:
        page = image.pages[0]
        assert isinstance(page, tifffile.TiffPage)
        return ({int(tag.code): tag.value for tag in page.tags}, dict(page.tags[0x8769].value))


def xml_text(root: ET.Element, path: str) -> str | None:
    element = root.find(path, NS)
    assert element is not None
    return element.text


def test_generated_minimal_maker_note_and_reference_packet() -> None:
    note = apple_maker_note()
    assert len(note) == 32
    assert note[:14] == b"Apple iOS\0\0\x01MM"
    assert struct.unpack(">HHHII", note[14:28]) == (1, 0x001F, 9, 1, 1)
    assert note[28:] == b"\0" * 4
    packet = rendering_parameters()
    assert packet[:8] == b"REND\x07\0\0\0"
    assert len(packet) == 1352
    assert hashlib.sha256(packet).hexdigest() == REND_SHA256


def test_source_exif_uses_capture_fields_and_preserves_identity() -> None:
    payload = exif_payload(source(), 1024, 768, srgb=False, portrait=True)
    entries, exif = exif_entries(payload)
    assert entries[0x010F] == "Lytro, Inc."
    assert entries[0x0110] == "F01"
    assert entries[0x0112] == 1
    assert exif["ExposureTime"] == (1, 250)
    assert exif["FNumber"] == (39, 20)
    assert exif["ISOSpeedRatings"] == 125
    assert exif["FocalLength"] == (159, 20)
    assert exif["DateTimeOriginal"] == "2013:01:07 20:51:31"
    assert exif["OffsetTimeOriginal"] == "+00:00"
    assert exif["SubsecTimeOriginal"] == "012300"
    assert exif["ColorSpace"] == 65535
    assert exif["PixelXDimension"] == 1024
    assert exif["PixelYDimension"] == 768
    assert exif["MakerNote"] == apple_maker_note()


def test_exif_omits_malformed_unknown_fields_without_inventing_values() -> None:
    malformed = {
        "camera": {"make": "bad\0value", "model": "é"},
        "devices": {
            "clock": {"zuluTime": "2013-02-30T20:51:31Z"},
            "shutter": {"pixelExposureDuration": -1, "frameExposureDuration": 1},
            "lens": {"fNumber": "2.8", "focalLength": float("inf")},
            "sensor": {"iso": 2.5},
        },
    }
    entries, exif = exif_entries(exif_payload(malformed, 16, 16, srgb=True, portrait=False))
    assert 0x010F not in entries and 0x0110 not in entries
    assert "ExposureTime" not in exif
    assert "DateTimeOriginal" not in exif
    assert "FNumber" not in exif and "FocalLength" not in exif
    assert "ISOSpeedRatings" not in exif and "MakerNote" not in exif
    assert exif["ColorSpace"] == 1


def test_xmp_preserves_original_json_bytes_and_distinct_recipe_provenance() -> None:
    raw = b'{ "camera": {"make":"Lytro, Inc.","model":"F01"}, "test": -0 }'
    recipe = b'{"viewFocus":-0}'
    bundle = build_metadata((raw,), recipe, {"center": {"u": 0}}, {"tnt": "2.0"}, 1134, 1134, -2, 3)
    assert b'xmlns:lff="https://light-field-foundry.org/ns/1.0/"' in bundle.primary_xmp
    root = ET.fromstring(bundle.primary_xmp)
    public = root.find(".//lff:SourcePublicJSONBase64/rdf:Seq/rdf:li", NS)
    assert public is not None and base64.b64decode(public.text or "") == raw
    recipe_field = root.find(".//lff:SourceRecipeJSONBase64", NS)
    assert recipe_field is not None and base64.b64decode(recipe_field.text or "") == recipe
    assert root.find(".//lff:SourcePrivateJSONBase64", NS) is None
    assert xml_text(root, ".//stDim:w") == "1134"
    assert xml_text(root, ".//stArea:x") == "0.5"
    assert xml_text(root, ".//stArea:w") == "1"
    provenance = root.find(".//lff:CompatibilityProvenanceJSON", NS)
    assert provenance is not None
    assert json.loads(provenance.text or "")["spatial"]["referenceWidth"] == 640


def test_depth_xmp_declares_exact_l016_native12_mapping_and_compatibility() -> None:
    root = ET.fromstring(depth_xmp(800, 600, 6.25, -3.0))
    for name, value in [
        ("NativeFormat", str(int.from_bytes(b"fdis", "big"))),
        ("StoredFormat", str(int.from_bytes(b"L016", "big"))),
        ("IntMinValue", "0"),
        ("IntMaxValue", "65535"),
        ("FloatMinValue", "6.25"),
        ("FloatMaxValue", "-3.0"),
    ]:
        assert xml_text(root, f".//apdi:{name}") == value
    for name, value in [
        ("Quality", "high"),
        ("Filtered", "True"),
        ("Accuracy", "relative"),
        ("DepthDataVersion", "65541"),
    ]:
        assert xml_text(root, f".//depthData:{name}") == value
    rend = root.find(".//depthBlurEffect:RenderingParameters", NS)
    assert rend is not None and base64.b64decode(rend.text or "") == rendering_parameters()
    assert xml_text(root, ".//depthData:IntrinsicMatrixReferenceWidth") == "800"


@pytest.mark.parametrize(("a", "b"), [(0, 0), (float("nan"), 1), (0, float("inf"))])
def test_depth_xmp_rejects_bad_endpoints(a: float, b: float) -> None:
    with pytest.raises(LytroError):
        depth_xmp(16, 16, a, b)


def test_spatial_property_defaults_match_reference_model() -> None:
    properties = spatial_properties(1134, 1134)
    assert properties.cmin[:12] == struct.pack(">I4sI", 24, b"cmin", 16 << 8)
    focal, cx, cy = struct.unpack(">iii", properties.cmin[12:])
    assert focal == 62371 and cx == cy == 32768
    assert struct.unpack(">I4sIiiihhh", properties.cmex_left) == (30, b"cmex", 15, 0, 0, 0, 0, 0, 0)
    assert struct.unpack(">I4sIiiihhh", properties.cmex_right) == (
        30,
        b"cmex",
        15,
        35000,
        0,
        0,
        0,
        0,
        0,
    )
    assert properties.dadj == struct.pack(">I4sIi", 16, b"dadj", 0, 0)


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(LytroError):
        build_metadata((b'{"test":NaN}',), b"{}", {}, {}, 16, 16, 0, 1)


def test_original_master_retained_without_private_payload_and_explicit_private_opt_in() -> None:
    master = (
        b'{ "picture":{"frameArray":[{"frame":{"privateMetadataRef":"sha1-digest"}}]},"value":-0 }'
    )
    public = json.dumps(source()).encode()
    private = b'{"camera":{"serialNumber":"PRIVATE-TEST-SERIAL"}}'
    default = build_metadata((public,), b"{}", {}, {}, 16, 16, 0, 1, master_json=master)
    root = ET.fromstring(default.primary_xmp)
    original = xml_text(root, ".//lff:SourceMasterJSONBase64")
    assert original is not None and base64.b64decode(original) == master
    assert root.find(".//lff:SourcePrivateJSONBase64", NS) is None
    assert private not in default.primary_xmp
    included = build_metadata((public,), b"{}", {}, {}, 16, 16, 0, 1, private_json=(private,))
    root = ET.fromstring(included.primary_xmp)
    field = xml_text(root, ".//lff:SourcePrivateJSONBase64/rdf:Seq/rdf:li")
    assert field is not None and base64.b64decode(field) == private


def test_depth_distinguishes_native_relative_disparity_from_descending_source_lambda() -> None:
    mapping = {
        "lambda_at_code_0": 30,
        "lambda_at_code_4095": -3.3,
        "polarity": "inverse lambda; higher code nearer",
    }
    bundle = build_metadata((b"{}",), b"{}", {}, {}, 16, 16, 0, 1, source_depth_mapping=mapping)
    root = ET.fromstring(bundle.depth_xmp)
    data = json.loads(xml_text(root, ".//lff:DepthInterpretation") or "")
    assert data["codeToRelativeDisparityEndpoints"] == [0, 1]
    assert data["sourceLambdaMapping"] == mapping
    assert "codeToLambdaEndpoints" not in data


@pytest.mark.parametrize("timestamp", ["2013-01-07T20:51:31+00:99", "2013-01-07T20:51:31", "bad"])
def test_malformed_capture_time_offset_is_omitted(timestamp: str) -> None:
    _, exif = exif_entries(
        exif_payload(
            {"devices": {"clock": {"zuluTime": timestamp}}}, 16, 16, srgb=False, portrait=False
        )
    )
    assert "DateTimeOriginal" not in exif
    assert "OffsetTimeOriginal" not in exif


def test_source_offset_and_fraction_retained_and_huge_numeric_fields_omitted() -> None:
    _, exif = exif_entries(
        exif_payload(
            {
                "devices": {
                    "clock": {"zuluTime": "2013-01-07T20:51:31.123456789-00:00"},
                    "shutter": {"pixelExposureDuration": 10**1000},
                }
            },
            16,
            16,
            srgb=False,
            portrait=False,
        )
    )
    assert exif["OffsetTimeOriginal"] == "-00:00"
    assert exif["SubsecTimeOriginal"] == "123456789"
    assert "ExposureTime" not in exif
