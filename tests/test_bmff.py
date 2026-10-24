"""Synthetic item-graph fixtures; no captures or vendor executables are bundled."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from lytro_avif import bmff

ICC = b"synthetic ICC bytes for container-only preservation tests"


def _bits(fields: list[tuple[int, int]]) -> bytes:
    binary = "".join(f"{value:0{width}b}" for value, width in fields)
    binary += "0" * (-len(binary) % 8)
    return int(binary, 2).to_bytes(len(binary) // 8, "big")


def _coded(depth: int, mono: bool = False) -> tuple[bytes, bytes]:
    profile = 2 if depth == 12 else 1
    fields = [
        (profile, 3),
        (1, 1),
        (1, 1),
        (0, 5),
        (3, 4),
        (3, 4),
        (15, 4),
        (15, 4),
        (0, 3),
        (0, 3),
        (int(depth > 8), 1),
    ]
    if profile == 2:
        fields += [(int(depth == 12), 1), (int(mono), 1)]
    fields += [(1, 1), (2, 8), (2, 8), (2 if mono else 0, 8), (1, 1)]
    if not mono:
        if profile == 2:
            fields += [(0, 1)]  # subsampling_x
        fields += [(0, 1)]  # separate_uv_delta_q
    fields += [(0, 1), (1, 1)]  # film grain, trailing bit
    sequence = _bits(fields)
    payload = bytes([0x0A, len(sequence)]) + sequence
    config_flags = (0x60 if depth == 12 else 0) | (0x1C if mono else 0)
    return payload, bytes([0x81, profile << 5, config_flags, 0])


def _property(kind: bytes, payload: bytes, *, full: bool = False) -> bytes:
    return bmff.full_box(kind, payload) if full else bmff.box(kind, payload)


def _color() -> bmff.Avif:
    high, high_config = _coded(12)
    low, low_config = _coded(8)
    avif = bmff.Avif(
        1,
        {
            1: bmff.Item(1, b"av01", "Base", False, high),
            2: bmff.Item(2, b"sato", "RGB16", False, bmff.SATO_12_4),
            3: bmff.Item(3, b"av01", "Suffix", True, low),
        },
        [],
        {},
        [bmff.Reference(b"dimg", 2, (1, 3))],
        [bmff.Group(b"altr", 4, (2, 1))],
    )
    ispe = _property(b"ispe", struct.pack(">II", 16, 16), full=True)
    for item, depth in ((1, 12), (2, 16), (3, 8)):
        avif.add_property(item, ispe)
        avif.add_property(item, _property(b"pixi", bytes([3, depth, depth, depth]), full=True))
    for item, config in ((1, high_config), (3, low_config)):
        avif.add_property(item, _property(b"av1C", config), True)
    avif.add_property(1, _property(b"colr", b"nclx" + struct.pack(">HHHB", 2, 2, 0, 128)))
    avif.add_property(1, _property(b"colr", b"prof" + ICC))
    return avif


def _depth() -> bmff.Avif:
    payload, config = _coded(12, True)
    avif = bmff.Avif(1, {1: bmff.Item(1, b"av01", "Disparity", False, payload)}, [], {})
    avif.add_property(1, _property(b"ispe", struct.pack(">II", 16, 16), full=True))
    avif.add_property(1, _property(b"pixi", bytes([1, 12]), full=True))
    avif.add_property(1, _property(b"av1C", config), True)
    avif.add_property(1, _property(b"colr", b"nclx" + struct.pack(">HHHB", 2, 2, 2, 128)))
    return avif


def test_sato_graph_roundtrip_preserves_payloads_and_roles() -> None:
    original = _color()
    restored = bmff.parse(bmff.serialize(original))
    assert bmff.validate_color(restored, ICC) == bmff.ColorRoles(1, 3, 2, 16, 16)
    assert restored.items == original.items
    assert restored.references == original.references
    assert restored.groups == original.groups
    assert restored.properties == original.properties
    assert restored.associations == original.associations


def test_monochrome_precision_comes_from_sequence_header() -> None:
    avif = bmff.parse(bmff.serialize(_depth()))
    assert bmff.validate_depth(avif) == bmff.DepthRoles(1, 16, 16)
    seq = bmff.sequence(avif, 1)
    assert seq.depth == 12 and seq.monochrome and seq.full_range


def test_decoder_primary_selection_patches_only_pitm() -> None:
    original = bmff.serialize(_color())
    changed = bmff.select_primary(original, 3)
    parsed = bmff.parse(changed)
    assert parsed.primary == 3
    assert parsed.items == bmff.parse(original).items
    assert sum(a != b for a, b in zip(original, changed, strict=True)) == 1


def test_missing_primary_selection_is_rejected() -> None:
    with pytest.raises(bmff.ContainerError, match="missing image"):
        bmff.select_primary(bmff.serialize(_color()), 100)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\0" * 7,
        struct.pack(">I4s", 7, b"ftyp"),
        struct.pack(">I4s", 200, b"ftyp"),
        struct.pack(">I4s", 1, b"ftyp"),
    ],
)
def test_truncated_or_invalid_boxes_fail(data: bytes) -> None:
    with pytest.raises(bmff.ContainerError):
        bmff.parse(data)


def test_unknown_box_is_not_silently_stripped() -> None:
    data = bmff.serialize(_color()) + bmff.box(b"moov", b"")
    with pytest.raises(bmff.ContainerError, match="unsupported top-level"):
        bmff.parse(data)


def test_unknown_property_fails_closed() -> None:
    avif = _color()
    avif.add_property(1, bmff.box(b"irot", b"\1"))
    with pytest.raises(bmff.ContainerError, match="unsupported image property"):
        bmff.parse(bmff.serialize(avif))


def test_extents_cannot_reference_box_headers_or_external_bytes() -> None:
    data = bytearray(bmff.serialize(_color()))
    outer = bmff.boxes(bytes(data))
    meta = next(b for b in outer if b.kind == b"meta")
    iloc = next(
        b for b in bmff.boxes(bytes(data), meta.payload_start + 4, meta.end) if b.kind == b"iloc"
    )
    # Full-box prefix + size fields + count + id/dataref/extentcount = first offset.
    struct.pack_into(">I", data, iloc.payload_start + 14, 0)
    with pytest.raises(bmff.ContainerError, match="extent lies outside"):
        bmff.parse(bytes(data))


def test_group_identity_cannot_collide_with_item() -> None:
    avif = _color()
    avif.groups = [bmff.Group(b"altr", 1, (2, 1))]
    with pytest.raises(bmff.ContainerError, match="disjoint"):
        bmff.serialize(avif)


def test_derived_image_cycle_is_rejected() -> None:
    avif = _color()
    avif.references = [bmff.Reference(b"dimg", 2, (2, 3))]
    with pytest.raises(bmff.ContainerError, match="cyclic"):
        bmff.serialize(avif)


def test_av1_configuration_must_match_actual_coding() -> None:
    avif = _depth()
    index = next(i for i, p in enumerate(avif.properties) if p.kind == b"av1C")
    avif.properties[index] = bmff.Property(b"av1C", _property(b"av1C", bytes.fromhex("81401c00")))
    with pytest.raises(bmff.ContainerError, match="disagrees"):
        bmff.validate_depth(avif)


def test_only_exact_sato_expression_is_admitted() -> None:
    avif = _color()
    avif.items[2] = bmff.Item(2, b"sato", "Wrong", False, b"\0")
    with pytest.raises(bmff.ContainerError, match="expression"):
        bmff.validate_color(avif)


def test_only_ordered_alternatives_are_admitted() -> None:
    avif = _color()
    avif.groups = [bmff.Group(b"altr", 4, (1, 2))]
    with pytest.raises(bmff.ContainerError, match="ordered"):
        bmff.validate_color(avif)


def test_complete_color_requires_icc_on_every_component() -> None:
    avif = _color()
    with pytest.raises(bmff.ContainerError, match="ICC"):
        bmff.validate_color(avif, ICC, complete_color=True)
    for target in (2, 3):
        for prop in avif.item_properties(1, b"colr"):
            avif.add_property(target, prop.raw)
    bmff.validate_color(avif, ICC, complete_color=True)


def test_assembly_imports_all_components_and_metadata(tmp_path: Path) -> None:
    from types import SimpleNamespace

    paths = [tmp_path / f"{name}.avif" for name in ("center", "left", "right", "depth")]
    for path in paths[:3]:
        path.write_bytes(bmff.serialize(_color()))
    paths[3].write_bytes(bmff.serialize(_depth()))
    spatial = SimpleNamespace(
        cmin=bmff.full_box(b"cmin", bytes(12)),
        cmex_left=bmff.full_box(b"cmex", bytes(18)),
        cmex_right=bmff.full_box(b"cmex", bytes(18)),
        dadj=bmff.full_box(b"dadj", bytes(4)),
    )
    metadata: Any = SimpleNamespace(
        center_exif=b"\0\0\0\0II*\0",
        eye_exif=b"\0\0\0\0II*\0",
        primary_xmp=b"<source/>",
        depth_xmp=b"<depth/>",
        spatial=spatial,
    )
    output = tmp_path / "final.avif"
    result = bmff.assemble(*paths, metadata, output)
    final = bmff.read(output)
    bmff.validate_final(final, result, ICC)
    assert len(final.items) == 14
    assert len(final.groups) == 4
    assert result.depth.item == 10
    assert final.items[result.depth.item].hidden
    assert result.size == output.stat().st_size
    assert [g.members for g in final.groups if g.kind == b"ster"] == [(4, 7)]
    assert [r.targets for r in final.references if r.kind == b"auxl"] == [(1, 2)]
    for roles in (result.center, result.left, result.right):
        for item in (roles.base, roles.suffix, roles.sato):
            assert bmff.icc_profile(final, item) == ICC


def _low4_leaf() -> bmff.Avif:
    color = _color()
    suffix = color.items[3]
    leaf = bmff.Avif(3, {3: suffix}, color.properties.copy(), {3: color.associations[3].copy()})
    for prop in color.item_properties(1, b"colr"):
        leaf.add_property(3, prop.raw)
    return leaf


def test_shifted_suffix_recipe_is_only_admitted_for_normalization() -> None:
    color = _color()
    original = color.items[2]
    color.items[2] = bmff.Item(
        original.id, original.kind, original.name, original.hidden, bmff.SATO_SHIFTED_12_4
    )
    with pytest.raises(bmff.ContainerError, match="expression"):
        bmff.validate_color(color)
    assert bmff.validate_color(color, allow_shifted=True).suffix == 3


def test_normalization_preserves_base_and_imports_literal_low4(tmp_path: Path) -> None:
    color = _color()
    original = color.items[2]
    color.items[2] = bmff.Item(
        original.id, original.kind, original.name, original.hidden, bmff.SATO_SHIFTED_12_4
    )
    color_path, low4_path = tmp_path / "color.avif", tmp_path / "low4.avif"
    color_path.write_bytes(bmff.serialize(color))
    low4 = _low4_leaf()
    low4_path.write_bytes(bmff.serialize(low4))
    bmff.normalize_suffix(color_path, low4_path)
    normalized = bmff.read(color_path)
    roles = bmff.validate_color(normalized, ICC, complete_color=True)
    assert normalized.items[roles.base].payload == color.items[1].payload
    assert normalized.items[roles.suffix].payload == low4.items[3].payload
    assert normalized.items[roles.sato].payload == bmff.SATO_12_4
    assert normalized.one_property(roles.suffix, b"av1C") == low4.one_property(3, b"av1C")
    assert normalized.references == color.references
    assert normalized.groups == color.groups


def test_normalization_rejects_color_mismatch_without_replacing_source(tmp_path: Path) -> None:
    color_path, low4_path = tmp_path / "color.avif", tmp_path / "low4.avif"
    original = bmff.serialize(_color())
    color_path.write_bytes(original)
    low4 = _low4_leaf()
    low4.add_property(3, bmff.box(b"colr", b"prof" + b"different ICC"))
    low4_path.write_bytes(bmff.serialize(low4))
    with pytest.raises(bmff.ContainerError, match="ICC"):
        bmff.normalize_suffix(color_path, low4_path)
    assert color_path.read_bytes() == original


def test_shared_derived_dependencies_are_checked_in_bounded_time() -> None:
    # Without a visited set, this valid DAG expands exponentially at validation.
    items = {
        item_id: bmff.Item(item_id, b"sato", "synthetic", False, b"x") for item_id in range(1, 100)
    }
    references = [
        bmff.Reference(b"dimg", item_id, (item_id + 1, item_id + 2)) for item_id in range(1, 98)
    ]
    graph = bmff.Avif(1, items, [], {}, references)
    bmff.validate_graph(graph)
