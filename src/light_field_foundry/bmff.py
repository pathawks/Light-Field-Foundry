"""Bounded AVIF item-graph import and assembly, without touching coded samples.

Only still-image structures used by the checked libavif encoder are admitted.
Unknown boxes, external extents, transformations and sequence tracks fail closed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import LytroError

if TYPE_CHECKING:
    from .metadata import MetadataBundle

MAX_FILE = 512 * 1024 * 1024
MAX_METADATA = 16 * 1024 * 1024
MAX_ITEMS = 256
MAX_PROPERTIES = 1024
MAX_BOXES = 4096
MAX_DIMENSION = 32768
DEPTH_AUX_TYPE = b"urn:mpeg:mpegB:cicp:systems:auxiliary:depth\0"
# libavif 1.4.2 left-shifts its four-bit suffix into an RGB8 component.
# Normalize that representation to the plan's literal low4 codes and expression.
SATO_SHIFTED_12_4 = bytes.fromhex("0207000000001001820200000000108380")
SATO_12_4 = bytes.fromhex("0205000000001001820280")


class ContainerError(LytroError, ValueError):
    """A container cannot establish the pipeline's still-image contract."""


@dataclass(frozen=True)
class Box:
    kind: bytes
    start: int
    payload_start: int
    end: int

    def body(self, data: bytes) -> bytes:
        return data[self.payload_start : self.end]

    def raw(self, data: bytes) -> bytes:
        return data[self.start : self.end]


def boxes(data: bytes, start: int = 0, end: int | None = None) -> list[Box]:
    end = len(data) if end is None else end
    if not 0 <= start <= end <= len(data):
        raise ContainerError("invalid box bounds")
    result: list[Box] = []
    while start < end:
        if len(result) >= MAX_BOXES or end - start < 8:
            raise ContainerError("too many boxes or truncated box header")
        size, kind = struct.unpack_from(">I4s", data, start)
        header = 8
        if size == 1:
            if end - start < 16:
                raise ContainerError("truncated extended box header")
            size = struct.unpack_from(">Q", data, start + 8)[0]
            header = 16
        elif size == 0:
            size = end - start
        if size < header or size > end - start:
            raise ContainerError("box length exceeds enclosing bounds")
        result.append(Box(kind, start, start + header, start + size))
        start += size
    return result


def box(kind: bytes, payload: bytes) -> bytes:
    if len(kind) != 4 or len(payload) > MAX_FILE - 8:
        raise ContainerError("invalid box type or size")
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def full_box(kind: bytes, payload: bytes, version: int = 0, flags: int = 0) -> bytes:
    return box(kind, bytes([version]) + flags.to_bytes(3, "big") + payload)


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, size: int) -> bytes:
        if size < 0 or size > len(self.data) - self.pos:
            raise ContainerError("truncated item structure")
        data = self.data[self.pos : self.pos + size]
        self.pos += size
        return data

    def integer(self, size: int) -> int:
        return int.from_bytes(self.take(size), "big")

    def string(self) -> str:
        end = self.data.find(b"\0", self.pos)
        if end < 0 or end - self.pos > 4096:
            raise ContainerError("missing or oversized item string")
        try:
            result = self.data[self.pos : end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContainerError("invalid UTF-8 item string") from error
        self.pos = end + 1
        return result

    def finish(self) -> None:
        if self.pos != len(self.data):
            raise ContainerError("unrecognized trailing item bytes")

    def full(self, versions: tuple[int, ...] = (0,), allowed_flags: int = 0) -> tuple[int, int]:
        version, flags = self.integer(1), self.integer(3)
        if version not in versions or flags & ~allowed_flags:
            raise ContainerError("unsupported full-box version or flags")
        return version, flags


@dataclass(frozen=True)
class Item:
    id: int
    kind: bytes
    name: str
    hidden: bool
    payload: bytes
    content_type: str | None = None


@dataclass(frozen=True)
class Property:
    kind: bytes
    raw: bytes

    @property
    def payload(self) -> bytes:
        parsed = boxes(self.raw)
        return parsed[0].body(self.raw)


@dataclass(frozen=True)
class Association:
    property_index: int  # one-based, as in ipma
    essential: bool = False


@dataclass(frozen=True)
class Reference:
    kind: bytes
    source: int
    targets: tuple[int, ...]


@dataclass(frozen=True)
class Group:
    kind: bytes
    id: int
    members: tuple[int, ...]


@dataclass
class Avif:
    primary: int
    items: dict[int, Item]
    properties: list[Property]
    associations: dict[int, list[Association]]
    references: list[Reference] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    source: bytes = b""
    primary_field: tuple[int, int] | None = None

    def item_properties(self, item_id: int, kind: bytes | None = None) -> list[Property]:
        return [
            self.properties[a.property_index - 1]
            for a in self.associations.get(item_id, [])
            if kind is None or self.properties[a.property_index - 1].kind == kind
        ]

    def one_property(self, item_id: int, kind: bytes) -> Property:
        matches = self.item_properties(item_id, kind)
        if len(matches) != 1:
            raise ContainerError(f"item {item_id} requires one {kind.decode()} property")
        return matches[0]

    def add_property(self, item_id: int, raw: bytes, essential: bool = False) -> None:
        parsed = boxes(raw)
        if len(parsed) != 1:
            raise ContainerError("property must contain exactly one box")
        prop = Property(parsed[0].kind, raw)
        try:
            index = self.properties.index(prop) + 1
        except ValueError:
            self.properties.append(prop)
            index = len(self.properties)
        association = Association(index, essential)
        if association not in self.associations.setdefault(item_id, []):
            self.associations[item_id].append(association)


def _one(parsed: list[Box], kind: bytes, *, optional: bool = False) -> Box | None:
    found = [b for b in parsed if b.kind == kind]
    if len(found) != 1:
        if optional and not found:
            return None
        raise ContainerError(f"expected one {kind.decode()} box")
    return found[0]


def _required(parsed: list[Box], kind: bytes) -> Box:
    value = _one(parsed, kind)
    assert value is not None
    return value


def parse(data: bytes) -> Avif:
    if not 16 <= len(data) <= MAX_FILE:
        raise ContainerError("AVIF file size outside bounds")
    top = boxes(data)
    if any(b.kind not in {b"ftyp", b"meta", b"mdat"} for b in top):
        raise ContainerError("unsupported top-level box; only still AVIFs are supported")
    ftyp = _required(top, b"ftyp").body(data)
    if (
        len(ftyp) < 12
        or len(ftyp) % 4
        or b"avif" not in [ftyp[:4], *[ftyp[n : n + 4] for n in range(8, len(ftyp), 4)]]
    ):
        raise ContainerError("file is not branded as AVIF")
    meta = _required(top, b"meta")
    reader = Reader(meta.body(data))
    reader.full()
    children = boxes(data, meta.payload_start + 4, meta.end)
    if any(
        b.kind not in {b"hdlr", b"pitm", b"iloc", b"iinf", b"iref", b"iprp", b"grpl", b"idat"}
        for b in children
    ):
        raise ContainerError("unsupported meta child box")
    hdlr = Reader(_required(children, b"hdlr").body(data))
    hdlr.full()
    if hdlr.integer(4) != 0 or hdlr.take(4) != b"pict":
        raise ContainerError("unsupported metadata handler")
    pitm = _required(children, b"pitm")
    r = Reader(pitm.body(data))
    version, _ = r.full((0, 1))
    id_size = 2 if version == 0 else 4
    primary = r.integer(id_size)
    r.finish()
    item_info = _required(children, b"iinf")
    r = Reader(item_info.body(data))
    version, _ = r.full((0, 1))
    count = r.integer(2 if version == 0 else 4)
    if not 1 <= count <= MAX_ITEMS:
        raise ContainerError("invalid item count")
    item_boxes = boxes(data, item_info.payload_start + r.pos, item_info.end)
    if len(item_boxes) != count or any(b.kind != b"infe" for b in item_boxes):
        raise ContainerError("item count does not match infe entries")
    infos: dict[int, tuple[bytes, str, bool, str | None]] = {}
    for entry in item_boxes:
        r = Reader(entry.body(data))
        version, flags = r.full((2, 3), 1)
        item_id = r.integer(2 if version == 2 else 4)
        if item_id == 0 or item_id in infos or r.integer(2):
            raise ContainerError("invalid, duplicate or protected item")
        kind, name = r.take(4), r.string()
        if kind not in {b"av01", b"sato", b"Exif", b"mime"}:
            raise ContainerError("unsupported image item type")
        content_type = r.string() if kind == b"mime" else None
        if kind == b"mime" and r.pos < len(r.data) and r.string():
            raise ContainerError("encoded MIME item is unsupported")
        r.finish()
        infos[item_id] = kind, name, bool(flags & 1), content_type
    iloc = Reader(_required(children, b"iloc").body(data))
    version, _ = iloc.full((0, 1, 2))
    sizes = iloc.integer(1)
    offset_size, length_size = sizes >> 4, sizes & 15
    sizes = iloc.integer(1)
    base_size, index_size = sizes >> 4, sizes & 15
    if any(
        size not in {0, 4, 8} for size in (offset_size, length_size, base_size)
    ) or index_size not in {0, 4, 8}:
        raise ContainerError("unsupported iloc integer width")
    if version == 0 and index_size:
        raise ContainerError("reserved iloc index field is nonzero")
    if iloc.integer(2 if version < 2 else 4) != count:
        raise ContainerError("iloc item count mismatch")
    items: dict[int, Item] = {}
    idat = _one(children, b"idat", optional=True)
    media = [b for b in top if b.kind == b"mdat"]
    for _ in range(count):
        item_id = iloc.integer(2 if version < 2 else 4)
        construction = iloc.integer(2) if version else 0
        if construction not in {0, 1} or iloc.integer(2):
            raise ContainerError("external or unsupported item extent")
        base_offset = iloc.integer(base_size)
        extent_count = iloc.integer(2)
        if item_id not in infos or item_id in items or not 1 <= extent_count <= 16:
            raise ContainerError("invalid item extent count or identity")
        payloads: list[bytes] = []
        for _ in range(extent_count):
            if version and index_size and iloc.integer(index_size):
                raise ContainerError("indexed extents are unsupported")
            offset = base_offset + iloc.integer(offset_size)
            length = iloc.integer(length_size)
            if not length:
                raise ContainerError("empty item extent")
            if construction == 1:
                if idat is None:
                    raise ContainerError("idat extent has no idat box")
                offset += idat.payload_start
                valid = idat.payload_start <= offset <= offset + length <= idat.end
            else:
                valid = any(b.payload_start <= offset <= offset + length <= b.end for b in media)
            if not valid:
                raise ContainerError("item extent lies outside its data box")
            payloads.append(data[offset : offset + length])
        kind, name, hidden, content_type = infos[item_id]
        payload = b"".join(payloads)
        if kind in {b"Exif", b"mime"} and len(payload) > MAX_METADATA:
            raise ContainerError("metadata item too large")
        items[item_id] = Item(item_id, kind, name, hidden, payload, content_type)
    iloc.finish()
    iprp = _required(children, b"iprp")
    property_children = boxes(data, iprp.payload_start, iprp.end)
    if any(b.kind not in {b"ipco", b"ipma"} for b in property_children):
        raise ContainerError("unsupported iprp child")
    ipco = _required(property_children, b"ipco")
    parsed_properties = boxes(data, ipco.payload_start, ipco.end)
    if len(parsed_properties) > MAX_PROPERTIES:
        raise ContainerError("too many image properties")
    allowed_properties = {b"ispe", b"pixi", b"av1C", b"colr", b"auxC", b"cmin", b"cmex", b"dadj"}
    if any(b.kind not in allowed_properties for b in parsed_properties):
        raise ContainerError("unsupported image property or transform")
    properties = [Property(b.kind, b.raw(data)) for b in parsed_properties]
    r = Reader(_required(property_children, b"ipma").body(data))
    version, flags = r.full((0, 1), 1)
    entries = r.integer(4)
    if entries > MAX_ITEMS + MAX_ITEMS:
        raise ContainerError("too many association entries")
    associations: dict[int, list[Association]] = {}
    for _ in range(entries):
        entity = r.integer(2 if version == 0 else 4)
        if entity == 0 or entity in associations:
            raise ContainerError("invalid duplicate association entity")
        entity_associations = []
        for _ in range(r.integer(1)):
            value = r.integer(2 if flags & 1 else 1)
            mask = 0x7FFF if flags & 1 else 0x7F
            index = value & mask
            if not 1 <= index <= len(properties):
                raise ContainerError("invalid property association index")
            association = Association(index, bool(value & ~mask))
            if association in entity_associations:
                raise ContainerError("duplicate property association")
            entity_associations.append(association)
        associations[entity] = entity_associations
    r.finish()
    references: list[Reference] = []
    iref = _one(children, b"iref", optional=True)
    if iref:
        r = Reader(iref.body(data))
        version, _ = r.full((0, 1))
        for entry in boxes(data, iref.payload_start + 4, iref.end):
            if entry.kind not in {b"dimg", b"auxl", b"cdsc"}:
                raise ContainerError("unsupported item reference")
            r = Reader(entry.body(data))
            source = r.integer(2 if version == 0 else 4)
            target_count = r.integer(2)
            if not 1 <= target_count <= MAX_ITEMS:
                raise ContainerError("invalid reference count")
            targets = tuple(r.integer(2 if version == 0 else 4) for _ in range(target_count))
            r.finish()
            references.append(Reference(entry.kind, source, targets))
    groups: list[Group] = []
    grpl = _one(children, b"grpl", optional=True)
    if grpl:
        for entry in boxes(data, grpl.payload_start, grpl.end):
            if entry.kind not in {b"altr", b"ster"}:
                raise ContainerError("unsupported entity group")
            r = Reader(entry.body(data))
            r.full()
            group_id, member_count = r.integer(4), r.integer(4)
            if not 1 <= member_count <= MAX_ITEMS:
                raise ContainerError("invalid group membership count")
            members = tuple(r.integer(4) for _ in range(member_count))
            r.finish()
            groups.append(Group(entry.kind, group_id, members))
    result = Avif(
        primary,
        items,
        properties,
        associations,
        references,
        groups,
        data,
        (pitm.payload_start + 4, id_size),
    )
    validate_graph(result)
    return result


def read(path: Path) -> Avif:
    if path.stat().st_size > MAX_FILE:
        raise ContainerError("AVIF exceeds file-size bound")
    return parse(path.read_bytes())


def validate_graph(avif: Avif) -> None:
    if avif.primary not in avif.items or not 1 <= len(avif.items) <= MAX_ITEMS:
        raise ContainerError("missing primary or invalid item count")
    group_ids = [group.id for group in avif.groups]
    if (
        len(set(group_ids)) != len(group_ids)
        or set(group_ids) & avif.items.keys()
        or 0 in group_ids
    ):
        raise ContainerError("group IDs must be unique and disjoint from item IDs")
    entities = set(avif.items) | set(group_ids)
    if any(entity not in entities for entity in avif.associations):
        raise ContainerError("association references nonexistent entity")
    if any(
        ref.source not in avif.items or any(target not in avif.items for target in ref.targets)
        for ref in avif.references
    ):
        raise ContainerError("reference targets nonexistent item")
    if any(
        any(member not in avif.items for member in group.members)
        or len(set(group.members)) != len(group.members)
        for group in avif.groups
    ):
        raise ContainerError("invalid group membership")
    ref_keys = [(ref.kind, ref.source) for ref in avif.references]
    if len(set(ref_keys)) != len(ref_keys):
        raise ContainerError("duplicate item reference list")
    # A derived-image cycle must never reach the independent decoder.
    dependencies = {ref.source: ref.targets for ref in avif.references if ref.kind == b"dimg"}

    visited: set[int] = set()
    visiting: set[int] = set()

    def visit(item_id: int) -> None:
        if item_id in visiting:
            raise ContainerError("cyclic derived-image dependency")
        if item_id in visited:
            return
        visiting.add(item_id)
        for target in dependencies.get(item_id, ()):
            visit(target)
        visiting.remove(item_id)
        visited.add(item_id)

    for item_id in dependencies:
        visit(item_id)


class Bits:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def get(self, count: int = 1) -> int:
        if count < 0 or self.position + count > len(self.data) * 8:
            raise ContainerError("truncated AV1 sequence header")
        value = 0
        for _ in range(count):
            value = (value << 1) | ((self.data[self.position // 8] >> (7 - self.position % 8)) & 1)
            self.position += 1
        return value


@dataclass(frozen=True)
class Sequence:
    profile: int
    level: int
    width: int
    height: int
    color_description_present: bool
    depth: int
    monochrome: bool
    full_range: bool
    subsampling_x: int
    subsampling_y: int
    primaries: int
    transfer: int
    matrix: int


def _sequence_header(payload: bytes) -> Sequence:
    bits = Bits(payload)
    profile = bits.get(3)
    still, reduced = bits.get(), bits.get()
    if not still or not reduced or profile > 2:
        raise ContainerError("only reduced still-picture AV1 sequence headers are admitted")
    level = bits.get(5)
    width_bits, height_bits = bits.get(4) + 1, bits.get(4) + 1
    width = bits.get(width_bits) + 1
    height = bits.get(height_bits) + 1
    bits.get(3)  # superblock, filter intra, intra edge filter
    bits.get(3)  # superres, CDEF, restoration
    high = bits.get()
    twelve = bits.get() if profile == 2 and high else 0
    depth = 12 if twelve else (10 if high else 8)
    monochrome = bool(bits.get()) if profile != 1 else False
    color_description_present = bool(bits.get())
    if color_description_present:
        primaries, transfer, matrix = bits.get(8), bits.get(8), bits.get(8)
    else:
        primaries = transfer = matrix = 2
    if monochrome:
        full_range = bool(bits.get())
        sub_x = sub_y = 1
    elif (primaries, transfer, matrix) == (1, 13, 0):
        full_range, sub_x, sub_y = True, 0, 0
        bits.get()  # separate_uv_delta_q
    else:
        full_range = bool(bits.get())
        if profile == 0:
            sub_x = sub_y = 1
        elif profile == 1:
            sub_x = sub_y = 0
        elif depth == 12:
            sub_x = bits.get()
            sub_y = bits.get() if sub_x else 0
        else:
            sub_x, sub_y = 1, 0
        if sub_x and sub_y:
            bits.get(2)
        bits.get()  # separate_uv_delta_q
    return Sequence(
        profile,
        level,
        width,
        height,
        color_description_present,
        depth,
        monochrome,
        full_range,
        sub_x,
        sub_y,
        primaries,
        transfer,
        matrix,
    )


def _obu_sequences(data: bytes) -> list[bytes]:
    reader = Reader(data)
    found: list[bytes] = []
    count = 0
    while reader.pos < len(data):
        count += 1
        if count > MAX_BOXES:
            raise ContainerError("too many AV1 OBUs")
        header = reader.integer(1)
        if header & 0x81:
            raise ContainerError("invalid AV1 OBU header")
        kind = (header >> 3) & 15
        if header & 4:
            extension = reader.integer(1)
            if extension & 7:
                raise ContainerError("invalid OBU extension")
        if header & 2:
            size = 0
            for shift in range(0, 56, 7):
                octet = reader.integer(1)
                size |= (octet & 127) << shift
                if not octet & 128:
                    break
            else:
                raise ContainerError("oversized AV1 LEB128")
        else:
            size = len(data) - reader.pos
        payload = reader.take(size)
        if kind == 1:
            found.append(payload)
    return found


def sequence(avif: Avif, item_id: int) -> Sequence:
    item = avif.items[item_id]
    if item.kind != b"av01":
        raise ContainerError("item is not coded AV1")
    config = avif.one_property(item_id, b"av1C").payload
    if len(config) < 4 or config[0] != 0x81 or config[3] != 0:
        raise ContainerError("invalid AV1 configuration record")
    headers = _obu_sequences(item.payload)
    if len(headers) != 1:
        raise ContainerError("coded item requires one sequence header")
    result = _sequence_header(headers[0])
    profile, level = config[1] >> 5, config[1] & 31
    high, twelve, mono = bool(config[2] & 0x40), bool(config[2] & 0x20), bool(config[2] & 0x10)
    depth = 12 if twelve and high else (10 if high else 8)
    if (profile, level, depth, mono, (config[2] >> 3) & 1, (config[2] >> 2) & 1) != (
        result.profile,
        result.level,
        result.depth,
        result.monochrome,
        result.subsampling_x,
        result.subsampling_y,
    ):
        raise ContainerError("av1C disagrees with actual AV1 sequence header")
    if dimensions(avif, item_id) != (result.width, result.height):
        raise ContainerError("AV1 sequence dimensions differ from ispe")
    config_headers = _obu_sequences(config[4:])
    if config_headers and config_headers != headers:
        raise ContainerError("av1C sequence header differs from coded item")
    return result


def dimensions(avif: Avif, item_id: int) -> tuple[int, int]:
    r = Reader(avif.one_property(item_id, b"ispe").payload)
    r.full()
    width, height = r.integer(4), r.integer(4)
    r.finish()
    if not 1 <= width <= MAX_DIMENSION or not 1 <= height <= MAX_DIMENSION:
        raise ContainerError("image dimensions outside admitted bounds")
    return width, height


def pixel_depths(avif: Avif, item_id: int) -> tuple[int, ...]:
    r = Reader(avif.one_property(item_id, b"pixi").payload)
    r.full()
    count = r.integer(1)
    if count not in {1, 3}:
        raise ContainerError("only opaque mono/RGB images are supported")
    depths = tuple(r.integer(1) for _ in range(count))
    r.finish()
    return depths


def icc_profile(avif: Avif, item_id: int) -> bytes:
    profiles = [
        p.payload[4:]
        for p in avif.item_properties(item_id, b"colr")
        if p.payload[:4] in {b"prof", b"rICC"}
    ]
    if len(profiles) != 1 or not profiles[0]:
        raise ContainerError("one nonempty ICC profile is required per color item")
    return profiles[0]


def cicp(avif: Avif, item_id: int) -> tuple[int, int, int, bool]:
    props = [p for p in avif.item_properties(item_id, b"colr") if p.payload[:4] == b"nclx"]
    if len(props) != 1 or len(props[0].payload) != 11:
        raise ContainerError("one valid nclx color property is required")
    p, t, m, flags = struct.unpack(">HHHB", props[0].payload[4:])
    if flags & 127:
        raise ContainerError("reserved nclx bits are nonzero")
    return p, t, m, bool(flags & 128)


@dataclass(frozen=True)
class ColorRoles:
    base: int
    suffix: int
    sato: int
    width: int
    height: int


@dataclass(frozen=True)
class DepthRoles:
    item: int
    width: int
    height: int


def validate_color(
    avif: Avif,
    icc: bytes | None = None,
    *,
    complete_color: bool = False,
    allow_shifted: bool = False,
) -> ColorRoles:
    sato_ids = [item.id for item in avif.items.values() if item.kind == b"sato"]
    if len(sato_ids) != 1 or len(avif.items) != 3:
        raise ContainerError("color input must contain exactly one 12+4 SATO graph")
    sato = sato_ids[0]
    dependencies = [r for r in avif.references if r.source == sato and r.kind == b"dimg"]
    if len(dependencies) != 1 or len(dependencies[0].targets) != 2:
        raise ContainerError("SATO requires two ordered inputs")
    base, suffix = dependencies[0].targets
    if base != avif.primary or len({base, suffix, sato}) != 3:
        raise ContainerError("SATO primary must be its base12 compatibility item")
    allowed_expressions = {SATO_12_4, SATO_SHIFTED_12_4} if allow_shifted else {SATO_12_4}
    if avif.items[sato].payload not in allowed_expressions:
        raise ContainerError("unsupported SATO expression; exact 12+4 recipe required")
    if [g.members for g in avif.groups if g.kind == b"altr"] != [(sato, base)] or len(
        avif.groups
    ) != 1:
        raise ContainerError("missing ordered SATO/base alternative group")
    size = dimensions(avif, base)
    for item_id, depth in ((base, 12), (suffix, 8), (sato, 16)):
        if dimensions(avif, item_id) != size or pixel_depths(avif, item_id) != (depth,) * 3:
            raise ContainerError("SATO dimension/channel/precision mismatch")
    for item_id, expected_depth in ((base, 12), (suffix, 8)):
        seq = sequence(avif, item_id)
        if (
            seq.depth != expected_depth
            or seq.monochrome
            or not seq.full_range
            or seq.subsampling_x
            or seq.subsampling_y
        ):
            raise ContainerError("color requires actual full-range RGB444 base12/suffix8 coding")
    base_cicp = cicp(avif, base)
    if base_cicp[2:] != (0, True):
        raise ContainerError("lossless RGB requires full-range identity-matrix CICP")
    for item_id in (base, suffix):
        seq = sequence(avif, item_id)
        if (
            seq.color_description_present
            and (seq.primaries, seq.transfer, seq.matrix, seq.full_range) != base_cicp
        ):
            raise ContainerError("AV1 sequence and container color signaling disagree")
    base_icc = icc_profile(avif, base) if icc is not None or complete_color else None
    if icc is not None and base_icc != icc:
        raise ContainerError("encoded ICC bytes differ from TNT TIFF")
    if complete_color:
        for item_id in (suffix, sato):
            if icc_profile(avif, item_id) != base_icc or cicp(avif, item_id) != base_cicp:
                raise ContainerError("all SATO component color properties must match")
    return ColorRoles(base, suffix, sato, *size)


def normalize_suffix(color_path: Path, low4_path: Path) -> None:
    """Replace libavif's shifted suffix with independently encoded literal low4.

    No coded item is patched internally: the replacement is a complete lossless
    RGB8 AV1 item supplied by the caller. Final independent decoders establish
    both its literal 0..15 samples and exact reconstructed RGB16 values.
    """
    color, low4 = read(color_path), read(low4_path)
    roles = validate_color(color, allow_shifted=True)
    if len(low4.items) != 1 or low4.groups or low4.references:
        raise ContainerError("low4 normalization requires one ordinary RGB8 AV1 item")
    item_id = low4.primary
    if dimensions(low4, item_id) != (roles.width, roles.height):
        raise ContainerError("low4 suffix dimensions differ from the color view")
    if pixel_depths(low4, item_id) != (8, 8, 8):
        raise ContainerError("low4 replacement must declare three 8-bit channels")
    seq = sequence(low4, item_id)
    if (
        seq.depth != 8
        or seq.monochrome
        or not seq.full_range
        or seq.subsampling_x
        or seq.subsampling_y
    ):
        raise ContainerError("low4 replacement must be full-range RGB444 AV1 at 8 bits")
    expected_cicp = cicp(color, roles.base)
    if cicp(low4, item_id) != expected_cicp or icc_profile(low4, item_id) != icc_profile(
        color, roles.base
    ):
        raise ContainerError("low4 replacement color properties differ from the base12 view")
    if (
        seq.color_description_present
        and (seq.primaries, seq.transfer, seq.matrix, seq.full_range) != expected_cicp
    ):
        raise ContainerError("low4 bitstream and container color descriptions disagree")
    suffix, sato = color.items[roles.suffix], color.items[roles.sato]
    color.items[roles.suffix] = Item(
        suffix.id, suffix.kind, suffix.name, suffix.hidden, low4.items[item_id].payload
    )
    color.items[roles.sato] = Item(sato.id, sato.kind, sato.name, sato.hidden, SATO_12_4)
    color.associations[roles.suffix] = [
        association
        for association in color.associations[roles.suffix]
        if color.properties[association.property_index - 1].kind != b"av1C"
    ]
    color.add_property(roles.suffix, low4.one_property(item_id, b"av1C").raw, True)
    for target in (roles.suffix, roles.sato):
        for prop in color.item_properties(roles.base, b"colr"):
            color.add_property(target, prop.raw)
    normalized = serialize(color)
    validate_color(parse(normalized), icc_profile(color, roles.base), complete_color=True)
    color_path.write_bytes(normalized)


def validate_depth(avif: Avif) -> DepthRoles:
    if len(avif.items) != 1 or avif.groups or avif.references:
        raise ContainerError("depth input must be one ordinary AV1 image")
    item_id = avif.primary
    size = dimensions(avif, item_id)
    if pixel_depths(avif, item_id) != (12,):
        raise ContainerError("depth pixi must declare monochrome12")
    seq = sequence(avif, item_id)
    if (
        seq.depth != 12
        or not seq.monochrome
        or not seq.full_range
        or (seq.primaries, seq.transfer, seq.matrix) != (2, 2, 2)
    ):
        raise ContainerError("depth must be actual full-range monochrome12, CICP 2/2/2")
    if cicp(avif, item_id) != (2, 2, 2, True):
        raise ContainerError("depth container CICP must be 2/2/2 full-range")
    if any(p.payload[:4] in {b"prof", b"rICC"} for p in avif.item_properties(item_id, b"colr")):
        raise ContainerError("depth must not carry a color ICC")
    return DepthRoles(item_id, *size)


def serialize(avif: Avif) -> bytes:
    validate_graph(avif)
    if len(avif.properties) > MAX_PROPERTIES or any(
        not 1 <= item_id <= 65535 for item_id in avif.items
    ):
        raise ContainerError("graph exceeds writer ID/property bounds")
    ftyp = box(b"ftyp", b"avif\0\0\0\0avifmif1miaf")

    def make_meta(data_start: int) -> bytes:
        hdlr = full_box(b"hdlr", b"\0\0\0\0pict" + bytes(12) + b"\0")
        pitm = full_box(b"pitm", struct.pack(">H", avif.primary))
        iloc_entries = bytearray()
        offset = data_start
        for item_id, item in avif.items.items():
            iloc_entries.extend(struct.pack(">HHHII", item_id, 0, 1, offset, len(item.payload)))
            offset += len(item.payload)
        iloc = full_box(b"iloc", b"\x44\0" + struct.pack(">H", len(avif.items)) + iloc_entries)
        infe_entries = []
        for item_id, item in avif.items.items():
            payload = struct.pack(">HH4s", item_id, 0, item.kind) + item.name.encode() + b"\0"
            if item.kind == b"mime":
                payload += (item.content_type or "application/rdf+xml").encode() + b"\0"
            infe_entries.append(full_box(b"infe", payload, 2, int(item.hidden)))
        iinf = full_box(b"iinf", struct.pack(">H", len(avif.items)) + b"".join(infe_entries))
        refs = b"".join(
            box(
                r.kind,
                struct.pack(">HH", r.source, len(r.targets))
                + b"".join(struct.pack(">H", target) for target in r.targets),
            )
            for r in avif.references
        )
        iref = full_box(b"iref", refs) if refs else b""
        ipco = box(b"ipco", b"".join(p.raw for p in avif.properties))
        entries = bytearray(struct.pack(">I", len(avif.associations)))
        for entity, assoc in avif.associations.items():
            if not 0 < entity <= 65535 or len(assoc) > 255:
                raise ContainerError("association exceeds writer bounds")
            entries.extend(struct.pack(">HB", entity, len(assoc)))
            for a in assoc:
                if not 1 <= a.property_index <= len(avif.properties):
                    raise ContainerError("invalid property index")
                entries.extend(struct.pack(">H", a.property_index | (0x8000 if a.essential else 0)))
        iprp = box(b"iprp", ipco + full_box(b"ipma", bytes(entries), flags=1))
        groups = b"".join(
            full_box(
                g.kind,
                struct.pack(">II", g.id, len(g.members))
                + b"".join(struct.pack(">I", member) for member in g.members),
            )
            for g in avif.groups
        )
        grpl = box(b"grpl", groups) if groups else b""
        return full_box(b"meta", hdlr + pitm + iloc + iinf + iref + iprp + grpl)

    placeholder = make_meta(0)
    meta = make_meta(len(ftyp) + len(placeholder) + 8)
    result = ftyp + meta + box(b"mdat", b"".join(item.payload for item in avif.items.values()))
    if len(result) > MAX_FILE:
        raise ContainerError("assembled AVIF exceeds file-size bound")
    return result


def select_primary(data: bytes, item_id: int) -> bytes:
    """Patch only pitm: all final payloads, extents and dependency bytes remain intact."""
    avif = parse(data)
    if item_id not in avif.items or avif.primary_field is None:
        raise ContainerError("cannot select missing image item")
    position, width = avif.primary_field
    if item_id >= 1 << (width * 8):
        raise ContainerError("item ID exceeds primary field width")
    return data[:position] + item_id.to_bytes(width, "big") + data[position + width :]


@dataclass(frozen=True)
class AssemblyResult:
    center: ColorRoles
    left: ColorRoles
    right: ColorRoles
    depth: DepthRoles
    stereo_group: int
    size: int


def assemble(
    center: Path,
    left: Path,
    right: Path,
    depth: Path,
    metadata: MetadataBundle,
    output: Path,
    *,
    separate_center_profile: bool = False,
) -> AssemblyResult:
    inputs = [read(path) for path in (center, left, right, depth)]
    source_roles = [validate_color(avif) for avif in inputs[:3]]
    source_depth = validate_depth(inputs[3])
    if len({(role.width, role.height) for role in source_roles}) != 1:
        raise ContainerError("color-view dimensions differ")
    profiles = tuple(
        icc_profile(avif, role.base) for avif, role in zip(inputs[:3], source_roles, strict=True)
    )
    source_icc = profiles[0]
    source_cicp = cicp(inputs[0], source_roles[0].base)
    for avif, roles in zip(inputs[:3], source_roles, strict=True):
        if (not separate_center_profile and icc_profile(avif, roles.base) != source_icc) or cicp(
            avif, roles.base
        ) != source_cicp:
            raise ContainerError("color interpretations differ between views")
    if profiles[1] != profiles[2]:
        raise ContainerError("stereo eye ICC profiles differ")
    merged = Avif(1, {}, [], {})
    new_roles: list[ColorRoles] = []
    next_id = 1
    for avif, roles in zip(inputs, [*source_roles, source_depth], strict=True):
        remap = {old: next_id + index for index, old in enumerate(avif.items)}
        next_id += len(remap)
        group_remap: dict[int, int] = {}  # groups allocated after all image IDs
        for old, item in avif.items.items():
            new = remap[old]
            merged.items[new] = Item(
                new, item.kind, item.name, item.hidden, item.payload, item.content_type
            )
            for association in avif.associations.get(old, []):
                merged.add_property(
                    new, avif.properties[association.property_index - 1].raw, association.essential
                )
        for ref in avif.references:
            merged.references.append(
                Reference(
                    ref.kind, remap[ref.source], tuple(remap[target] for target in ref.targets)
                )
            )
        if isinstance(roles, ColorRoles):
            new_roles.append(
                ColorRoles(
                    remap[roles.base],
                    remap[roles.suffix],
                    remap[roles.sato],
                    roles.width,
                    roles.height,
                )
            )
            for target in (remap[roles.suffix], remap[roles.sato]):
                for prop in avif.item_properties(roles.base, b"colr"):
                    merged.add_property(target, prop.raw)
        for group in avif.groups:
            # Reserve high entity IDs to avoid collisions with later metadata items.
            group_remap[group.id] = 1000 + len(merged.groups)
            merged.groups.append(
                Group(
                    group.kind,
                    group_remap[group.id],
                    tuple(remap[member] for member in group.members),
                )
            )
            for association in avif.associations.get(group.id, []):
                merged.add_property(
                    group_remap[group.id],
                    avif.properties[association.property_index - 1].raw,
                    association.essential,
                )
    center_roles, left_roles, right_roles = new_roles
    depth_item = next(
        item.id
        for item in merged.items.values()
        if item.id > max(right_roles.base, right_roles.suffix, right_roles.sato)
    )
    old_depth = merged.items[depth_item]
    merged.items[depth_item] = Item(
        depth_item, old_depth.kind, "Center disparity (12-bit)", True, old_depth.payload
    )
    merged.primary = center_roles.base
    merged.add_property(depth_item, full_box(b"auxC", DEPTH_AUX_TYPE), True)
    merged.references.append(Reference(b"auxl", depth_item, (center_roles.base, center_roles.sato)))
    stereo_id = 1000 + len(merged.groups)
    merged.groups.append(Group(b"ster", stereo_id, (left_roles.base, right_roles.base)))
    for entity, raw in (
        (left_roles.base, metadata.spatial.cmin),
        (right_roles.base, metadata.spatial.cmin),
        (left_roles.base, metadata.spatial.cmex_left),
        (right_roles.base, metadata.spatial.cmex_right),
        (stereo_id, metadata.spatial.dadj),
    ):
        merged.add_property(entity, raw)
    center_targets = (center_roles.base, center_roles.sato)
    eye_targets = (left_roles.base, left_roles.sato, right_roles.base, right_roles.sato)
    for name, kind, payload, targets in (
        ("Center source Exif", b"Exif", metadata.center_exif, center_targets),
        ("Eye source Exif", b"Exif", metadata.eye_exif, eye_targets),
        ("Source and render XMP", b"mime", metadata.primary_xmp, center_targets + eye_targets),
        ("Apple depth interpretation", b"mime", metadata.depth_xmp, (depth_item,)),
    ):
        if not payload or len(payload) > MAX_METADATA:
            raise ContainerError("missing or oversized output metadata")
        merged.items[next_id] = Item(
            next_id, kind, name, False, payload, "application/rdf+xml" if kind == b"mime" else None
        )
        merged.references.append(Reference(b"cdsc", next_id, targets))
        next_id += 1
    final = serialize(merged)
    final_graph = parse(final)
    result = AssemblyResult(
        center_roles,
        left_roles,
        right_roles,
        DepthRoles(depth_item, source_depth.width, source_depth.height),
        stereo_id,
        len(final),
    )
    validate_final(final_graph, result, profiles if separate_center_profile else source_icc)
    output.write_bytes(final)
    return result


def validate_final(avif: Avif, result: AssemblyResult, icc: bytes | tuple[bytes, ...]) -> None:
    validate_graph(avif)
    if avif.primary != result.center.base or len(avif.items) != 14:
        raise ContainerError("unexpected final primary/item count")
    profiles = (icc, icc, icc) if isinstance(icc, bytes) else icc
    for roles, profile in zip((result.center, result.left, result.right), profiles, strict=True):
        # Validate each imported color graph with only its own entities.
        ids = {roles.base, roles.suffix, roles.sato}
        sub = Avif(
            roles.base,
            {i: avif.items[i] for i in ids},
            avif.properties,
            {i: avif.associations[i] for i in ids},
            [r for r in avif.references if r.kind == b"dimg" and r.source in ids],
            [g for g in avif.groups if g.kind == b"altr" and g.members == (roles.sato, roles.base)],
        )
        validate_color(sub, profile, complete_color=True)
    depth_id = result.depth.item
    depth_sub = Avif(
        depth_id,
        {depth_id: avif.items[depth_id]},
        avif.properties,
        {depth_id: avif.associations[depth_id]},
    )
    validate_depth(depth_sub)
    if not avif.items[depth_id].hidden:
        raise ContainerError("depth item must be hidden")
    aux = avif.one_property(depth_id, b"auxC")
    if aux.raw != full_box(b"auxC", DEPTH_AUX_TYPE):
        raise ContainerError("depth auxiliary declaration differs from required recipe")
    aux_association = [
        a
        for a in avif.associations[depth_id]
        if avif.properties[a.property_index - 1].kind == b"auxC"
    ]
    if len(aux_association) != 1 or not aux_association[0].essential:
        raise ContainerError("depth auxC must be essential")
    if [r.targets for r in avif.references if r.kind == b"auxl"] != [
        (result.center.base, result.center.sato)
    ]:
        raise ContainerError("depth must reference both center representations")
    stereo = [g for g in avif.groups if g.kind == b"ster"]
    if stereo != [Group(b"ster", result.stereo_group, (result.left.base, result.right.base))]:
        raise ContainerError("stereo order must be left12 then right12, excluding center")
    if (
        len(avif.groups) != 4
        or len([item for item in avif.items.values() if item.kind == b"sato"]) != 3
    ):
        raise ContainerError("final graph requires three independent SATO alternatives")
    for roles in (result.left, result.right):
        for kind in (b"cmin", b"cmex"):
            avif.one_property(roles.base, kind)
    avif.one_property(result.stereo_group, b"dadj")
    for item in avif.items.values():
        if item.kind in {b"mime", b"Exif"} and not any(
            r.kind == b"cdsc" and r.source == item.id for r in avif.references
        ):
            raise ContainerError("unassociated output metadata")
