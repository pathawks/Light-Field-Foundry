"""Bounded, read-only LFP/LFR framing, metadata, and TNT Float32 depth.

LFP section references identify SHA-1 of their exact bytes. Original JSON bytes
are retained independently of parsed objects, including signed numeric zero.
Only the measured F01 upright, full-frame center geometry is admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import LytroError

FILE_MAGIC = b"\x89LFP\r\n\x1a\n"
MASTER_MAGIC = b"\x89LFM\r\n\x1a\n"
CHUNK_MAGIC = b"\x89LFC\r\n\x1a\n"
MAX_FILE_BYTES = 1024 * 1024 * 1024
MAX_SECTION_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_SECTIONS = 4096
MAX_PIXELS = 16 * 1024 * 1024
REFERENCE = re.compile(r"sha1-[0-9a-f]{40}\Z")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LytroError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise LytroError(f"Nonfinite JSON number: {value}")


def read_json_bytes(data: bytes, description: str = "JSON") -> dict[str, Any]:
    """Strict bounded JSON object reader, also used for user recipes."""
    if len(data) > MAX_JSON_BYTES:
        raise LytroError(f"{description} exceeds the 4 MiB JSON limit")
    try:
        obj = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise LytroError(f"Malformed {description}: {exc}") from exc
    if not isinstance(obj, dict):
        raise LytroError(f"{description} must contain a JSON object")

    def check(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise LytroError(f"Nonfinite numeric value in {description}")

    try:
        check(obj)
    except RecursionError as exc:
        raise LytroError(f"Excessive JSON nesting in {description}") from exc
    return obj


def _json_prefix(data: bytes) -> tuple[dict[str, Any] | None, bytes]:
    if not data.lstrip().startswith(b"{"):
        return None, b""
    try:
        # Binary suffixes can follow a JSON prefix in some LFP assets. Decode
        # reversibly first, then validate just the actual JSON as strict UTF-8.
        text = data[:MAX_JSON_BYTES].decode("utf-8", errors="surrogateescape")
        start = len(text) - len(text.lstrip())
        _, end = json.JSONDecoder().raw_decode(text, start)
        raw = text[:end].encode("utf-8", errors="surrogateescape")
        obj = read_json_bytes(raw, "LFP section JSON")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise LytroError(f"Malformed or oversized LFP JSON section: {exc}") from exc
    return obj, raw


@dataclass(frozen=True)
class Section:
    reference: str
    offset: int
    length: int
    metadata: dict[str, Any] | None
    json_bytes: bytes
    is_master: bool = False


@dataclass(frozen=True)
class CaptureInfo:
    public: dict[str, Any]
    private: dict[str, Any]
    make: str
    model: str
    width: int
    height: int
    orientation: int
    camera_serial: str | None
    raw_reference: str
    master_json: bytes
    public_json: tuple[bytes, ...]
    private_json: tuple[bytes, ...]

    def raw_jsons(self, *, include_private: bool = False) -> dict[str, Any]:
        """Exact UTF-8 JSON text; private serials remain opt-in for embedding."""
        result: dict[str, Any] = {
            "master": self.master_json.decode("utf-8"),
            "public": [value.decode("utf-8") for value in self.public_json],
        }
        if include_private:
            result["private"] = [value.decode("utf-8") for value in self.private_json]
        return result


@dataclass(frozen=True)
class LfpContainer:
    path: Path
    version: int
    flags: int
    master: dict[str, Any]
    master_json: bytes
    sections: dict[str, Section]

    def payload(self, reference: str) -> bytes:
        try:
            section = self.sections[reference]
        except KeyError as exc:
            raise LytroError(f"Unresolved LFP asset reference: {reference}") from exc
        with self.path.open("rb") as stream:
            stream.seek(section.offset)
            data = stream.read(section.length)
        if len(data) != section.length:
            raise LytroError("LFP changed while being read")
        _verify_digest(reference, data)
        return data

    def _frames(self) -> list[dict[str, Any]]:
        picture = self.master.get("picture", self.master)
        if not isinstance(picture, dict):
            raise LytroError("LFP picture must be an object")
        frames = picture.get("frames", picture.get("frameArray", []))
        if not isinstance(frames, list) or any(not isinstance(f, dict) for f in frames):
            raise LytroError("LFP frames must be objects")
        return frames

    def _metadata(self, key: str) -> tuple[bytes, ...]:
        values: list[bytes] = []
        for item in self._frames():
            frame = item.get("frame", {})
            if not isinstance(frame, dict):
                raise LytroError("LFP frame must be an object")
            reference = frame.get(key)
            if reference is not None:
                if not isinstance(reference, str):
                    raise LytroError(f"LFP {key} is not a reference")
                value = self.payload(reference)
                read_json_bytes(value, key)
                values.append(value)
        return tuple(values)

    @property
    def public_json(self) -> tuple[bytes, ...]:
        return self._metadata("metadataRef")

    @property
    def private_json(self) -> tuple[bytes, ...]:
        return self._metadata("privateMetadataRef")

    def capture_metadata(self) -> dict[str, Any]:
        metadata = self.public_json
        if len(metadata) != 1:
            raise LytroError("Exactly one capture metadata frame is required")
        return read_json_bytes(metadata[0], "public capture metadata")

    def admit_capture(self) -> CaptureInfo:
        if self.path.suffix.lower() not in {".lfr", ".lfp"}:
            raise LytroError("Capture must have an .lfr or .lfp extension")
        frames = self._frames()
        if len(frames) != 1:
            raise LytroError("Only one raw capture frame is supported")
        frame = frames[0].get("frame", {})
        reference = frame.get("imageRef") if isinstance(frame, dict) else None
        if not isinstance(reference, str) or reference not in self.sections:
            raise LytroError(
                "Capture does not contain raw image bytes; processed-only LFP is unsupported"
            )
        public = self.capture_metadata()
        private_json = self.private_json
        private = read_json_bytes(private_json[0]) if private_json else {}
        image, camera = public.get("image", {}), public.get("camera", {})
        if not isinstance(image, dict) or not isinstance(camera, dict):
            raise LytroError("Capture image/camera metadata is malformed")
        width, height, orientation = (
            image.get("width"),
            image.get("height"),
            image.get("orientation"),
        )
        if camera.get("model") != "F01" or (width, height, orientation) != (3280, 3280, 1):
            raise LytroError("Admitted capture scope is F01 3280×3280, upright orientation 1")
        details = image.get("rawDetails", {})
        if not isinstance(details, dict) or not isinstance(details.get("pixelPacking"), dict):
            raise LytroError("Malformed raw sample packing metadata")
        packing = details["pixelPacking"]
        if (
            image.get("representation") != "rawPacked"
            or packing.get("bitsPerPixel") != 12
            or packing.get("endianness") != "big"
        ):
            raise LytroError("Admitted F01 capture must contain big-endian packed12 raw samples")
        if self.sections[reference].length != width * height * 3 // 2:
            raise LytroError("F01 packed12 raw byte extent does not match sensor dimensions")
        private_camera = private.get("camera", {})
        if not isinstance(private_camera, dict):
            raise LytroError("Malformed private camera metadata")
        serial = private_camera.get("serialNumber")
        return CaptureInfo(
            public=public,
            private=private,
            make=str(camera.get("make", "Lytro, Inc.")),
            model="F01",
            width=width,
            height=height,
            orientation=orientation,
            camera_serial=serial if isinstance(serial, str) else None,
            raw_reference=reference,
            master_json=self.master_json,
            public_json=self.public_json,
            private_json=private_json,
        )


def _verify_digest(reference: str, data: bytes) -> None:
    if not REFERENCE.fullmatch(reference):
        raise LytroError(f"Unsupported LFP chunk reference: {reference}")
    if hashlib.sha1(data).hexdigest() != reference[5:]:
        raise LytroError(f"LFP SHA-1 digest mismatch: {reference}")


def read_lfp(path: Path) -> LfpContainer:
    path = Path(path)
    size = path.stat().st_size
    if not 16 <= size <= MAX_FILE_BYTES:
        raise LytroError("LFP size is outside the supported 16-byte..1-GiB range")
    sections: dict[str, Section] = {}
    masters: list[Section] = []
    with path.open("rb") as stream:
        header = stream.read(16)
        magic, version, flags = struct.unpack(">8sII", header)
        if magic != FILE_MAGIC or version != 1 or flags != 0:
            raise LytroError("Unsupported LFP file header (expected version 1, flags 0)")
        position = 16
        while position < size:
            if len(sections) >= MAX_SECTIONS or size - position < 96:
                raise LytroError("Too many LFP chunks or truncated chunk header")
            stream.seek(position)
            marker, length, reference_bytes = struct.unpack(">8sq80s", stream.read(96))
            if marker not in {MASTER_MAGIC, CHUNK_MAGIC} or not 0 <= length <= MAX_SECTION_BYTES:
                raise LytroError("Invalid LFP chunk marker or extent")
            offset, end = position + 96, position + 96 + length
            aligned = (end + 15) // 16 * 16
            if end > size or aligned > size:
                raise LytroError("Truncated LFP chunk data or alignment padding")
            try:
                reference = reference_bytes.rstrip(b"\0").decode("ascii")
            except UnicodeError as exc:
                raise LytroError("Non-ASCII LFP chunk reference") from exc
            data = stream.read(length)
            _verify_digest(reference, data)
            if reference in sections:
                raise LytroError("Duplicate LFP chunk reference")
            try:
                metadata, json_bytes = _json_prefix(data)
            except LytroError:
                # An opaque binary sample can begin with the byte 0x7b. Only
                # master/metadata references require JSON interpretation.
                if marker == MASTER_MAGIC:
                    raise
                metadata, json_bytes = None, b""
            if metadata is not None and not data[len(json_bytes) :].strip():
                json_bytes = data
            if marker == MASTER_MAGIC and (metadata is None or data[len(json_bytes) :].strip()):
                raise LytroError("LFP master must contain only JSON")
            stream.seek(end)
            if any(stream.read(aligned - end)):
                raise LytroError("Nonzero LFP alignment padding")
            section = Section(
                reference, offset, length, metadata, json_bytes, marker == MASTER_MAGIC
            )
            sections[reference] = section
            if section.is_master:
                masters.append(section)
            position = aligned
    if len(masters) != 1:
        raise LytroError("LFP must contain exactly one master JSON chunk")
    master = masters[0]
    assert master.metadata is not None
    return LfpContainer(path, version, flags, master.metadata, master.json_bytes, sections)


@dataclass(frozen=True)
class FloatDepth:
    samples: NDArray[np.float32]
    width: int
    height: int
    min_lambda: float
    max_lambda: float
    map_to_image: dict[str, Any] | None
    view: dict[str, Any]
    descriptor: dict[str, Any]
    witness_image: bytes
    witness_perspective: tuple[float, float]
    image_size: tuple[int, int]

    @property
    def mapping_provenance(self) -> dict[str, Any]:
        return {
            "source": "TNT edofWarp packedFloat Float32 little-endian",
            "lambda_at_code_0": self.max_lambda,
            "lambda_at_code_4095": self.min_lambda,
            "quantization": "nearest integer, ties to even; Float64 evaluation",
            "native_raster": [self.width, self.height],
            "alignment": "upright canonical center, full-frame normalized coordinates",
            "mapToImage": self.map_to_image,
            "apple_disparity": "relative inverse lambda; code increases toward near; no metric calibration",
        }


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise LytroError(f"{name} must be a finite number")
    return float(value)


def _dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16384:
        raise LytroError(f"{name} must be a positive bounded dimension")
    return value


def _read_owned_asset(value: Any, asset_root: Path, maximum: int) -> bytes:
    """Read only regular, nonsymlink files below the explicit current-run root.

    File-descriptor-relative opens prevent a path component from being replaced
    by an escaping symlink between validation and reading. No source-directory
    or home-directory lookup is attempted.
    """
    if not isinstance(value, str) or not value or "\0" in value:
        raise LytroError("Unpacked TNT imagePath must be a nonempty path")
    candidate = Path(value)
    if ".." in candidate.parts:
        raise LytroError("Unpacked TNT asset path traversal is unsupported")
    lexical_root = asset_root.absolute()
    try:
        if lexical_root.is_symlink() or not lexical_root.is_dir():
            raise LytroError("TNT asset root must be a nonsymlink working directory")
        root = lexical_root.resolve(strict=True)
        relative = candidate
        if candidate.is_absolute():
            for base in (lexical_root, root):
                try:
                    relative = candidate.relative_to(base)
                    break
                except ValueError:
                    continue
            else:
                raise LytroError("Unpacked TNT asset path escapes the current working directory")
        if not relative.parts:
            raise LytroError("Unpacked TNT asset must name a regular file")
        with ExitStack() as stack:
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, directory)
            for part in relative.parts[:-1]:
                directory = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                stack.callback(os.close, directory)
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            stack.callback(os.close, descriptor)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
                raise LytroError("Unpacked TNT asset is not a bounded nonempty regular file")
            data = bytearray()
            while len(data) <= maximum:
                chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            after = os.fstat(descriptor)
            if len(data) != before.st_size or after.st_size != before.st_size:
                raise LytroError(
                    "Unpacked TNT asset changed while being read or exceeded its bound"
                )
            return bytes(data)
    except (OSError, ValueError) as exc:
        raise LytroError(f"Cannot read current-run TNT asset: {exc}") from exc


def _asset_payload(
    lfp: LfpContainer, descriptor: dict[str, Any], asset_root: Path | None, maximum: int
) -> bytes:
    if "imageRef" in descriptor:
        if "imagePath" in descriptor or not isinstance(descriptor["imageRef"], str):
            raise LytroError("TNT asset must have exactly one imageRef or imagePath")
        reference = descriptor["imageRef"]
        section = lfp.sections.get(reference)
        if section is not None and section.length > maximum:
            raise LytroError("Embedded TNT asset exceeds its descriptor bound")
        return lfp.payload(reference)
    if asset_root is None:
        raise LytroError("TNT assets must be embedded unless a current-run asset_root is supplied")
    return _read_owned_asset(descriptor.get("imagePath"), asset_root, maximum)


def extract_depth(path: Path, *, asset_root: Path | None = None) -> FloatDepth:
    lfp = read_lfp(path)
    picture = lfp.master.get("picture", lfp.master)
    if not isinstance(picture, dict):
        raise LytroError("Processed LFP picture must be an object")
    views = picture.get("views", picture.get("viewArray", []))
    if not isinstance(views, list):
        raise LytroError("Processed LFP views must be an array")
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for view in views:
        if not isinstance(view, dict):
            raise LytroError("Malformed processed LFP view")
        accelerations = view.get("accelerations", view.get("accelerationArray", []))
        if not isinstance(accelerations, list):
            raise LytroError("Processed LFP accelerations must be an array")
        for acceleration in accelerations:
            if isinstance(acceleration, dict) and "depthMap" in acceleration:
                matches.append((view, acceleration))
    if len(matches) != 1:
        raise LytroError("Processed LFP must contain exactly one depth-bearing view acceleration")
    view, acceleration = matches[0]
    if acceleration.get("kind", acceleration.get("type")) != "edofWarp":
        raise LytroError("Only TNT edofWarp depth is admitted")
    descriptor = acceleration["depthMap"]
    if not isinstance(descriptor, dict) or descriptor.get("representation") != "packedFloat":
        raise LytroError("TNT depth must explicitly declare packedFloat")
    width = _dimension(descriptor.get("width"), "depth width")
    height = _dimension(descriptor.get("height"), "depth height")
    if width * height > MAX_PIXELS:
        raise LytroError("Depth dimensions exceed the 16-Mpixel bound")
    payload = _asset_payload(lfp, descriptor, asset_root, width * height * 4)
    if len(payload) != width * height * 4:
        raise LytroError("packedFloat depth extent is not width × height × 4")
    samples = np.frombuffer(payload, dtype="<f4").reshape(height, width).copy()
    a = _number(descriptor.get("minLambda"), "depth minLambda")
    b = _number(descriptor.get("maxLambda"), "depth maxLambda")
    if (
        a == b
        or not np.isfinite(samples).all()
        or np.any(samples < min(a, b))
        or np.any(samples > max(a, b))
    ):
        raise LytroError("Depth has degenerate range, nonfinite/sentinel or out-of-range samples")
    center: list[dict[str, Any]] = []
    images = acceleration.get("perImage", acceleration.get("imageArray", []))
    if not isinstance(images, list):
        raise LytroError("Processed LFP perImage must be an array")
    for image in images:
        if not isinstance(image, dict):
            raise LytroError("Processed LFP image must be an object")
        perspective = image.get("perspective", {})
        if not isinstance(perspective, dict):
            raise LytroError("Processed LFP perspective must be an object")
        if perspective.get("u") == 0 and perspective.get("v") == 0:
            center.append(image)
    if len(center) != 1:
        raise LytroError("Processed LFP requires one canonical-center alignment witness")
    if acceleration.get("representation") != "tiff":
        raise LytroError("Processed LFP alignment witness must be TIFF")
    image_size = (
        _dimension(acceleration.get("width"), "view width"),
        _dimension(acceleration.get("height"), "view height"),
    )
    mapping = descriptor.get("mapToImage")
    if mapping is not None and not isinstance(mapping, dict):
        raise LytroError("Malformed depth mapToImage")
    return FloatDepth(
        samples,
        width,
        height,
        a,
        b,
        mapping,
        view,
        descriptor,
        _asset_payload(lfp, center[0], asset_root, MAX_SECTION_BYTES),
        (0.0, 0.0),
        image_size,
    )


def quantize_depth(depth: FloatDepth) -> NDArray[np.uint16]:
    samples = depth.samples.astype(np.float64)
    # TNT F01 lambda increases toward infinity (lens.infinityLambda), and
    # the recovered native depth projection resolves occlusions by MIN lambda.
    # Reverse the descriptor endpoints so Apple disparity increases near.
    a, b = depth.max_lambda, depth.min_lambda
    if not math.isfinite(a) or not math.isfinite(b) or a == b:
        raise LytroError("Depth range must be finite and nondegenerate")
    if not np.isfinite(samples).all() or np.any(samples < min(a, b)) or np.any(samples > max(a, b)):
        raise LytroError("Depth contains unsupported invalid or out-of-range samples")
    return np.rint(4095.0 * (samples - a) / (b - a)).astype(np.uint16)


def validate_alignment(depth: FloatDepth, resolved: Any, color_size: tuple[int, int]) -> None:
    """Admit only the probed F01 canonical-center, full-frame depth geometry.

    TNT's F01 ComputeOutputDepthMap center route is a full-frame ScaleSignal
    operation. At zero viewpoint no depth reprojection/pivot warp is applied.
    The native depth raster is preserved; an AVIF auxiliary may be lower-res.
    """
    if resolved.center.get("viewPerspectiveU") != 0 or resolved.center.get("viewPerspectiveV") != 0:
        raise LytroError("Depth alignment is validated only for center perspective U=V=0")
    if color_size != depth.image_size:
        raise LytroError("Rendered view does not match its TNT alignment witness dimensions")
    if (depth.width, depth.height) != (330, 330) or depth.image_size != (1134, 1134):
        raise LytroError("Admitted F01 depth alignment is 330² depth with a 1134² full-frame view")
    if depth.map_to_image is not None:
        expected = {
            "origin": {"x": 0, "y": 0},
            "xStep": {"dx": 1 / depth.width, "dy": 0},
            "yStep": {"dx": 0, "dy": 1 / depth.height},
        }
        if depth.map_to_image != expected:
            raise LytroError("Explicit mapToImage differs from the admitted full-frame mapping")
    recipe = next(
        (value for key, value in depth.view.items() if re.fullmatch(r"recipe\d*", key)), None
    )
    if not isinstance(recipe, dict):
        raise LytroError("Depth LFP is missing its effective view recipe")
    keys = (
        "viewFocus",
        "viewAperture",
        "viewFocusSpread",
        "viewPivot",
        "viewStereoPivot",
        "viewPerspectiveU",
        "viewPerspectiveV",
        "viewCrop",
        "viewOrientation",
        "viewPanX",
        "viewPanY",
        "viewZoom",
        "viewTiltX",
        "viewTiltY",
    )
    for key in keys:
        actual, wanted = recipe.get(key), resolved.center.get(key)
        if isinstance(actual, (int, float)) and isinstance(wanted, (int, float)):
            if float(np.float32(actual)) == float(np.float32(wanted)):
                continue
        elif actual == wanted:
            continue
        raise LytroError(f"Depth LFP effective {key} differs from the resolved center recipe")
