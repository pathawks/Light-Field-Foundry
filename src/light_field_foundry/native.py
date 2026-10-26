"""Read-only Apple public-API validation; never interacts with Photos."""

from __future__ import annotations

import json
import sys
from importlib.resources import files
from pathlib import Path

from .errors import LytroError
from .images import U16
from .tools import Tools, executable


def verify_native(
    tools: Tools,
    work: Path,
    avif: Path,
    codes: U16,
    a: float = 0,
    b: float = 1,
) -> dict[str, object]:
    if sys.platform != "darwin":
        raise LytroError("This release requires the validated macOS native depth path.")
    compiler = executable("swiftc")
    source = work / "native_depth.swift"
    source.write_bytes(
        files("light_field_foundry").joinpath("resources/native_depth.swift").read_bytes()
    )
    binary = work / "native-depth"
    tools.run(
        [
            compiler,
            "-module-cache-path",
            work / "swift-cache",
            source,
            "-o",
            binary,
        ],
        work,
    )
    prepared = work / "native-depth-codes.u16le"
    prepared.write_bytes(codes.astype("<u2").tobytes())
    height, width = codes.shape
    result = tools.run([binary, avif, prepared, str(width), str(height), str(a), str(b)], work)
    try:
        decoded = json.loads(result)
    except ValueError as exc:
        raise LytroError("Native depth verifier returned malformed diagnostics.") from exc
    if not isinstance(decoded, dict) or decoded.get("verified") is not True:
        raise LytroError("Native depth verification did not succeed.")
    return decoded
