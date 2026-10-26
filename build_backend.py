"""Package verified Kaitai sources; regenerate or compare them explicitly for CI."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCHEMA = Path("formats/lfp_framing.ksy")
PARSER = Path("src/light_field_foundry/_generated/lfp_framing.py")
MANIFEST = Path("formats/lfp_framing.generated.json")


def _compiler_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["tool"]["kaitai"]["compiler-version"]
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError(
            "tool.kaitai.compiler-version must pin an exact major.minor.patch version"
        )
    return version


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        "compiler": f"kaitai-struct-compiler {_compiler_version(root)}",
        "schema_sha256": hashlib.sha256((root / SCHEMA).read_bytes()).hexdigest(),
        "parser_sha256": hashlib.sha256((root / PARSER).read_bytes()).hexdigest(),
    }


def ensure_generated(root: Path = ROOT) -> None:
    """Verify the checked-in schema/parser pair without requiring Java or a compiler."""
    try:
        current = json.loads((root / MANIFEST).read_text()) == _fingerprint(root)
    except (OSError, ValueError):
        current = False
    if not current:
        raise RuntimeError(
            "Kaitai parser is missing or stale; install the pinned compiler and run "
            "python build_backend.py --generate, then commit the schema, parser, and manifest"
        )


def regenerate(root: Path = ROOT, *, check: bool = False) -> None:
    """Compile the schema, optionally comparing real compiler output without writing files."""
    expected = _compiler_version(root)
    compiler = (
        shutil.which(os.environ["KAITAI_STRUCT_COMPILER"])
        if "KAITAI_STRUCT_COMPILER" in os.environ
        else shutil.which("kaitai-struct-compiler") or shutil.which("ksc")
    )
    if compiler is None:
        raise RuntimeError(
            f"Install kaitai-struct-compiler {expected}, or set KAITAI_STRUCT_COMPILER "
            "to its executable. The compiler is only needed for regeneration and CI checks."
        )
    # The compiler is an explicitly selected executable; arguments never invoke a shell.
    version = subprocess.run(  # noqa: S603
        [compiler, "--version"], check=True, capture_output=True, text=True
    )
    match = re.fullmatch(r"kaitai-struct-compiler (\d+)\.(\d+)(?:\.(\d+))?", version.stdout.strip())
    # Upstream labels the 0.11.0 release as 0.11; accept only that exact release.
    actual = ".".join(part or "0" for part in match.groups()) if match else None
    if actual != expected:
        raise RuntimeError(f"Parser generation requires kaitai-struct-compiler {expected}")
    with tempfile.TemporaryDirectory(prefix="lff-kaitai-") as directory:
        subprocess.run(  # noqa: S603 -- pinned compiler, explicit arguments, no shell
            [compiler, "--target", "python", "--outdir", directory, str(root / SCHEMA)],
            check=True,
        )
        raw = (Path(directory) / PARSER.name).read_bytes()
    # Kaitai indents blank docstring lines. Normalize trailing whitespace for Git.
    generated = b"\n".join(line.rstrip() for line in raw.splitlines()).rstrip() + b"\n"
    destination = root / PARSER
    if check:
        stored = destination.read_bytes() if destination.exists() else b""
        if generated != stored:
            difference = "".join(
                difflib.unified_diff(
                    stored.decode().splitlines(keepends=True),
                    generated.decode().splitlines(keepends=True),
                    fromfile=str(PARSER),
                    tofile="fresh Kaitai output",
                )
            )
            raise RuntimeError(f"Generated Kaitai parser differs; run --generate:\n{difference}")
        ensure_generated(root)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(generated)
        (root / MANIFEST).write_text(json.dumps(_fingerprint(root), indent=2) + "\n")


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    ensure_generated()
    backend = importlib.import_module("uv_build")
    result = backend.build_wheel(wheel_directory, config_settings, metadata_directory)
    if not isinstance(result, str):
        raise TypeError("uv_build.build_wheel must return a filename")
    return result


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    ensure_generated()
    backend = importlib.import_module("uv_build")
    result = backend.build_sdist(sdist_directory, config_settings)
    if not isinstance(result, str):
        raise TypeError("uv_build.build_sdist must return a filename")
    return result


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    ensure_generated()
    backend = importlib.import_module("uv_build")
    result = backend.build_editable(wheel_directory, config_settings, metadata_directory)
    if not isinstance(result, str):
        raise TypeError("uv_build.build_editable must return a filename")
    return result


def __getattr__(name: str) -> Any:  # noqa: ANN401 -- PEP 517 hooks have distinct signatures
    """Forward optional PEP 517/660 hooks to the underlying backend."""
    if name not in {
        "get_requires_for_build_wheel",
        "get_requires_for_build_sdist",
        "get_requires_for_build_editable",
        "prepare_metadata_for_build_wheel",
        "prepare_metadata_for_build_editable",
    }:
        raise AttributeError(name)
    return getattr(importlib.import_module("uv_build"), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Verify manifest without a compiler")
    mode.add_argument("--check", action="store_true", help="Compare against fresh compiler output")
    mode.add_argument(
        "--generate", action="store_true", help="Regenerate using the pinned compiler"
    )
    options = parser.parse_args()
    if options.generate or options.check:
        regenerate(check=options.check)
    else:
        ensure_generated()
