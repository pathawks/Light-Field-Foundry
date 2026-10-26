"""Build verification and CI comparison of checked-in Kaitai sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARSER = Path("src/light_field_foundry/_generated/lfp_framing.py")
SCHEMA = Path("formats/lfp_framing.ksy")
MANIFEST = Path("formats/lfp_framing.generated.json")


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    for relative in (Path("build_backend.py"), Path("pyproject.toml"), SCHEMA, PARSER, MANIFEST):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def run_check(
    root: Path, *args: str, compiler: Path | None = None, version: str = "0.11"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "build_backend.py"), *args],
        env={
            **os.environ,
            "KAITAI_STRUCT_COMPILER": str(compiler or root / "absent-compiler"),
            "LFF_TEST_COMPILER_VERSION": version,
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("args", [(), ("--verify",)])
def test_current_sources_need_no_compiler(source_tree: Path, args: tuple[str, ...]) -> None:
    result = run_check(source_tree, *args)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("relative", [SCHEMA, PARSER, MANIFEST])
def test_changed_schema_parser_or_manifest_is_rejected(source_tree: Path, relative: Path) -> None:
    with (source_tree / relative).open("a") as stream:
        stream.write("\n# changed\n")
    result = run_check(source_tree, "--verify")
    assert result.returncode != 0
    assert "missing or stale" in result.stderr
    assert "--generate" in result.stderr


@pytest.mark.parametrize("relative", [PARSER, MANIFEST])
def test_build_rejects_missing_generated_sources(source_tree: Path, relative: Path) -> None:
    (source_tree / relative).unlink()
    result = run_check(source_tree)
    assert result.returncode != 0
    assert "missing or stale" in result.stderr
    assert "--generate" in result.stderr


def test_changed_schema_is_rejected_during_build(source_tree: Path) -> None:
    with (source_tree / SCHEMA).open("a") as stream:
        stream.write("\n# schema update\n")
    result = run_check(source_tree)
    assert result.returncode != 0
    assert "missing or stale" in result.stderr


def test_ci_check_requires_a_real_compiler_even_when_manifest_matches(source_tree: Path) -> None:
    result = run_check(source_tree, "--check")
    assert result.returncode != 0
    assert "Install kaitai-struct-compiler 0.11.0" in result.stderr


@pytest.fixture
def compiler(source_tree: Path) -> Path:
    """Stand in for compiler output to isolate the comparison/version contracts."""
    snapshot = source_tree / "fresh-parser.py"
    shutil.copyfile(ROOT / PARSER, snapshot)
    executable = source_tree / "compiler"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, shutil, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('kaitai-struct-compiler ' + os.environ['LFF_TEST_COMPILER_VERSION'])\n"
        "else:\n"
        "    output = Path(sys.argv[sys.argv.index('--outdir') + 1])\n"
        f"    shutil.copyfile({str(snapshot)!r}, output / 'lfp_framing.py')\n"
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize("version", ["0.11", "0.11.0"])
def test_ci_accepts_only_the_pinned_release(
    source_tree: Path, compiler: Path, version: str
) -> None:
    result = run_check(source_tree, "--check", compiler=compiler, version=version)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("version", ["0.11.1", "0.12", "0.11-SNAPSHOT"])
def test_ci_rejects_other_compiler_versions(
    source_tree: Path, compiler: Path, version: str
) -> None:
    result = run_check(source_tree, "--check", compiler=compiler, version=version)
    assert result.returncode != 0
    assert "requires kaitai-struct-compiler 0.11.0" in result.stderr


def test_ci_detects_edited_parser_even_if_manifest_was_updated(
    source_tree: Path, compiler: Path
) -> None:
    parser = source_tree / PARSER
    parser.write_bytes(parser.read_bytes() + b"# edited by hand\n")
    edited = parser.read_bytes()
    manifest = source_tree / MANIFEST
    record = json.loads(manifest.read_text())
    record["parser_sha256"] = hashlib.sha256(edited).hexdigest()
    manifest.write_text(json.dumps(record))
    assert run_check(source_tree, "--verify").returncode == 0
    result = run_check(source_tree, "--check", compiler=compiler)
    assert result.returncode != 0
    assert "fresh Kaitai output" in result.stderr
    assert parser.read_bytes() == edited
