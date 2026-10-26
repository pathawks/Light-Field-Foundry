import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from light_field_foundry import library
from light_field_foundry.errors import LytroError
from light_field_foundry.lfp import LfpContainer

PICTURE = "405589c3-e8d7-4648-ade4-81abbf155a9a"
BIN = "efa39c92-67ef-4420-8ac7-e41ff2a43a98"


def test_library_lookup_checks_identity_and_never_changes_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "test.lytrolibrary"
    raw = root / BIN / PICTURE / "raw.lfp"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    database = root / "lytrolibrary.db"
    with sqlite3.connect(database) as db:
        db.execute("create table picture(uuid text, imagebin_uuid text)")
        db.execute("insert into picture values (?, ?)", (PICTURE, BIN))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    stack = LfpContainer(
        tmp_path / PICTURE / "stack.lfp",
        1,
        0,
        {"frames": [{"frame": {"imageRef": "sha1-" + "1" * 40}}]},
        b"{}",
        {},
    )
    info = SimpleNamespace(raw_reference="sha1-" + "1" * 40)
    monkeypatch.setattr(library, "read_lfp", lambda _: SimpleNamespace(admit_capture=lambda: info))
    assert library.find_raw(stack, root) == raw
    info.raw_reference = "sha1-" + "2" * 40
    with pytest.raises(LytroError, match="identity"):
        library.find_raw(stack, root)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    raw.unlink()
    raw.symlink_to(tmp_path / "outside.lfp")
    (tmp_path / "outside.lfp").write_bytes(b"outside")
    with pytest.raises(LytroError, match="escapes"):
        library.find_raw(stack, root)


def test_library_lookup_allows_unbound_recipe_only_when_explicitly_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "test.lytrolibrary"
    raw = root / BIN / PICTURE / "raw.lfp"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    database = root / "lytrolibrary.db"
    with sqlite3.connect(database) as db:
        db.execute("create table picture(uuid text, imagebin_uuid text)")
        db.execute("insert into picture values (?, ?)", (PICTURE, BIN))
    original_files = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    recipe = LfpContainer(
        tmp_path / PICTURE / "stack.view.lfp",
        1,
        0,
        {"frames": [], "views": [{"recipe4": {"viewFocus": 3}}]},
        b"{}",
        {},
    )
    info = SimpleNamespace(raw_reference="sha1-" + "1" * 40)
    monkeypatch.setattr(library, "read_lfp", lambda _: SimpleNamespace(admit_capture=lambda: info))
    with pytest.raises(LytroError):
        library.find_raw(recipe, root)
    assert library.find_raw(recipe, root, allow_unbound_recipe=True) == raw
    # The opt-in must not suppress a conflicting identity when one is present.
    bound_recipe = LfpContainer(
        recipe.path,
        1,
        0,
        {
            "frames": [{"frame": {"imageRef": "sha1-" + "2" * 40}}],
            "views": [{"recipe4": {"viewFocus": 3}}],
        },
        b"{}",
        {},
    )
    with pytest.raises(LytroError, match="identity"):
        library.find_raw(bound_recipe, root, allow_unbound_recipe=True)
    assert original_files == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
