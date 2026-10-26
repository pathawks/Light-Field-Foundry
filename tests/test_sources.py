"""Generated LFP role selection and raw-only staging; no photographic assets."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from light_field_foundry import sources
from light_field_foundry.errors import LytroError
from light_field_foundry.lfp import (
    CHUNK_MAGIC,
    FILE_MAGIC,
    MASTER_MAGIC,
    LfpContainer,
    capture_reference,
    read_lfp,
)

RAW = b"\x11generated raw sample bytes\x00\xff"


def _ref(payload: bytes) -> str:
    return "sha1-" + hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def _write(path: Path, master: dict[str, Any], *payloads: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytearray(struct.pack(">8sII", FILE_MAGIC, 1, 0))
    encoded = json.dumps(master).encode()
    for index, payload in enumerate((encoded, *payloads)):
        data.extend(
            struct.pack(
                ">8sq80s",
                MASTER_MAGIC if index == 0 else CHUNK_MAGIC,
                len(payload),
                _ref(payload).encode(),
            )
        )
        data.extend(payload)
        data.extend(b"\0" * (-len(data) % 16))
    path.write_bytes(data)
    return path


def _role(
    path: Path,
    *,
    reference: str | None = None,
    embedded: bool = False,
    recipe: bool = False,
    kind: str | None = None,
) -> Path:
    master: dict[str, Any] = {"frames": [{"frame": {"imageRef": reference or _ref(RAW)}}]}
    if recipe or kind is not None:
        view: dict[str, Any] = {}
        if recipe:
            view["recipe4"] = {"viewFocus": 3, "viewFocusSpread": 2}
        if kind is not None:
            # Selection needs declarations/identity, not any decoded image assets.
            view["accelerations"] = [
                {
                    "kind": kind,
                    "depthMap": {"imagePath": "missing-depth.dat"},
                    "perImage": [{"imagePath": "missing-center.tiff"}],
                }
            ]
        master["views"] = [view]
    return _write(path, master, *((RAW,) if embedded else ()))


@pytest.fixture
def tiny_raw_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    def admit(container: LfpContainer) -> SimpleNamespace:
        reference = capture_reference(container)
        if reference not in container.sections:
            raise LytroError("Generated raw role lacks its raw payload")
        return SimpleNamespace(raw_reference=reference)

    monkeypatch.setattr(LfpContainer, "admit_capture", admit)


def test_independent_matching_roles_are_selected_without_loading_assets(
    tmp_path: Path, tiny_raw_admission: None
) -> None:
    raw = _role(tmp_path / "raw.lfr", embedded=True)
    recipe = _role(tmp_path / "recipe.lfp", recipe=True)
    depth = _role(tmp_path / "depth.lfp", kind="edofWarp")
    edof = _role(tmp_path / "edof.lfp", kind="edofWarp")
    before = {path: path.read_bytes() for path in (raw, recipe, depth, edof)}
    selected = sources.select_sources(None, raw=raw, recipe=recipe, depth=depth, edof=edof)
    assert selected.raw.path == raw
    assert selected.recipe is not None
    assert selected.recipe.path == recipe
    assert selected.depth is not None
    assert selected.depth.path == depth
    assert selected.edof is not None
    assert selected.edof.path == edof
    assert selected.metadata is selected.raw
    for role, entry in selected.provenance.items():
        assert entry["rawImageRef"] == _ref(RAW), role
        assert len(entry["masterSHA256"]) == 64
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize("role", ["recipe", "depth", "edof"])
def test_independent_role_identity_mismatch_is_rejected(
    tmp_path: Path, tiny_raw_admission: None, role: str
) -> None:
    raw = _role(tmp_path / "raw.lfp", embedded=True)
    mismatch = _role(
        tmp_path / "different.lfp",
        reference=_ref(b"a different raw"),
        recipe=role == "recipe",
        kind="edofWarp" if role != "recipe" else None,
    )
    with pytest.raises(LytroError, match=rf"{role} LFP.*imageRef mismatch"):
        sources.select_sources(
            None,
            raw=raw,
            recipe=mismatch if role == "recipe" else None,
            depth=mismatch if role == "depth" else None,
            edof=mismatch if role == "edof" else None,
        )


def test_explicit_raw_never_opens_a_library_index(
    tmp_path: Path, tiny_raw_admission: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _role(tmp_path / "fresh-export" / "img00581.lfp", embedded=True)
    monkeypatch.setattr(sources, "find_raw", lambda *_: pytest.fail("library must not be queried"))
    selected = sources.select_sources(None, raw=raw, library=tmp_path / "missing.lytrolibrary")
    assert selected.raw.path == raw


@pytest.mark.parametrize("cache_kind", ["missing", "malformed"])
def test_recomputed_roles_do_not_open_ignored_cache_files(
    tmp_path: Path, tiny_raw_admission: None, cache_kind: str
) -> None:
    raw = _role(tmp_path / "raw.lfp", embedded=True)
    ignored_depth, ignored_edof = tmp_path / "depth.txt", tmp_path / "edof.txt"
    if cache_kind == "malformed":
        ignored_depth.write_bytes(b"not an LFP container")
        ignored_edof.write_bytes(b"not an LFP container")
    selected = sources.select_sources(
        None,
        raw=raw,
        depth=ignored_depth,
        edof=ignored_edof,
        recompute_depth=True,
        recompute_edof=True,
    )
    assert selected.depth is None
    assert selected.edof is None


def test_recipe_only_input_can_anchor_library_fallback_and_ignore_unusable_caches(
    tmp_path: Path, tiny_raw_admission: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _role(tmp_path / "raw.lfp", embedded=True)
    recipe = _role(tmp_path / "picture-uuid" / "stack.view.lfp", recipe=True)
    library = tmp_path / "readonly.lytrolibrary"
    lookups: list[tuple[Path, Path]] = []

    def find(anchor: LfpContainer, root: Path, *, allow_unbound_recipe: bool = False) -> Path:
        lookups.append((anchor.path, root))
        return raw

    monkeypatch.setattr(sources, "find_raw", find)
    selected = sources.select_sources(
        None,
        recipe=recipe,
        depth=tmp_path / "missing-depth.lfp",
        edof=tmp_path / "missing-edof.lfp",
        library=library,
        recompute_depth=True,
        recompute_edof=True,
    )
    assert lookups == [(recipe, library)]
    assert selected.raw.path == raw
    assert selected.recipe is not None
    assert selected.recipe.path == recipe
    assert selected.depth is None
    assert selected.edof is None


@pytest.mark.parametrize("kind", ["edofWarpPreAdjust", "limitedDepth", "refocused"])
def test_edof_role_requires_native_edof_acceleration(
    tmp_path: Path, tiny_raw_admission: None, kind: str
) -> None:
    raw = _role(tmp_path / "raw.lfp", embedded=True)
    limited = _role(tmp_path / "limited.lfp", kind=kind)
    with pytest.raises(LytroError, match="native edofWarp"):
        sources.select_sources(None, raw=raw, edof=limited)


def test_positional_processed_defaults_can_be_recomputed_independently(
    tmp_path: Path, tiny_raw_admission: None
) -> None:
    capture = _role(tmp_path / "combined.lfp", embedded=True, recipe=True, kind="edofWarp")
    cached = sources.select_sources(capture)
    assert cached.recipe is not None
    assert cached.depth is not None
    assert cached.edof is not None
    fresh = sources.select_sources(capture, recompute_depth=True, recompute_edof=True)
    assert fresh.raw.path == capture
    assert fresh.recipe is not None
    assert fresh.depth is None
    assert fresh.edof is None


@pytest.mark.parametrize("nested_picture", [False, True])
def test_raw_staging_keeps_exact_reachable_chunks_and_removes_all_processed_caches(
    tmp_path: Path, nested_picture: bool
) -> None:
    retained_blob = b"\x22original metadata-referenced calibration bytes\x00"
    public = (' {"signedZero": -0.0, "calibrationRef": "' + _ref(retained_blob) + '" }\n').encode()
    private = b'{ "camera": {"serialNumber": "generated-fixture"}, "value": -0 }\n'
    cache_json = b'{"processed": true}'
    orphan_cache = b"\x99orphan rendered image bytes\x00"
    cache = {"imageRef": _ref(cache_json)}
    frame = {
        "frame": {
            "imageRef": _ref(RAW),
            "metadataRef": _ref(public),
            "privateMetadataRef": _ref(private),
            "accelerations": [cache],
            "accelerationArray": [cache],
        },
        "assignedProperties": {"darkFrame": False},
        "accelerations": [cache],
        "accelerationArray": [cache],
    }
    processed = {
        "views": [{"accelerations": [cache]}],
        "viewArray": [{"accelerationArray": [cache]}],
        "accelerations": [cache],
        "accelerationArray": [cache],
        "thumbnails": [{"imageRef": _ref(orphan_cache)}],
        "derivations": [_ref(cache_json)],
    }
    picture = {"frameArray": [frame], **copy.deepcopy(processed)}
    master = (
        {"schema": "generated", "picture": picture, **copy.deepcopy(processed)}
        if nested_picture
        else {"schema": "generated", **picture}
    )
    path = _write(
        tmp_path / "combined.lfp",
        master,
        RAW,
        public,
        private,
        retained_blob,
        cache_json,
        orphan_cache,
    )
    original_bytes = path.read_bytes()
    source = read_lfp(path)
    original_master = copy.deepcopy(source.master)
    target = tmp_path / "raw-only.lfp"
    sources.stage_raw(source, target)
    staged = read_lfp(target)
    assert capture_reference(staged) == _ref(RAW)
    for payload in (RAW, public, private, retained_blob):
        assert staged.payload(_ref(payload)) == payload
    assert set(staged.sections) == {
        *(_ref(payload) for payload in (RAW, public, private, retained_blob)),
        _ref(staged.master_json),
    }
    assert _ref(cache_json) not in staged.sections
    assert _ref(orphan_cache) not in staged.sections
    processed_keys = {
        "views",
        "viewArray",
        "accelerations",
        "accelerationArray",
        "thumbnails",
        "derivations",
    }
    # TNT's raw schema requires these arrays to remain present. Clearing them
    # removes every processed-cache reference while retaining native structure.
    for parent in (staged.master, staged.master.get("picture", staged.master)):
        for key in processed_keys:
            assert parent[key] == []
        assert parent["views"] == []
        assert parent["accelerations"] == []
    staged_frame = staged._frames()[0]
    for parent in (staged_frame, staged_frame["frame"]):
        assert parent["accelerations"] == []
        assert parent["accelerationArray"] == []
    assert staged_frame["assignedProperties"] == {"darkFrame": False}
    assert source.master == original_master
    assert path.read_bytes() == original_bytes


def test_raw_staging_refuses_missing_raw_and_existing_output(tmp_path: Path) -> None:
    processed = read_lfp(_role(tmp_path / "processed.lfp", kind="edofWarp"))
    target = tmp_path / "raw-only.lfp"
    with pytest.raises(LytroError, match="raw image payload"):
        sources.stage_raw(processed, target)
    assert not target.exists()
    raw = read_lfp(_role(tmp_path / "raw.lfp", embedded=True))
    original = raw.path.read_bytes()
    with pytest.raises(FileExistsError):
        sources.stage_raw(raw, raw.path)
    assert raw.path.read_bytes() == original


def test_frame_less_recipe_is_reusable_with_explicit_raw_and_has_no_capture_identity(
    tmp_path: Path, tiny_raw_admission: None
) -> None:
    raw = _role(tmp_path / "raw.lfp", embedded=True)
    recipe = _write(
        tmp_path / "stack.view.lfp",
        {"frames": [], "views": [{"recipe4": {"viewFocus": 3, "viewFocusSpread": 2}}]},
    )
    original = recipe.read_bytes()
    selected = sources.select_sources(None, raw=raw, recipe=recipe)
    assert selected.recipe is not None
    assert selected.recipe.path == recipe
    assert selected.provenance["recipe"]["rawImageRef"] is None
    assert selected.provenance["raw"]["rawImageRef"] == _ref(RAW)
    assert selected.metadata is selected.raw
    assert recipe.read_bytes() == original


def test_frame_less_recipe_only_library_fallback_explicitly_allows_unbound_recipe(
    tmp_path: Path, tiny_raw_admission: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _role(tmp_path / "raw.lfp", embedded=True)
    recipe = _write(
        tmp_path / "picture-uuid" / "stack.view.lfp",
        {"frames": [], "views": [{"recipe4": {"viewFocus": 3}}]},
    )
    library = tmp_path / "readonly.lytrolibrary"
    calls: list[tuple[Path, Path, bool]] = []

    def find(anchor: LfpContainer, root: Path, *, allow_unbound_recipe: bool) -> Path:
        calls.append((anchor.path, root, allow_unbound_recipe))
        return raw

    monkeypatch.setattr(sources, "find_raw", find)
    selected = sources.select_sources(None, recipe=recipe, library=library)
    assert calls == [(recipe, library, True)]
    assert selected.raw.path == raw
    assert selected.metadata is selected.raw
    assert selected.provenance["recipe"]["rawImageRef"] is None


@pytest.mark.parametrize("role", ["depth", "edof"])
def test_capture_identity_is_still_required_for_unbound_image_and_depth_roles(
    tmp_path: Path, tiny_raw_admission: None, role: str
) -> None:
    raw = _role(tmp_path / "raw.lfp", embedded=True)
    unbound = _write(
        tmp_path / "unbound.lfp",
        {"frames": [], "views": [{"accelerations": [{"kind": "edofWarp"}]}]},
    )
    with pytest.raises(LytroError):
        sources.select_sources(
            None,
            raw=raw,
            depth=unbound if role == "depth" else None,
            edof=unbound if role == "edof" else None,
        )
