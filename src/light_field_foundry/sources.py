"""Select independent, identity-matched LFP roles without modifying inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LytroError
from .lfp import (
    CHUNK_MAGIC,
    FILE_MAGIC,
    MASTER_MAGIC,
    REFERENCE,
    LfpContainer,
    capture_reference,
    extract_recipe,
    has_edof,
    read_lfp,
)
from .library import find_raw


def _reference(source: LfpContainer, *, recipe: bool = False) -> str | None:
    # Desktop stack.view.lfp can contain only a recipe (frames=[]). Those
    # explicitly selected controls are reusable, just like --recipe JSON.
    if recipe and not source._frames():
        return None
    return capture_reference(source)


@dataclass(frozen=True)
class Sources:
    raw: LfpContainer
    recipe: LfpContainer | None
    depth: LfpContainer | None
    edof: LfpContainer | None
    metadata: LfpContainer

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            role: {
                "rawImageRef": _reference(source, recipe=role == "recipe"),
                "masterSHA256": hashlib.sha256(source.master_json).hexdigest(),
            }
            for role, source in (
                ("raw", self.raw),
                ("recipe", self.recipe),
                ("depth", self.depth),
                ("edof", self.edof),
            )
            if source is not None
        }


def select_sources(
    capture: Path | None,
    *,
    raw: Path | None = None,
    recipe: Path | None = None,
    depth: Path | None = None,
    edof: Path | None = None,
    library: Path | None = None,
    recompute_depth: bool = False,
    recompute_edof: bool = False,
) -> Sources:
    loaded: dict[Path, LfpContainer] = {}

    def load(path: Path) -> LfpContainer:
        path = path.expanduser().resolve(strict=True)
        if path.suffix.lower() not in {".lfp", ".lfr"}:
            raise LytroError("LFP roles require an .lfp or .lfr file")
        if path not in loaded:
            loaded[path] = read_lfp(path)
        return loaded[path]

    positional = load(capture) if capture is not None else None
    raw_source = load(raw) if raw is not None else None
    if (
        raw_source is None
        and positional is not None
        and _reference(positional, recipe=True) in positional.sections
    ):
        raw_source = positional
    recipe_source = load(recipe) if recipe is not None else None
    if recipe_source is None and positional is not None:
        picture = positional.master.get("picture", positional.master)
        if picture.get("views", picture.get("viewArray")):
            # An explicitly supplied recipe must contain a recipe. A positional
            # depth/EDOF-only container is also useful without one.
            views = picture.get("views", picture.get("viewArray", []))
            if any(isinstance(v, dict) and any(k.startswith("recipe") for k in v) for v in views):
                extract_recipe(positional)
                recipe_source = positional
    depth_source = None if recompute_depth else (load(depth) if depth is not None else None)
    edof_source = None if recompute_edof else (load(edof) if edof is not None else None)
    if positional is not None and has_edof(positional):
        if depth_source is None and depth is None and not recompute_depth:
            picture = positional.master.get("picture", positional.master)
            if any(
                "depthMap" in a
                for v in picture.get("views", picture.get("viewArray", []))
                for a in v.get("accelerations", v.get("accelerationArray", []))
            ):
                depth_source = positional
        if edof_source is None and edof is None and not recompute_edof:
            edof_source = positional
    if raw_source is None:
        candidates = [
            source
            for source in (recipe_source, depth_source, edof_source, positional)
            if source is not None
        ]
        anchor = next(
            (source for source in candidates if _reference(source, recipe=True) is not None), None
        )
        anchor = anchor or recipe_source or positional
        if anchor is None:
            # Ignored caches may still provide identity for a library lookup.
            anchor_path = depth or edof
            anchor = load(anchor_path) if anchor_path is not None else None
        if anchor is None or library is None:
            raise LytroError(
                "Supply --lfp-raw (or a raw positional input), or --library for raw lookup"
            )
        if _reference(anchor, recipe=True) is None:
            raw_source = load(find_raw(anchor, library, allow_unbound_recipe=True))
        else:
            raw_source = load(find_raw(anchor, library))
    raw_source.admit_capture()
    reference = capture_reference(raw_source)
    for role, source in (("recipe", recipe_source), ("depth", depth_source), ("edof", edof_source)):
        if source is not None and _reference(source, recipe=role == "recipe") not in (
            None,
            reference,
        ):
            raise LytroError(f"{role} LFP belongs to a different raw capture (imageRef mismatch)")
    if positional is not None and _reference(
        positional, recipe=positional is recipe_source
    ) not in (None, reference):
        raise LytroError("Positional LFP belongs to a different raw capture (imageRef mismatch)")
    if recipe_source is not None:
        extract_recipe(recipe_source)
    if edof_source is not None and not has_edof(edof_source):
        raise LytroError("--lfp-edof must contain a native edofWarp acceleration")
    return Sources(
        raw_source,
        recipe_source,
        depth_source,
        edof_source,
        raw_source,
    )


def stage_raw(source: LfpContainer, target: Path) -> None:
    """Keep exact raw/metadata chunks, remove every processed cache reference.

    TNT receives a fresh raw-only container even when the selected raw LFP also
    carries accelerations. Recompute cannot silently reuse an embedded cache.
    """
    reference = capture_reference(source)
    if reference not in source.sections:
        raise LytroError("Raw role does not contain its raw image payload")
    master = copy.deepcopy(source.master)
    picture = master.get("picture", master)
    for parent in (master, picture):
        for key in (
            "views",
            "viewArray",
            "accelerations",
            "accelerationArray",
            "thumbnails",
            "derivations",
        ):
            if key in parent:
                parent[key] = []
    for frame in picture.get("frames", picture.get("frameArray", [])):
        for parent in (frame, frame.get("frame", {})):
            for key in ("accelerations", "accelerationArray"):
                if key in parent:
                    parent[key] = []
    payloads: dict[str, bytes] = {}

    def include(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                include(child)
        elif isinstance(value, list):
            for child in value:
                include(child)
        elif (
            isinstance(value, str)
            and REFERENCE.fullmatch(value)
            and value in source.sections
            and value not in payloads
        ):
            payloads[value] = source.payload(value)
            metadata = source.sections[value].metadata
            if metadata is not None:
                include(metadata)

    include(master)
    encoded = json.dumps(master, allow_nan=False, separators=(",", ":")).encode()
    with target.open("xb") as output:
        output.write(struct.pack(">8sII", FILE_MAGIC, 1, 0))
        for index, payload in enumerate((encoded, *payloads.values())):
            digest = ("sha1-" + hashlib.sha1(payload).hexdigest()).encode()
            output.write(
                struct.pack(
                    ">8sq80s", MASTER_MAGIC if index == 0 else CHUNK_MAGIC, len(payload), digest
                )
            )
            output.write(payload)
            output.write(b"\0" * (-output.tell() % 16))
    staged = read_lfp(target)
    if capture_reference(staged) != reference:
        raise LytroError("Staged raw identity differs from its source")
