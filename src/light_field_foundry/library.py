"""Read-only UUID lookup; verify imageRef whenever the source carries one."""

import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from .errors import LytroError
from .lfp import LfpContainer, read_lfp


def find_raw(stack: LfpContainer, library: Path, *, allow_unbound_recipe: bool = False) -> Path:
    root = library.expanduser().resolve(strict=True)
    try:
        picture = str(uuid.UUID(stack.path.parent.name))
    except ValueError as exc:
        raise LytroError(
            "Stack parent directory must be its picture UUID for library lookup"
        ) from exc
    database = root / "lytrolibrary.db"
    # immutable + ro never creates journals, locks, WAL or SHM files.
    try:
        with closing(
            sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as connection:
            rows = connection.execute(
                "SELECT imagebin_uuid FROM picture WHERE uuid = ?", (picture,)
            ).fetchall()
    except sqlite3.Error as exc:
        raise LytroError(f"Cannot read library index: {exc}") from exc
    if len(rows) != 1:
        raise LytroError("Picture UUID does not identify exactly one library capture")
    try:
        imagebin = str(uuid.UUID(rows[0][0]))
    except (ValueError, TypeError) as exc:
        raise LytroError("Invalid library image-bin UUID") from exc
    candidate = root / imagebin / picture / "raw.lfp"
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise LytroError("Library raw path escapes the library")
    raw = read_lfp(resolved)
    info = raw.admit_capture()
    frames = stack._frames()
    if allow_unbound_recipe and not frames:
        return resolved
    if len(frames) != 1 or frames[0].get("frame", {}).get("imageRef") != info.raw_reference:
        raise LytroError("Library raw payload identity differs from the exported stack")
    return resolved
