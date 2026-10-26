"""User-owned executables, bounded processes, and disposable workspaces."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import LytroError


def executable(name: str, explicit: Path | str | None = None) -> Path:
    """An explicit bad path is never silently replaced by another executable."""
    value = str(explicit) if explicit is not None else shutil.which(name)
    if not value:
        raise LytroError(f"Missing {name}; install it yourself and put it on PATH.")
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise LytroError(f"{name} is not an executable file: {path}")
    return path


@dataclass(frozen=True)
class Tools:
    tnt: Path
    avifenc: Path
    avifdec: Path
    threads: int = 4
    timeout: float = 600

    @classmethod
    def discover(
        cls,
        avifenc: Path | None = None,
        avifdec: Path | None = None,
        threads: int = 4,
        timeout: float = 600,
    ) -> Tools:
        if not 1 <= threads <= 32 or not 0 < timeout <= 86400:
            raise LytroError("Threads must be 1..32 and timeout must be in (0, 86400] seconds.")
        return cls(
            executable("tnt", os.environ.get("TNT_PATH")),
            executable("avifenc", avifenc),
            executable("avifdec", avifdec),
            threads,
            timeout,
        )

    def run(self, args: Sequence[str | Path], cwd: Path) -> str:
        """Drain output continuously, retaining at most the last 64 KiB in memory."""
        command = [str(arg) for arg in args]
        output = bytearray()
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise LytroError(f"Could not launch {Path(command[0]).name}: {exc}") from exc
        assert process.stdout is not None
        stream = process.stdout

        def drain() -> None:
            while chunk := stream.read(8192):
                output.extend(chunk)
                if len(output) > 65536:
                    del output[:-65536]

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            process.wait(timeout=self.timeout)
        except BaseException as exc:
            if process.poll() is None:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            process.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise LytroError(
                    f"{Path(command[0]).name} exceeded {self.timeout:g} seconds."
                ) from exc
            raise
        finally:
            reader.join(timeout=5)
            stream.close()
        message = output.decode("utf-8", errors="replace")
        if process.returncode:
            raise LytroError(
                f"{Path(command[0]).name} exited {process.returncode}:\n{message.strip()}"
            )
        return message

    def capabilities(self, cwd: Path) -> dict[str, str]:
        enc = self.run([self.avifenc, "--help"], cwd)
        dec = self.run([self.avifdec, "--help"], cwd)
        for required in ("12,4", "--lossless", "--cicp", "--icc"):
            if required not in enc:
                raise LytroError(f"avifenc lacks required capability {required}.")
        if "--sato" not in dec or "y4m" not in dec:
            raise LytroError("avifdec must support --sato and native Y4M output.")
        return {
            "tnt": self.run([self.tnt, "--version"], cwd).strip(),
            "avifenc": self.run([self.avifenc, "--version"], cwd).strip(),
            "avifdec": self.run([self.avifdec, "--version"], cwd).strip(),
        }


@contextlib.contextmanager
def workspace() -> Generator[Path]:
    """Use a unique system temporary directory, cleaned on context exit."""
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(_signum: int, _frame: object) -> None:
        raise InterruptedError("Terminated; current-run temporary files were removed.")

    signal.signal(signal.SIGTERM, terminate)
    try:
        with tempfile.TemporaryDirectory(prefix="lff-") as directory:
            yield Path(directory)
    finally:
        signal.signal(signal.SIGTERM, previous)


def require_output(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise LytroError(f"Native tool returned without creating {path.name}.")
