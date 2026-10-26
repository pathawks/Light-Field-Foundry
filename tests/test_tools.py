import signal
import sys
import tempfile
from pathlib import Path
from typing import Never

import pytest

from light_field_foundry.errors import LytroError
from light_field_foundry.tools import Tools, executable, workspace


def test_bad_explicit_tool_never_falls_back(tmp_path: Path) -> None:
    with pytest.raises(LytroError, match="not an executable"):
        executable("python3", tmp_path / "missing")


def test_tnt_environment_path_is_not_a_shell_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNT_PATH", "tnt --version")
    with pytest.raises(LytroError, match="not an executable"):
        Tools.discover()


def test_process_timeout_kills_child_and_workspace_is_removed() -> None:
    tools = Tools(Path(sys.executable), Path(sys.executable), Path(sys.executable), timeout=0.05)
    directories: list[Path] = []

    def timeout(work: Path) -> None:
        directories.append(work)
        tools.run([sys.executable, "-c", "import time; time.sleep(10)"], work)

    with pytest.raises(LytroError, match="exceeded"), workspace() as work:
        timeout(work)
    assert len(directories) == 1
    assert not directories[0].exists()


def test_process_output_is_bounded() -> None:
    tools = Tools(Path(sys.executable), Path(sys.executable), Path(sys.executable))
    with workspace() as work:
        output = tools.run([sys.executable, "-c", "print('x'*100000)"], work)
        assert len(output.encode()) == 65536


def test_workspace_removes_files_on_handled_interruption() -> None:
    directories: list[Path] = []

    def interrupt(work: Path) -> Never:
        directories.append(work)
        (work / "current-run.tiff").write_bytes(b"temporary")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), workspace() as work:
        interrupt(work)
    assert len(directories) == 1
    assert not directories[0].exists()


def test_workspace_uses_unique_empty_system_temporary_directories() -> None:
    with workspace() as first, workspace() as second:
        assert first != second
        assert first.parent.resolve() == Path(tempfile.gettempdir()).resolve()
        assert second.parent.resolve() == Path(tempfile.gettempdir()).resolve()
        assert list(first.iterdir()) == []
        assert list(second.iterdir()) == []
    assert not first.exists()
    assert not second.exists()


def test_workspace_cleans_up_on_handled_sigterm_and_restores_handler() -> None:
    previous = signal.getsignal(signal.SIGTERM)

    directories: list[Path] = []

    def terminate(work: Path) -> None:
        directories.append(work)
        (work / "current-run.lfp").write_bytes(b"temporary")
        signal.raise_signal(signal.SIGTERM)

    with pytest.raises(InterruptedError, match="Terminated"), workspace() as work:
        terminate(work)
    assert len(directories) == 1
    assert not directories[0].exists()
    assert signal.getsignal(signal.SIGTERM) == previous
