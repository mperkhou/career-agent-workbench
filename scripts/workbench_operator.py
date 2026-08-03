"""Narrow local website lifecycle and exact public-skill link helper."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

EXACT_SKILLS = (
    "career-agent-workbench",
    "master-resume-yaml",
    "manual-resume-passthrough",
    "agentic-workflow-init",
    "agentic-workflow-controller",
)
_PID_FILE = "website.json"
_LOG_FILE = "website.log"
_TOKEN_RE = re.compile(r"[0-9a-f]{32}\Z")
_ERROR = "Operator command failed."


class OperatorError(Exception):
    """Stable, content-free operator failure."""

    __slots__ = ()


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file() or not (root / "Makefile").is_file():
        raise OperatorError(_ERROR)
    return root


def _state_directory(root: Path) -> Path:
    state = root / "tmp" / "website"
    try:
        current = root
        for part in ("tmp", "website"):
            current /= part
            if current.is_symlink():
                raise ValueError
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        state.resolve(strict=True).relative_to(root.resolve(strict=True))
        if os.name == "posix":
            os.chmod(state, 0o700)
    except (OSError, RuntimeError, ValueError):
        raise OperatorError(_ERROR) from None
    return state


def _read_record(path: Path) -> tuple[int, str] | None:
    if not path.exists():
        return None
    try:
        if path.is_symlink() or path.stat().st_size > 4_096:
            raise ValueError
        value = json.loads(path.read_text(encoding="utf-8"))
        pid = value["pid"]
        token = value["token"]
        if type(pid) is not int or pid <= 1:
            raise ValueError
        if type(token) is not str or _TOKEN_RE.fullmatch(token) is None:
            raise ValueError
        return pid, token
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise OperatorError(_ERROR) from None


def _write_record(path: Path, pid: int, token: str) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        if path.is_symlink() or temporary.exists() or temporary.is_symlink():
            raise ValueError
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": pid, "token": token}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            os.chmod(path, 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise OperatorError(_ERROR) from None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_command(pid: int) -> str:
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0 or len(completed.stdout) > 16_384:
        return ""
    return completed.stdout.strip()


def _owns_process(pid: int, token: str) -> bool:
    command = _process_command(pid)
    script = str(Path(__file__).resolve())
    return (
        bool(command)
        and script in command
        and " _serve " in f" {command} "
        and f"--token {token}" in command
    )


def _remove_record(path: Path) -> None:
    try:
        if path.is_symlink():
            raise ValueError
        path.unlink(missing_ok=True)
    except (OSError, ValueError):
        raise OperatorError(_ERROR) from None


def start_website(*, host: str, port: int, open_browser: bool) -> str:
    """Start one recorded local web child or return its current state."""

    if host not in {"127.0.0.1", "localhost", "::1"} or not 1 <= port <= 65535:
        raise OperatorError(_ERROR)
    root = _project_root()
    state = _state_directory(root)
    record_path = state / _PID_FILE
    record = _read_record(record_path)
    if record is not None:
        pid, token = record
        if _pid_is_alive(pid) and _owns_process(pid, token):
            return "Website is already running."
        _remove_record(record_path)

    token = secrets.token_hex(16)
    log_path = state / _LOG_FILE
    try:
        if log_path.is_symlink():
            raise ValueError
        log = log_path.open("ab", buffering=0)
        if os.name == "posix":
            os.chmod(log_path, 0o600)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_serve",
            "--token",
            token,
            "--host",
            host,
            "--port",
            str(port),
        ]
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        log.close()
        _write_record(record_path, process.pid, token)
        time.sleep(0.1)
        if process.poll() is not None:
            _remove_record(record_path)
            raise ValueError
        if open_browser:
            webbrowser.open(f"http://{host}:{port}/")
    except OperatorError:
        raise
    except Exception:
        raise OperatorError(_ERROR) from None
    return "Website started."


def stop_website() -> str:
    """Stop only the exact child bound by the current record."""

    state = _state_directory(_project_root())
    record_path = state / _PID_FILE
    record = _read_record(record_path)
    if record is None:
        return "Website is not running."
    pid, token = record
    if not _pid_is_alive(pid) or not _owns_process(pid, token):
        _remove_record(record_path)
        return "Stale website state removed."
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10.0
        while _pid_is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _pid_is_alive(pid):
            if not _owns_process(pid, token):
                raise ValueError
            os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 2.0
            while _pid_is_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
        if _pid_is_alive(pid):
            raise ValueError
        _remove_record(record_path)
    except Exception:
        raise OperatorError(_ERROR) from None
    return "Website stopped."


def link_skills(destination: Path) -> int:
    """Idempotently install exactly the five public repository skills."""

    root = _project_root()
    source_root = root / "skills"
    try:
        destination = destination.expanduser()
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError
        sources = tuple(source_root / name for name in EXACT_SKILLS)
        targets = tuple(destination / name for name in EXACT_SKILLS)
        if any(not (source / "SKILL.md").is_file() for source in sources):
            raise ValueError
        if any(target.exists() and not target.is_symlink() for target in targets):
            raise ValueError
        for source, target in zip(sources, targets, strict=True):
            expected = str(source.resolve(strict=True))
            if target.is_symlink():
                if os.readlink(target) == expected:
                    continue
                target.unlink()
            target.symlink_to(expected, target_is_directory=True)
    except (OSError, RuntimeError, ValueError):
        raise OperatorError(_ERROR) from None
    return len(EXACT_SKILLS)


def _port(value: str) -> int:
    try:
        selected = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(_ERROR) from None
    if not 1 <= selected <= 65535:
        raise argparse.ArgumentTypeError(_ERROR)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded local operator tasks.")
    commands = parser.add_subparsers(dest="command", required=True)
    website = commands.add_parser("website")
    website_commands = website.add_subparsers(dest="website_command", required=True)
    start = website_commands.add_parser("start")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=_port, default=8765)
    start.add_argument("--open-browser", action="store_true")
    website_commands.add_parser("stop")
    skills = commands.add_parser("skills")
    skills_commands = skills.add_subparsers(dest="skills_command", required=True)
    link = skills_commands.add_parser("link")
    link.add_argument("--destination", type=Path, required=True)
    serve = commands.add_parser("_serve")
    serve.add_argument("--token", required=True)
    serve.add_argument("--host", required=True)
    serve.add_argument("--port", type=_port, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "website" and args.website_command == "start":
            print(
                start_website(
                    host=args.host,
                    port=args.port,
                    open_browser=args.open_browser,
                )
            )
        elif args.command == "website" and args.website_command == "stop":
            print(stop_website())
        elif args.command == "skills" and args.skills_command == "link":
            count = link_skills(args.destination)
            print(f"Linked {count} public skills.")
        elif args.command == "_serve":
            if _TOKEN_RE.fullmatch(args.token) is None:
                raise OperatorError(_ERROR)
            from career_agent_workbench.webapp_archive_runtime import (
                main as webapp_main,
            )

            return webapp_main(
                [
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--project-root",
                    str(_project_root()),
                ]
            )
        else:
            raise OperatorError(_ERROR)
    except OperatorError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXACT_SKILLS",
    "OperatorError",
    "link_skills",
    "start_website",
    "stop_website",
]
