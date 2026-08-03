"""Proportional public-tree and distribution safety checks."""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
import tomllib
import urllib.parse
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath

_MAX_GIT_OUTPUT = 2_000_000
_MAX_PATHS = 10_000
_MAX_TEXT_FILE = 2_000_000
_MAX_TREE_TEXT = 32_000_000
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_MEMBER = 8_000_000
_MAX_ARCHIVE_TEXT = 64_000_000

_TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".css",
        ".csv",
        ".example",
        ".html",
        ".ini",
        ".j2",
        ".js",
        ".json",
        ".md",
        ".py",
        ".pyi",
        ".rst",
        ".sh",
        ".svg",
        ".toml",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_PRIVATE_ROOTS = frozenset({"artifacts", "generated", "output", "profile", "tmp"})
_ROOT_RESUME_NAMES = frozenset({"master-resume.yml", "mp-master-resume.txt"})
_DATABASE_SUFFIXES = (
    ".db",
    ".sqlite",
    ".sqlite3",
    "-journal",
    "-wal",
    "-shm",
)
_DOCUMENT_SUFFIXES = frozenset({".docx", ".egg", ".odt", ".pdf", ".whl"})
_GENERATED_DIRECTORIES = frozenset(
    {
        "artifact",
        "artifacts",
        "cover-letter",
        "cover-letters",
        "generated-artifact",
        "generated-artifacts",
        "generated-cover-letter",
        "generated-cover-letters",
        "generated-resume",
        "generated-resumes",
        "generated_resume",
        "generated_resumes",
    }
)
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+:-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})\.)+"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})|localhost)"
)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[ .-]?)?\(?(\d{3})\)?[ .-](\d{3})[ .-](\d{4})(?!\d)"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_TICKET_RE = re.compile(
    r"(?i)(?:\b(?:jira\s*:\s*|ticket\s+)"
    r"(?!(?:CVE|ISO|SHA)-)"
    r"[A-Z][A-Z0-9]{1,9}-\d{2,8}\b|"
    r"/browse/(?!(?:CVE|ISO|SHA)-)"
    r"[A-Z][A-Z0-9]{1,9}-\d{2,8}\b)"
)
_POSIX_HOME_RE = re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[A-Za-z0-9._-]{1,64}/")
_WINDOWS_HOME_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Z]:\\Users\\[A-Za-z0-9._-]{1,64}\\"
)
_PRIVATE_KEY_RE = re.compile(
    ("-" * 5) + r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY" + ("-" * 5)
)
_TOKEN_RES = (
    re.compile(r"\bghp_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{24,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
)
_HTTP_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_MAILTO_RE = re.compile(r"mailto:([^\s\"'<>]+)", re.IGNORECASE)
_SDIST_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*-\d[0-9A-Za-z._-]*")
_WHEEL_DIST_INFO_RE = re.compile(r"career_agent_workbench-\d[^/]*\.dist-info")
_WHEEL_REQUIRED = frozenset(
    {
        "career_agent_workbench/_archived_flask_source.py",
        "career_agent_workbench/__init__.py",
        "career_agent_workbench/__main__.py",
        "career_agent_workbench/cover_letter_rendering.py",
        "career_agent_workbench/static/webapp/app.js",
        "career_agent_workbench/templates/resume/master_resume.html.j2",
        "career_agent_workbench/templates/webapp/add.html",
        "career_agent_workbench/templates/webapp/cover_letter_edit.html",
        "career_agent_workbench/templates/webapp/index.html",
        "career_agent_workbench/templates/webapp/jod.html",
        "career_agent_workbench/templates/webapp/resume_edit.html",
        "career_agent_workbench/templates/webapp/variant_review.html",
        "career_agent_workbench/webapp_actions.py",
        "career_agent_workbench/webapp_archive_runtime.py",
        "career_agent_workbench/webapp_artifacts.py",
        "career_agent_workbench/webapp_editors.py",
        "career_agent_workbench/webapp_ingestion.py",
        "career_agent_workbench/webapp_tracker.py",
    }
)


class SafetyCheckError(Exception):
    """Generic public-safety invocation failure."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, "Public-safety check failed.\n")


@dataclass(frozen=True, order=True, slots=True)
class Finding:
    """One content-hidden, deterministic finding."""

    rule: str
    path: str


def _normalize_relative(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        return None
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    normalized = posixpath.normpath(raw)
    if normalized in {"", "."} or normalized.startswith("../"):
        return None
    return normalized


def _path_findings(relative: str) -> set[Finding]:
    findings: set[Finding] = set()
    path = PurePosixPath(relative)
    parts = path.parts
    lowered = tuple(part.casefold() for part in parts)
    name = lowered[-1]

    if lowered[0] in _PRIVATE_ROOTS or (
        len(parts) == 1 and name in {".blacklist", *_ROOT_RESUME_NAMES}
    ):
        findings.add(Finding("PATH_PRIVATE_STATE", relative))
    if relative != ".env.example" and (
        name == ".env" or name.startswith(".env.") or name.endswith(".env")
    ):
        findings.add(Finding("PATH_DOTENV", relative))
    if name.endswith(_DATABASE_SUFFIXES):
        findings.add(Finding("PATH_DATABASE", relative))
    if path.suffix.casefold() in _DOCUMENT_SUFFIXES or any(
        part in _GENERATED_DIRECTORIES for part in lowered[:-1]
    ):
        findings.add(Finding("PATH_GENERATED_ARTIFACT", relative))
    return findings


def _reserved_host(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    if normalized == "users.noreply.github.com":
        return True
    exact = {"example.com", "example.net", "example.org", "localhost"}
    if normalized in exact:
        return True
    return normalized.endswith(
        (
            ".example.com",
            ".example.net",
            ".example.org",
            ".invalid",
            ".localhost",
            ".test",
        )
    )


def _text_findings(relative: str, data: bytes) -> set[Finding]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {Finding("TEXT_ENCODING", relative)}

    findings: set[Finding] = set()
    if any(not _reserved_host(match.group(1)) for match in _EMAIL_RE.finditer(text)):
        findings.add(Finding("TEXT_EMAIL", relative))

    unsafe_phone = False
    for match in _PHONE_RE.finditer(text):
        exchange = match.group(2)
        subscriber = int(match.group(3))
        if exchange != "555" or not 100 <= subscriber <= 199:
            unsafe_phone = True
            break
    if unsafe_phone or _SSN_RE.search(text):
        findings.add(Finding("TEXT_PERSONAL_NUMBER", relative))
    if _TICKET_RE.search(text):
        findings.add(Finding("TEXT_INTERNAL_TICKET", relative))
    if _POSIX_HOME_RE.search(text) or _WINDOWS_HOME_RE.search(text):
        findings.add(Finding("TEXT_MACHINE_PATH", relative))
    if _PRIVATE_KEY_RE.search(text) or any(
        pattern.search(text) for pattern in _TOKEN_RES
    ):
        findings.add(Finding("TEXT_CREDENTIAL", relative))

    if relative == "examples" or relative.startswith("examples/"):
        unsafe_example = False
        for match in _HTTP_RE.finditer(text):
            try:
                host = urllib.parse.urlsplit(match.group(0)).hostname
            except ValueError:
                host = None
            if host is None or not _reserved_host(host):
                unsafe_example = True
                break
        if not unsafe_example:
            for match in _MAILTO_RE.finditer(text):
                address = urllib.parse.unquote(match.group(1)).split("?", 1)[0]
                domain = address.rsplit("@", 1)[-1] if "@" in address else ""
                if not domain or not _reserved_host(domain):
                    unsafe_example = True
                    break
        if unsafe_example:
            findings.add(Finding("TEXT_EXAMPLE_DOMAIN", relative))
    return findings


def _looks_text(relative: str) -> bool:
    path = PurePosixPath(relative)
    return path.suffix.casefold() in _TEXT_SUFFIXES or not path.suffix


def _regular_without_symlinks(root: Path, relative: str) -> bool:
    current = root
    parts = PurePosixPath(relative).parts
    try:
        for index, part in enumerate(parts):
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                return False
            if index == len(parts) - 1:
                return stat.S_ISREG(metadata.st_mode)
            if not stat.S_ISDIR(metadata.st_mode):
                return False
    except OSError:
        return False
    return False


def scan_paths(root: Path, relative_paths: Iterable[str]) -> tuple[Finding, ...]:
    """Apply the public path and bounded text rules to explicit relative paths."""

    try:
        candidate_root = Path(root).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise SafetyCheckError("Public-safety check failed.") from None
    if not candidate_root.is_dir():
        raise SafetyCheckError("Public-safety check failed.")

    findings: set[Finding] = set()
    aggregate = 0
    count = 0
    for raw in relative_paths:
        count += 1
        if count > _MAX_PATHS:
            raise SafetyCheckError("Public-safety check failed.")
        relative = _normalize_relative(raw)
        if relative is None:
            findings.add(Finding("PATH_INVALID", "<invalid>"))
            continue
        path_results = _path_findings(relative)
        findings.update(path_results)
        if path_results:
            continue
        if not _regular_without_symlinks(candidate_root, relative):
            findings.add(Finding("PATH_NON_REGULAR", relative))
            continue
        if not _looks_text(relative):
            continue
        candidate = candidate_root.joinpath(*PurePosixPath(relative).parts)
        try:
            size = candidate.stat().st_size
            if size > _MAX_TEXT_FILE:
                findings.add(Finding("TEXT_SIZE", relative))
                continue
            aggregate += size
            if aggregate > _MAX_TREE_TEXT:
                raise SafetyCheckError("Public-safety check failed.")
            with candidate.open("rb") as stream:
                data = stream.read(_MAX_TEXT_FILE + 1)
        except OSError:
            findings.add(Finding("PATH_NON_REGULAR", relative))
            continue
        if len(data) > _MAX_TEXT_FILE:
            findings.add(Finding("TEXT_SIZE", relative))
            continue
        findings.update(_text_findings(relative, data))
    return tuple(sorted(findings))


def tracked_paths(root: Path) -> tuple[str, ...]:
    """Return only the index/worktree candidate set reported by git ls-files."""

    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        raise SafetyCheckError("Public-safety check failed.") from None
    if result.returncode != 0 or result.stderr or len(result.stdout) > _MAX_GIT_OUTPUT:
        raise SafetyCheckError("Public-safety check failed.")
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise SafetyCheckError("Public-safety check failed.") from None
    if decoded and not decoded.endswith("\x00"):
        raise SafetyCheckError("Public-safety check failed.")
    paths = tuple(decoded.split("\x00")[:-1]) if decoded else ()
    if len(paths) > _MAX_PATHS:
        raise SafetyCheckError("Public-safety check failed.")
    return paths


def scan_tree(root: Path) -> tuple[Finding, ...]:
    """Scan the tracked candidate tree only."""

    try:
        candidate_root = Path(root).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise SafetyCheckError("Public-safety check failed.") from None
    return scan_paths(candidate_root, tracked_paths(candidate_root))


def _archive_content_findings(relative: str, data: bytes) -> set[Finding]:
    findings = _path_findings(relative)
    if not findings and _looks_text(relative):
        findings.update(_text_findings(relative, data))
    return findings


def scan_wheel(wheel: Path) -> tuple[Finding, ...]:
    """Scan one wheel in place without extraction."""

    archive_path = Path(wheel)
    if archive_path.suffix.casefold() != ".whl" or archive_path.is_symlink():
        raise SafetyCheckError("Public-safety check failed.")
    findings: set[Finding] = set()
    files: set[str] = set()
    normalized_names: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
                raise SafetyCheckError("Public-safety check failed.")
            for member in members:
                relative = _normalize_relative(member.filename.rstrip("/"))
                if relative is None or relative in normalized_names:
                    findings.add(Finding("ARCHIVE_MEMBER", "<archive>"))
                    continue
                normalized_names.add(relative)
                if member.is_dir():
                    continue
                mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type and file_type != stat.S_IFREG:
                    findings.add(Finding("ARCHIVE_MEMBER", relative))
                    continue
                if member.file_size > _MAX_ARCHIVE_MEMBER:
                    findings.add(Finding("ARCHIVE_SIZE", relative))
                    continue
                with archive.open(member) as stream:
                    data = stream.read(_MAX_ARCHIVE_MEMBER + 1)
                if len(data) > _MAX_ARCHIVE_MEMBER:
                    findings.add(Finding("ARCHIVE_SIZE", relative))
                    continue
                total += len(data)
                if total > _MAX_ARCHIVE_TEXT:
                    raise SafetyCheckError("Public-safety check failed.")
                files.add(relative)
                findings.update(_archive_content_findings(relative, data))
    except SafetyCheckError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        raise SafetyCheckError("Public-safety check failed.") from None

    dist_info: set[str] = set()
    for relative in files:
        top = relative.split("/", 1)[0]
        if top == "career_agent_workbench":
            if "/" not in relative:
                findings.add(Finding("WHEEL_MEMBERSHIP", relative))
                continue
            package_relative = relative.split("/", 1)[1]
            for finding in _path_findings(package_relative):
                findings.add(Finding(finding.rule, relative))
            if package_relative.split("/", 1)[0] in {
                "docs",
                "examples",
                "scripts",
                "tests",
            }:
                findings.add(Finding("WHEEL_MEMBERSHIP", relative))
            continue
        if _WHEEL_DIST_INFO_RE.fullmatch(top):
            dist_info.add(top)
            continue
        findings.add(Finding("WHEEL_MEMBERSHIP", relative))
    if not _WHEEL_REQUIRED <= files or len(dist_info) != 1:
        findings.add(Finding("WHEEL_MEMBERSHIP", "<archive>"))
    return tuple(sorted(findings))


def _sdist_declared_paths(root: Path) -> frozenset[str]:
    try:
        metadata = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
        values = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        raise SafetyCheckError("Public-safety check failed.") from None
    if not isinstance(values, list) or not values:
        raise SafetyCheckError("Public-safety check failed.")
    declared: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.startswith("/"):
            raise SafetyCheckError("Public-safety check failed.")
        relative = _normalize_relative(value[1:])
        if relative is None:
            raise SafetyCheckError("Public-safety check failed.")
        declared.add(relative)
    return frozenset(declared)


def scan_sdist(root: Path, sdist: Path) -> tuple[Finding, ...]:
    """Scan one source archive in place without extraction."""

    archive_path = Path(sdist)
    if (
        not archive_path.name.casefold().endswith(".tar.gz")
        or archive_path.is_symlink()
    ):
        raise SafetyCheckError("Public-safety check failed.")
    findings: set[Finding] = set()
    stripped_files: set[str] = set()
    total = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
                raise SafetyCheckError("Public-safety check failed.")
            normalized: list[tuple[tarfile.TarInfo, str]] = []
            prefixes: set[str] = set()
            raw_names: set[str] = set()
            for member in members:
                relative = _normalize_relative(member.name.rstrip("/"))
                if relative is None or relative in raw_names:
                    findings.add(Finding("ARCHIVE_MEMBER", "<archive>"))
                    continue
                raw_names.add(relative)
                normalized.append((member, relative))
                if "/" in relative:
                    prefixes.add(relative.split("/", 1)[0])
            if len(prefixes) != 1:
                raise SafetyCheckError("Public-safety check failed.")
            prefix = next(iter(prefixes))
            if _SDIST_PREFIX_RE.fullmatch(prefix) is None:
                raise SafetyCheckError("Public-safety check failed.")

            normalized_members: set[str] = set()
            for member, relative in normalized:
                if relative == prefix and member.isdir():
                    continue
                if not relative.startswith(f"{prefix}/"):
                    findings.add(Finding("ARCHIVE_MEMBER", "<archive>"))
                    continue
                stripped = relative[len(prefix) + 1 :]
                if stripped in normalized_members:
                    findings.add(Finding("ARCHIVE_MEMBER", "<archive>"))
                    continue
                normalized_members.add(stripped)
                if member.isdir():
                    continue
                if not member.isreg():
                    findings.add(Finding("ARCHIVE_MEMBER", stripped))
                    continue
                if member.size > _MAX_ARCHIVE_MEMBER:
                    findings.add(Finding("ARCHIVE_SIZE", stripped))
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    findings.add(Finding("ARCHIVE_MEMBER", stripped))
                    continue
                with stream:
                    data = stream.read(_MAX_ARCHIVE_MEMBER + 1)
                if len(data) > _MAX_ARCHIVE_MEMBER:
                    findings.add(Finding("ARCHIVE_SIZE", stripped))
                    continue
                total += len(data)
                if total > _MAX_ARCHIVE_TEXT:
                    raise SafetyCheckError("Public-safety check failed.")
                stripped_files.add(stripped)
                findings.update(_archive_content_findings(stripped, data))
    except SafetyCheckError:
        raise
    except (OSError, ValueError, tarfile.TarError, RuntimeError):
        raise SafetyCheckError("Public-safety check failed.") from None

    for missing in sorted(_sdist_declared_paths(Path(root)) - stripped_files):
        findings.add(Finding("SDIST_MEMBERSHIP", missing))
    return tuple(sorted(findings))


def scan_artifacts(root: Path, wheel: Path, sdist: Path) -> tuple[Finding, ...]:
    """Scan one wheel and one sdist using the public path/content rules."""

    return tuple(sorted({*scan_wheel(wheel), *scan_sdist(root, sdist)}))


def installed_smoke(expected_prefix: Path) -> None:
    """Run a tiny offline smoke against an installed package prefix."""

    try:
        prefix = Path(expected_prefix).resolve(strict=True)
        import career_agent_workbench
        from importlib.metadata import distribution

        from career_agent_workbench import (
            cover_letter_rendering,
            webapp_actions,
            webapp_artifacts,
            webapp_editors,
            webapp_ingestion,
            webapp_tracker,
        )
        from career_agent_workbench.resume_rendering import (
            render_resume_html_from_mapping,
        )

        module_file = Path(career_agent_workbench.__file__).resolve(strict=True)
        module_file.relative_to(prefix)
        if career_agent_workbench.__version__ != "2.0.0":
            raise SafetyCheckError("Public-safety check failed.")
        package = resources.files("career_agent_workbench")
        resume_template = package.joinpath(
            "templates", "resume", "master_resume.html.j2"
        ).read_text("utf-8")
        web_template = package.joinpath("templates", "webapp", "index.html").read_text(
            "utf-8"
        )
        web_resources = (
            package.joinpath("templates", "webapp", name).read_text("utf-8")
            for name in (
                "add.html",
                "cover_letter_edit.html",
                "jod.html",
                "resume_edit.html",
                "variant_review.html",
            )
        )
        static_script = package.joinpath("static", "webapp", "app.js").read_text(
            "utf-8"
        )
        if (
            not resume_template
            or not web_template
            or not all(web_resources)
            or not static_script
        ):
            raise SafetyCheckError("Public-safety check failed.")
        modules = (
            cover_letter_rendering,
            webapp_actions,
            webapp_artifacts,
            webapp_editors,
            webapp_ingestion,
            webapp_tracker,
        )
        if any(
            not Path(module.__file__).resolve().is_relative_to(prefix)
            for module in modules
        ):
            raise SafetyCheckError("Public-safety check failed.")
        expected_entries = {
            "career-agent-workbench": "career_agent_workbench.__main__:main",
            "career-agent-workbench-audit-jods": (
                "career_agent_workbench.jod_cleaner_audit:main"
            ),
            "career-agent-workbench-refine-resume": (
                "career_agent_workbench.resume_refinement_cli:main"
            ),
            "career-agent-workbench-seed-jobs": (
                "career_agent_workbench.workflows.matching:main"
            ),
            "career-agent-workbench-webapp": (
                "career_agent_workbench.webapp_archive_runtime:main"
            ),
            "career-agent-workbench-mcp": "career_agent_workbench.server:main",
        }
        installed_entries = {
            entry.name: entry.value
            for entry in distribution("career-agent-workbench").entry_points
            if entry.group == "console_scripts" and entry.name in expected_entries
        }
        if installed_entries != expected_entries:
            raise SafetyCheckError("Public-safety check failed.")
        name = "Jules <Example>"
        rendered = render_resume_html_from_mapping(
            resume={"header_top": {"line_1_name_header_text": name}}
        )
        if "Jules &lt;Example&gt;" not in rendered or name in rendered:
            raise SafetyCheckError("Public-safety check failed.")
    except SafetyCheckError:
        raise
    except Exception:
        raise SafetyCheckError("Public-safety check failed.") from None


def _print_findings(findings: Sequence[Finding]) -> int:
    for finding in findings:
        print(f"{finding.rule} {finding.path}")
    return 1 if findings else 0


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Check public repository safety.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    tree = subparsers.add_parser("tree")
    tree.add_argument("--root", required=True, type=Path)

    artifacts = subparsers.add_parser("artifacts")
    artifacts.add_argument("--root", required=True, type=Path)
    artifacts.add_argument("--wheel", required=True, type=Path)
    artifacts.add_argument("--sdist", required=True, type=Path)

    smoke = subparsers.add_parser("installed-smoke")
    smoke.add_argument("--expected-prefix", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "tree":
            return _print_findings(scan_tree(args.root))
        if args.mode == "artifacts":
            return _print_findings(scan_artifacts(args.root, args.wheel, args.sdist))
        installed_smoke(args.expected_prefix)
        return 0
    except SafetyCheckError:
        print("Public-safety check failed.", file=sys.stderr)
        return 2
    except Exception:
        print("Public-safety check failed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
