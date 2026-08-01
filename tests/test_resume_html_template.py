from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import types
from collections.abc import Iterator, Mapping
from io import BytesIO, StringIO
from pathlib import Path
from typing import Self

import pytest
import yaml
from pypdf import PdfReader, PdfWriter

from career_agent_workbench import resume_rendering
from career_agent_workbench.ats import (
    calculate_ats_diagnostics,
    calculate_ats_proxy_score,
)
from career_agent_workbench.resume_rendering import (
    MAX_RENDERED_HTML_CHARS,
    MAX_RESUME_DATA_CHARS,
    MAX_RESUME_DATA_DEPTH,
    MAX_RESUME_DATA_NODES,
    MAX_RESUME_PDF_BYTES,
    MAX_RESUME_PDF_PAGES,
    MAX_RESUME_YAML_BYTES,
    MAX_TEMPLATE_CHARS,
    ResumeRenderingError,
    linkify,
    render_core_skill_rows,
    render_resume_html,
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
    rich_text,
    sanitize_resume_url,
)


def _synthetic_pdf_bytes(page_count: int = 1) -> bytes:
    writer = PdfWriter()
    for _index in range(page_count):
        writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


_VALID_ONE_PAGE_PDF = _synthetic_pdf_bytes()


def _emit_test_diagnostics(logger: logging.Logger, label: str) -> None:
    print(f"{label}-stdout")
    print(f"{label}-stderr", file=sys.stderr)
    logger.debug("%s-log", label)


def _spawn_joined_diagnostic_tree(
    logger: logging.Logger,
    label: str,
) -> None:
    failures: list[BaseException] = []

    def grandchild_target() -> None:
        try:
            _emit_test_diagnostics(logger, f"{label}-grandchild")
        except BaseException as error:
            failures.append(error)

    def child_target() -> None:
        try:
            _emit_test_diagnostics(logger, f"{label}-child")
            grandchild = threading.Thread(target=grandchild_target)
            grandchild.start()
            grandchild.join(timeout=5)
            if grandchild.is_alive():
                failures.append(RuntimeError("grandchild timeout"))
        except BaseException as error:
            failures.append(error)

    child = threading.Thread(target=child_target)
    child.start()
    child.join(timeout=5)
    if child.is_alive():
        failures.append(RuntimeError("child timeout"))
    if failures:
        raise RuntimeError("diagnostic descendant failure")


def _diagnostic_surfaces() -> tuple[object, object, object, object]:
    return (
        sys.stdout,
        sys.stderr,
        vars(logging.Logger)["handle"],
        vars(threading.Thread)["start"],
    )


def _restore_diagnostic_surfaces(
    surfaces: tuple[object, object, object, object],
) -> None:
    sys.stdout = surfaces[0]  # type: ignore[assignment]
    sys.stderr = surfaces[1]  # type: ignore[assignment]
    logging.Logger.handle = surfaces[2]  # type: ignore[assignment]
    threading.Thread.start = surfaces[3]  # type: ignore[assignment]


def _spawn_joined_mixed_diagnostic_tree(
    logger: logging.Logger,
    label: str,
    *,
    execution_ids: list[int] | None = None,
    ownership: dict[str, frozenset[object]] | None = None,
) -> None:
    failures: list[BaseException] = []

    def emit(body_label: str) -> None:
        try:
            if execution_ids is not None:
                execution_ids.append(threading.get_ident())
            if ownership is not None:
                with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                    ownership[body_label] = frozenset(
                        resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                            id(threading.current_thread()),
                            set(),
                        ),
                    )
            _emit_test_diagnostics(logger, body_label)
        except BaseException as error:
            failures.append(error)

    def target_grandchild() -> None:
        emit(f"{label}-target-grandchild")

    class DiagnosticRunThread(threading.Thread):
        def __init__(self, body_label: str, *, spawn_target: bool = False) -> None:
            super().__init__()
            self._body_label = body_label
            self._spawn_target = spawn_target

        def run(self) -> None:
            emit(self._body_label)
            if self._spawn_target:
                grandchild = threading.Thread(target=target_grandchild)
                grandchild.start()
                grandchild.join(timeout=5)
                if grandchild.is_alive():
                    failures.append(RuntimeError("target grandchild timeout"))

    def target_child() -> None:
        emit(f"{label}-target-child")
        grandchild = DiagnosticRunThread(f"{label}-run-grandchild")
        grandchild.start()
        grandchild.join(timeout=5)
        if grandchild.is_alive():
            failures.append(RuntimeError("run grandchild timeout"))

    children: list[threading.Thread] = [
        threading.Thread(target=target_child),
        DiagnosticRunThread(f"{label}-run-child", spawn_target=True),
    ]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=5)
        if child.is_alive():
            failures.append(RuntimeError("mixed child timeout"))
    if failures:
        raise RuntimeError("mixed diagnostic descendant failure")


def _start_coordinated_replacer(
    logger: logging.Logger,
    ordinary: tuple[object, object, object, object],
    *,
    emit_unrelated: bool,
    pause_thread_start: tuple[threading.Event, threading.Event] | None = None,
) -> types.SimpleNamespace:
    request = threading.Event()
    installed = threading.Event()
    emit_request = threading.Event()
    emitted = threading.Event()
    replacement_stdout = StringIO()
    replacement_stderr = StringIO()
    failures: list[BaseException] = []
    start_calls: list[tuple[int, threading.Thread]] = []
    captured: list[tuple[object, object, object, object]] = []

    def replacement_logger_handle(
        target_logger: logging.Logger,
        record: logging.LogRecord,
    ) -> object:
        return ordinary[2](target_logger, record)  # type: ignore[operator]

    def replacement_thread_start(
        thread: threading.Thread,
        *args: object,
        **kwargs: object,
    ) -> object:
        start_calls.append((threading.get_ident(), thread))
        if pause_thread_start is not None:
            entered, release = pause_thread_start
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("replacement thread-start release timeout")
        return ordinary[3](thread, *args, **kwargs)  # type: ignore[operator]

    def worker() -> None:
        try:
            if not request.wait(timeout=5):
                raise RuntimeError("replacement request timeout")
            captured.append(_diagnostic_surfaces())
            sys.stdout = replacement_stdout
            sys.stderr = replacement_stderr
            logging.Logger.handle = replacement_logger_handle
            threading.Thread.start = replacement_thread_start
            installed.set()
            if emit_unrelated:
                if not emit_request.wait(timeout=5):
                    raise RuntimeError("unrelated emission timeout")
                _emit_test_diagnostics(logger, "unrelated-after-rearm")
        except BaseException as error:
            failures.append(error)
        finally:
            installed.set()
            emitted.set()

    thread = threading.Thread(target=worker)
    controller = types.SimpleNamespace(
        captured=captured,
        emit_request=emit_request,
        emitted=emitted,
        failures=failures,
        installed=installed,
        replacement_logger_handle=replacement_logger_handle,
        replacement_stderr=replacement_stderr,
        replacement_stdout=replacement_stdout,
        replacement_thread_start=replacement_thread_start,
        request=request,
        start_calls=start_calls,
        thread=thread,
    )
    thread.start()
    return controller


def _owned_surface_depths() -> tuple[int, int, int, int]:
    surfaces = _diagnostic_surfaces()
    depths: list[int] = []
    classifiers = (
        resume_rendering._is_owned_stream,
        resume_rendering._is_owned_stream,
        resume_rendering._is_owned_logger_handle,
        resume_rendering._is_owned_thread_start,
    )
    target_names = ("_stream", "_stream", "_target", "_target")
    for surface, classifier, target_name in zip(
        surfaces,
        classifiers,
        target_names,
        strict=True,
    ):
        depth = 0
        seen: set[int] = set()
        while classifier(surface) and id(surface) not in seen:
            seen.add(id(surface))
            depth += 1
            surface = object.__getattribute__(surface, target_name)
        depths.append(depth)
    return tuple(depths)  # type: ignore[return-value]


class _ObservableBuffer:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.line_batches: list[tuple[bytes, ...]] = []
        self.flushes = 0

    def write(self, value: bytes) -> int:
        self.writes.append(value)
        return len(value)

    def writelines(self, values: object) -> None:
        self.line_batches.append(tuple(values))  # type: ignore[arg-type]

    def flush(self) -> None:
        self.flushes += 1

    def contains(self, marker: str) -> bool:
        encoded = marker.encode()
        return any(encoded in value for value in self.writes) or any(
            encoded in value for batch in self.line_batches for value in batch
        )


class _ObservableTextStream:
    encoding = "utf-8"
    errors = "strict"

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.line_batches: list[tuple[str, ...]] = []
        self.flushes = 0
        self.buffer = _ObservableBuffer()

    def write(self, value: str) -> int:
        self.writes.append(value)
        return len(value)

    def writelines(self, values: object) -> None:
        self.line_batches.append(tuple(values))  # type: ignore[arg-type]

    def flush(self) -> None:
        self.flushes += 1

    def isatty(self) -> bool:
        return False

    def contains(self, marker: str) -> bool:
        return any(marker in value for value in self.writes) or any(
            marker in value for batch in self.line_batches for value in batch
        )


_INFLIGHT_DIAGNOSTIC_CASES = (
    ("stdout", "write"),
    ("stdout", "writelines"),
    ("stdout", "flush"),
    ("stderr", "write"),
    ("stderr", "writelines"),
    ("stderr", "flush"),
    ("stdout-buffer", "write"),
    ("stdout-buffer", "writelines"),
    ("stdout-buffer", "flush"),
    ("stderr-buffer", "write"),
    ("stderr-buffer", "writelines"),
    ("stderr-buffer", "flush"),
    ("logger", "handle"),
    ("target-thread", "start"),
    ("run-thread", "start"),
)


def _prepare_inflight_diagnostic_operation(
    hooks: object,
    case: tuple[str, str],
    logger: logging.Logger,
    child_ownership: list[frozenset[object]],
) -> types.SimpleNamespace:
    surface_name, method_name = case
    marker = f"owned-inflight-{surface_name}-{method_name}"

    if surface_name in {"stdout", "stderr"}:
        target = (
            hooks.stdout_proxy  # type: ignore[attr-defined]
            if surface_name == "stdout"
            else hooks.stderr_proxy  # type: ignore[attr-defined]
        )
        if method_name == "write":
            value = marker

            def invoke() -> object:
                return target.write(value)

            expected_result: object = len(value)
        elif method_name == "writelines":
            values = (f"{marker}-one", f"{marker}-two")

            def invoke() -> object:
                return target.writelines(values)

            expected_result = None
        else:

            def invoke() -> object:
                return target.flush()

            expected_result = None
        return types.SimpleNamespace(
            case=case,
            child=None,
            expected_result=expected_result,
            gate_thread=None,
            gate_wrapper=target,
            invoke=invoke,
            marker=marker,
        )

    if surface_name in {"stdout-buffer", "stderr-buffer"}:
        owner = (
            hooks.stdout_proxy  # type: ignore[attr-defined]
            if surface_name == "stdout-buffer"
            else hooks.stderr_proxy  # type: ignore[attr-defined]
        )
        target = owner.buffer
        if method_name == "write":
            value = marker.encode()

            def invoke() -> object:
                return target.write(value)

            expected_result = len(value)
        elif method_name == "writelines":
            values = (f"{marker}-one".encode(), f"{marker}-two".encode())

            def invoke() -> object:
                return target.writelines(values)

            expected_result = None
        else:

            def invoke() -> object:
                return target.flush()

            expected_result = None
        return types.SimpleNamespace(
            case=case,
            child=None,
            expected_result=expected_result,
            gate_thread=None,
            gate_wrapper=owner,
            invoke=invoke,
            marker=marker,
        )

    if surface_name == "logger":
        record = logger.makeRecord(
            logger.name,
            logging.DEBUG,
            __file__,
            1,
            marker,
            (),
            None,
        )

        def invoke() -> object:
            return hooks.logger_handle(logger, record)  # type: ignore[attr-defined]

        return types.SimpleNamespace(
            case=case,
            child=None,
            expected_result=None,
            gate_thread=None,
            gate_wrapper=hooks.logger_handle,  # type: ignore[attr-defined]
            invoke=invoke,
            marker=marker,
        )

    def child_body() -> None:
        with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
            child_ownership.append(
                frozenset(
                    resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                        id(threading.current_thread()),
                        set(),
                    ),
                ),
            )
        _emit_test_diagnostics(logger, marker)

    if surface_name == "target-thread":
        child = threading.Thread(target=child_body)
    else:

        class InflightRunThread(threading.Thread):
            def run(self) -> None:
                child_body()

        child = InflightRunThread()

    def invoke() -> object:
        result = child.start()
        child.join(timeout=5)
        if child.is_alive():
            raise RuntimeError("in-flight child timeout")
        return result

    return types.SimpleNamespace(
        case=case,
        child=child,
        expected_result=None,
        gate_thread=child,
        gate_wrapper=hooks.thread_start,  # type: ignore[attr-defined]
        invoke=invoke,
        marker=marker,
    )


def _assert_marker_absent_from_probes(
    marker: str,
    *streams: _ObservableTextStream,
) -> None:
    for stream in streams:
        assert stream.contains(marker) is False
        assert stream.buffer.contains(marker) is False


def _assert_operation_delegated_once(
    operation: types.SimpleNamespace,
    stdout: _ObservableTextStream,
    stderr: _ObservableTextStream,
    messages: list[str],
) -> None:
    surface_name, method_name = operation.case
    marker = operation.marker
    if surface_name in {"target-thread", "run-thread"}:
        assert stdout.writes.count(f"{marker}-stdout") == 1
        assert stderr.writes.count(f"{marker}-stderr") == 1
        assert messages.count(f"{marker}-log") == 1
        return
    if surface_name == "logger":
        assert messages.count(marker) == 1
        return

    stream = stdout if surface_name.startswith("stdout") else stderr
    target = stream.buffer if surface_name.endswith("-buffer") else stream
    if method_name == "write":
        value = marker.encode() if surface_name.endswith("-buffer") else marker
        assert target.writes.count(value) == 1
    elif method_name == "writelines":
        values = (
            (f"{marker}-one".encode(), f"{marker}-two".encode())
            if surface_name.endswith("-buffer")
            else (f"{marker}-one", f"{marker}-two")
        )
        assert target.line_batches.count(values) == 1
    else:
        assert target.flushes == 1


def _assert_operation_not_delegated(
    operation: types.SimpleNamespace,
    stdout: _ObservableTextStream,
    stderr: _ObservableTextStream,
    messages: list[str],
) -> None:
    surface_name, method_name = operation.case
    marker = operation.marker
    if surface_name in {"target-thread", "run-thread"}:
        assert stdout.writes.count(f"{marker}-stdout") == 0
        assert stderr.writes.count(f"{marker}-stderr") == 0
        assert messages.count(f"{marker}-log") == 0
        return
    if surface_name == "logger":
        assert messages.count(marker) == 0
        return

    stream = stdout if surface_name.startswith("stdout") else stderr
    target = stream.buffer if surface_name.endswith("-buffer") else stream
    if method_name == "write":
        value = marker.encode() if surface_name.endswith("-buffer") else marker
        assert target.writes.count(value) == 0
    elif method_name == "writelines":
        values = (
            (f"{marker}-one".encode(), f"{marker}-two".encode())
            if surface_name.endswith("-buffer")
            else (f"{marker}-one", f"{marker}-two")
        )
        assert target.line_batches.count(values) == 0
    else:
        assert target.flushes == 0


def _fictional_resume() -> dict[str, object]:
    return {
        "header_top": {
            "line_1_name_header_text": "Jules Example",
            "line_2_header_text": "Public-safe systems engineer",
            "contact_items": [
                "jules@example.test",
                "Portfolio",
                "Unsafe",
            ],
            "links": [
                {
                    "label": "jules@example.test",
                    "url": "mailto:jules@example.test",
                },
                {
                    "label": "Portfolio",
                    "url": "https://portfolio.example.test/?a=1&b=2",
                },
                {"label": "Unsafe", "url": "javascript:private-value"},
            ],
        },
        "professional_summary": {
            "paragraph": (
                "Builds <strong>bounded tools</strong> for fictional teams. "
                '<a href="https://summary.example.test/a?x=1&y=2">Sample</a> '
                '<a href="data:text/plain,private-value">Unsafe data</a>.'
            )
        },
        "core_technical_skills": {
            "bullet_points": [
                {
                    "category": "Languages",
                    "items": {
                        "primary": ["Python", "SQL"],
                        "additional": ["Rust"],
                        "match_terms": {"Rust": ["rustlang"]},
                    },
                    "jod_matched_items": ["rustlang"],
                }
            ]
        },
        "professional_experience": {
            "jobs": [
                {
                    "line_1": {
                        "company_name_text": "Example Systems Cooperative",
                        "position_name_text": "Systems Engineer",
                        "position_dates_text": "2024–Present",
                    },
                    "line_2": {"position_intro_text": "Fictional public-safe work."},
                    "bullet_points": [
                        "Improved a synthetic workflow by 25%.",
                    ],
                }
            ]
        },
        "education": {
            "entries": [
                {
                    "line_1": {"institution_name_text": "Example Technical Institute"},
                    "line_2": {"degree_name_text": "B.S., Example Studies"},
                }
            ]
        },
        "certifications": {"bullet_points": ["Synthetic Systems Certificate"]},
        "portfolio": {
            "projects": [
                {
                    "title": "Example Project",
                    "url": "https://project.example.test",
                    "description": "A fictional public demonstration.",
                }
            ]
        },
    }


def _assert_stable_error(
    captured: pytest.ExceptionInfo[ResumeRenderingError],
    message: str,
) -> None:
    error = captured.value
    assert str(error) == message
    assert error.__cause__ is None
    assert error.__context__ is None


def test_packaged_template_renders_fictional_mapping_with_safe_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    rendered = render_resume_html_from_mapping(resume=_fictional_resume())

    assert set(tmp_path.iterdir()) == before
    assert "Jules Example" in rendered
    assert "Example Systems Cooperative" in rendered
    assert 'href="mailto:jules@example.test"' in rendered
    assert 'href="https://portfolio.example.test/?a=1&amp;b=2"' in rendered
    assert 'href="https://summary.example.test/a?x=1&amp;y=2"' in rendered
    assert 'href="https://project.example.test"' in rendered
    assert "<b>bounded tools</b>" in rendered
    assert "javascript:" not in rendered
    assert "data:text/plain" not in rendered


@pytest.mark.parametrize(
    "value",
    [
        "javascript:private-value",
        "data:text/plain,private-value",
        "//private.example.test/path",
        "HTTPS://safe.example.test/path",
        " https://safe.example.test/path",
        "https ://safe.example.test/path",
        "java\tscript:private-value",
        "java\x00script:private-value",
        'https://safe.example.test/"onload="private-value',
        "ftp://safe.example.test/file",
        "mailto:not-an-address",
        "https://safe.example.test:",
        "https://safe.example.test\\private",
        "https://safe.example.test/%",
        "https://safe.example.test/%0",
        "https://safe.example.test/%GG",
        "https://safe.example.test/%00",
        "https://safe.example.test/%1F",
        "https://safe.example.test/%7f",
        "https://safe.example.test/%C2%80",
        "https://safe.example.test/%C2%9F",
        "mailto:jules%0A@example.test",
    ],
)
def test_shared_url_policy_rejects_unsafe_and_obfuscated_values(value: str) -> None:
    assert sanitize_resume_url(value) == ""
    if value not in {
        " https://safe.example.test/path",
        "https://safe.example.test:",
    }:
        assert 'href="' not in str(linkify(value))
    assert 'href="' not in str(rich_text(f'<a href="{value}">label</a>'))


@pytest.mark.parametrize(
    "value",
    [
        "http://safe.example.test/path",
        "https://safe.example.test/a?x=1&y=2",
        "https://safe.example.test/a%20b",
        "https://safe.example.test/%2F",
        "https://safe.example.test/%E2%9C%93",
        "mailto:jules@example.test",
        "mailto:jules%2Btag@example.test",
    ],
)
def test_shared_url_policy_retains_and_escapes_clean_values(value: str) -> None:
    assert sanitize_resume_url(value) == value
    linked = str(linkify(value))
    assert 'href="' in linked
    if "&" in value:
        assert "&amp;" in linked


@pytest.mark.parametrize(
    "value",
    [
        "https://safe.example.test/%0A",
        "https://safe.example.test/%C2%85",
        "https://safe.example.test/%",
        "mailto:jules%0A@example.test",
    ],
)
def test_template_header_uses_percent_encoding_safety_policy(value: str) -> None:
    rendered = render_resume_html_from_mapping(
        resume={
            "header_top": {
                "line_1_name_header_text": "Jules Example",
                "contact_items": ["Contact"],
                "links": [{"label": "Contact", "url": value}],
            }
        }
    )

    assert 'href="' not in rendered
    assert "Contact" in rendered


def test_skill_rows_keep_primary_alias_order_deduplication_and_eight_matches() -> None:
    additional = [f"Additional {index}" for index in range(1, 11)]
    value = {
        "bullet_points": [
            {
                "category": "Tools",
                "items": {
                    "primary": ["Python", "SQL", "Python 3", "C++", "C#"],
                    "additional": additional,
                    "match_terms": {"Additional 2": ["alias-two"]},
                },
                "jod_matched_items": [
                    "alias-two",
                    "Additional 1",
                    "Additional 3",
                    "Additional 4",
                    "Additional 5",
                    "Additional 6",
                    "Additional 7",
                    "Additional 8",
                    "Additional 9",
                    "Additional 10",
                ],
            },
            {
                "category": "Repeated",
                "items": ["SQL", "New Skill"],
            },
        ]
    }

    rows = render_core_skill_rows(value)

    assert rows[0] == {
        "category": "Tools",
        "text": (
            "Python, SQL, C++, C#, Additional 2, Additional 1, "
            "Additional 3, Additional 4, Additional 5, Additional 6, "
            "Additional 7, Additional 8"
        ),
    }
    assert rows[1] == {"category": "Repeated", "text": "New Skill"}
    assert "Additional 9" not in rows[0]["text"]
    assert "Additional 10" not in rows[0]["text"]


def test_exact_yaml_and_exact_template_override_are_honored(tmp_path: Path) -> None:
    yaml_path = tmp_path / "resume-input.yml"
    yaml_path.write_text(yaml.safe_dump(_fictional_resume()), encoding="utf-8")
    template_path = tmp_path / "only-this-template.j2"
    template_path.write_text(
        "<p>{{ data.header_top.line_1_name_header_text }}</p>",
        encoding="utf-8",
    )

    rendered = render_resume_html(
        yaml_path=yaml_path,
        template_path=template_path,
    )

    assert rendered == "<p>Jules Example</p>"


@pytest.mark.parametrize(
    "directive",
    [
        '{% include "private-sibling.j2" %}',
        '{% import "private-sibling.j2" as private %}',
        '{% extends "private-sibling.j2" %}',
    ],
)
def test_override_has_no_loader_for_include_import_or_parent_search(
    tmp_path: Path,
    directive: str,
) -> None:
    sibling = tmp_path / "private-sibling.j2"
    sibling.write_text("private-value", encoding="utf-8")
    override = tmp_path / "override.j2"
    override.write_text(directive, encoding="utf-8")

    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_html_from_mapping(
            resume=_fictional_resume(),
            template_path=override,
        )

    _assert_stable_error(captured, "Resume HTML could not be rendered.")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("- not\n- a\n- mapping\n", "Resume YAML must contain a mapping."),
        (
            "value: !!python/object/apply:builtins.print [private-value]\n",
            "Resume YAML could not be loaded.",
        ),
        ("value: [unterminated\n", "Resume YAML could not be loaded."),
    ],
)
def test_yaml_failures_are_stable_silent_and_content_free(
    tmp_path: Path,
    payload: str,
    message: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    yaml_path = tmp_path / "private-name.yml"
    yaml_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_html(yaml_path=yaml_path)

    _assert_stable_error(captured, message)
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert caplog.text == ""
    assert "private-value" not in str(captured.value)
    assert str(yaml_path) not in str(captured.value)


def test_yaml_template_and_rendered_html_limits_fail_closed(tmp_path: Path) -> None:
    yaml_path = tmp_path / "large.yml"
    yaml_path.write_bytes(b"x" * (MAX_RESUME_YAML_BYTES + 1))
    with pytest.raises(ResumeRenderingError) as yaml_error:
        render_resume_html(yaml_path=yaml_path)
    _assert_stable_error(
        yaml_error,
        "Resume YAML exceeds the supported size.",
    )

    template_path = tmp_path / "large.j2"
    template_path.write_text("x" * (MAX_TEMPLATE_CHARS + 1), encoding="utf-8")
    with pytest.raises(ResumeRenderingError) as template_error:
        render_resume_html_from_mapping(
            resume=_fictional_resume(),
            template_path=template_path,
        )
    _assert_stable_error(
        template_error,
        "Resume template exceeds the supported size.",
    )

    expanding_template = tmp_path / "expanding.j2"
    expanding_template.write_text(
        "{{ data.payload }}{{ data.payload }}",
        encoding="utf-8",
    )
    with pytest.raises(ResumeRenderingError) as html_error:
        render_resume_html_from_mapping(
            resume={"payload": "x" * (MAX_RENDERED_HTML_CHARS // 2 + 1)},
            template_path=expanding_template,
        )
    _assert_stable_error(
        html_error,
        "Resume HTML exceeds the supported size.",
    )


def test_jinja_generation_stops_at_the_html_limit(tmp_path: Path) -> None:
    template_path = tmp_path / "streaming.j2"
    template_path.write_text(
        "{% for chunk in data.chunks %}{{ chunk }}{{ chunk }}{% endfor %}"
        "{{ 1 / data.zero }}",
        encoding="utf-8",
    )

    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_html_from_mapping(
            resume={"chunks": ["x" * 100_001 for _ in range(10)], "zero": 0},
            template_path=template_path,
        )

    _assert_stable_error(
        captured,
        "Resume HTML exceeds the supported size.",
    )


def test_yaml_and_template_reads_are_bounded_binary_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_sizes: list[int] = []

    class Reader:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return self._payload[:size]

    def open_bounded(path: Path, mode: str) -> Reader:
        assert mode == "rb"
        payload = b"{}\n" if path.suffix == ".yml" else b"<p>bounded</p>"
        return Reader(payload)

    monkeypatch.setattr(Path, "open", open_bounded)

    assert resume_rendering.load_resume(Path("resume.yml")) == {}
    assert (
        render_resume_html_from_mapping(
            resume={},
            template_path=Path("template.j2"),
        )
        == "<p>bounded</p>"
    )
    assert read_sizes == [
        MAX_RESUME_YAML_BYTES + 1,
        MAX_TEMPLATE_CHARS * 4 + 1,
    ]


def test_invalid_paths_and_resource_failures_are_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ResumeRenderingError) as yaml_error:
        render_resume_html(yaml_path=tmp_path / "private-resume.yml")
    _assert_stable_error(yaml_error, "Resume YAML could not be loaded.")

    with pytest.raises(ResumeRenderingError) as template_error:
        render_resume_html_from_mapping(
            resume=_fictional_resume(),
            template_path=tmp_path / "private-template.j2",
        )
    _assert_stable_error(
        template_error,
        "Resume template could not be loaded.",
    )

    def fail_resource(_package: str) -> object:
        raise RuntimeError("private-resource-detail")

    monkeypatch.setattr(resume_rendering.resources, "files", fail_resource)
    with pytest.raises(ResumeRenderingError) as resource_error:
        render_resume_html_from_mapping(resume=_fictional_resume())
    _assert_stable_error(
        resource_error,
        "Resume template could not be loaded.",
    )


def test_fresh_import_and_packaged_resource_access_create_no_state(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = (
        "from career_agent_workbench.resume_rendering import "
        "render_resume_html_from_mapping\n"
        "result = render_resume_html_from_mapping("
        "resume={'header_top': {'line_1_name_header_text': 'Jules Example'}})\n"
        "assert 'Jules Example' in result\n"
    )
    environment = {
        "PATH": os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(source_root),
    }
    before = set(tmp_path.iterdir())

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert set(tmp_path.iterdir()) == before


def test_invalid_input_types_and_pdf_html_bound_fail_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ResumeRenderingError) as yaml_error:
        render_resume_html(yaml_path="private-name.yml")  # type: ignore[arg-type]
    _assert_stable_error(yaml_error, "Resume YAML could not be loaded.")

    with pytest.raises(ResumeRenderingError) as template_error:
        render_resume_html_from_mapping(
            resume=_fictional_resume(),
            template_path="private-template.j2",  # type: ignore[arg-type]
        )
    _assert_stable_error(
        template_error,
        "Resume template could not be loaded.",
    )

    browser_called = False

    def observe_browser(_html: str) -> bytes:
        nonlocal browser_called
        browser_called = True
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        observe_browser,
    )
    with pytest.raises(ResumeRenderingError) as html_error:
        render_resume_pdf_from_html("x" * (MAX_RENDERED_HTML_CHARS + 1))
    _assert_stable_error(
        html_error,
        "Resume HTML exceeds the supported size.",
    )
    assert browser_called is False


def test_capability_objects_and_local_paths_never_reach_override_jinja(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Capability:
        def __init__(self) -> None:
            self.called = False

        def touch(self) -> str:
            self.called = True
            return "private-capability-value"

    capability = Capability()
    capability_template = tmp_path / "capability.j2"
    capability_template.write_text(
        "{{ data.probe.touch() }}",
        encoding="utf-8",
    )
    with pytest.raises(ResumeRenderingError) as capability_error:
        render_resume_html_from_mapping(
            resume={"probe": capability},
            template_path=capability_template,
        )
    _assert_stable_error(
        capability_error,
        "Resume data contains unsupported values.",
    )
    assert capability.called is False

    local_path = tmp_path / "local-value.txt"
    local_path.write_text("private-local-value", encoding="utf-8")
    path_template = tmp_path / "path.j2"
    path_template.write_text(
        "{{ data.local_path.read_text() }}",
        encoding="utf-8",
    )
    path_read_called = False
    original_read_text = Path.read_text

    def observe_read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal path_read_called
        if path == local_path:
            path_read_called = True
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observe_read_text)
    with pytest.raises(ResumeRenderingError) as path_error:
        render_resume_html_from_mapping(
            resume={"local_path": local_path},
            template_path=path_template,
        )
    _assert_stable_error(
        path_error,
        "Resume data contains unsupported values.",
    )
    assert path_read_called is False


def test_resume_data_materialization_rejects_cycles_and_iteration_failures(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "plain.j2"
    template_path.write_text("bounded", encoding="utf-8")
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(ResumeRenderingError) as cycle_error:
        render_resume_html_from_mapping(
            resume={"cycle": cyclic},
            template_path=template_path,
        )
    _assert_stable_error(
        cycle_error,
        "Resume data contains unsupported values.",
    )

    class FailingMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError(key)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("private-iteration-detail")

        def __len__(self) -> int:
            return 1

    with pytest.raises(ResumeRenderingError) as iteration_error:
        render_resume_html_from_mapping(
            resume=FailingMapping(),
            template_path=template_path,
        )
    _assert_stable_error(
        iteration_error,
        "Resume data contains unsupported values.",
    )


def test_active_mappings_and_spoofed_class_are_rejected_without_callbacks(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-active-mapping")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    marker = tmp_path / "synthetic-callback-marker.txt"
    marker.write_text("synthetic-callback-value", encoding="utf-8")
    template_path = tmp_path / "plain.j2"
    template_path.write_text("bounded", encoding="utf-8")
    callbacks: list[str] = []

    def activate(name: str) -> None:
        callbacks.append(name)
        value = marker.read_text(encoding="utf-8")
        print(value)
        print(value, file=sys.stderr)
        logger.debug("%s:%s", name, value)

    class ActiveMapping(Mapping[str, object]):
        @property
        def __class__(self) -> type[dict[str, object]]:
            activate("__class__")
            return dict

        def items(self) -> object:
            activate("items")
            return {}.items()

        def __getitem__(self, key: str) -> object:
            activate(f"getitem:{key}")
            return "value"

        def __iter__(self) -> Iterator[str]:
            activate("iter")
            return iter(("value",))

        def __len__(self) -> int:
            activate("len")
            return 1

    class SpoofedClass:
        @property
        def __class__(self) -> type[dict[str, object]]:
            activate("spoofed-class")
            return dict

    class SpoofingMeta(type):
        @property
        def __mro__(cls) -> tuple[type[object], ...]:
            activate("metaclass-mro")
            return ()

        def __instancecheck__(cls, instance: object) -> bool:
            activate("metaclass-instancecheck")
            return True

        def __subclasscheck__(cls, subclass: type[object]) -> bool:
            activate("metaclass-subclasscheck")
            return True

    class MetaclassProbe(metaclass=SpoofingMeta):
        pass

    callbacks.clear()
    capsys.readouterr()
    probes: list[object] = [
        ActiveMapping(),
        {"nested": ActiveMapping()},
        SpoofedClass(),
        {"nested": SpoofedClass()},
        MetaclassProbe(),
        {"nested": MetaclassProbe()},
    ]
    for probe in probes:
        with pytest.raises(ResumeRenderingError) as captured:
            render_resume_html_from_mapping(
                resume=probe,  # type: ignore[arg-type]
                template_path=template_path,
            )
        _assert_stable_error(
            captured,
            "Resume data contains unsupported values.",
        )

    captured_io = capsys.readouterr()
    assert callbacks == []
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert caplog.text == ""


def test_dict_subclasses_use_only_builtin_inert_operations(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-dict-subclass")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    marker = tmp_path / "synthetic-dict-marker.txt"
    marker.write_text("synthetic-dict-value", encoding="utf-8")
    template_path = tmp_path / "dict-subclass.j2"
    template_path.write_text(
        "{{ data.safe }}|{{ data.nested.value }}",
        encoding="utf-8",
    )
    callbacks: list[str] = []

    def activate(name: str) -> None:
        callbacks.append(name)
        value = marker.read_text(encoding="utf-8")
        print(value)
        print(value, file=sys.stderr)
        logger.debug("%s:%s", name, value)

    class SpoofingDictMeta(type):
        @property
        def __mro__(cls) -> tuple[type[object], ...]:
            activate("metaclass-mro")
            return ()

        def __instancecheck__(cls, instance: object) -> bool:
            activate("metaclass-instancecheck")
            return False

        def __subclasscheck__(cls, subclass: type[object]) -> bool:
            activate("metaclass-subclasscheck")
            return False

    class InertDictSubclass(dict[str, object], metaclass=SpoofingDictMeta):
        @property
        def __class__(self) -> type[object]:
            activate("__class__")
            return object

        def __call__(self) -> None:
            activate("call")

        def items(self) -> object:
            activate("items")
            return {}.items()

        def __iter__(self) -> Iterator[str]:
            activate("iter")
            return iter(())

        def __getitem__(self, key: str) -> object:
            activate(f"getitem:{key}")
            return "active"

        def __len__(self) -> int:
            activate("len")
            return 0

    nested = InertDictSubclass()
    dict.__setitem__(nested, "value", "Nested")
    resume = InertDictSubclass()
    dict.__setitem__(resume, "safe", "Synthetic")
    dict.__setitem__(resume, "nested", nested)

    callbacks.clear()
    capsys.readouterr()
    rendered = render_resume_html_from_mapping(
        resume=resume,
        template_path=template_path,
    )
    normal_rendered = render_resume_html_from_mapping(
        resume={"safe": "Normal", "nested": {"value": "Dictionary"}},
        template_path=template_path,
    )

    captured_io = capsys.readouterr()
    assert rendered == "Synthetic|Nested"
    assert normal_rendered == "Normal|Dictionary"
    assert callbacks == []
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert caplog.text == ""


def test_resume_data_materialization_enforces_depth_node_and_size_bounds(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "plain.j2"
    template_path.write_text("bounded", encoding="utf-8")
    deeply_nested: dict[str, object] = {}
    cursor = deeply_nested
    for _ in range(MAX_RESUME_DATA_DEPTH + 1):
        child: dict[str, object] = {}
        cursor["next"] = child
        cursor = child

    values = [
        deeply_nested,
        {"nodes": [None] * MAX_RESUME_DATA_NODES},
        {"large": "x" * (MAX_RESUME_DATA_CHARS + 1)},
    ]
    for value in values:
        with pytest.raises(ResumeRenderingError) as captured:
            render_resume_html_from_mapping(
                resume={"value": value},
                template_path=template_path,
            )
        _assert_stable_error(
            captured,
            "Resume data contains unsupported values.",
        )


def test_reportlab_fallback_returns_parseable_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_browser(_html: str) -> bytes:
        raise RuntimeError("private-browser-detail")

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        fail_browser,
    )
    pdf_bytes = render_resume_pdf_from_html(
        "<html><body><h1>Jules Example</h1><p>Synthetic resume.</p></body></html>"
    )

    assert pdf_bytes.startswith(b"%PDF")
    reader = PdfReader(resume_rendering.BytesIO(pdf_bytes))
    assert len(reader.pages) == 1
    assert "Jules Example" in (reader.pages[0].extract_text() or "")
    job_description = (
        "A fictional team requires Python automation and reliable systems."
    )
    score = calculate_ats_proxy_score(
        resume_pdf=pdf_bytes,
        job_description=job_description,
    )
    diagnostics = calculate_ats_diagnostics(
        resume_pdf=pdf_bytes,
        job_description=job_description,
    )
    assert diagnostics.score == score
    assert score.parsing_score > 0


def test_empty_reportlab_fallback_does_not_invent_resume_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_browser(_html: str) -> bytes:
        raise RuntimeError("private-browser-detail")

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        fail_browser,
    )

    pdf_bytes = render_resume_pdf_from_html("")
    reader = PdfReader(resume_rendering.BytesIO(pdf_bytes))

    assert len(reader.pages) == 1
    assert (reader.pages[0].extract_text() or "").strip() == ""


def test_valid_one_page_browser_pdf_returns_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_called = False

    def fail_if_fallback_runs(_html: str) -> bytes:
        nonlocal fallback_called
        fallback_called = True
        raise AssertionError("fallback should not run")

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: _VALID_ONE_PAGE_PDF,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        fail_if_fallback_runs,
    )

    result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")

    assert result == _VALID_ONE_PAGE_PDF
    assert len(PdfReader(BytesIO(result)).pages) == 1
    assert fallback_called is False


@pytest.mark.parametrize(
    "invalid_kind",
    ["empty", "malformed", "oversized", "fifty-one-pages"],
)
def test_invalid_browser_pdf_uses_validated_fallback(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    if invalid_kind == "empty":
        browser_result = b""
    elif invalid_kind == "malformed":
        browser_result = b"%PDF-synthetic-malformed"
    elif invalid_kind == "oversized":
        browser_result = b"x" * (MAX_RESUME_PDF_BYTES + 1)
    else:
        browser_result = _synthetic_pdf_bytes(MAX_RESUME_PDF_PAGES + 1)
    fallback_calls = 0

    def valid_fallback(_html: str) -> bytes:
        nonlocal fallback_calls
        fallback_calls += 1
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: browser_result,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        valid_fallback,
    )

    result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")

    assert result == _VALID_ONE_PAGE_PDF
    assert fallback_calls == 1


def test_zero_page_browser_pdf_uses_one_page_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_calls = 0

    def valid_fallback(_html: str) -> bytes:
        nonlocal fallback_calls
        fallback_calls += 1
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: _synthetic_pdf_bytes(0),
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        valid_fallback,
    )

    result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")

    assert result == _VALID_ONE_PAGE_PDF
    assert fallback_calls == 1
    assert len(PdfReader(BytesIO(result)).pages) == 1


def test_zero_page_final_pdf_fails_with_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: (_ for _ in ()).throw(RuntimeError("browser detail")),
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: _synthetic_pdf_bytes(0),
    )

    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_pdf_from_html("<p>Synthetic resume.</p>")

    _assert_stable_error(captured, "Resume PDF could not be rendered.")


def test_generated_pdf_page_limit_accepts_fifty_and_rejects_fifty_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fifty_page_pdf = _synthetic_pdf_bytes(MAX_RESUME_PDF_PAGES)
    fifty_one_page_pdf = _synthetic_pdf_bytes(MAX_RESUME_PDF_PAGES + 1)

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: fifty_page_pdf,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: (_ for _ in ()).throw(
            AssertionError("fallback should not run"),
        ),
    )
    accepted = render_resume_pdf_from_html("<p>Synthetic resume.</p>")
    assert accepted == fifty_page_pdf
    assert len(PdfReader(BytesIO(accepted)).pages) == MAX_RESUME_PDF_PAGES

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: (_ for _ in ()).throw(RuntimeError("browser failure")),
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: fifty_one_page_pdf,
    )
    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_pdf_from_html("<p>Synthetic resume.</p>")
    _assert_stable_error(captured, "Resume PDF could not be rendered.")


def test_real_reportlab_output_above_page_limit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounded_html = "".join("<p>Synthetic line</p>" for _index in range(2_800))
    assert len(bounded_html) < MAX_RENDERED_HTML_CHARS
    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: (_ for _ in ()).throw(RuntimeError("browser unavailable")),
    )

    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_pdf_from_html(bounded_html)

    _assert_stable_error(captured, "Resume PDF could not be rendered.")


def test_pdf_validation_diagnostics_are_silent_and_content_free(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-pdf-validation")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    def noisy_reader(*_args: object, **_kwargs: object) -> object:
        _emit_test_diagnostics(logger, "synthetic-parser")
        raise RuntimeError("synthetic parser detail")

    monkeypatch.setattr(resume_rendering, "PdfReader", noisy_reader)
    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: _VALID_ONE_PAGE_PDF,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: _VALID_ONE_PAGE_PDF,
    )

    capsys.readouterr()
    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_pdf_from_html("<p>Synthetic resume.</p>")

    _assert_stable_error(captured, "Resume PDF could not be rendered.")
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert caplog.text == ""


class _FakeRoute:
    def __init__(self, events: list[object], url: str) -> None:
        self._events = events
        self._url = url

    def abort(self) -> None:
        self._events.append(("abort", self._url))


class _FakePage:
    def __init__(self, events: list[object], *, fail_stage: str | None) -> None:
        self._events = events
        self._handler: object | None = None
        self._fail_stage = fail_stage

    def route(self, pattern: str, handler: object) -> None:
        self._events.append(("route", pattern))
        self._handler = handler
        if self._fail_stage == "route":
            raise RuntimeError("private-route-detail")

    def set_content(self, html: str, *, wait_until: str) -> None:
        self._events.append(("set_content", html, wait_until))
        assert callable(self._handler)
        for scheme in ("file", "http", "https", "ftp", "ws", "wss"):
            self._handler(_FakeRoute(self._events, f"{scheme}://blocked.example.test"))
        if self._fail_stage == "set_content":
            raise RuntimeError("private-content-detail")

    def pdf(self, **options: object) -> bytes:
        self._events.append(("pdf", options))
        if self._fail_stage == "pdf":
            raise RuntimeError("private-pdf-detail")
        return _VALID_ONE_PAGE_PDF

    def close(self) -> None:
        self._events.append("page.close")
        if self._fail_stage == "page.close":
            raise RuntimeError("private-page-close-detail")


class _FakeContext:
    def __init__(self, events: list[object], *, fail_stage: str | None) -> None:
        self._events = events
        self._fail_stage = fail_stage

    def new_page(self) -> _FakePage:
        self._events.append("new_page")
        if self._fail_stage == "new_page":
            raise RuntimeError("private-new-page-detail")
        return _FakePage(self._events, fail_stage=self._fail_stage)

    def close(self) -> None:
        self._events.append("context.close")
        if self._fail_stage == "context.close":
            raise RuntimeError("private-context-close-detail")


class _FakeBrowser:
    def __init__(self, events: list[object], *, fail_stage: str | None) -> None:
        self._events = events
        self._fail_stage = fail_stage

    def new_context(self, **options: object) -> _FakeContext:
        self._events.append(("new_context", options))
        if self._fail_stage == "new_context":
            raise RuntimeError("private-new-context-detail")
        return _FakeContext(self._events, fail_stage=self._fail_stage)

    def close(self) -> None:
        self._events.append("browser.close")
        if self._fail_stage == "browser.close":
            raise RuntimeError("private-browser-close-detail")


class _FakeChromium:
    def __init__(self, events: list[object], *, fail_stage: str | None) -> None:
        self._events = events
        self._fail_stage = fail_stage

    def launch(self) -> _FakeBrowser:
        self._events.append("launch")
        if self._fail_stage == "launch":
            raise RuntimeError("private-launch-detail")
        return _FakeBrowser(self._events, fail_stage=self._fail_stage)


class _FakePlaywright:
    def __init__(self, events: list[object], *, fail_stage: str | None) -> None:
        self.chromium = _FakeChromium(events, fail_stage=fail_stage)


class _FakePlaywrightManager:
    def __init__(self, events: list[object], *, fail_stage: str | None) -> None:
        self._events = events
        self._fail_stage = fail_stage
        self._playwright = _FakePlaywright(events, fail_stage=fail_stage)

    def __enter__(self) -> _FakePlaywright:
        self._events.append("manager.enter")
        return self._playwright

    def __exit__(self, *args: object) -> None:
        self._events.append("manager.exit")
        if self._fail_stage == "manager.exit":
            raise RuntimeError("private-manager-exit-detail")


def _install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch,
    events: list[object],
    *,
    fail_stage: str | None = None,
) -> None:
    package = types.ModuleType("playwright")
    package.__path__ = []  # type: ignore[attr-defined]
    sync_api = types.ModuleType("playwright.sync_api")

    def sync_playwright() -> _FakePlaywrightManager:
        return _FakePlaywrightManager(events, fail_stage=fail_stage)

    sync_api.sync_playwright = sync_playwright  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)


def test_mocked_browser_is_isolated_aborts_requests_and_closes_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_playwright(monkeypatch, events)
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    html = (
        "<html><body><p>Jules Example</p>"
        '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yw=">'
        "</body></html>"
    )

    result = render_resume_pdf_from_html(html)

    assert result == _VALID_ONE_PAGE_PDF
    assert set(tmp_path.iterdir()) == before
    context_event = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "new_context"
    )
    assert context_event == (
        "new_context",
        {"java_script_enabled": False, "service_workers": "block"},
    )
    assert events.index(("route", "**/*")) < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "set_content"
    )
    aborted = [
        event[1] for event in events if isinstance(event, tuple) and event[0] == "abort"
    ]
    assert aborted == [
        f"{scheme}://blocked.example.test"
        for scheme in ("file", "http", "https", "ftp", "ws", "wss")
    ]
    assert events[-4:] == [
        "page.close",
        "context.close",
        "browser.close",
        "manager.exit",
    ]


@pytest.mark.parametrize(
    "fail_stage",
    [
        "launch",
        "new_context",
        "new_page",
        "route",
        "set_content",
        "pdf",
        "page.close",
        "context.close",
        "browser.close",
        "manager.exit",
    ],
)
def test_mocked_browser_failures_close_created_resources_then_use_fallback(
    monkeypatch: pytest.MonkeyPatch,
    fail_stage: str,
) -> None:
    events: list[object] = []
    _install_fake_playwright(monkeypatch, events, fail_stage=fail_stage)

    result = render_resume_pdf_from_html("<p>Jules Example</p>")

    assert result.startswith(b"%PDF")
    assert result != _VALID_ONE_PAGE_PDF
    assert "manager.exit" in events
    if fail_stage != "launch":
        assert "browser.close" in events
    if fail_stage not in {"launch", "new_context"}:
        assert "context.close" in events
    if fail_stage not in {"launch", "new_context", "new_page"}:
        assert "page.close" in events


def test_backend_diagnostic_suppression_is_thread_selective_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-rendering-concurrency")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    logger_handle_before = logging.Logger.handle
    thread_start_before = threading.Thread.start
    root_handlers_before = tuple(logging.getLogger().handlers)
    root_filters_before = tuple(logging.getLogger().filters)
    worker_start = threading.Event()
    worker_done = threading.Event()
    worker_failures: list[BaseException] = []

    def unrelated_worker() -> None:
        try:
            assert worker_start.wait(timeout=5)
            _emit_test_diagnostics(logger, "unrelated-root")
            _spawn_joined_diagnostic_tree(logger, "unrelated-tree")
        except Exception as error:  # noqa: BLE001 - relay worker assertions.
            worker_failures.append(error)
        finally:
            worker_done.set()

    worker = threading.Thread(target=unrelated_worker)
    worker.start()

    def noisy_browser(_html: str) -> bytes:
        _emit_test_diagnostics(logger, "backend-browser-root")
        _spawn_joined_diagnostic_tree(logger, "backend-browser-tree")
        worker_start.set()
        assert worker_done.wait(timeout=5)
        raise RuntimeError("synthetic browser detail")

    def noisy_reportlab(_html: str) -> bytes:
        _emit_test_diagnostics(logger, "backend-fallback-root")
        _spawn_joined_diagnostic_tree(logger, "backend-fallback-tree")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        noisy_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        noisy_reportlab,
    )

    result = render_resume_pdf_from_html("<p>Jules Example</p>")
    worker.join(timeout=5)

    assert result == _VALID_ONE_PAGE_PDF
    assert worker.is_alive() is False
    assert worker_failures == []
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    assert logging.Logger.handle is logger_handle_before
    assert threading.Thread.start is thread_start_before
    assert tuple(logging.getLogger().handlers) == root_handlers_before
    assert tuple(logging.getLogger().filters) == root_filters_before
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    captured_io = capsys.readouterr()
    assert "unrelated-root-stdout" in captured_io.out
    assert "unrelated-tree-child-stdout" in captured_io.out
    assert "unrelated-tree-grandchild-stdout" in captured_io.out
    assert "unrelated-root-stderr" in captured_io.err
    assert "unrelated-tree-child-stderr" in captured_io.err
    assert "unrelated-tree-grandchild-stderr" in captured_io.err
    assert "backend-" not in captured_io.out
    assert "backend-" not in captured_io.err
    messages = [record.getMessage() for record in caplog.records]
    assert "unrelated-root-log" in messages
    assert "unrelated-tree-child-log" in messages
    assert "unrelated-tree-grandchild-log" in messages
    assert all("backend-" not in message for message in messages)


def test_two_stage_pdf_failure_is_stable_silent_and_has_no_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    logger_handle_before = logging.Logger.handle
    thread_start_before = threading.Thread.start
    root_handlers_before = tuple(logging.getLogger().handlers)
    root_filters_before = tuple(logging.getLogger().filters)
    logger = logging.getLogger("resume-rendering-failure")

    def fail_browser(_html: str) -> bytes:
        _emit_test_diagnostics(logger, "failed-browser-root")
        _spawn_joined_diagnostic_tree(logger, "failed-browser-tree")
        raise RuntimeError("synthetic browser detail")

    def fail_reportlab(_html: str) -> bytes:
        _emit_test_diagnostics(logger, "failed-fallback-root")
        _spawn_joined_diagnostic_tree(logger, "failed-fallback-tree")
        raise RuntimeError("synthetic fallback detail")

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        fail_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        fail_reportlab,
    )

    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_pdf_from_html("<p>private-html-value</p>")

    _assert_stable_error(captured, "Resume PDF could not be rendered.")
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    assert logging.Logger.handle is logger_handle_before
    assert threading.Thread.start is thread_start_before
    assert tuple(logging.getLogger().handlers) == root_handlers_before
    assert tuple(logging.getLogger().filters) == root_filters_before
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert caplog.text == ""
    assert "synthetic" not in str(captured.value)


def test_overlapping_render_scopes_do_not_deadlock_or_cross_suppress(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-overlapping-scopes")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    logger_handle_before = logging.Logger.handle
    thread_start_before = threading.Thread.start
    overlap = threading.Barrier(2)
    release_second = threading.Event()
    first_scope_exited = threading.Event()
    results: dict[str, bytes] = {}
    failures: list[BaseException] = []

    def overlapping_browser(html: str) -> bytes:
        overlap.wait(timeout=5)
        label = "overlap-first" if "first" in html else "overlap-second"
        _spawn_joined_diagnostic_tree(logger, label)
        if "second" in html and not release_second.wait(timeout=5):
            raise RuntimeError("second scope timeout")
        return _VALID_ONE_PAGE_PDF

    def fail_fallback(_html: str) -> bytes:
        raise RuntimeError("fallback should not run")

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        overlapping_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        fail_fallback,
    )

    def render_worker(label: str) -> None:
        try:
            results[label] = render_resume_pdf_from_html(f"<p>{label}</p>")
            if label == "first":
                _emit_test_diagnostics(logger, "first-after-scope")
                first_scope_exited.set()
        except BaseException as error:
            failures.append(error)
            first_scope_exited.set()

    capsys.readouterr()
    first = threading.Thread(target=render_worker, args=("first",))
    second = threading.Thread(target=render_worker, args=("second",))
    first.start()
    second.start()
    assert first_scope_exited.wait(timeout=5)
    assert second.is_alive()
    release_second.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert failures == []
    assert results == {
        "first": _VALID_ONE_PAGE_PDF,
        "second": _VALID_ONE_PAGE_PDF,
    }
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    assert logging.Logger.handle is logger_handle_before
    assert threading.Thread.start is thread_start_before
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    captured_io = capsys.readouterr()
    assert captured_io.out == "first-after-scope-stdout\n"
    assert captured_io.err == "first-after-scope-stderr\n"
    assert "overlap-" not in captured_io.out
    assert "overlap-" not in captured_io.err
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["first-after-scope-log"]


def test_nested_render_scope_in_backend_descendant_is_reentrant(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-nested-scope")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    logger_handle_before = logging.Logger.handle
    thread_start_before = threading.Thread.start
    inner_results: list[bytes] = []
    failures: list[BaseException] = []

    def nested_browser(html: str) -> bytes:
        if "inner" in html:
            _spawn_joined_diagnostic_tree(logger, "nested-inner-tree")
            return _VALID_ONE_PAGE_PDF

        def nested_render_target() -> None:
            try:
                _emit_test_diagnostics(logger, "nested-child-before")
                inner_results.append(
                    render_resume_pdf_from_html("<p>inner</p>"),
                )
                _emit_test_diagnostics(logger, "nested-child-after")
            except BaseException as error:
                failures.append(error)

        child = threading.Thread(target=nested_render_target)
        child.start()
        child.join(timeout=5)
        if child.is_alive():
            failures.append(RuntimeError("nested scope timeout"))
        if failures:
            raise RuntimeError("nested render failed")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        nested_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: (_ for _ in ()).throw(
            AssertionError("fallback should not run"),
        ),
    )

    capsys.readouterr()
    result = render_resume_pdf_from_html("<p>outer</p>")

    assert result == _VALID_ONE_PAGE_PDF
    assert inner_results == [_VALID_ONE_PAGE_PDF]
    assert failures == []
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    assert logging.Logger.handle is logger_handle_before
    assert threading.Thread.start is thread_start_before
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert caplog.text == ""


@pytest.mark.parametrize("failure_kind", ["target", "overridden-run"])
def test_descendant_failure_and_backend_cleanup_failure_restore_every_surface(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    logger = logging.getLogger("resume-descendant-failure")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    logger_handle_before = logging.Logger.handle
    thread_start_before = threading.Thread.start
    root_handlers_before = tuple(logging.getLogger().handlers)
    root_filters_before = tuple(logging.getLogger().filters)

    def synthetic_excepthook(_args: threading.ExceptHookArgs) -> None:
        _emit_test_diagnostics(logger, "descendant-excepthook")

    monkeypatch.setattr(threading, "excepthook", synthetic_excepthook)

    def failing_target() -> None:
        _emit_test_diagnostics(logger, "descendant-target")
        raise RuntimeError("synthetic target detail")

    class FailingRunThread(threading.Thread):
        def run(self) -> None:
            _emit_test_diagnostics(logger, "descendant-overridden-run")
            raise RuntimeError("synthetic run detail")

    def browser_with_cleanup_failure(_html: str) -> bytes:
        child = (
            threading.Thread(target=failing_target)
            if failure_kind == "target"
            else FailingRunThread()
        )
        child.start()
        child.join(timeout=5)
        if child.is_alive():
            raise RuntimeError("child timeout")
        raise RuntimeError("synthetic backend cleanup detail")

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        browser_with_cleanup_failure,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: (_ for _ in ()).throw(
            RuntimeError("synthetic fallback cleanup detail"),
        ),
    )

    capsys.readouterr()
    with pytest.raises(ResumeRenderingError) as captured:
        render_resume_pdf_from_html("<p>Synthetic resume.</p>")

    _assert_stable_error(captured, "Resume PDF could not be rendered.")
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    assert logging.Logger.handle is logger_handle_before
    assert threading.Thread.start is thread_start_before
    assert tuple(logging.getLogger().handlers) == root_handlers_before
    assert tuple(logging.getLogger().filters) == root_filters_before
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert caplog.text == ""
    assert "synthetic" not in str(captured.value)


def test_mid_scope_third_party_global_replacements_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    logger_handle_before = logging.Logger.handle
    thread_start_before = threading.Thread.start
    root_handlers_before = tuple(logging.getLogger().handlers)
    root_filters_before = tuple(logging.getLogger().filters)
    replacement_stdout = StringIO()
    replacement_stderr = StringIO()
    replace_now = threading.Event()
    replaced = threading.Event()
    replacement_failures: list[BaseException] = []

    def replacement_logger_handle(
        logger: logging.Logger,
        record: logging.LogRecord,
    ) -> None:
        logger_handle_before(logger, record)

    def replacement_thread_start(
        thread: threading.Thread,
        *args: object,
        **kwargs: object,
    ) -> object:
        return thread_start_before(thread, *args, **kwargs)

    def replacer() -> None:
        try:
            assert replace_now.wait(timeout=5)
            sys.stdout = replacement_stdout
            sys.stderr = replacement_stderr
            logging.Logger.handle = replacement_logger_handle
            threading.Thread.start = replacement_thread_start
        except BaseException as error:
            replacement_failures.append(error)
        finally:
            replaced.set()

    replacement_thread = threading.Thread(target=replacer)
    replacement_thread.start()

    def browser_during_replacement(_html: str) -> bytes:
        replace_now.set()
        if not replaced.wait(timeout=5):
            raise RuntimeError("replacement timeout")
        raise RuntimeError("synthetic browser detail")

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        browser_during_replacement,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: _VALID_ONE_PAGE_PDF,
    )

    preserved = False
    result = b""
    try:
        result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")
        replacement_thread.join(timeout=5)
        preserved = (
            sys.stdout is replacement_stdout
            and sys.stderr is replacement_stderr
            and logging.Logger.handle is replacement_logger_handle
            and threading.Thread.start is replacement_thread_start
        )
    finally:
        sys.stdout = stdout_before
        sys.stderr = stderr_before
        logging.Logger.handle = logger_handle_before
        threading.Thread.start = thread_start_before

    assert replacement_thread.is_alive() is False
    assert replacement_failures == []
    assert result == _VALID_ONE_PAGE_PDF
    assert preserved is True
    assert tuple(logging.getLogger().handlers) == root_handlers_before
    assert tuple(logging.getLogger().filters) == root_filters_before
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        lambda _html: _VALID_ONE_PAGE_PDF,
    )
    reusable = render_resume_pdf_from_html("<p>Reusable render.</p>")
    assert reusable == _VALID_ONE_PAGE_PDF
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    assert logging.Logger.handle is logger_handle_before
    assert threading.Thread.start is thread_start_before


@pytest.mark.parametrize(
    "transition",
    [
        "browser-return-to-validation",
        "browser-validation-failure-to-fallback",
        "browser-exception-to-fallback",
        "fallback-return-to-validation",
    ],
)
def test_diagnostic_generation_rearms_at_every_stage_transition(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    transition: str,
) -> None:
    logger = logging.getLogger(f"resume-stage-rearm-{transition}")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    root_handlers_before = tuple(logging.getLogger().handlers)
    root_filters_before = tuple(logging.getLogger().filters)
    controller = _start_coordinated_replacer(
        logger,
        ordinary,
        emit_unrelated=True,
    )
    real_validator = resume_rendering._validated_pdf_bytes
    validation_calls = 0
    execution_ids: list[int] = []
    renderer_ident = threading.get_ident()

    def trigger_replacement() -> None:
        controller.request.set()
        if not controller.installed.wait(timeout=5):
            raise RuntimeError("replacement installation timeout")
        if controller.failures:
            raise RuntimeError("replacement installation failed")

    def emit_from_following_stage() -> None:
        controller.emit_request.set()
        _emit_test_diagnostics(logger, "owned-after-rearm")
        _spawn_joined_mixed_diagnostic_tree(
            logger,
            "owned-after-rearm-tree",
            execution_ids=execution_ids,
        )
        if not controller.emitted.wait(timeout=5):
            raise RuntimeError("unrelated emission timeout")

    def staged_browser(_html: str) -> bytes:
        if transition == "browser-return-to-validation":
            trigger_replacement()
            return _VALID_ONE_PAGE_PDF
        if transition == "browser-exception-to-fallback":
            trigger_replacement()
            raise RuntimeError("synthetic browser detail")
        if transition == "fallback-return-to-validation":
            raise RuntimeError("synthetic browser detail")
        return _VALID_ONE_PAGE_PDF

    def staged_validator(value: object) -> bytes:
        nonlocal validation_calls
        validation_calls += 1
        if (
            transition == "browser-validation-failure-to-fallback"
            and validation_calls == 1
        ):
            trigger_replacement()
            raise RuntimeError("synthetic validation detail")
        if transition in {
            "browser-return-to-validation",
            "fallback-return-to-validation",
        }:
            emit_from_following_stage()
        return real_validator(value)

    def staged_fallback(_html: str) -> bytes:
        if transition in {
            "browser-validation-failure-to-fallback",
            "browser-exception-to-fallback",
        }:
            emit_from_following_stage()
        elif transition == "fallback-return-to-validation":
            trigger_replacement()
        else:
            raise AssertionError("fallback should not run")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        staged_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        staged_fallback,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_validated_pdf_bytes",
        staged_validator,
    )

    result = b""
    preserved = False
    replacement_out = ""
    replacement_err = ""
    capsys.readouterr()
    try:
        result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")
        preserved = _diagnostic_surfaces() == (
            controller.replacement_stdout,
            controller.replacement_stderr,
            controller.replacement_logger_handle,
            controller.replacement_thread_start,
        )
        replacement_out = controller.replacement_stdout.getvalue()
        replacement_err = controller.replacement_stderr.getvalue()
    finally:
        controller.request.set()
        controller.emit_request.set()
        controller.thread.join(timeout=5)
        _restore_diagnostic_surfaces(ordinary)

    assert controller.thread.is_alive() is False
    assert controller.failures == []
    assert result == _VALID_ONE_PAGE_PDF
    assert preserved is True
    assert "unrelated-after-rearm-stdout" in replacement_out
    assert replacement_out.count("unrelated-after-rearm-stdout") == 1
    assert "unrelated-after-rearm-stderr" in replacement_err
    assert replacement_err.count("unrelated-after-rearm-stderr") == 1
    assert "owned-after-rearm" not in replacement_out
    assert "owned-after-rearm" not in replacement_err
    assert all(thread_ident != renderer_ident for thread_ident in execution_ids)
    assert len(execution_ids) == 4
    assert len(controller.start_calls) == 4
    assert all(
        caller_ident == renderer_ident or caller_ident in execution_ids
        for caller_ident, _thread in controller.start_calls
    )
    assert tuple(logging.getLogger().handlers) == root_handlers_before
    assert tuple(logging.getLogger().filters) == root_filters_before
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)
    assert "owned-after-rearm" not in result.decode("latin-1")
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["unrelated-after-rearm-log"]


def test_partial_replacement_carries_forward_unchanged_surface_targets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-partial-stage-rearm")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    replacement_stdout = StringIO()
    replacement_calls: list[threading.Thread] = []
    request = threading.Event()
    installed = threading.Event()
    failures: list[BaseException] = []
    active_targets: list[tuple[object, object, object, object]] = []

    def replacement_thread_start(
        thread: threading.Thread,
        *args: object,
        **kwargs: object,
    ) -> object:
        replacement_calls.append(thread)
        return ordinary[3](thread, *args, **kwargs)  # type: ignore[operator]

    def replacer() -> None:
        try:
            if not request.wait(timeout=5):
                raise RuntimeError("partial replacement request timeout")
            sys.stdout = replacement_stdout
            threading.Thread.start = replacement_thread_start
        except BaseException as error:
            failures.append(error)
        finally:
            installed.set()

    replacement_worker = threading.Thread(target=replacer)
    replacement_worker.start()

    def replaced_browser(_html: str) -> bytes:
        request.set()
        if not installed.wait(timeout=5):
            raise RuntimeError("partial replacement timeout")
        if failures:
            raise RuntimeError("partial replacement failed")
        raise RuntimeError("synthetic browser detail")

    def fallback_after_partial_replacement(_html: str) -> bytes:
        hooks = resume_rendering._DIAGNOSTIC_HOOKS
        if hooks is None:
            raise RuntimeError("missing fresh hook generation")
        active_targets.append(
            (
                hooks.stdout_proxy._stream,
                hooks.stderr_proxy._stream,
                hooks.logger_handle._target,
                hooks.thread_start._target,
            ),
        )
        _emit_test_diagnostics(logger, "owned-partial-fallback")
        _spawn_joined_mixed_diagnostic_tree(logger, "owned-partial-tree")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        replaced_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        fallback_after_partial_replacement,
    )

    result = b""
    post_surfaces: tuple[object, object, object, object] | None = None
    capsys.readouterr()
    try:
        result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")
        post_surfaces = _diagnostic_surfaces()
    finally:
        request.set()
        replacement_worker.join(timeout=5)
        _restore_diagnostic_surfaces(ordinary)

    assert replacement_worker.is_alive() is False
    assert failures == []
    assert result == _VALID_ONE_PAGE_PDF
    assert active_targets == [
        (
            replacement_stdout,
            ordinary[1],
            ordinary[2],
            replacement_thread_start,
        ),
    ]
    assert post_surfaces == (
        replacement_stdout,
        ordinary[1],
        ordinary[2],
        replacement_thread_start,
    )
    assert len(replacement_calls) == 4
    assert replacement_stdout.getvalue() == ""
    assert caplog.text == ""
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)


def test_final_opaque_stage_third_party_replacement_is_left_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("resume-final-stage-replacement")
    ordinary = _diagnostic_surfaces()
    first = _start_coordinated_replacer(
        logger,
        ordinary,
        emit_unrelated=False,
    )
    second = _start_coordinated_replacer(
        logger,
        ordinary,
        emit_unrelated=False,
    )
    real_validator = resume_rendering._validated_pdf_bytes

    def trigger(controller: types.SimpleNamespace) -> None:
        controller.request.set()
        if not controller.installed.wait(timeout=5):
            raise RuntimeError("replacement installation timeout")
        if controller.failures:
            raise RuntimeError("replacement installation failed")

    def replaced_browser(_html: str) -> bytes:
        trigger(first)
        raise RuntimeError("synthetic browser detail")

    def final_validator(value: object) -> bytes:
        trigger(second)
        return real_validator(value)

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        replaced_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        lambda _html: _VALID_ONE_PAGE_PDF,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_validated_pdf_bytes",
        final_validator,
    )

    result = b""
    preserved = False
    try:
        result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")
        preserved = _diagnostic_surfaces() == (
            second.replacement_stdout,
            second.replacement_stderr,
            second.replacement_logger_handle,
            second.replacement_thread_start,
        )
    finally:
        for controller in (first, second):
            controller.request.set()
            controller.emit_request.set()
            controller.thread.join(timeout=5)
        _restore_diagnostic_surfaces(ordinary)

    assert first.thread.is_alive() is False
    assert second.thread.is_alive() is False
    assert first.failures == []
    assert second.failures == []
    assert result == _VALID_ONE_PAGE_PDF
    assert preserved is True
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}


@pytest.mark.parametrize(
    "case",
    _INFLIGHT_DIAGNOSTIC_CASES,
    ids=lambda case: "-".join(case),
)
def test_active_path_operation_linearizes_against_live_scope_after_rollover(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: tuple[str, str],
) -> None:
    logger = logging.getLogger(f"resume-inflight-rollover-{'-'.join(case)}")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    old_stdout = _ObservableTextStream()
    old_stderr = _ObservableTextStream()
    replacement_stdout = _ObservableTextStream()
    replacement_stderr = _ObservableTextStream()
    replacement_logs: list[str] = []
    replacement_starts: list[threading.Thread] = []
    gate_entered = threading.Event()
    release_gate = threading.Event()
    operation_completed = threading.Event()
    facts_checked = threading.Event()
    gate_hits: list[tuple[str, str]] = []
    gate_active_values: list[bool] = []
    rollover_failures: list[BaseException] = []
    backend_failures: list[BaseException] = []
    fallback_calls: list[str] = []
    operations: list[types.SimpleNamespace] = []
    child_ownership: list[frozenset[object]] = []
    parent_scopes: list[frozenset[object]] = []
    generations: dict[str, object] = {}
    nominated: dict[str, object] = {}
    operation_results: list[object] = []
    unrelated_marker = f"unrelated-inflight-{'-'.join(case)}"
    real_suppression_decision = resume_rendering._wrapper_suppresses_current_thread
    real_thread_start_decision = resume_rendering._start_thread_with_diagnostic_scopes

    def gate() -> None:
        gate_hits.append(case)
        gate_active_values.append(
            nominated["generation"].state.active,  # type: ignore[attr-defined]
        )
        gate_entered.set()
        if not release_gate.wait(timeout=5):
            raise RuntimeError("in-flight decision release timeout")

    def gated_suppression_decision(wrapper: object) -> bool:
        if wrapper is nominated.get(
            "wrapper"
        ) and threading.current_thread() is nominated.get("owner"):
            gate()
        return real_suppression_decision(wrapper)

    def gated_thread_start_decision(
        hook: object,
        thread: threading.Thread,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        if (
            hook is nominated.get("wrapper")
            and thread is nominated.get("thread")
            and threading.current_thread() is nominated.get("owner")
        ):
            gate()
        return real_thread_start_decision(  # type: ignore[arg-type]
            hook,
            thread,
            args,
            kwargs,
        )

    def replacement_logger_handle(
        target_logger: logging.Logger,
        record: logging.LogRecord,
    ) -> object:
        replacement_logs.append(record.getMessage())
        return resume_rendering._invoke_descriptor_target(
            ordinary[2],
            target_logger,
            record,
        )

    def replacement_thread_start(
        thread: threading.Thread,
        *args: object,
        **kwargs: object,
    ) -> object:
        replacement_starts.append(thread)
        return resume_rendering._invoke_descriptor_target(
            ordinary[3],
            thread,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        resume_rendering,
        "_wrapper_suppresses_current_thread",
        gated_suppression_decision,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_start_thread_with_diagnostic_scopes",
        gated_thread_start_decision,
    )

    def browser_backend(_html: str) -> bytes:
        try:
            with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                old_hooks = resume_rendering._DIAGNOSTIC_HOOKS
                assert old_hooks is not None
                generations["old"] = old_hooks
                scopes = frozenset(
                    resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                        id(threading.current_thread()),
                        set(),
                    ),
                )
                assert scopes
                parent_scopes.append(scopes)
            operation = _prepare_inflight_diagnostic_operation(
                old_hooks,
                case,
                logger,
                child_ownership,
            )
            operations.append(operation)
            nominated.update(
                generation=old_hooks,
                owner=threading.current_thread(),
                thread=operation.gate_thread,
                wrapper=operation.gate_wrapper,
            )
            try:
                operation_results.append(operation.invoke())
            finally:
                operation_completed.set()
            return _VALID_ONE_PAGE_PDF
        except Exception as error:
            backend_failures.append(error)
            raise

    def fallback_backend(_html: str) -> bytes:
        fallback_calls.append("called")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        browser_backend,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        fallback_backend,
    )

    def unrelated_body() -> None:
        sys.stdout.write(f"{unrelated_marker}-stdout")
        sys.stderr.write(f"{unrelated_marker}-stderr")
        logger.debug("%s-log", unrelated_marker)

    def rollover_worker() -> None:
        try:
            if not gate_entered.wait(timeout=5):
                raise RuntimeError("in-flight decision gate timeout")
            sys.stdout = replacement_stdout  # type: ignore[assignment]
            sys.stderr = replacement_stderr  # type: ignore[assignment]
            logging.Logger.handle = replacement_logger_handle
            threading.Thread.start = replacement_thread_start
            resume_rendering._diagnostic_stage_checkpoint()
            with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                old_hooks = generations["old"]
                fresh_hooks = resume_rendering._DIAGNOSTIC_HOOKS
                assert fresh_hooks is not None
                generations["fresh"] = fresh_hooks
                assert operation_completed.is_set() is False
                assert gate_hits == [case]
                assert gate_active_values == [True]
                assert old_hooks is not fresh_hooks
                assert old_hooks.retired is True  # type: ignore[attr-defined]
                assert old_hooks.state.active is False  # type: ignore[attr-defined]
                assert fresh_hooks.retired is False
                assert fresh_hooks.state.active is True
                assert resume_rendering._hook_generation_is_current(
                    fresh_hooks,
                )
                assert fresh_hooks.stdout_proxy._stream is replacement_stdout
                assert fresh_hooks.stderr_proxy._stream is replacement_stderr
                assert fresh_hooks.logger_handle._target is replacement_logger_handle
                assert fresh_hooks.thread_start._target is replacement_thread_start
                assert parent_scopes[0].issubset(
                    resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES,
                )
                assert (
                    frozenset(
                        resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                            id(nominated["owner"]),
                            set(),
                        ),
                    )
                    == parent_scopes[0]
                )
                if case[0].endswith("thread"):
                    assert child_ownership == []
            unrelated = threading.Thread(target=unrelated_body)
            unrelated.start()
            unrelated.join(timeout=5)
            if unrelated.is_alive():
                raise RuntimeError("unrelated diagnostic thread timeout")
            facts_checked.set()
        except Exception as error:
            rollover_failures.append(error)
        finally:
            release_gate.set()

    sys.stdout = old_stdout  # type: ignore[assignment]
    sys.stderr = old_stderr  # type: ignore[assignment]
    rollover = threading.Thread(target=rollover_worker)
    rollover.start()
    result = b""
    post_surfaces: tuple[object, object, object, object] | None = None
    try:
        result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")
        rollover.join(timeout=5)
        post_surfaces = _diagnostic_surfaces()
    finally:
        release_gate.set()
        rollover.join(timeout=5)
        _restore_diagnostic_surfaces(ordinary)

    assert rollover.is_alive() is False
    assert rollover_failures == []
    assert backend_failures == []
    assert fallback_calls == []
    assert facts_checked.is_set()
    assert gate_hits == [case]
    assert gate_active_values == [True]
    assert operation_completed.is_set()
    assert len(operations) == 1
    operation = operations[0]
    assert operation_results == [operation.expected_result]
    assert result == _VALID_ONE_PAGE_PDF
    assert operation.marker.encode() not in result
    assert post_surfaces == (
        replacement_stdout,
        replacement_stderr,
        replacement_logger_handle,
        replacement_thread_start,
    )
    _assert_marker_absent_from_probes(
        operation.marker,
        old_stdout,
        old_stderr,
        replacement_stdout,
        replacement_stderr,
    )
    messages = [record.getMessage() for record in caplog.records]
    assert operation.marker not in "\n".join(messages)
    assert operation.marker not in "\n".join(replacement_logs)
    _assert_operation_not_delegated(
        operation,
        old_stdout,
        old_stderr,
        messages,
    )
    _assert_operation_not_delegated(
        operation,
        replacement_stdout,
        replacement_stderr,
        replacement_logs,
    )
    assert replacement_stdout.writes.count(f"{unrelated_marker}-stdout") == 1
    assert replacement_stderr.writes.count(f"{unrelated_marker}-stderr") == 1
    assert messages.count(f"{unrelated_marker}-log") == 1
    assert replacement_logs.count(f"{unrelated_marker}-log") == 1
    assert len(replacement_starts) == 1
    if operation.child is not None:
        assert operation.child not in replacement_starts
        assert child_ownership == [parent_scopes[0]]
    else:
        assert child_ownership == []
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "case",
    _INFLIGHT_DIAGNOSTIC_CASES,
    ids=lambda case: "-".join(case),
)
def test_active_path_operation_delegates_once_after_final_scope_exit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: tuple[str, str],
) -> None:
    logger = logging.getLogger(f"resume-inflight-final-exit-{'-'.join(case)}")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    stdout = _ObservableTextStream()
    stderr = _ObservableTextStream()
    gate_entered = threading.Event()
    release_gate = threading.Event()
    operation_completed = threading.Event()
    gate_hits: list[tuple[str, str]] = []
    gate_active_values: list[bool] = []
    operator_ownership: list[frozenset[object]] = []
    child_ownership: list[frozenset[object]] = []
    failures: list[BaseException] = []
    operation_results: list[object] = []
    nominated: dict[str, object] = {}
    old_hooks_holder: list[object] = []
    operation_holder: list[types.SimpleNamespace] = []
    real_suppression_decision = resume_rendering._wrapper_suppresses_current_thread
    real_thread_start_decision = resume_rendering._start_thread_with_diagnostic_scopes

    def gate() -> None:
        gate_hits.append(case)
        gate_active_values.append(
            nominated["generation"].state.active,  # type: ignore[attr-defined]
        )
        gate_entered.set()
        if not release_gate.wait(timeout=5):
            raise RuntimeError("final-exit decision release timeout")

    def gated_suppression_decision(wrapper: object) -> bool:
        if wrapper is nominated.get(
            "wrapper"
        ) and threading.current_thread() is nominated.get("owner"):
            gate()
        return real_suppression_decision(wrapper)

    def gated_thread_start_decision(
        hook: object,
        thread: threading.Thread,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        if (
            hook is nominated.get("wrapper")
            and thread is nominated.get("thread")
            and threading.current_thread() is nominated.get("owner")
        ):
            gate()
        return real_thread_start_decision(  # type: ignore[arg-type]
            hook,
            thread,
            args,
            kwargs,
        )

    monkeypatch.setattr(
        resume_rendering,
        "_wrapper_suppresses_current_thread",
        gated_suppression_decision,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_start_thread_with_diagnostic_scopes",
        gated_thread_start_decision,
    )

    def operator_body() -> None:
        try:
            with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                operator_ownership.append(
                    frozenset(
                        resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                            id(threading.current_thread()),
                            set(),
                        ),
                    ),
                )
            operation_results.append(operation_holder[0].invoke())
        except Exception as error:
            failures.append(error)
        finally:
            operation_completed.set()

    sys.stdout = stdout  # type: ignore[assignment]
    sys.stderr = stderr  # type: ignore[assignment]
    operator: threading.Thread | None = None
    try:
        with resume_rendering._suppress_backend_diagnostics():
            with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                old_hooks = resume_rendering._DIAGNOSTIC_HOOKS
                assert old_hooks is not None
                old_hooks_holder.append(old_hooks)
                parent_scopes = frozenset(
                    resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                        id(threading.current_thread()),
                        set(),
                    ),
                )
                assert parent_scopes
            operation = _prepare_inflight_diagnostic_operation(
                old_hooks,
                case,
                logger,
                child_ownership,
            )
            operation_holder.append(operation)
            operator = threading.Thread(target=operator_body)
            nominated.update(
                generation=old_hooks,
                owner=operator,
                thread=operation.gate_thread,
                wrapper=operation.gate_wrapper,
            )
            operator.start()
            assert gate_entered.wait(timeout=5)
            assert operation_completed.is_set() is False
            assert gate_hits == [case]
            assert gate_active_values == [True]
            assert old_hooks.retired is False
            assert old_hooks.state.active is True
            assert operator_ownership == [parent_scopes]
            assert child_ownership == []
            with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                assert parent_scopes.issubset(
                    resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES,
                )
                assert (
                    frozenset(
                        resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                            id(operator),
                            set(),
                        ),
                    )
                    == parent_scopes
                )

        assert operation_completed.is_set() is False
        assert old_hooks.retired is True
        assert old_hooks.state.active is False
        assert resume_rendering._DIAGNOSTIC_HOOKS is None
        assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
        assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
        assert _diagnostic_surfaces() == (
            stdout,
            stderr,
            ordinary[2],
            ordinary[3],
        )
        release_gate.set()
        operator.join(timeout=5)
        if operator.is_alive():
            raise RuntimeError("final-exit operation thread timeout")
        unrelated_marker = f"unrelated-final-exit-{'-'.join(case)}"
        sys.stdout.write(f"{unrelated_marker}-stdout")
        sys.stderr.write(f"{unrelated_marker}-stderr")
        logger.debug("%s-log", unrelated_marker)
    finally:
        release_gate.set()
        if operator is not None:
            operator.join(timeout=5)
        _restore_diagnostic_surfaces(ordinary)

    assert operator is not None
    assert operator.is_alive() is False
    assert failures == []
    assert gate_hits == [case]
    assert gate_active_values == [True]
    assert operation_completed.is_set()
    assert len(old_hooks_holder) == 1
    assert len(operation_holder) == 1
    operation = operation_holder[0]
    assert operation_results == [operation.expected_result]
    messages = [record.getMessage() for record in caplog.records]
    _assert_operation_delegated_once(operation, stdout, stderr, messages)
    if operation.child is not None:
        assert child_ownership == [frozenset()]
    else:
        assert child_ownership == []
    unrelated_marker = f"unrelated-final-exit-{'-'.join(case)}"
    assert stdout.writes.count(f"{unrelated_marker}-stdout") == 1
    assert stderr.writes.count(f"{unrelated_marker}-stderr") == 1
    assert messages.count(f"{unrelated_marker}-log") == 1
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)


def test_overlapping_scope_rollover_keeps_one_fresh_shared_generation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-overlapping-rollover")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    controller = _start_coordinated_replacer(
        logger,
        ordinary,
        emit_unrelated=True,
    )
    both_in_browser = threading.Barrier(2)
    rollover_done = threading.Event()
    second_emitted = threading.Event()
    release_second = threading.Event()
    first_exited = threading.Event()
    results: dict[str, bytes] = {}
    failures: list[BaseException] = []
    ownership: dict[str, frozenset[object]] = {}
    generation_ids: dict[str, int] = {}

    def parent_ownership(label: str) -> None:
        with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
            ownership[label] = frozenset(
                resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                    id(threading.current_thread()),
                    set(),
                ),
            )

    def overlapping_browser(html: str) -> bytes:
        both_in_browser.wait(timeout=5)
        if "first" in html:
            with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                assert resume_rendering._DIAGNOSTIC_HOOKS is not None
                generation_ids["old"] = id(resume_rendering._DIAGNOSTIC_HOOKS)
            controller.request.set()
            if not controller.installed.wait(timeout=5):
                raise RuntimeError("replacement installation timeout")
            raise RuntimeError("synthetic browser detail")

        if not rollover_done.wait(timeout=5):
            raise RuntimeError("rollover timeout")
        with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
            assert resume_rendering._DIAGNOSTIC_HOOKS is not None
            generation_ids["second"] = id(
                resume_rendering._DIAGNOSTIC_HOOKS,
            )
        parent_ownership("second-parent")
        _emit_test_diagnostics(logger, "owned-second-after-rollover")
        _spawn_joined_mixed_diagnostic_tree(
            logger,
            "owned-second-tree",
            ownership=ownership,
        )
        second_emitted.set()
        if not release_second.wait(timeout=5):
            raise RuntimeError("second release timeout")
        return _VALID_ONE_PAGE_PDF

    def first_fallback(_html: str) -> bytes:
        with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
            assert resume_rendering._DIAGNOSTIC_HOOKS is not None
            generation_ids["fresh"] = id(resume_rendering._DIAGNOSTIC_HOOKS)
        parent_ownership("first-parent")
        _emit_test_diagnostics(logger, "owned-first-after-rollover")
        _spawn_joined_mixed_diagnostic_tree(
            logger,
            "owned-first-tree",
            ownership=ownership,
        )
        controller.emit_request.set()
        if not controller.emitted.wait(timeout=5):
            raise RuntimeError("unrelated emission timeout")
        rollover_done.set()
        if not second_emitted.wait(timeout=5):
            raise RuntimeError("second emission timeout")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        overlapping_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        first_fallback,
    )

    def render_worker(label: str) -> None:
        try:
            results[label] = render_resume_pdf_from_html(f"<p>{label}</p>")
        except BaseException as error:
            failures.append(error)
        finally:
            if label == "first":
                first_exited.set()

    first = threading.Thread(target=render_worker, args=("first",))
    second = threading.Thread(target=render_worker, args=("second",))
    capsys.readouterr()
    first.start()
    second.start()
    current_hooks: object | None = None
    generation_remained_active = False
    preserved = False
    replacement_out = ""
    replacement_err = ""
    try:
        assert first_exited.wait(timeout=5)
        first.join(timeout=5)
        assert first.is_alive() is False
        assert second.is_alive()
        with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
            current_hooks = resume_rendering._DIAGNOSTIC_HOOKS
            assert current_hooks is not None
            generation_remained_active = (
                id(current_hooks) == generation_ids["fresh"]
                and current_hooks.retired is False  # type: ignore[attr-defined]
                and len(resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES) == 1
                and resume_rendering._hook_generation_is_current(
                    current_hooks,  # type: ignore[arg-type]
                )
            )
        _emit_test_diagnostics(logger, "unrelated-after-first-exit")
    finally:
        controller.request.set()
        controller.emit_request.set()
        rollover_done.set()
        second_emitted.set()
        release_second.set()
        first.join(timeout=5)
        second.join(timeout=5)
        controller.thread.join(timeout=5)
        preserved = _diagnostic_surfaces() == (
            controller.replacement_stdout,
            controller.replacement_stderr,
            controller.replacement_logger_handle,
            controller.replacement_thread_start,
        )
        replacement_out = controller.replacement_stdout.getvalue()
        replacement_err = controller.replacement_stderr.getvalue()
        _restore_diagnostic_surfaces(ordinary)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert controller.thread.is_alive() is False
    assert controller.failures == []
    assert failures == []
    assert results == {
        "first": _VALID_ONE_PAGE_PDF,
        "second": _VALID_ONE_PAGE_PDF,
    }
    assert generation_ids["fresh"] != generation_ids["old"]
    assert generation_ids["second"] == generation_ids["fresh"]
    assert generation_remained_active is True
    assert preserved is True
    assert ownership["first-parent"]
    assert ownership["second-parent"]
    assert ownership["first-parent"].isdisjoint(ownership["second-parent"])
    for label, scopes in ownership.items():
        expected = (
            ownership["first-parent"]
            if label.startswith("owned-first") or label == "first-parent"
            else ownership["second-parent"]
        )
        assert scopes == expected
    assert replacement_out.count("unrelated-after-rearm-stdout") == 1
    assert replacement_err.count("unrelated-after-rearm-stderr") == 1
    assert replacement_out.count("unrelated-after-first-exit-stdout") == 1
    assert replacement_err.count("unrelated-after-first-exit-stderr") == 1
    assert "owned-first" not in replacement_out
    assert "owned-first" not in replacement_err
    assert "owned-second" not in replacement_out
    assert "owned-second" not in replacement_err
    assert len(controller.start_calls) == 8
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert sorted(record.getMessage() for record in caplog.records) == [
        "unrelated-after-first-exit-log",
        "unrelated-after-rearm-log",
    ]


def test_inflight_thread_start_keeps_scope_ownership_across_rollover(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-inflight-start-rollover")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    start_entered = threading.Event()
    release_start = threading.Event()
    first = _start_coordinated_replacer(
        logger,
        ordinary,
        emit_unrelated=False,
        pause_thread_start=(start_entered, release_start),
    )
    second = _start_coordinated_replacer(
        logger,
        ordinary,
        emit_unrelated=False,
    )
    both_in_browser = threading.Barrier(2)
    results: dict[str, bytes] = {}
    failures: list[BaseException] = []
    child_ownership: list[frozenset[object]] = []
    parent_ownership: list[frozenset[object]] = []

    def trigger(controller: types.SimpleNamespace) -> None:
        controller.request.set()
        if not controller.installed.wait(timeout=5):
            raise RuntimeError("replacement installation timeout")
        if controller.failures:
            raise RuntimeError("replacement installation failed")

    def racing_browser(html: str) -> bytes:
        both_in_browser.wait(timeout=5)
        if "starter" in html:
            trigger(first)
            raise RuntimeError("synthetic starter browser detail")
        if not start_entered.wait(timeout=5):
            raise RuntimeError("thread start did not pause")
        trigger(second)
        raise RuntimeError("synthetic roller browser detail")

    def racing_fallback(html: str) -> bytes:
        if "roller" in html:
            release_start.set()
            return _VALID_ONE_PAGE_PDF

        with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
            parent_ownership.append(
                frozenset(
                    resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                        id(threading.current_thread()),
                        set(),
                    ),
                ),
            )

        def child_target() -> None:
            with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
                child_ownership.append(
                    frozenset(
                        resume_rendering._THREAD_DIAGNOSTIC_SCOPES.get(
                            id(threading.current_thread()),
                            set(),
                        ),
                    ),
                )
            _emit_test_diagnostics(logger, "owned-inflight-child")

        child = threading.Thread(target=child_target)
        child.start()
        child.join(timeout=5)
        if child.is_alive():
            raise RuntimeError("inflight child timeout")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        racing_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        racing_fallback,
    )

    def render_worker(label: str) -> None:
        try:
            results[label] = render_resume_pdf_from_html(f"<p>{label}</p>")
        except BaseException as error:
            failures.append(error)

    starter = threading.Thread(target=render_worker, args=("starter",))
    roller = threading.Thread(target=render_worker, args=("roller",))
    capsys.readouterr()
    starter.start()
    roller.start()
    preserved = False
    replacement_out = ""
    replacement_err = ""
    try:
        starter.join(timeout=5)
        roller.join(timeout=5)
    finally:
        first.request.set()
        second.request.set()
        first.emit_request.set()
        second.emit_request.set()
        release_start.set()
        starter.join(timeout=5)
        roller.join(timeout=5)
        first.thread.join(timeout=5)
        second.thread.join(timeout=5)
        preserved = _diagnostic_surfaces() == (
            second.replacement_stdout,
            second.replacement_stderr,
            second.replacement_logger_handle,
            second.replacement_thread_start,
        )
        replacement_out = second.replacement_stdout.getvalue()
        replacement_err = second.replacement_stderr.getvalue()
        _restore_diagnostic_surfaces(ordinary)

    assert starter.is_alive() is False
    assert roller.is_alive() is False
    assert first.thread.is_alive() is False
    assert second.thread.is_alive() is False
    assert first.failures == []
    assert second.failures == []
    assert failures == []
    assert results == {
        "starter": _VALID_ONE_PAGE_PDF,
        "roller": _VALID_ONE_PAGE_PDF,
    }
    assert parent_ownership and child_ownership
    assert child_ownership == parent_ownership
    assert len(first.start_calls) == 1
    assert second.start_calls == []
    assert preserved is True
    assert "owned-inflight-child" not in replacement_out
    assert "owned-inflight-child" not in replacement_err
    assert caplog.text == ""
    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)


def test_delayed_restored_owned_wrappers_flatten_for_eight_cycles(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-delayed-wrapper-restoration")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    active_controller: list[types.SimpleNamespace] = []
    retained_wrappers: list[object] = []
    reusable_surfaces: list[tuple[object, object, object, object]] = []

    def cycling_browser(html: str) -> bytes:
        if "replace" not in html:
            reusable_surfaces.append(_diagnostic_surfaces())
            _emit_test_diagnostics(logger, "owned-reusable-render")
            return _VALID_ONE_PAGE_PDF
        controller = active_controller[0]
        controller.request.set()
        if not controller.installed.wait(timeout=5):
            raise RuntimeError("replacement installation timeout")
        if controller.failures:
            raise RuntimeError("replacement installation failed")
        raise RuntimeError("synthetic browser detail")

    def cycling_fallback(_html: str) -> bytes:
        _emit_test_diagnostics(logger, "owned-cycle-fallback")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        cycling_browser,
    )
    monkeypatch.setattr(
        resume_rendering,
        "_render_text_pdf_from_html",
        cycling_fallback,
    )

    for cycle in range(8):
        controller = _start_coordinated_replacer(
            logger,
            ordinary,
            emit_unrelated=False,
        )
        active_controller[:] = [controller]
        result = render_resume_pdf_from_html(f"<p>replace-{cycle}</p>")
        controller.thread.join(timeout=5)

        assert controller.thread.is_alive() is False
        assert controller.failures == []
        assert result == _VALID_ONE_PAGE_PDF
        assert _diagnostic_surfaces() == (
            controller.replacement_stdout,
            controller.replacement_stderr,
            controller.replacement_logger_handle,
            controller.replacement_thread_start,
        )
        assert len(controller.captured) == 1
        stale = controller.captured[0]
        retained_wrappers.extend(stale)
        assert all(
            wrapper._state.active is False  # type: ignore[attr-defined]
            for wrapper in stale
        )

        _restore_diagnostic_surfaces(stale)
        caplog.clear()
        capsys.readouterr()
        execution_ids: list[int] = []

        def target_body() -> None:
            execution_ids.append(threading.get_ident())
            _emit_test_diagnostics(logger, f"cycle-{cycle}-target")

        class RunBodyThread(threading.Thread):
            def run(self) -> None:
                execution_ids.append(threading.get_ident())
                _emit_test_diagnostics(logger, f"cycle-{cycle}-run")

        print(f"cycle-{cycle}-direct-stdout")
        print(f"cycle-{cycle}-direct-stderr", file=sys.stderr)
        logger.debug("cycle-%s-direct-log", cycle)
        children = [threading.Thread(target=target_body), RunBodyThread()]
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=5)
            assert child.is_alive() is False

        captured_io = capsys.readouterr()
        assert captured_io.out.count(f"cycle-{cycle}-direct-stdout") == 1
        assert captured_io.out.count(f"cycle-{cycle}-target-stdout") == 1
        assert captured_io.out.count(f"cycle-{cycle}-run-stdout") == 1
        assert captured_io.err.count(f"cycle-{cycle}-direct-stderr") == 1
        assert captured_io.err.count(f"cycle-{cycle}-target-stderr") == 1
        assert captured_io.err.count(f"cycle-{cycle}-run-stderr") == 1
        assert sorted(record.getMessage() for record in caplog.records) == sorted(
            [
                f"cycle-{cycle}-direct-log",
                f"cycle-{cycle}-target-log",
                f"cycle-{cycle}-run-log",
            ],
        )
        assert len(execution_ids) == 2
        assert all(
            thread_ident != threading.get_ident() for thread_ident in execution_ids
        )
        assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
        assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}

        caplog.clear()
        reusable = render_resume_pdf_from_html(f"<p>reuse-{cycle}</p>")
        assert reusable == _VALID_ONE_PAGE_PDF
        active = reusable_surfaces[cycle]
        assert resume_rendering._is_owned_stream(active[0])
        assert resume_rendering._is_owned_stream(active[1])
        assert resume_rendering._is_owned_logger_handle(active[2])
        assert resume_rendering._is_owned_thread_start(active[3])
        assert active[0]._stream is ordinary[0]  # type: ignore[attr-defined]
        assert active[1]._stream is ordinary[1]  # type: ignore[attr-defined]
        assert active[2]._target is ordinary[2]  # type: ignore[attr-defined]
        assert active[3]._target is ordinary[3]  # type: ignore[attr-defined]
        assert _diagnostic_surfaces() == ordinary
        assert _owned_surface_depths() == (0, 0, 0, 0)
        assert resume_rendering._DIAGNOSTIC_HOOKS is None
        assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
        assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
        reusable_io = capsys.readouterr()
        assert reusable_io.out == ""
        assert reusable_io.err == ""
        assert caplog.text == ""

    assert len(retained_wrappers) == 32
    assert all(
        wrapper._state.active is False  # type: ignore[attr-defined]
        for wrapper in retained_wrappers
    )
    assert _diagnostic_surfaces() == ordinary


def test_retired_wrappers_delegate_without_waiting_for_diagnostic_lock() -> None:
    ordinary = _diagnostic_surfaces()
    stream_calls: list[tuple[str, object]] = []
    logger_calls: list[str] = []

    class ProbeBuffer:
        def write(self, value: object) -> str:
            stream_calls.append(("buffer-write", value))
            return "buffer-write-result"

        def writelines(self, values: object) -> str:
            stream_calls.append(("buffer-writelines", tuple(values)))  # type: ignore[arg-type]
            return "buffer-writelines-result"

        def flush(self) -> str:
            stream_calls.append(("buffer-flush", ""))
            return "buffer-flush-result"

    class ProbeStream:
        def __init__(self, label: str) -> None:
            self.buffer = ProbeBuffer()
            self.label = label

        def write(self, value: object) -> str:
            stream_calls.append((f"{self.label}-write", value))
            return f"{self.label}-write-result"

        def writelines(self, values: object) -> str:
            stream_calls.append(
                (f"{self.label}-writelines", tuple(values)),  # type: ignore[arg-type]
            )
            return f"{self.label}-writelines-result"

        def flush(self) -> str:
            stream_calls.append((f"{self.label}-flush", ""))
            return f"{self.label}-flush-result"

    stdout_target = ProbeStream("stdout")
    stderr_target = ProbeStream("stderr")

    def logger_target(
        _logger: logging.Logger,
        record: logging.LogRecord,
    ) -> str:
        logger_calls.append(record.getMessage())
        return "logger-result"

    _restore_diagnostic_surfaces(
        (
            stdout_target,
            stderr_target,
            logger_target,
            ordinary[3],
        ),
    )
    active_buffer: object
    with resume_rendering._suppress_backend_diagnostics():
        stale = _diagnostic_surfaces()
        active_buffer = stale[0].buffer  # type: ignore[attr-defined]
    _restore_diagnostic_surfaces(ordinary)

    assert all(
        wrapper._state.active is False  # type: ignore[attr-defined]
        for wrapper in stale
    )
    lock_held = threading.Event()
    release_lock = threading.Event()
    invocation_done = threading.Event()
    child_ran = threading.Event()
    failures: list[BaseException] = []
    results: list[object] = []

    def hold_lock() -> None:
        with resume_rendering._DIAGNOSTIC_SCOPE_LOCK:
            lock_held.set()
            release_lock.wait(timeout=10)

    def invoke_retired_surfaces() -> None:
        try:
            results.extend(
                [
                    sys.stdout.write("stdout-value"),
                    sys.stderr.writelines(["stderr-value"]),
                    sys.stderr.flush(),
                    sys.stdout.buffer.write(b"buffer-value"),
                    active_buffer.write(b"captured-buffer-value"),  # type: ignore[attr-defined]
                    active_buffer.writelines([b"captured-line"]),  # type: ignore[attr-defined]
                    active_buffer.flush(),  # type: ignore[attr-defined]
                ],
            )
            record = logger.makeRecord(
                logger.name,
                logging.DEBUG,
                __file__,
                1,
                "retired-log",
                (),
                None,
            )
            results.append(logger.handle(record))
            child = threading.Thread(target=child_ran.set)
            child.start()
            child.join(timeout=5)
            if child.is_alive():
                raise RuntimeError("retired thread-start child timeout")
        except BaseException as error:
            failures.append(error)
        finally:
            invocation_done.set()

    logger = logging.getLogger("resume-retired-lock-free")
    holder = threading.Thread(target=hold_lock)
    invoker = threading.Thread(target=invoke_retired_surfaces)
    ordinary[3](holder)  # type: ignore[operator]
    assert lock_held.wait(timeout=5)
    _restore_diagnostic_surfaces(stale)
    ordinary[3](invoker)  # type: ignore[operator]
    completed_while_locked = invocation_done.wait(timeout=5)
    try:
        release_lock.set()
        holder.join(timeout=5)
        invoker.join(timeout=5)
    finally:
        release_lock.set()
        _restore_diagnostic_surfaces(ordinary)

    assert completed_while_locked is True
    assert holder.is_alive() is False
    assert invoker.is_alive() is False
    assert child_ran.is_set()
    assert failures == []
    assert results == [
        "stdout-write-result",
        "stderr-writelines-result",
        "stderr-flush-result",
        "buffer-write-result",
        "buffer-write-result",
        "buffer-writelines-result",
        "buffer-flush-result",
        "logger-result",
    ]
    assert stream_calls == [
        ("stdout-write", "stdout-value"),
        ("stderr-writelines", ("stderr-value",)),
        ("stderr-flush", ""),
        ("buffer-write", b"buffer-value"),
        ("buffer-write", b"captured-buffer-value"),
        ("buffer-writelines", (b"captured-line",)),
        ("buffer-flush", ""),
    ]
    assert logger_calls == ["retired-log"]
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)


def test_marker_lookalikes_remain_third_party_owned(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("resume-marker-lookalikes")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    ordinary = _diagnostic_surfaces()
    captured_genuine: list[tuple[object, object, object, object]] = []

    def capture_generation(_html: str) -> bytes:
        captured_genuine.append(_diagnostic_surfaces())
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        capture_generation,
    )
    assert render_resume_pdf_from_html("<p>capture</p>") == _VALID_ONE_PAGE_PDF
    genuine = captured_genuine[0]

    class LookalikeStream:
        def __init__(self, copied_state: object) -> None:
            self._active = False
            self._state = copied_state
            self._token = resume_rendering._DIAGNOSTIC_WRAPPER_TOKEN
            self._stream = StringIO()

        def write(self, value: str) -> int:
            return self._stream.write(value)

        def writelines(self, values: object) -> None:
            self._stream.writelines(values)  # type: ignore[arg-type]

        def flush(self) -> None:
            self._stream.flush()

        def getvalue(self) -> str:
            return self._stream.getvalue()

    lookalike_stdout = LookalikeStream(
        genuine[0]._state,  # type: ignore[attr-defined]
    )
    lookalike_stderr = LookalikeStream(
        genuine[1]._state,  # type: ignore[attr-defined]
    )
    logger_calls: list[str] = []
    thread_start_calls: list[threading.Thread] = []

    def lookalike_logger_handle(
        target_logger: logging.Logger,
        record: logging.LogRecord,
    ) -> object:
        logger_calls.append(record.getMessage())
        return ordinary[2](target_logger, record)  # type: ignore[operator]

    def lookalike_thread_start(
        thread: threading.Thread,
        *args: object,
        **kwargs: object,
    ) -> object:
        thread_start_calls.append(thread)
        return ordinary[3](thread, *args, **kwargs)  # type: ignore[operator]

    for function, wrapper in (
        (lookalike_logger_handle, genuine[2]),
        (lookalike_thread_start, genuine[3]),
    ):
        function._active = False  # type: ignore[attr-defined]
        function._state = wrapper._state  # type: ignore[attr-defined]
        function._token = resume_rendering._DIAGNOSTIC_WRAPPER_TOKEN  # type: ignore[attr-defined]
        function.__wrapped__ = wrapper  # type: ignore[attr-defined]

    lookalikes = (
        lookalike_stdout,
        lookalike_stderr,
        lookalike_logger_handle,
        lookalike_thread_start,
    )
    _restore_diagnostic_surfaces(lookalikes)

    def noisy_success(_html: str) -> bytes:
        _emit_test_diagnostics(logger, "owned-lookalike-render")
        _spawn_joined_mixed_diagnostic_tree(logger, "owned-lookalike-tree")
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        noisy_success,
    )

    results: list[bytes] = []
    preserved: list[bool] = []
    execution_ids: list[int] = []
    try:
        for cycle in range(2):
            results.append(render_resume_pdf_from_html(f"<p>{cycle}</p>"))
            preserved.append(_diagnostic_surfaces() == lookalikes)
        print("lookalike-direct-stdout")
        print("lookalike-direct-stderr", file=sys.stderr)
        logger.debug("lookalike-direct-log")

        def target_body() -> None:
            execution_ids.append(threading.get_ident())

        class RunBodyThread(threading.Thread):
            def run(self) -> None:
                execution_ids.append(threading.get_ident())

        children = [threading.Thread(target=target_body), RunBodyThread()]
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=5)
            assert child.is_alive() is False
    finally:
        _restore_diagnostic_surfaces(ordinary)

    assert results == [_VALID_ONE_PAGE_PDF, _VALID_ONE_PAGE_PDF]
    assert preserved == [True, True]
    assert lookalike_stdout.getvalue() == "lookalike-direct-stdout\n"
    assert lookalike_stderr.getvalue() == "lookalike-direct-stderr\n"
    assert logger_calls == ["lookalike-direct-log"]
    assert [record.getMessage() for record in caplog.records] == [
        "lookalike-direct-log",
    ]
    assert len(thread_start_calls) == 10
    assert len(execution_ids) == 2
    assert all(thread_ident != threading.get_ident() for thread_ident in execution_ids)
    assert resume_rendering._DIAGNOSTIC_HOOKS is None
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    assert _owned_surface_depths() == (0, 0, 0, 0)


def test_scope_does_not_join_or_retain_an_escaped_descendant(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = logging.getLogger("resume-escaped-descendant")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    release_child = threading.Event()
    child_started = threading.Event()
    children: list[threading.Thread] = []
    stdout_before = sys.stdout
    stderr_before = sys.stderr
    logger_handle_before = logging.Logger.handle
    thread_start_before = threading.Thread.start

    def escaped_target() -> None:
        child_started.set()
        assert release_child.wait(timeout=5)
        _emit_test_diagnostics(logger, "escaped-after-scope")

    def browser_with_escaped_child(_html: str) -> bytes:
        child = threading.Thread(target=escaped_target)
        children.append(child)
        child.start()
        assert child_started.wait(timeout=5)
        return _VALID_ONE_PAGE_PDF

    monkeypatch.setattr(
        resume_rendering,
        "_render_pdf_with_playwright",
        browser_with_escaped_child,
    )

    capsys.readouterr()
    result = render_resume_pdf_from_html("<p>Synthetic resume.</p>")

    assert result == _VALID_ONE_PAGE_PDF
    assert len(children) == 1
    assert children[0].is_alive()
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before
    assert logging.Logger.handle is logger_handle_before
    assert threading.Thread.start is thread_start_before
    assert resume_rendering._ACTIVE_DIAGNOSTIC_SCOPES == set()
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}

    release_child.set()
    children[0].join(timeout=5)

    assert children[0].is_alive() is False
    assert resume_rendering._THREAD_DIAGNOSTIC_SCOPES == {}
    captured_io = capsys.readouterr()
    assert captured_io.out == "escaped-after-scope-stdout\n"
    assert captured_io.err == "escaped-after-scope-stderr\n"
    assert [record.getMessage() for record in caplog.records] == [
        "escaped-after-scope-log",
    ]
