"""Bounded, caller-injected resume rendering helpers."""

from __future__ import annotations

import html as html_lib
import logging
import re
import sys
import threading
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import resources
from io import BytesIO
from pathlib import Path
from types import MethodType
from typing import Any
from urllib.parse import urlsplit

import yaml
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from jinja2 import ChainableUndefined
from jinja2.sandbox import SandboxedEnvironment
from markupsafe import Markup, escape
from pypdf import PdfReader
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from career_agent_workbench.errors import WorkflowError

MAX_RESUME_YAML_BYTES = 2_000_000
MAX_TEMPLATE_CHARS = 1_000_000
MAX_RENDERED_HTML_CHARS = 2_000_000
MAX_RESUME_PDF_BYTES = 20_000_000
MAX_RESUME_PDF_PAGES = 50
MAX_RENDERED_JOD_MATCHED_SKILLS_PER_CATEGORY = 8
MAX_RESUME_DATA_DEPTH = 64
MAX_RESUME_DATA_NODES = 100_000
MAX_RESUME_DATA_CHARS = 2_000_000

_DEFAULT_TEMPLATE_PARTS = ("templates", "resume", "master_resume.html.j2")
# A Unicode scalar value occupies at most four bytes in well-formed UTF-8.
_MAX_TEMPLATE_UTF8_BYTES = MAX_TEMPLATE_CHARS * 4
_RESUME_RICH_TAG_RE = re.compile(
    r"</?\s*(a|b|br|div|em|i|p|span|strong)\b",
    re.IGNORECASE,
)
_RESUME_LINK_RE = re.compile(
    r"(?<![A-Za-z0-9_@])"
    r"(?:https?://|mailto:|javascript:|data:|//)"
    r"[^\s<>()]*",
    re.IGNORECASE,
)
_MAILTO_RE = re.compile(
    r"mailto:[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
)
_ATTRIBUTE_BREAKERS = frozenset("\"'`<>\\")
_DIAGNOSTIC_SCOPE_LOCK = threading.RLock()
_ACTIVE_DIAGNOSTIC_SCOPES: set[object] = set()
_THREAD_DIAGNOSTIC_SCOPES: dict[int, set[object]] = {}
_DIAGNOSTIC_HOOKS: _DiagnosticHooks | None = None
_DIAGNOSTIC_WRAPPER_TOKEN = object()


class ResumeRenderingError(WorkflowError):
    """Raised when bounded resume rendering cannot complete safely."""

    __slots__ = ()


class _ResumeDataBoundaryError(Exception):
    __slots__ = ()


class _ResumeDataState:
    __slots__ = ("active_container_ids", "characters", "nodes")

    def __init__(self) -> None:
        self.active_container_ids: set[int] = set()
        self.characters = 0
        self.nodes = 0

    def add_node(self) -> None:
        self.nodes += 1
        if self.nodes > MAX_RESUME_DATA_NODES:
            raise _ResumeDataBoundaryError

    def add_characters(self, count: int) -> None:
        self.characters += count
        if self.characters > MAX_RESUME_DATA_CHARS:
            raise _ResumeDataBoundaryError


class _DiagnosticGenerationState:
    __slots__ = ("active", "token")

    def __init__(self, token: object) -> None:
        self.active = True
        self.token = token


class _ThreadSelectiveBuffer:
    __slots__ = ("_owner", "_stream")

    def __init__(self, owner: _ThreadSelectiveStream, stream: Any) -> None:
        self._owner = owner
        self._stream = stream

    def write(self, value: Any) -> Any:
        if not self._owner._state.active:
            return self._stream.write(value)
        if _wrapper_suppresses_current_thread(self._owner):
            return len(value)
        return self._stream.write(value)

    def writelines(self, values: Any) -> Any:
        if not self._owner._state.active:
            return self._stream.writelines(values)
        if _wrapper_suppresses_current_thread(self._owner):
            for _value in values:
                pass
            return
        return self._stream.writelines(values)

    def flush(self) -> Any:
        if not self._owner._state.active:
            return self._stream.flush()
        if _wrapper_suppresses_current_thread(self._owner):
            return None
        return self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _ThreadSelectiveStream:
    __slots__ = ("_state", "_stream")

    def __init__(
        self,
        stream: Any,
        state: _DiagnosticGenerationState,
    ) -> None:
        self._state = state
        self._stream = stream

    def write(self, value: Any) -> Any:
        if not self._state.active:
            return self._stream.write(value)
        if _wrapper_suppresses_current_thread(self):
            return len(value)
        return self._stream.write(value)

    def writelines(self, values: Any) -> Any:
        if not self._state.active:
            return self._stream.writelines(values)
        if _wrapper_suppresses_current_thread(self):
            for _value in values:
                pass
            return
        return self._stream.writelines(values)

    def flush(self) -> Any:
        if not self._state.active:
            return self._stream.flush()
        if _wrapper_suppresses_current_thread(self):
            return None
        return self._stream.flush()

    @property
    def buffer(self) -> Any:
        if not self._state.active:
            return self._stream.buffer
        with _DIAGNOSTIC_SCOPE_LOCK:
            active = self._state.active
        if not active:
            return self._stream.buffer
        return _ThreadSelectiveBuffer(self, self._stream.buffer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _flatten_target(self) -> None:
        self._stream = _flatten_stream_target(self._stream)


class _OwnedLoggerHandle:
    __slots__ = ("_state", "_target")

    def __init__(
        self,
        target: Any,
        state: _DiagnosticGenerationState,
    ) -> None:
        self._state = state
        self._target = target

    def __get__(
        self,
        instance: logging.Logger | None,
        _owner: type[logging.Logger],
    ) -> Any:
        if instance is None:
            return self
        return MethodType(self, instance)

    def __call__(
        self,
        logger: logging.Logger,
        record: logging.LogRecord,
    ) -> Any:
        if not self._state.active:
            return _invoke_descriptor_target(self._target, logger, record)
        if _wrapper_suppresses_current_thread(self):
            return None
        return _invoke_descriptor_target(self._target, logger, record)

    def _flatten_target(self) -> None:
        self._target = _flatten_logger_target(self._target)


class _OwnedThreadStart:
    __slots__ = ("_state", "_target")

    def __init__(
        self,
        target: Any,
        state: _DiagnosticGenerationState,
    ) -> None:
        self._state = state
        self._target = target

    def __get__(
        self,
        instance: threading.Thread | None,
        _owner: type[threading.Thread],
    ) -> Any:
        if instance is None:
            return self
        return MethodType(self, instance)

    def __call__(
        self,
        thread: threading.Thread,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not self._state.active:
            return _invoke_descriptor_target(
                self._target,
                thread,
                *args,
                **kwargs,
            )
        return _start_thread_with_diagnostic_scopes(
            self,
            thread,
            args,
            kwargs,
        )

    def _flatten_target(self) -> None:
        self._target = _flatten_thread_start_target(self._target)


class _DiagnosticHooks:
    __slots__ = (
        "logger_handle",
        "retired",
        "state",
        "stderr_proxy",
        "stdout_proxy",
        "thread_start",
    )

    def __init__(
        self,
        *,
        stdout_target: Any,
        stderr_target: Any,
        logger_target: Any,
        thread_start_target: Any,
    ) -> None:
        self.state = _DiagnosticGenerationState(_DIAGNOSTIC_WRAPPER_TOKEN)
        self.stdout_proxy = _ThreadSelectiveStream(
            stdout_target,
            self.state,
        )
        self.stderr_proxy = _ThreadSelectiveStream(
            stderr_target,
            self.state,
        )
        self.logger_handle = _OwnedLoggerHandle(
            logger_target,
            self.state,
        )
        self.thread_start = _OwnedThreadStart(
            thread_start_target,
            self.state,
        )
        self.retired = False

    def retire(self) -> None:
        if self.retired:
            return
        self.retired = True
        self.state.active = False
        self.stdout_proxy._flatten_target()
        self.stderr_proxy._flatten_target()
        self.logger_handle._flatten_target()
        self.thread_start._flatten_target()


def _wrapper_suppresses_current_thread(_wrapper: Any) -> bool:
    with _DIAGNOSTIC_SCOPE_LOCK:
        thread_key = id(threading.current_thread())
        owned_scopes = _THREAD_DIAGNOSTIC_SCOPES.get(thread_key)
        return bool(
            owned_scopes and not owned_scopes.isdisjoint(_ACTIVE_DIAGNOSTIC_SCOPES)
        )


def _invoke_descriptor_target(
    target: Any,
    instance: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    descriptor = getattr(type(target), "__get__", None)
    bound_target = (
        target if descriptor is None else descriptor(target, instance, type(instance))
    )
    return bound_target(*args, **kwargs)


def _is_owned_stream(value: Any) -> bool:
    return (
        type(value) is _ThreadSelectiveStream
        and type(value._state) is _DiagnosticGenerationState
        and value._state.token is _DIAGNOSTIC_WRAPPER_TOKEN
    )


def _is_owned_logger_handle(value: Any) -> bool:
    return (
        type(value) is _OwnedLoggerHandle
        and type(value._state) is _DiagnosticGenerationState
        and value._state.token is _DIAGNOSTIC_WRAPPER_TOKEN
    )


def _is_owned_thread_start(value: Any) -> bool:
    return (
        type(value) is _OwnedThreadStart
        and type(value._state) is _DiagnosticGenerationState
        and value._state.token is _DIAGNOSTIC_WRAPPER_TOKEN
    )


def _flatten_stream_target(value: Any) -> Any:
    seen: set[int] = set()
    while _is_owned_stream(value) and id(value) not in seen:
        seen.add(id(value))
        value = value._stream
    return value


def _flatten_logger_target(value: Any) -> Any:
    seen: set[int] = set()
    while _is_owned_logger_handle(value) and id(value) not in seen:
        seen.add(id(value))
        value = value._target
    return value


def _flatten_thread_start_target(value: Any) -> Any:
    seen: set[int] = set()
    while _is_owned_thread_start(value) and id(value) not in seen:
        seen.add(id(value))
        value = value._target
    return value


def _current_logger_handle() -> Any:
    return vars(logging.Logger)["handle"]


def _current_thread_start() -> Any:
    return vars(threading.Thread)["start"]


def _start_thread_with_diagnostic_scopes(
    hook: _OwnedThreadStart,
    thread: threading.Thread,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    inherited_scopes: set[object] = set()
    with _DIAGNOSTIC_SCOPE_LOCK:
        parent_scopes = _THREAD_DIAGNOSTIC_SCOPES.get(
            id(threading.current_thread()),
        )
        if parent_scopes:
            inherited_scopes = parent_scopes.intersection(
                _ACTIVE_DIAGNOSTIC_SCOPES,
            )
    if not inherited_scopes:
        return _invoke_descriptor_target(hook._target, thread, *args, **kwargs)

    child_key = id(thread)
    thread_state = object.__getattribute__(thread, "__dict__")
    missing = object()
    previous_bootstrap = dict.get(thread_state, "_bootstrap_inner", missing)
    original_bootstrap_inner = object.__getattribute__(
        thread,
        "_bootstrap_inner",
    )

    def scoped_bootstrap_inner(
        *bootstrap_args: Any,
        **bootstrap_kwargs: Any,
    ) -> Any:
        registered_scopes: set[object] = set()
        with _DIAGNOSTIC_SCOPE_LOCK:
            registered_scopes = inherited_scopes.intersection(
                _ACTIVE_DIAGNOSTIC_SCOPES,
            )
            if registered_scopes:
                _THREAD_DIAGNOSTIC_SCOPES.setdefault(
                    child_key,
                    set(),
                ).update(registered_scopes)
        try:
            return original_bootstrap_inner(
                *bootstrap_args,
                **bootstrap_kwargs,
            )
        finally:
            with _DIAGNOSTIC_SCOPE_LOCK:
                owned_scopes = _THREAD_DIAGNOSTIC_SCOPES.get(child_key)
                if owned_scopes is not None:
                    owned_scopes.difference_update(registered_scopes)
                    if not owned_scopes:
                        _THREAD_DIAGNOSTIC_SCOPES.pop(child_key, None)

    dict.__setitem__(
        thread_state,
        "_bootstrap_inner",
        scoped_bootstrap_inner,
    )
    try:
        return _invoke_descriptor_target(
            hook._target,
            thread,
            *args,
            **kwargs,
        )
    finally:
        current_bootstrap = dict.get(
            thread_state,
            "_bootstrap_inner",
            missing,
        )
        if current_bootstrap is scoped_bootstrap_inner:
            if previous_bootstrap is missing:
                dict.pop(thread_state, "_bootstrap_inner", None)
            else:
                dict.__setitem__(
                    thread_state,
                    "_bootstrap_inner",
                    previous_bootstrap,
                )


def _hook_generation_is_current(hooks: _DiagnosticHooks) -> bool:
    return (
        sys.stdout is hooks.stdout_proxy
        and sys.stderr is hooks.stderr_proxy
        and _current_logger_handle() is hooks.logger_handle
        and _current_thread_start() is hooks.thread_start
    )


def _rearm_diagnostic_hooks_locked() -> None:
    global _DIAGNOSTIC_HOOKS

    previous = _DIAGNOSTIC_HOOKS
    if previous is not None and _hook_generation_is_current(previous):
        return

    current_stdout = sys.stdout
    current_stderr = sys.stderr
    current_logger_handle = _current_logger_handle()
    current_thread_start = _current_thread_start()

    stdout_target = (
        previous.stdout_proxy._stream
        if previous is not None and current_stdout is previous.stdout_proxy
        else current_stdout
    )
    stderr_target = (
        previous.stderr_proxy._stream
        if previous is not None and current_stderr is previous.stderr_proxy
        else current_stderr
    )
    logger_target = (
        previous.logger_handle._target
        if previous is not None and current_logger_handle is previous.logger_handle
        else current_logger_handle
    )
    thread_start_target = (
        previous.thread_start._target
        if previous is not None and current_thread_start is previous.thread_start
        else current_thread_start
    )

    replacement = _DiagnosticHooks(
        stdout_target=_flatten_stream_target(stdout_target),
        stderr_target=_flatten_stream_target(stderr_target),
        logger_target=_flatten_logger_target(logger_target),
        thread_start_target=_flatten_thread_start_target(thread_start_target),
    )
    sys.stdout = replacement.stdout_proxy
    sys.stderr = replacement.stderr_proxy
    logging.Logger.handle = replacement.logger_handle
    threading.Thread.start = replacement.thread_start
    _DIAGNOSTIC_HOOKS = replacement
    if previous is not None:
        previous.retire()


def _diagnostic_stage_checkpoint() -> None:
    with _DIAGNOSTIC_SCOPE_LOCK:
        if _ACTIVE_DIAGNOSTIC_SCOPES:
            _rearm_diagnostic_hooks_locked()


def _restore_diagnostic_hooks_locked(hooks: _DiagnosticHooks) -> None:
    hooks.retire()
    current_stdout = sys.stdout
    current_stderr = sys.stderr
    current_logger_handle = _current_logger_handle()
    current_thread_start = _current_thread_start()

    if current_stdout is hooks.stdout_proxy:
        sys.stdout = _flatten_stream_target(hooks.stdout_proxy._stream)
    elif _is_owned_stream(current_stdout) and not current_stdout._state.active:
        sys.stdout = _flatten_stream_target(current_stdout)

    if current_stderr is hooks.stderr_proxy:
        sys.stderr = _flatten_stream_target(hooks.stderr_proxy._stream)
    elif _is_owned_stream(current_stderr) and not current_stderr._state.active:
        sys.stderr = _flatten_stream_target(current_stderr)

    if current_logger_handle is hooks.logger_handle:
        logging.Logger.handle = _flatten_logger_target(hooks.logger_handle._target)
    elif (
        _is_owned_logger_handle(current_logger_handle)
        and not current_logger_handle._state.active
    ):
        logging.Logger.handle = _flatten_logger_target(current_logger_handle)

    if current_thread_start is hooks.thread_start:
        threading.Thread.start = _flatten_thread_start_target(
            hooks.thread_start._target,
        )
    elif (
        _is_owned_thread_start(current_thread_start)
        and not current_thread_start._state.active
    ):
        threading.Thread.start = _flatten_thread_start_target(current_thread_start)


@contextmanager
def _suppress_backend_diagnostics() -> Iterator[None]:
    global _DIAGNOSTIC_HOOKS

    scope = object()
    thread_key = id(threading.current_thread())
    with _DIAGNOSTIC_SCOPE_LOCK:
        _ACTIVE_DIAGNOSTIC_SCOPES.add(scope)
        _THREAD_DIAGNOSTIC_SCOPES.setdefault(thread_key, set()).add(scope)
        _rearm_diagnostic_hooks_locked()
    try:
        yield
    finally:
        with _DIAGNOSTIC_SCOPE_LOCK:
            _ACTIVE_DIAGNOSTIC_SCOPES.discard(scope)
            for owner_key, owned_scopes in list(_THREAD_DIAGNOSTIC_SCOPES.items()):
                owned_scopes.discard(scope)
                if not owned_scopes:
                    _THREAD_DIAGNOSTIC_SCOPES.pop(owner_key, None)
            if _ACTIVE_DIAGNOSTIC_SCOPES:
                _rearm_diagnostic_hooks_locked()
            elif _DIAGNOSTIC_HOOKS is not None:
                hooks = _DIAGNOSTIC_HOOKS
                _restore_diagnostic_hooks_locked(hooks)
                _DIAGNOSTIC_HOOKS = None


def sanitize_resume_url(value: object) -> str:
    """Return a clean supported URL, or an empty string when it is unsafe."""

    if not isinstance(value, str) or not value:
        return ""
    if value != value.strip() or value.startswith("//"):
        return ""
    if any(
        character.isspace()
        or unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        or character in _ATTRIBUTE_BREAKERS
        for character in value
    ):
        return ""
    if not _has_safe_percent_encoding(value):
        return ""

    colon = value.find(":")
    if colon <= 0:
        return ""
    scheme = value[:colon]
    if scheme not in {"http", "https", "mailto"}:
        return ""

    if scheme == "mailto":
        return value if _MAILTO_RE.fullmatch(value) else ""

    parse_failed = False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        parse_failed = True
        parsed = None
        port = None
    if parse_failed or parsed is None:
        return ""
    if (
        parsed.scheme != scheme
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.netloc.endswith(":")
        or port is not None
        and not (1 <= port <= 65535)
    ):
        return ""
    return value


def _has_safe_percent_encoding(value: str) -> bool:
    index = 0
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue

        octets: list[int] = []
        while index < len(value) and value[index] == "%":
            if (
                index + 2 >= len(value)
                or value[index + 1] not in hexadecimal
                or value[index + 2] not in hexadecimal
            ):
                return False
            octets.append(int(value[index + 1 : index + 3], 16))
            index += 3

        if any(octet <= 0x1F or octet == 0x7F for octet in octets):
            return False
        if any(0x80 <= octet <= 0x9F for octet in octets):
            try:
                decoded = bytes(octets).decode("utf-8")
            except UnicodeDecodeError:
                return False
            if any(
                ord(character) <= 0x1F
                or 0x7F <= ord(character) <= 0x9F
                or unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
                for character in decoded
            ):
                return False
    return True


def linkify(value: object) -> Markup:
    """Escape plain text and link only clean URLs under the shared policy."""

    text = "" if value is None else str(value)
    parts: list[str | Markup] = []
    cursor = 0
    for match in _RESUME_LINK_RE.finditer(text):
        candidate = match.group(0)
        url = candidate.rstrip(".,;:!?")
        trailing = candidate[len(url) :]
        parts.append(escape(text[cursor : match.start()]))
        safe_url = sanitize_resume_url(url)
        if safe_url:
            parts.append(
                Markup(
                    f'<a href="{escape(safe_url)}">{escape(url)}</a>',
                )
            )
        else:
            parts.append(escape(url))
        parts.append(escape(trailing))
        cursor = match.end()
    parts.append(escape(text[cursor:]))
    return Markup("").join(parts)


def rich_text(value: object) -> Markup:
    """Return sanitized resume rich text as trusted rendered markup."""

    return Markup(sanitize_resume_rich_text(value))


def sanitize_resume_rich_text(value: object) -> str:
    """Allow a tiny formatting subset while applying the shared URL policy."""

    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    text = html_lib.unescape(text)
    if not _RESUME_RICH_TAG_RE.search(text):
        return str(linkify(text))

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    return "".join(_resume_rich_node_markup(child) for child in soup.contents).strip()


def _resume_rich_node_markup(node: object) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(linkify(str(node)))
    if not isinstance(node, Tag):
        return ""

    name = (node.name or "").lower()
    if name == "br":
        return "<br/>"

    inner = "".join(_resume_rich_node_markup(child) for child in node.children)
    if name in {"b", "strong"}:
        return f"<b>{inner}</b>"
    if name == "span" and _span_is_semantic_strong(node):
        return f"<b>{inner}</b>"
    if name in {"i", "em"}:
        return f"<i>{inner}</i>"
    if name == "a":
        if set(node.attrs) != {"href"}:
            return inner
        safe_href = sanitize_resume_url(str(node.get("href") or ""))
        if safe_href:
            return f'<a href="{escape(safe_href)}">{inner}</a>'
        return inner
    return inner


def _span_is_semantic_strong(node: object) -> bool:
    if not isinstance(node, Tag):
        return False
    if str(node.get("data-streamdown") or "").lower() == "strong":
        return True
    classes = node.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    return "font-semibold" in classes or "font-bold" in classes


def render_skill_items(value: object) -> str:
    """Render a skill collection with deterministic duplicate suppression."""

    return ", ".join(_rendered_skill_items(value, seen=set()))


def render_core_skill_rows(value: object) -> list[dict[str, str]]:
    """Build data-only rows for the resume template's skills section."""

    if not isinstance(value, Mapping):
        return []
    raw_bullets = value.get("bullet_points")
    if not isinstance(raw_bullets, list):
        return []

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for bullet in raw_bullets:
        if isinstance(bullet, str):
            text = bullet.strip()
            if text:
                rows.append({"category": "", "text": text})
            continue
        if not isinstance(bullet, Mapping):
            continue
        items = _rendered_skill_items(bullet, seen=seen)
        if items:
            rows.append(
                {
                    "category": str(bullet.get("category") or "").strip(),
                    "text": ", ".join(items),
                }
            )
    return rows


def _rendered_skill_items(value: object, *, seen: set[str]) -> list[str]:
    if not isinstance(value, Mapping):
        return []

    items = value.get("items")
    if isinstance(items, Mapping):
        primary = _string_list(items.get("primary"))
        additional = _string_list(items.get("additional"))
        display_by_key = {
            _skill_key(item): item for item in primary + additional if _skill_key(item)
        }
        aliases = _skill_aliases(items, display_by_key)
        additional_by_key = {
            _skill_key(item): item for item in additional if _skill_key(item)
        }
        matched: list[str] = []
        matched_keys: set[str] = set()
        for item in _string_list(value.get("jod_matched_items")):
            item_key = _skill_key(item)
            canonical = aliases.get(item_key, display_by_key.get(item_key, item))
            canonical_key = _skill_key(canonical)
            additional_item = additional_by_key.get(canonical_key)
            if additional_item is None or canonical_key in matched_keys:
                continue
            matched.append(additional_item)
            matched_keys.add(canonical_key)
            if len(matched) == MAX_RENDERED_JOD_MATCHED_SKILLS_PER_CATEGORY:
                break
        candidates = primary + matched
    else:
        candidates = _string_list(items)

    rendered: list[str] = []
    for item in candidates:
        key = _skill_key(item)
        if key and key not in seen:
            rendered.append(item)
            seen.add(key)
    return rendered


def _skill_aliases(
    items: Mapping[str, object],
    display_by_key: Mapping[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    raw_match_terms = items.get("match_terms")
    if not isinstance(raw_match_terms, Mapping):
        return aliases
    for raw_skill, raw_terms in raw_match_terms.items():
        skill = display_by_key.get(_skill_key(raw_skill))
        if skill is None:
            continue
        for term in _string_list(raw_terms):
            key = _skill_key(term)
            if key:
                aliases[key] = skill
    return aliases


def _skill_key(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("c++", "cplusplus").replace("c#", "csharp")
    replacements = {
        "python 3": "python",
        "rest api": "restful api",
        "rest apis": "restful apis",
        "managed postgresql": "postgresql",
        "linux environments": "linux",
        "cloud monitoring": "observability",
    }
    text = replacements.get(text, text)
    return "".join(character for character in text if character.isalnum())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def load_resume(path: Path) -> dict[str, Any]:
    """Load an exact caller-supplied bounded YAML mapping."""

    if not isinstance(path, Path):
        raise ResumeRenderingError("Resume YAML could not be loaded.")
    read_failed = False
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_RESUME_YAML_BYTES + 1)
    except Exception:  # noqa: BLE001 - exact path read failures stay content-free.
        read_failed = True
        payload = b""
    if read_failed or not isinstance(payload, bytes):
        raise ResumeRenderingError("Resume YAML could not be loaded.")
    if len(payload) > MAX_RESUME_YAML_BYTES:
        raise ResumeRenderingError("Resume YAML exceeds the supported size.")

    decode_failed = False
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        decode_failed = True
        text = ""
    if decode_failed:
        raise ResumeRenderingError("Resume YAML could not be loaded.")

    parse_failed = False
    try:
        value = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - normalize every parser failure at the boundary.
        parse_failed = True
        value = None
    if parse_failed:
        raise ResumeRenderingError("Resume YAML could not be loaded.")
    if type(value) is not dict:
        raise ResumeRenderingError("Resume YAML must contain a mapping.")
    return dict.copy(value)


def render_resume_html(
    *,
    yaml_path: Path,
    template_path: Path | None = None,
) -> str:
    """Render an exact caller-supplied YAML resume to bounded HTML."""

    return render_resume_html_from_mapping(
        resume=load_resume(yaml_path),
        template_path=template_path,
    )


def render_resume_html_from_mapping(
    *,
    resume: Mapping[str, Any],
    template_path: Path | None = None,
) -> str:
    """Render a caller-supplied mapping with an exact or packaged template."""

    safe_resume = _materialize_resume_mapping(resume)
    template_source = _read_template(template_path)

    environment = SandboxedEnvironment(
        loader=None,
        autoescape=True,
        undefined=ChainableUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["linkify"] = linkify
    environment.filters["rich_text"] = rich_text
    environment.filters["render_core_skill_rows"] = render_core_skill_rows
    environment.filters["render_skill_items"] = render_skill_items
    environment.filters["safe_url"] = sanitize_resume_url

    render_failed = False
    html_too_large = False
    stream: Any = None
    chunks: list[str] = []
    try:
        template = environment.from_string(template_source)
        stream = template.generate(data=safe_resume)
        rendered_chars = 0
        for chunk in stream:
            rendered_chars += len(chunk)
            if rendered_chars > MAX_RENDERED_HTML_CHARS:
                html_too_large = True
                break
            chunks.append(chunk)
    except Exception:  # noqa: BLE001 - keep Jinja and mapping details private.
        render_failed = True
    close_failed = False
    if stream is not None:
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - generator cleanup details stay private.
            close_failed = True
    render_failed = render_failed or close_failed
    if render_failed:
        raise ResumeRenderingError("Resume HTML could not be rendered.")
    if html_too_large:
        raise ResumeRenderingError("Resume HTML exceeds the supported size.")
    return "".join(chunks)


def _materialize_resume_mapping(resume: object) -> dict[str, Any]:
    materialization_failed = False
    try:
        value = _materialize_resume_value(
            resume,
            state=_ResumeDataState(),
            depth=0,
        )
    except Exception:  # noqa: BLE001 - mapping traversal details stay private.
        materialization_failed = True
        value = None
    if materialization_failed or type(value) is not dict:
        raise ResumeRenderingError("Resume data contains unsupported values.")
    return value


def _materialize_resume_value(
    value: object,
    *,
    state: _ResumeDataState,
    depth: int,
) -> Any:
    if depth > MAX_RESUME_DATA_DEPTH:
        raise _ResumeDataBoundaryError
    state.add_node()

    value_type = type(value)
    if value is None or value_type is bool or value_type is float:
        return value
    if value_type is int:
        approximate_digits = (abs(value).bit_length() * 30103) // 100000 + 2
        state.add_characters(approximate_digits)
        return value
    if value_type is str:
        state.add_characters(len(value))
        return value

    if value_type is list or value_type is tuple:
        container_id = id(value)
        if container_id in state.active_container_ids:
            raise _ResumeDataBoundaryError
        state.active_container_ids.add(container_id)
        try:
            materialized = [
                _materialize_resume_value(
                    item,
                    state=state,
                    depth=depth + 1,
                )
                for item in value
            ]
            return materialized if value_type is list else tuple(materialized)
        finally:
            state.active_container_ids.remove(container_id)

    try:
        inert_items = dict.items(value)
    except TypeError:
        raise _ResumeDataBoundaryError from None

    if inert_items is not None:
        container_id = id(value)
        if container_id in state.active_container_ids:
            raise _ResumeDataBoundaryError
        state.active_container_ids.add(container_id)
        try:
            result: dict[str, Any] = {}
            for key, nested_value in inert_items:
                if type(key) is not str or not _is_safe_resume_key(key):
                    raise _ResumeDataBoundaryError
                state.add_characters(len(key))
                result[key] = _materialize_resume_value(
                    nested_value,
                    state=state,
                    depth=depth + 1,
                )
            return result
        finally:
            state.active_container_ids.remove(container_id)

    raise _ResumeDataBoundaryError


def _is_safe_resume_key(value: str) -> bool:
    return not any(
        ord(character) <= 0x1F
        or 0x7F <= ord(character) <= 0x9F
        or unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in value
    )


def _read_template(template_path: Path | None) -> str:
    if template_path is not None and not isinstance(template_path, Path):
        raise ResumeRenderingError("Resume template could not be loaded.")
    read_failed = False
    if template_path is None:
        try:
            resource = resources.files("career_agent_workbench").joinpath(
                *_DEFAULT_TEMPLATE_PARTS
            )
            with resource.open("rb") as handle:
                payload = handle.read(_MAX_TEMPLATE_UTF8_BYTES + 1)
        except Exception:  # noqa: BLE001 - resource internals must never escape.
            read_failed = True
            payload = b""
    else:
        try:
            with template_path.open("rb") as handle:
                payload = handle.read(_MAX_TEMPLATE_UTF8_BYTES + 1)
        except Exception:  # noqa: BLE001 - exact path read failures stay content-free.
            read_failed = True
            payload = b""
    if read_failed or not isinstance(payload, bytes):
        raise ResumeRenderingError("Resume template could not be loaded.")
    if len(payload) > _MAX_TEMPLATE_UTF8_BYTES:
        raise ResumeRenderingError("Resume template exceeds the supported size.")
    decode_failed = False
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        decode_failed = True
        text = ""
    if decode_failed:
        raise ResumeRenderingError("Resume template could not be loaded.")
    if len(text) > MAX_TEMPLATE_CHARS:
        raise ResumeRenderingError("Resume template exceeds the supported size.")
    return text


def render_resume_pdf_from_html(html: str) -> bytes:
    """Render bounded HTML to PDF using an isolated browser or ReportLab."""

    if not isinstance(html, str):
        raise ResumeRenderingError("Resume HTML could not be rendered.")
    if len(html) > MAX_RENDERED_HTML_CHARS:
        raise ResumeRenderingError("Resume HTML exceeds the supported size.")

    with _suppress_backend_diagnostics():
        browser_failed = False
        try:
            _diagnostic_stage_checkpoint()
            result = _render_pdf_with_playwright(html)
            _diagnostic_stage_checkpoint()
            result = _validated_pdf_bytes(result)
        except Exception:  # noqa: BLE001 - optional browser failures trigger fallback.
            browser_failed = True
            result = b""
        if not browser_failed:
            return result

        fallback_failed = False
        try:
            _diagnostic_stage_checkpoint()
            result = _render_text_pdf_from_html(html)
            _diagnostic_stage_checkpoint()
            result = _validated_pdf_bytes(result)
        except Exception:  # noqa: BLE001 - ReportLab details are private diagnostics.
            fallback_failed = True
            result = b""
    if fallback_failed:
        raise ResumeRenderingError("Resume PDF could not be rendered.")
    return result


def _validated_pdf_bytes(value: object) -> bytes:
    if type(value) is not bytes or not value:
        raise ValueError
    if len(value) > MAX_RESUME_PDF_BYTES:
        raise ValueError
    reader = PdfReader(BytesIO(value), strict=False)
    if not 1 <= len(reader.pages) <= MAX_RESUME_PDF_PAGES:
        raise ValueError
    return value


def _render_pdf_with_playwright(html: str) -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = browser.new_context(
                java_script_enabled=False,
                service_workers="block",
            )
            try:
                page = context.new_page()
                try:
                    page.route("**/*", lambda route: route.abort())
                    page.set_content(html, wait_until="load")
                    return page.pdf(format="Letter", print_background=True)
                finally:
                    page.close()
            finally:
                context.close()
        finally:
            browser.close()


def _render_text_pdf_from_html(html: str) -> bytes:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]

    buffer = BytesIO()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ResumeHtmlFallbackBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.4,
        leading=11.2,
        spaceAfter=2,
    )
    story: list[Any] = []
    for line in lines:
        story.append(Paragraph(html_lib.escape(line), body))
        story.append(Spacer(1, 1))
    if not story:
        story.append(Spacer(1, 1))

    document = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=54,
        leftMargin=54,
        topMargin=58,
        bottomMargin=52,
    )
    document.build(story)
    return buffer.getvalue()
