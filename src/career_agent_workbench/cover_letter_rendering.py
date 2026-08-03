"""Inert cover-letter sanitation, text normalization, and PDF rendering."""

from __future__ import annotations

import html
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate

MAX_COVER_LETTER_HTML_CHARS = 500_000
MAX_COVER_LETTER_PDF_BYTES = 20_000_000
_ALLOWED_TAGS = frozenset({"p", "div", "br", "strong", "b", "em", "i", "a"})
_BLOCK_TAGS = frozenset({"p", "div"})
_REMOVED_WITH_CONTENT = frozenset({"script", "style", "iframe", "object", "embed"})
_SAFE_LINK_SCHEMES = frozenset({"http", "https", "mailto"})
_PAGE_MARGIN = 54
_BODY_FONT_SIZE = 10
_BODY_LEADING = 13
_PARAGRAPH_SPACE_AFTER = 6


class CoverLetterRenderingError(ValueError):
    """Content-free cover-letter validation or rendering failure."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class RenderedCoverLetter:
    """One sanitized cover-letter object and deterministic PDF."""

    value: dict[str, Any]
    pdf: bytes


def blank_cover_letter() -> dict[str, Any]:
    """Return the inert editor object for a row without stored CLO state."""

    return {
        "schema_version": 1,
        "source": "manual",
        "body_html": "",
        "body_text": "",
    }


def sanitize_cover_letter_html(value: object) -> str:
    """Keep only bounded formatting and safe explicit links."""

    if type(value) is not str or len(value) > MAX_COVER_LETTER_HTML_CHARS:
        raise CoverLetterRenderingError("Cover letter input is invalid.")
    try:
        soup = BeautifulSoup(value, "html.parser")
        for comment in soup.find_all(string=lambda item: isinstance(item, Comment)):
            comment.extract()
        for tag in tuple(soup.find_all(True)):
            if not isinstance(tag, Tag):
                continue
            name = tag.name.casefold()
            if name in _REMOVED_WITH_CONTENT:
                tag.decompose()
                continue
            if name not in _ALLOWED_TAGS:
                tag.unwrap()
                continue
            if name == "a":
                href = tag.get("href")
                tag.attrs = {}
                if type(href) is not str or not _safe_href(href):
                    tag.unwrap()
                else:
                    tag.attrs = {"href": href}
            else:
                tag.attrs = {}
        rendered = str(soup)
    except CoverLetterRenderingError:
        raise
    except Exception:  # noqa: BLE001 - parser details remain private.
        raise CoverLetterRenderingError("Cover letter input is invalid.") from None
    if len(rendered) > MAX_COVER_LETTER_HTML_CHARS:
        raise CoverLetterRenderingError("Cover letter input is invalid.")
    return rendered


def cover_letter_plain_text(sanitized_html: object) -> str:
    """Return deterministic line-oriented text from already-sanitized HTML."""

    if type(sanitized_html) is not str:
        raise CoverLetterRenderingError("Cover letter input is invalid.")
    try:
        soup = BeautifulSoup(sanitized_html, "html.parser")
        pieces: list[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, NavigableString):
                pieces.append(str(node))
                return
            if not isinstance(node, Tag):
                return
            name = node.name.casefold()
            if name == "br":
                pieces.append("\n")
                return
            block = name in _BLOCK_TAGS
            if block and pieces and not pieces[-1].endswith("\n"):
                pieces.append("\n")
            for child in node.children:
                visit(child)
            if block:
                pieces.append("\n")

        for child in soup.children:
            visit(child)
        lines = [" ".join(line.split()) for line in "".join(pieces).splitlines()]
        normalized = "\n".join(line for line in lines if line).strip()
    except Exception:  # noqa: BLE001 - parser details remain private.
        raise CoverLetterRenderingError("Cover letter input is invalid.") from None
    return normalized


def render_cover_letter(body_html: object) -> RenderedCoverLetter:
    """Sanitize, normalize, and render one cover-letter edit."""

    sanitized = sanitize_cover_letter_html(body_html)
    plain_text = cover_letter_plain_text(sanitized)
    value = {
        "schema_version": 1,
        "source": "manual",
        "body_html": sanitized,
        "body_text": plain_text,
    }
    try:
        pdf = _render_pdf(sanitized, plain_text)
    except Exception:  # noqa: BLE001 - ReportLab details remain private.
        raise CoverLetterRenderingError("Cover letter could not be rendered.") from None
    return RenderedCoverLetter(value=value, pdf=pdf)


def _safe_href(value: str) -> bool:
    if len(value) > 4_096 or any(ord(character) < 32 for character in value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.scheme.casefold() not in _SAFE_LINK_SCHEMES:
        return False
    if parsed.scheme.casefold() in {"http", "https"}:
        return bool(parsed.netloc)
    return bool(parsed.path) and "@" in parsed.path


def _render_pdf(sanitized_html: str, plain_text: str) -> bytes:
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=LETTER,
        rightMargin=_PAGE_MARGIN,
        leftMargin=_PAGE_MARGIN,
        topMargin=_PAGE_MARGIN,
        bottomMargin=_PAGE_MARGIN,
        title="Cover Letter",
        author="Career Agent Workbench",
    )
    style = getSampleStyleSheet()["BodyText"]
    style.fontName = "Helvetica"
    style.fontSize = _BODY_FONT_SIZE
    style.leading = _BODY_LEADING
    style.spaceAfter = _PARAGRAPH_SPACE_AFTER
    story: list[Any] = []
    soup = BeautifulSoup(sanitized_html, "html.parser")
    inline_nodes: list[Any] = []

    def flush_inline_nodes() -> None:
        markup = "".join(_reportlab_markup(node) for node in inline_nodes).strip()
        inline_nodes.clear()
        if markup:
            story.append(Paragraph(markup, style))

    for node in soup.children:
        if isinstance(node, Tag) and node.name.casefold() in _BLOCK_TAGS:
            flush_inline_nodes()
            markup = _reportlab_markup(node).strip()
            if markup:
                story.append(Paragraph(markup, style))
        else:
            inline_nodes.append(node)
    flush_inline_nodes()
    if not story:
        for line in plain_text.splitlines() or ("",):
            story.append(Paragraph(html.escape(line), style))

    def invariant_canvas(*args, **kwargs):
        kwargs["invariant"] = 1
        return Canvas(*args, **kwargs)

    document.build(story, canvasmaker=invariant_canvas)
    pdf = output.getvalue()
    if not pdf.startswith(b"%PDF-") or not pdf or len(pdf) > MAX_COVER_LETTER_PDF_BYTES:
        raise ValueError
    return pdf


def _reportlab_markup(node: Any) -> str:
    parts: list[str] = []

    def visit(current: Any) -> None:
        if isinstance(current, NavigableString):
            parts.append(html.escape(str(current)))
            return
        if not isinstance(current, Tag):
            return
        name = current.name.casefold()
        if name == "br":
            parts.append("<br/>")
            return
        rendered_name = {"strong": "b", "em": "i"}.get(name, name)
        if rendered_name in {"b", "i"}:
            parts.append(f"<{rendered_name}>")
        elif rendered_name == "a":
            href = html.escape(str(current.get("href") or ""), quote=True)
            parts.append(f'<a href="{href}">')
        for child in current.children:
            visit(child)
        if rendered_name in {"b", "i", "a"}:
            parts.append(f"</{rendered_name}>")

    if isinstance(node, Tag) and node.name.casefold() in _BLOCK_TAGS:
        for child in node.children:
            visit(child)
    else:
        visit(node)
    return "".join(parts)


__all__ = [
    "CoverLetterRenderingError",
    "MAX_COVER_LETTER_HTML_CHARS",
    "RenderedCoverLetter",
    "blank_cover_letter",
    "cover_letter_plain_text",
    "render_cover_letter",
    "sanitize_cover_letter_html",
]
