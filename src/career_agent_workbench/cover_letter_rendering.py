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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

MAX_COVER_LETTER_HTML_CHARS = 500_000
MAX_COVER_LETTER_PDF_BYTES = 20_000_000
_ALLOWED_TAGS = frozenset({"p", "div", "br", "strong", "b", "em", "i", "a"})
_BLOCK_TAGS = frozenset({"p", "div"})
_REMOVED_WITH_CONTENT = frozenset({"script", "style", "iframe", "object", "embed"})
_SAFE_LINK_SCHEMES = frozenset({"http", "https", "mailto"})


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
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="Cover Letter",
        author="Career Agent Workbench",
    )
    style = getSampleStyleSheet()["BodyText"]
    style.fontName = "Helvetica"
    style.fontSize = 11
    style.leading = 16
    story: list[Any] = []
    soup = BeautifulSoup(sanitized_html, "html.parser")
    blocks = tuple(
        tag for tag in soup.find_all(_BLOCK_TAGS) if not tag.find_parent(_BLOCK_TAGS)
    )
    if blocks:
        for block in blocks:
            markup = _reportlab_markup(block)
            if markup:
                story.extend((Paragraph(markup, style), Spacer(1, 9)))
    else:
        for line in plain_text.splitlines() or ("",):
            story.extend((Paragraph(html.escape(line), style), Spacer(1, 9)))

    def invariant_canvas(*args, **kwargs):
        kwargs["invariant"] = 1
        return Canvas(*args, **kwargs)

    document.build(story, canvasmaker=invariant_canvas)
    pdf = output.getvalue()
    if not pdf.startswith(b"%PDF-") or not pdf or len(pdf) > MAX_COVER_LETTER_PDF_BYTES:
        raise ValueError
    return pdf


def _reportlab_markup(block: Tag) -> str:
    parts: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, NavigableString):
            parts.append(html.escape(str(node)))
            return
        if not isinstance(node, Tag):
            return
        name = node.name.casefold()
        if name == "br":
            parts.append("<br/>")
            return
        rendered_name = {"strong": "b", "em": "i"}.get(name, name)
        if rendered_name in {"b", "i"}:
            parts.append(f"<{rendered_name}>")
        elif rendered_name == "a":
            href = html.escape(str(node.get("href") or ""), quote=True)
            parts.append(f'<a href="{href}">')
        for child in node.children:
            visit(child)
        if rendered_name in {"b", "i", "a"}:
            parts.append(f"</{rendered_name}>")

    for child in block.children:
        visit(child)
    return "".join(parts).strip()


__all__ = [
    "CoverLetterRenderingError",
    "MAX_COVER_LETTER_HTML_CHARS",
    "RenderedCoverLetter",
    "blank_cover_letter",
    "cover_letter_plain_text",
    "render_cover_letter",
    "sanitize_cover_letter_html",
]
