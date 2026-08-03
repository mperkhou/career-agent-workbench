from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfReader
from reportlab.pdfbase.pdfmetrics import stringWidth

from career_agent_workbench.cover_letter_rendering import (
    MAX_COVER_LETTER_PDF_BYTES,
    CoverLetterRenderingError,
    blank_cover_letter,
    cover_letter_plain_text,
    render_cover_letter,
    sanitize_cover_letter_html,
)


def test_blank_cover_letter_is_inert_and_independent() -> None:
    first = blank_cover_letter()
    second = blank_cover_letter()
    assert first == {
        "schema_version": 1,
        "source": "manual",
        "body_html": "",
        "body_text": "",
    }
    assert first is not second


def test_sanitation_keeps_only_supported_formatting_and_safe_links() -> None:
    source = (
        "<!-- remove -->"
        '<p class="private"><strong>Fictional</strong> <em>letter</em><br>'
        '<a href="https://example.com/path" target="_blank">Public link</a> '
        '<a href="mailto:operator@example.com" style="display:none">Email</a> '
        '<a href="javascript:alert(1)">Unsafe link</a></p>'
        '<section data-hidden="value">Supported text</section>'
        "<script>privateExecutableMarker()</script>"
    )
    sanitized = sanitize_cover_letter_html(source)
    assert "<!--" not in sanitized
    assert "privateExecutableMarker" not in sanitized
    assert "class=" not in sanitized
    assert "target=" not in sanitized
    assert "style=" not in sanitized
    assert "javascript:" not in sanitized
    assert "<section" not in sanitized
    assert "Supported text" in sanitized
    assert '<a href="https://example.com/path">Public link</a>' in sanitized
    assert '<a href="mailto:operator@example.com">Email</a>' in sanitized
    assert "Unsafe link" in sanitized


def test_plain_text_and_pdf_rendering_are_normalized_and_deterministic() -> None:
    rendered = render_cover_letter(
        "<div>Hello   Example</div><p><b>Second</b><br>Line</p>"
    )
    repeated = render_cover_letter(
        "<div>Hello   Example</div><p><b>Second</b><br>Line</p>"
    )
    assert rendered.value["body_text"] == "Hello Example\nSecond\nLine"
    assert cover_letter_plain_text(rendered.value["body_html"]) == (
        "Hello Example\nSecond\nLine"
    )
    assert rendered.pdf.startswith(b"%PDF-")
    assert rendered.pdf == repeated.pdf


def test_mixed_top_level_content_is_rendered_in_source_order() -> None:
    source = (
        "Dear <strong>Example Team</strong>,"
        "<p>Thank you for the fictional opportunity.</p>"
        "<em>Sincerely</em><br>Example Operator"
    )
    rendered = render_cover_letter(source)
    repeated = render_cover_letter(source)

    assert rendered.value["body_text"] == (
        "Dear Example Team,\n"
        "Thank you for the fictional opportunity.\n"
        "Sincerely\n"
        "Example Operator"
    )
    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(rendered.pdf)).pages
    )
    expected = (
        "Dear Example Team,",
        "Thank you for the fictional opportunity.",
        "Sincerely",
        "Example Operator",
    )
    positions = [extracted.index(text) for text in expected]
    assert positions == sorted(positions)
    assert rendered.pdf == repeated.pdf


def test_safe_links_are_the_only_pdf_link_annotations() -> None:
    rendered = render_cover_letter(
        '<p>Use <a href="https://example.com/path">the public reference</a> '
        'and <a href="javascript:privateMarker()">plain text</a>.</p>'
    )
    page = PdfReader(BytesIO(rendered.pdf)).pages[0]
    annotations = tuple(page.get("/Annots", ()))

    assert len(annotations) == 1
    assert annotations[0].get_object()["/A"]["/URI"] == "https://example.com/path"
    assert "privateMarker" not in rendered.value["body_html"]
    assert "plain text" in rendered.value["body_text"]


def test_representative_letter_is_exact_readable_and_bounded_on_one_page() -> None:
    words = ["signals"] * 268 + ["signal"] * 243
    paragraph_sizes = [47] * 10 + [41]
    offsets = [sum(paragraph_sizes[:index]) for index in range(12)]
    source = "".join(
        f"<p>{' '.join(words[offsets[index] : offsets[index + 1]])}</p>"
        for index in range(11)
    )

    rendered = render_cover_letter(source)
    repeated = render_cover_letter(source)
    reader = PdfReader(BytesIO(rendered.pdf))
    assert len(rendered.value["body_text"].split()) == 511
    assert len(rendered.value["body_text"]) == 3_844
    assert len(reader.pages) == 1
    assert rendered.pdf == repeated.pdf
    assert len(rendered.pdf) <= MAX_COVER_LETTER_PDF_BYTES

    page = reader.pages[0]
    extracted = page.extract_text() or ""
    assert " ".join(extracted.split()) == " ".join(rendered.value["body_text"].split())
    assert tuple(float(value) for value in page.mediabox) == (0.0, 0.0, 612.0, 792.0)
    text_runs: list[tuple[float, float, float]] = []
    page.extract_text(
        visitor_text=lambda text, matrix, _text_matrix, _font, size: (
            text_runs.append((matrix[4], matrix[5], size)) if text.strip() else None
        )
    )
    assert text_runs
    assert {size for _x, _y, size in text_runs} == {10.0}
    assert all(54 <= x <= 558 and 54 <= y <= 738 for x, y, _size in text_runs)
    assert (
        max(
            stringWidth(line, "Helvetica", 10)
            for line in extracted.splitlines()
            if line
        )
        <= 504
    )


@pytest.mark.parametrize("value", (None, object(), "x" * 500_001))
def test_invalid_cover_letter_input_is_content_free(value: object) -> None:
    with pytest.raises(CoverLetterRenderingError) as caught:
        render_cover_letter(value)
    assert str(caught.value) in {
        "Cover letter input is invalid.",
        "Cover letter could not be rendered.",
    }
    assert "x" not in str(caught.value)
