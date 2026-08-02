from __future__ import annotations

import pytest

from career_agent_workbench.cover_letter_rendering import (
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


@pytest.mark.parametrize("value", (None, object(), "x" * 500_001))
def test_invalid_cover_letter_input_is_content_free(value: object) -> None:
    with pytest.raises(CoverLetterRenderingError) as caught:
        render_cover_letter(value)
    assert str(caught.value) in {
        "Cover letter input is invalid.",
        "Cover letter could not be rendered.",
    }
    assert "x" not in str(caught.value)
