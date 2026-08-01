from __future__ import annotations

import logging
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from io import BytesIO
from threading import Event

import pytest
from pypdf import PdfWriter
from pypdf import _reader as pypdf_reader
from pypdf import _utils as pypdf_utils
from reportlab.pdfgen import canvas

from career_agent_workbench import ats
from career_agent_workbench.ats import (
    MAX_ATS_JOB_DESCRIPTION_CHARS,
    MAX_EXTRACTED_PDF_TEXT_CHARS,
    MAX_RESUME_PDF_BYTES,
    MAX_RESUME_PDF_PAGES,
    AtsDiagnostics,
    AtsError,
    AtsProxyScore,
    calculate_ats_diagnostics,
    calculate_ats_proxy_score,
    extract_pdf_text,
)


def _pdf_bytes(*pages: tuple[str, ...]) -> bytes:
    buffer = BytesIO()
    document = canvas.Canvas(buffer)
    for lines in pages:
        y = 800
        for line in lines:
            document.drawString(48, y, line)
            y -= 18
        document.showPage()
    document.save()
    return buffer.getvalue()


def _blank_pdf_bytes(page_count: int) -> bytes:
    buffer = BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    writer.write(buffer)
    return buffer.getvalue()


def _synthetic_resume_pdf() -> bytes:
    return _pdf_bytes(
        (
            "Jules Example",
            "jules@example.test",
            "312-555-0100",
            "https://portfolio.example.test/jules",
            "Professional Summary",
            "Engineer building reliable automation.",
            "Core Technical Skills",
            "Python, Docker, automation, REST API",
            "Professional Experience",
            "Fictional Systems LLC",
            "Built Python automation.",
            "Improved Docker delivery.",
            "Designed REST API integrations.",
            "Maintained reliable services.",
            "Collaborated on testing.",
            "Documented system behavior.",
            "Reviewed deployment changes.",
            "Monitored service health.",
            "Investigated defects.",
            "Simplified build workflows.",
            "Supported peer reviews.",
            "Education",
            "Example Institute",
            "Certifications",
            "Synthetic Systems Certificate",
        ),
    )


def _assert_sanitized(error: AtsError, *, secret: str | None = None) -> None:
    if secret is not None:
        assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_empty_malformed_and_valid_pdf_extraction_is_quiet_and_bounded(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "do-not-emit-this-pdf-marker"
    malformed = f"%PDF-1.7\n{secret}\nnot-a-pdf".encode()
    caplog.set_level(logging.DEBUG)
    pypdf_logger = logging.getLogger("pypdf")
    original_logger_state = (
        pypdf_logger.handlers,
        pypdf_logger.level,
        pypdf_logger.propagate,
        pypdf_logger.disabled,
    )

    assert extract_pdf_text(b"") == ""
    assert extract_pdf_text(malformed) == ""
    malformed_score = calculate_ats_proxy_score(
        resume_pdf=malformed,
        job_description="Python is required.",
    )
    assert isinstance(malformed_score, AtsProxyScore)
    assert malformed_score.parsing_score == 0

    valid_text = extract_pdf_text(_synthetic_resume_pdf())
    assert "Jules Example" in valid_text
    assert "Python" in valid_text
    assert len(valid_text) <= MAX_EXTRACTED_PDF_TEXT_CHARS

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert caplog.text == ""
    assert secret not in captured.out
    assert secret not in captured.err
    assert secret not in caplog.text
    assert pypdf_logger.handlers is original_logger_state[0]
    assert (
        pypdf_logger.level,
        pypdf_logger.propagate,
        pypdf_logger.disabled,
    ) == original_logger_state[1:]


def test_blocked_parser_suppresses_only_current_pypdf_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser_secret = "blocked-parser-secret"
    visible_stdout = "unrelated-stdout-visible"
    visible_stderr = "unrelated-stderr-visible"
    visible_log = "unrelated-log-visible"
    visible_helper_log = "unrelated-helper-log-visible"
    parser_entered = Event()
    release_parser = Event()

    caplog.set_level(logging.DEBUG)
    pypdf_logger = logging.getLogger("pypdf")
    original_logger_state = (
        pypdf_logger.handlers,
        pypdf_logger.filters,
        pypdf_logger.level,
        pypdf_logger.propagate,
        pypdf_logger.disabled,
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_utils_warning = pypdf_utils.logger_warning
    original_utils_error = pypdf_utils.logger_error
    original_reader_warning = pypdf_reader.logger_warning

    class BlockedMalformedReader:
        def __init__(self, stream: object, *, strict: bool) -> None:
            del stream
            assert strict is False
            pypdf_utils.logger_warning(parser_secret, source="pypdf.synthetic")
            pypdf_utils.logger_error(parser_secret, source="pypdf.synthetic")
            parser_entered.set()
            if not release_parser.wait(timeout=2):
                raise AssertionError("Timed out coordinating the blocked parser.")
            raise ValueError(parser_secret)

    monkeypatch.setattr(ats, "PdfReader", BlockedMalformedReader)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            extract_pdf_text,
            b"%PDF-blocked-malformed",
        )
        try:
            assert parser_entered.wait(timeout=2)
            assert ats._PDF_PARSE_LOCK.locked()
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr
            assert (
                pypdf_logger.handlers,
                pypdf_logger.filters,
                pypdf_logger.level,
                pypdf_logger.propagate,
                pypdf_logger.disabled,
            ) == original_logger_state
            assert pypdf_utils.logger_warning is not original_utils_warning
            assert pypdf_utils.logger_error is not original_utils_error
            assert pypdf_reader.logger_warning is not original_reader_warning

            print(visible_stdout)
            print(visible_stderr, file=sys.stderr)
            logging.getLogger("unrelated.application").warning(visible_log)
            pypdf_utils.logger_warning(
                visible_helper_log,
                source="unrelated.pypdf",
            )
        finally:
            release_parser.set()
        assert future.result(timeout=2) == ""

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert (
        pypdf_logger.handlers,
        pypdf_logger.filters,
        pypdf_logger.level,
        pypdf_logger.propagate,
        pypdf_logger.disabled,
    ) == original_logger_state
    assert pypdf_utils.logger_warning is original_utils_warning
    assert pypdf_utils.logger_error is original_utils_error
    assert pypdf_reader.logger_warning is original_reader_warning
    captured = capsys.readouterr()
    assert visible_stdout in captured.out
    assert visible_stderr in captured.err
    assert visible_log in caplog.text
    assert visible_helper_log in caplog.text
    assert parser_secret not in captured.out
    assert parser_secret not in captured.err
    assert parser_secret not in caplog.text


def test_pdf_byte_limit_precedes_parser_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_calls = 0

    def forbidden_reader(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal parser_calls
        parser_calls += 1
        raise AssertionError("The PDF parser must not receive oversized input.")

    monkeypatch.setattr(ats, "PdfReader", forbidden_reader)
    with pytest.raises(AtsError) as captured:
        extract_pdf_text(b"x" * (MAX_RESUME_PDF_BYTES + 1))

    _assert_sanitized(captured.value)
    assert parser_calls == 0


def test_pdf_page_limit_precedes_page_text_extraction() -> None:
    over_page_limit = _blank_pdf_bytes(MAX_RESUME_PDF_PAGES + 1)

    with pytest.raises(AtsError) as captured:
        extract_pdf_text(over_page_limit)

    _assert_sanitized(captured.value)


def test_extracted_text_limit_fails_closed_incrementally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyntheticPage:
        def __init__(self, text: str) -> None:
            self.text = text
            self.calls = 0

        def extract_text(self) -> str:
            self.calls += 1
            return self.text

    first_page = SyntheticPage("x" * MAX_EXTRACTED_PDF_TEXT_CHARS)
    second_page = SyntheticPage("private-overflow-marker")

    class SyntheticReader:
        def __init__(self, stream: object, *, strict: bool) -> None:
            del stream
            assert strict is False
            self.pages = [first_page, second_page]

    monkeypatch.setattr(ats, "PdfReader", SyntheticReader)
    with pytest.raises(AtsError) as captured:
        extract_pdf_text(b"%PDF-synthetic")

    _assert_sanitized(captured.value, secret="private-overflow-marker")
    assert first_page.calls == 1
    assert second_page.calls == 1


def test_job_text_limit_precedes_pdf_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_calls = 0

    def forbidden_extract(_: bytes) -> str:
        nonlocal extraction_calls
        extraction_calls += 1
        raise AssertionError("PDF extraction must not run for oversized job text.")

    monkeypatch.setattr(ats, "extract_pdf_text", forbidden_extract)
    with pytest.raises(AtsError) as captured:
        calculate_ats_diagnostics(
            resume_pdf=b"%PDF-unused",
            job_description="z" * (MAX_ATS_JOB_DESCRIPTION_CHARS + 1),
        )

    _assert_sanitized(captured.value)
    assert extraction_calls == 0


@pytest.mark.parametrize(
    ("pdf_content", "job_description"),
    [
        (bytearray(b"%PDF"), "Python"),
        (memoryview(b"%PDF"), "Python"),
        (b"%PDF", b"Python"),
    ],
)
def test_non_contract_input_types_fail_with_sanitized_errors(
    pdf_content: object,
    job_description: object,
) -> None:
    with pytest.raises(AtsError) as captured:
        calculate_ats_diagnostics(
            resume_pdf=pdf_content,  # type: ignore[arg-type]
            job_description=job_description,  # type: ignore[arg-type]
        )

    _assert_sanitized(captured.value)


def test_weighting_negation_aliases_and_missing_term_ordering() -> None:
    weighted = dict(
        ats._extract_weighted_terms(
            "Python is required. Docker supports delivery. "
            "Continuous integration experience is useful."
        )
    )
    assert weighted["python"] == 1.75
    assert weighted["docker"] == 1.0
    assert weighted["ci/cd"] == 1.0

    negated = dict(
        ats._extract_weighted_terms(
            "No Python experience is needed. Terraform is required."
        )
    )
    assert "python" not in negated
    assert negated["terraform"] == 1.75

    diagnostics = calculate_ats_diagnostics(
        resume_pdf=_synthetic_resume_pdf(),
        job_description=(
            "Terraform is required. Kubernetes is required. "
            "Python is required. Docker supports delivery."
        ),
    )
    assert diagnostics.score.missing_high_value_terms == (
        "kubernetes",
        "terraform",
    )
    assert tuple(item.term for item in diagnostics.matched_terms) == (
        "python",
        "docker",
    )
    assert tuple(item.weight for item in diagnostics.matched_terms) == (1.75, 1.0)


def test_repeated_phrase_noise_is_reported_without_distorting_score() -> None:
    job_text = (
        "Hardware productivity solutions support teams. "
        "Hardware productivity solutions support teams."
    )
    diagnostics = calculate_ats_diagnostics(
        resume_pdf=_synthetic_resume_pdf(),
        job_description=job_text,
    )

    repeated = tuple(item.term for item in diagnostics.repeated_phrase_terms)
    noisy = tuple(item.term for item in diagnostics.likely_noisy_phrase_matches)
    assert "hardware productivity solutions" in repeated
    assert "hardware productivity solutions" in noisy
    assert set(noisy).issubset(repeated)
    assert all(item.term not in noisy for item in diagnostics.unmatched_weighted_terms)


def test_semantic_cluster_parsing_formatting_and_score_clamping() -> None:
    semantic = ats._semantic_score(
        resume_normalized="docker container delivery",
        job_normalized="kubernetes platform",
        job_terms=[("kubernetes", 1.0)],
        resume_terms=set(),
    )
    assert 0 < semantic <= 100

    resume_pdf = _synthetic_resume_pdf()
    score = calculate_ats_proxy_score(
        resume_pdf=resume_pdf,
        job_description="Python and Docker automation are required.",
    )
    assert score.parsing_score >= 55
    assert score.formatting_risk in {"Low", "Medium"}
    for value in (
        score.overall_score,
        score.parsing_score,
        score.keyword_match_score,
        score.semantic_match_score,
    ):
        assert 0 <= value <= 100
    assert ats._clamp_score(-1000) == 0
    assert ats._clamp_score(1000) == 100


def test_results_are_frozen_and_deterministic() -> None:
    resume_pdf = _synthetic_resume_pdf()
    job_text = (
        "Python is required. Experience with Kubernetes and reliability. "
        "Docker automation supports delivery."
    )

    first = calculate_ats_diagnostics(
        resume_pdf=resume_pdf,
        job_description=job_text,
    )
    second = calculate_ats_diagnostics(
        resume_pdf=resume_pdf,
        job_description=job_text,
    )
    assert isinstance(first, AtsDiagnostics)
    assert first == second
    assert first.score == calculate_ats_proxy_score(
        resume_pdf=resume_pdf,
        job_description=job_text,
    )
    with pytest.raises(FrozenInstanceError):
        first.score.overall_score = 0  # type: ignore[misc]


def test_ats_processing_never_uses_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("ATS processing must remain offline.")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)
    result = calculate_ats_diagnostics(
        resume_pdf=_synthetic_resume_pdf(),
        job_description="Python and REST API experience are required.",
    )
    assert result.score.parsing_score > 0
    assert network_calls == 0
