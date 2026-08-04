from pathlib import Path

import fitz

from app.chunking import chunk_sections
from app.models import Source
from app.pdf import extract_document_metadata, extract_sections


def make_structured_pdf(path: Path) -> None:
    document = fitz.open()
    document.set_metadata({
        "title": "Mobile Devices for English Language Study",
        "author": "Mariusz Kruk",
    })
    page = document.new_page()
    page.insert_text((60, 70), "Mobile Devices for English Language Study", fontsize=21)
    page.insert_text((60, 105), "Mariusz Kruk", fontsize=12)
    page.insert_text((60, 125), "University of Zielona Gora, Poland", fontsize=10)
    page.insert_text((60, 170), "Abstract", fontsize=15)
    page.insert_textbox(
        fitz.Rect(60, 190, 540, 300),
        "This study examines how advanced learners use mobile devices for English "
        "language study. Interview evidence identifies recurring learning strategies "
        "and practical constraints experienced by participants in everyday settings.",
        fontsize=10,
    )
    page.insert_text((60, 340), "1 Introduction", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 360, 540, 440),
        "Mobile learning has become an important component of independent language "
        "study. This section introduces the research problem and its motivation.",
        fontsize=10,
    )
    page = document.new_page()
    page.insert_text((60, 70), "2 Methodology", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 95, 540, 190),
        "The researchers conducted semi-structured interviews with advanced learners. "
        "Interview transcripts were coded thematically and reviewed for recurring patterns.",
        fontsize=10,
    )
    page.insert_text((60, 230), "3 Results", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 255, 540, 350),
        "Participants used dictionaries, videos, and messaging applications. They valued "
        "convenience but reported distraction and limited opportunities for feedback.",
        fontsize=10,
    )
    document.save(path)
    document.close()


def test_author_and_affiliation_are_separated(tmp_path):
    path = tmp_path / "paper.pdf"
    make_structured_pdf(path)

    metadata = extract_document_metadata(path)

    assert metadata.authors == "Mariusz Kruk"
    assert "University of Zielona Gora" in metadata.affiliations
    assert "University" not in metadata.authors


def test_font_aware_section_detection(tmp_path):
    path = tmp_path / "paper.pdf"
    make_structured_pdf(path)
    metadata = extract_document_metadata(path)

    sections = extract_sections(path, metadata)
    names = [section.title for section in sections]

    assert "Introduction" in names
    assert "Methodology" in names
    assert "Results" in names
    assert next(section for section in sections if section.title == "Methodology").page_start == 2


def test_chunks_preserve_section_hierarchy_and_citation_metadata(tmp_path):
    path = tmp_path / "paper.pdf"
    make_structured_pdf(path)
    metadata = extract_document_metadata(path)
    sections = extract_sections(path, metadata)

    chunks = chunk_sections(path.name, sections, chunk_size=160, overlap=20)
    method_chunks = [chunk for chunk in chunks if chunk.section_name == "Methodology"]

    assert method_chunks
    assert all(chunk.page == 2 and chunk.section_level == 1 for chunk in method_chunks)
    source = Source(
        paper=method_chunks[0].paper,
        page=method_chunks[0].page,
        section_name=method_chunks[0].section_name,
        text=method_chunks[0].text,
        score=1.0,
    )
    assert source.model_dump()["section_name"] == "Methodology"

