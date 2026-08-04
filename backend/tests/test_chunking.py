import pytest

from app.chunking import chunk_pages
from app.models import PageText


def test_chunking_preserves_page_metadata_and_overlap():
    text = " ".join(f"Sentence {i} explains a result." for i in range(30))
    chunks = chunk_pages([PageText(paper="study.pdf", page=7, text=text)], chunk_size=180, overlap=30)

    assert len(chunks) > 2
    assert all(chunk.paper == "study.pdf" and chunk.page == 7 for chunk in chunks)
    assert all(len(chunk.text) <= 180 for chunk in chunks)


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, overlap=100)

