import re
import uuid

from .models import Chunk, PageText, Section


def _find_boundary(text: str, start: int, ideal_end: int, minimum_end: int) -> int:
    if ideal_end >= len(text):
        return len(text)
    window = text[minimum_end:ideal_end]
    matches = list(re.finditer(r"(?<=[.!?])\s+|\n+|\s+", window))
    return minimum_end + matches[-1].end() if matches else ideal_end


def chunk_pages(
    pages: list[PageText], chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            ideal_end = min(start + chunk_size, len(text))
            minimum_end = min(start + max(chunk_size // 2, 1), ideal_end)
            end = _find_boundary(text, start, ideal_end, minimum_end)
            body = text[start:end].strip()
            if body:
                stable_key = f"{page.paper}:{page.page}:{start}:{body[:40]}"
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                        paper=page.paper,
                        page=page.page,
                        section_name="Full document",
                        section_level=1,
                        text=body,
                    )
                )
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def chunk_sections(
    paper: str,
    sections: list[Section],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Chunk within section and page boundaries so hierarchy and citations survive."""
    chunks: list[Chunk] = []
    for section in sections:
        by_page: dict[int, list[str]] = {}
        for block in section.blocks:
            by_page.setdefault(block.page, []).append(block.text)
        for page, parts in by_page.items():
            page_chunks = chunk_pages(
                [PageText(paper=paper, page=page, text="\n\n".join(parts))],
                chunk_size=chunk_size,
                overlap=overlap,
            )
            for chunk in page_chunks:
                chunk.section_name = section.title
                chunk.section_level = section.level
                chunk.id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{paper}:{section.id}:{page}:{chunk.text[:50]}",
                    )
                )
            chunks.extend(page_chunks)
    return chunks
