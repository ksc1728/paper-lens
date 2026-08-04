from pathlib import Path

import fitz

import re

import statistics
import uuid

from .models import DocumentMetadata, PageText, Section, SectionBlock


def extract_pages(path: Path, display_name: str | None = None) -> list[PageText]:
    """Extract non-empty pages while preserving human-readable 1-based page numbers."""
    pages: list[PageText] = []
    name = display_name or path.name
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            text = " ".join(page.get_text("text").split())
            if text:
                pages.append(PageText(paper=name, page=page_index + 1, text=text))
    return pages


def _clean(text: str) -> str:
    return " ".join(text.replace("\u2009", " ").replace("\xa0", " ").split())


def _first_page_blocks(document: fitz.Document) -> list[tuple]:
    return sorted(document[0].get_text("blocks"), key=lambda block: (block[1], block[0]))


AFFILIATION_WORDS = re.compile(
    r"\b(university|institute|department|faculty|school|college|laboratory|laboratories|"
    r"centre|center|academy|hospital|corporation|company|poland|china|india|usa|uk)\b",
    re.I,
)


def _infer_author_info(
    blocks: list[tuple], title: str, metadata_author: str
) -> tuple[str, str]:
    """Best-effort author extraction for common journal first-page layouts."""
    title_words = set(re.findall(r"[a-z]{4,}", title.lower()))
    title_matches: list[tuple[int, float]] = []
    for block in blocks:
        if float(block[1]) > 180:
            continue
        words = set(re.findall(r"[a-z]{4,}", block[4].lower()))
        if title_words and len(title_words & words) >= max(2, len(title_words) // 2):
            title_matches.append((len(title_words & words), float(block[3])))
    title_bottom = max(title_matches, default=(0, 0.0), key=lambda item: item[0])[1]

    candidates: list[tuple[float, str]] = []
    for block in blocks:
        y0, text = float(block[1]), block[4].strip()
        if y0 < title_bottom or y0 > 300 or len(text) > 500:
            continue
        if re.search(r"doi|received|accepted|published|open access|abstract", text, re.I):
            continue
        name_signals = text.count(",") + text.count("&")
        if (name_signals or metadata_author.lower() in text.lower()) and re.search(r"[A-Z][a-z]+", text):
            candidates.append((y0, text))

    if not candidates:
        return metadata_author or "Not available in PDF metadata", ""

    candidate_y, raw = min(candidates, key=lambda item: item[0])
    lines = [_clean(line) for line in raw.splitlines() if _clean(line)]
    affiliation_lines = [line for line in lines if AFFILIATION_WORDS.search(line)]
    for block in blocks:
        block_y, block_text = float(block[1]), _clean(block[4])
        if candidate_y < block_y <= candidate_y + 90 and AFFILIATION_WORDS.search(block_text):
            affiliation_lines.append(block_text)
    affiliation_lines = list(dict.fromkeys(affiliation_lines))
    author_lines = [line for line in lines if line not in affiliation_lines]
    author_text = _clean(" ".join(author_lines))
    author_text = re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹⁰*✉]+", "", author_text)
    author_text = re.sub(r"(?:\s*,\s*){2,}", ", ", author_text)
    author_text = re.sub(r"\s+([,&])", r"\1", author_text).strip(" ,")
    author_text = re.sub(r",?\s*&\s*", " & ", author_text)
    if metadata_author and not ("," in author_text or "&" in author_text):
        author_text = metadata_author
    return author_text or metadata_author or "Not available in PDF metadata", _clean("; ".join(affiliation_lines))


def _infer_abstract(blocks: list[tuple], title: str) -> str:
    for block in blocks:
        text = _clean(block[4])
        match = re.search(r"\babstract\b[:.\s-]*(.+)", text, re.I)
        if match and len(match.group(1).split()) >= 25:
            return match.group(1).strip()

    # Journals such as Nature often show an unlabelled abstract near the top.
    candidates = []
    for block in blocks:
        y0, text = float(block[1]), _clean(block[4])
        word_count = len(text.split())
        if 120 <= y0 <= 500 and 70 <= word_count <= 450 and title.lower() not in text.lower():
            candidates.append((y0, word_count, text))
    if candidates:
        return min(candidates, key=lambda item: item[0])[2]
    return ""


def extract_document_metadata(path: Path, display_name: str | None = None) -> DocumentMetadata:
    with fitz.open(path) as document:
        raw = document.metadata or {}
        blocks = _first_page_blocks(document) if document.page_count else []
        title = _clean(raw.get("title") or "")
        if not title and blocks:
            title = _clean(max(blocks, key=lambda block: len(block[4]) if block[1] < 180 else 0)[4])
        authors, affiliations = _infer_author_info(
            blocks, title, _clean(raw.get("author") or "")
        )
        return DocumentMetadata(
            filename=display_name or path.name,
            title=title or Path(display_name or path.name).stem,
            authors=authors,
            affiliations=affiliations,
            page_count=document.page_count,
            abstract=_infer_abstract(blocks, title),
        )


KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "literature review",
    "materials and methods", "methods", "methodology", "experimental methods",
    "experimental section", "results", "results and discussion", "discussion",
    "conclusion", "conclusions", "limitations", "future work", "acknowledgements",
    "acknowledgments", "references", "supplementary information",
}


def _block_info(page: fitz.Page) -> list[dict]:
    result = []
    height = page.rect.height
    for block in page.get_text("dict").get("blocks", []):
        if "lines" not in block:
            continue
        spans = [span for line in block["lines"] for span in line.get("spans", []) if span.get("text", "").strip()]
        text = _clean(" ".join(span["text"] for span in spans))
        if not text:
            continue
        if re.match(
            r"^(?:\d+\s*)?\|?\s*(article|nature\s*\||www\.nature\.com)",
            text,
            re.I,
        ):
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        if bbox[1] < 18 or bbox[3] > height - 18:
            continue
        result.append({
            "text": text,
            "size": max((float(span.get("size", 0)) for span in spans), default=0),
            "bold": any("bold" in str(span.get("font", "")).lower() for span in spans),
            "y": float(bbox[1]),
        })
    return result


def _is_heading(block: dict, body_size: float) -> bool:
    text = block["text"].strip().rstrip(":")
    lower = re.sub(r"^\d+(?:\.\d+)*\s*", "", text.lower()).strip()
    words = text.split()
    if not (1 <= len(words) <= 16) or len(text) > 140:
        return False
    if text.endswith((".", ",", ";", "?")) or re.match(r"^(figure|fig\.|table)\s+\d", lower):
        return False
    known = lower in KNOWN_HEADINGS
    numbered = bool(re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text))
    formatted = block["size"] >= body_size * 1.16 or (
        block["bold"] and block["size"] >= body_size * 0.98
    )
    return known or numbered or formatted


def extract_sections(path: Path, metadata: DocumentMetadata) -> list[Section]:
    """Detect headings from font size/bold formatting and preserve page-aware blocks."""
    with fitz.open(path) as document:
        pages = [_block_info(page) for page in document]
    sizes = [
        block["size"]
        for page in pages
        for block in page
        for _ in range(min(len(block["text"]), 200))
        if block["size"] > 0 and len(block["text"].split()) >= 8
    ]
    body_size = statistics.median(sizes) if sizes else 10.0

    sections: list[Section] = []
    current_title = "Front matter"
    current_level = 1
    current_blocks: list[SectionBlock] = []

    def flush() -> None:
        nonlocal current_blocks
        if not current_blocks:
            return
        section_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{metadata.filename}:{current_title}"))
        sections.append(Section(
            id=section_id,
            title=current_title,
            level=current_level,
            page_start=current_blocks[0].page,
            page_end=current_blocks[-1].page,
            blocks=current_blocks,
        ))
        current_blocks = []

    for page_number, blocks in enumerate(pages, start=1):
        for block in blocks:
            text = block["text"]
            if page_number == 1 and metadata.title.lower() in text.lower():
                continue
            if _is_heading(block, body_size):
                flush()
                current_title = re.sub(r"^\d+(?:\.\d+)*\s*", "", text).strip().rstrip(":")
                current_level = 1 if block["size"] >= body_size * 1.28 or current_title.lower() in KNOWN_HEADINGS else 2
            else:
                current_blocks.append(SectionBlock(page=page_number, text=text))
    flush()

    useful = [section for section in sections if len(section.text.split()) >= 8]
    return useful or [Section(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{metadata.filename}:full")),
        title="Full document",
        level=1,
        page_start=1,
        page_end=metadata.page_count,
        blocks=[SectionBlock(page=page.page, text=page.text) for page in extract_pages(path, metadata.filename)],
    )]
