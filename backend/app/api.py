import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .chunking import chunk_sections
from .config import settings
from .generation import generate_answer
from .models import AskRequest, AskResponse
from .pdf import extract_document_metadata, extract_pages, extract_sections
from .retrieval import SentenceTransformerEmbedder, VectorStore


store: VectorStore | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global store
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore(settings.index_dir, SentenceTransformerEmbedder(settings.embedding_model))
    yield


app = FastAPI(title="PaperLens API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store() -> VectorStore:
    if store is None:
        raise HTTPException(503, "Embedding index is still starting")
    return store


def safe_name(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", base)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": get_store().count}


@app.get("/api/documents")
def list_documents():
    current = get_store()
    return {"documents": current.papers(), "chunks": current.count}


@app.post("/api/documents", status_code=201)
async def upload_documents(files: list[UploadFile] = File(...)):
    current = get_store()
    if len(files) != 1:
        raise HTTPException(400, "PaperLens accepts exactly one PDF at a time")
    processed = []
    for file in files:
        original_name = file.filename or "upload.pdf"
        if file.content_type != "application/pdf" and not original_name.lower().endswith(".pdf"):
            raise HTTPException(415, f"{original_name} is not a PDF")
        content = await file.read()
        if len(content) > settings.max_file_mb * 1024 * 1024:
            raise HTTPException(413, f"{original_name} exceeds {settings.max_file_mb} MB")
        name = safe_name(original_name)
        path = settings.upload_dir / name
        path.write_bytes(content)
        try:
            pages = extract_pages(path, name)
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, f"Could not read {name} as a PDF") from exc
        metadata = extract_document_metadata(path, name)
        sections = extract_sections(path, metadata)
        chunks = chunk_sections(name, sections)
        if not chunks:
            path.unlink(missing_ok=True)
            raise HTTPException(422, f"{name} has no extractable text; it may require OCR")
        for old_pdf in settings.upload_dir.glob("*.pdf"):
            if old_pdf != path:
                old_pdf.unlink(missing_ok=True)
        current.replace_document(metadata, sections, chunks)
        processed.append({
            "name": name,
            "pages": metadata.page_count,
            "sections": len(sections),
            "chunks": len(chunks),
        })
    return {"processed": processed, "total_chunks": current.count}


@app.delete("/api/documents")
def clear_documents():
    current = get_store()
    current.clear()
    for path in settings.upload_dir.glob("*.pdf"):
        path.unlink(missing_ok=True)
    return {"status": "cleared"}


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    current = get_store()
    question = request.question.lower().strip()
    documents = current.documents

    if not documents:
        return AskResponse(answer="Upload at least one paper first.", sources=[], mode="none")

    document = documents[0]
    sources = current.document_sources()
    if re.search(r"\b(title|paper name|name of (?:the )?(?:paper|article))\b", question):
        answer = f'“{document.title}” [1]'
        return AskResponse(answer=answer, sources=sources, mode="metadata")

    if re.search(r"\b(author|authors|wrote|written by)\b", question):
        answer = f"Author information extracted from the PDF:\n\nAuthors: {document.authors}"
        if document.affiliations:
            answer += f"\n\nAffiliations: {document.affiliations}"
        answer += " [1]"
        return AskResponse(answer=answer, sources=sources, mode="metadata")

    if re.search(r"\b(how many pages|page count|number of pages|pages in)\b", question):
        answer = f"The uploaded PDF contains {document.page_count} pages. [1]"
        return AskResponse(answer=answer, sources=sources, mode="metadata")

    if re.search(r"\b(section names|list (?:the )?sections|what (?:are )?(?:the )?sections|headings|table of contents)\b", question):
        visible = [section for section in current.sections if section.title.lower() != "front matter"]
        answer = "Detected sections:\n\n" + "\n".join(
            f"{'  ' if section.level > 1 else ''}• {section.title} (pages {section.page_start}–{section.page_end})"
            for section in visible
        )
        return AskResponse(answer=answer, sources=[], mode="structure")

    if re.search(r"\b(section[- ]wise|each section|all sections)\b", question) and re.search(
        r"\b(summary|summarize|summarise|overview)\b", question
    ):
        summaries = []
        section_sources = []
        for section in current.sections:
            if section.title.lower() == "front matter" or len(section.text.split()) < 25:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", section.text)
            summary = " ".join(sentences[:2]).strip()
            if summary:
                citation_number = len(section_sources) + 1
                summaries.append(f"{section.title}\n{summary} [{citation_number}]")
                section_sources.extend(current.section_sources(section)[:1])
        return AskResponse(
            answer="\n\n".join(summaries),
            sources=section_sources,
            mode="section-summary",
        )

    section_request = re.search(
        r"\b(section|extract|show|give|summarize|summarise|summary of|what does)\b",
        question,
    )
    if section_request:
        section = current.find_section(question)
        if section and section.title.lower() != "front matter":
            section_sources = current.section_sources(section)
            wants_summary = bool(re.search(r"\b(summary|summarize|summarise|overview)\b", question))
            if wants_summary:
                sentences = re.split(r"(?<=[.!?])\s+", section.text)
                answer = f"{section.title}\n\n" + " ".join(sentences[:3]).strip() + " [1]"
                mode = "section-summary"
            else:
                answer = f"{section.title}\n\n{section.text}\n\nSource: [1]"
                mode = "section-extract"
            return AskResponse(answer=answer, sources=section_sources, mode=mode)

    if re.search(r"\b(summary|summarize|summarise|what is (?:the )?(?:paper|article) about|overview)\b", question):
        if document.abstract:
            answer = f"{document.title}\n\n{document.abstract} [1]"
            return AskResponse(answer=answer, sources=sources, mode="abstract")

    sources = current.search(request.question, request.top_k or settings.top_k)
    answer, mode = await generate_answer(request.question, sources, settings)
    return AskResponse(answer=answer, sources=sources, mode=mode)
