import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from difflib import SequenceMatcher

from .models import Chunk, DocumentMetadata, Section, Source


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )


class VectorStore:
    def __init__(self, directory: Path, embedder: Embedder):
        self.directory = directory
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self.documents: list[DocumentMetadata] = []
        self.sections: list[Section] = []
        self.directory.mkdir(parents=True, exist_ok=True)
        self.load()

    @property
    def count(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        faiss.normalize_L2(vectors)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match the existing index")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.save()

    def register_document(self, document: DocumentMetadata) -> None:
        self.documents = [item for item in self.documents if item.filename != document.filename]
        self.documents.append(document)
        self.save()

    def search(self, query: str, k: int = 5) -> list[Source]:
        if self.index is None or not self.chunks:
            return []
        vector = self.embedder.encode([query])
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        results: list[Source] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                Source(
                    paper=chunk.paper,
                    page=chunk.page,
                    section_name=chunk.section_name,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

    def papers(self) -> list[dict[str, int | str]]:
        grouped: dict[str, dict[str, int | str]] = {}
        for chunk in self.chunks:
            item = grouped.setdefault(chunk.paper, {"name": chunk.paper, "chunks": 0})
            item["chunks"] = int(item["chunks"]) + 1
        metadata = {item.filename: item for item in self.documents}
        result = []
        for item in grouped.values():
            document = metadata.get(str(item["name"]))
            result.append({
                **item,
                "title": document.title if document else str(item["name"]),
                "pages": document.page_count if document else 0,
                "sections": len(self.sections),
            })
        return sorted(result, key=lambda item: str(item["name"]).lower())

    def document_sources(self) -> list[Source]:
        sources = []
        for document in self.documents:
            first = next(
                (chunk for chunk in self.chunks if chunk.paper == document.filename and chunk.page == 1),
                None,
            )
            if first:
                sources.append(Source(paper=first.paper, page=1, section_name=first.section_name, text=first.text, score=1.0))
        return sources

    def replace_document(
        self, document: DocumentMetadata, sections: list[Section], chunks: list[Chunk]
    ) -> None:
        self.clear()
        self.documents = [document]
        self.sections = sections
        self.add(chunks)

    def find_section(self, query: str) -> Section | None:
        normalized = query.lower()
        exact = [section for section in self.sections if section.title.lower() in normalized]
        if exact:
            return max(exact, key=lambda section: len(section.title))
        scored = [
            (SequenceMatcher(None, normalized, section.title.lower()).ratio(), section)
            for section in self.sections
        ]
        best_score, best = max(scored, default=(0.0, None), key=lambda item: item[0])
        return best if best_score >= 0.42 else None

    def section_sources(self, section: Section) -> list[Source]:
        seen: set[int] = set()
        sources = []
        for block in section.blocks:
            if block.page in seen:
                continue
            seen.add(block.page)
            page_text = "\n\n".join(item.text for item in section.blocks if item.page == block.page)
            sources.append(Source(
                paper=self.documents[0].filename,
                page=block.page,
                section_name=section.title,
                text=page_text,
                score=1.0,
            ))
        return sources

    def clear(self) -> None:
        self.index = None
        self.chunks = []
        self.documents = []
        self.sections = []
        for path in (
            self.directory / "vectors.faiss",
            self.directory / "chunks.json",
            self.directory / "documents.json",
            self.directory / "sections.json",
        ):
            path.unlink(missing_ok=True)

    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.directory / "vectors.faiss"))
        (self.directory / "chunks.json").write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (self.directory / "documents.json").write_text(
            json.dumps([document.model_dump() for document in self.documents], ensure_ascii=False),
            encoding="utf-8",
        )
        (self.directory / "sections.json").write_text(
            json.dumps([section.model_dump() for section in self.sections], ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> None:
        index_path = self.directory / "vectors.faiss"
        metadata_path = self.directory / "chunks.json"
        if index_path.exists() and metadata_path.exists():
            self.index = faiss.read_index(str(index_path))
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.chunks = [Chunk.model_validate(item) for item in payload]
        documents_path = self.directory / "documents.json"
        if documents_path.exists():
            payload = json.loads(documents_path.read_text(encoding="utf-8"))
            self.documents = [DocumentMetadata.model_validate(item) for item in payload]
        sections_path = self.directory / "sections.json"
        if sections_path.exists():
            payload = json.loads(sections_path.read_text(encoding="utf-8"))
            self.sections = [Section.model_validate(item) for item in payload]
