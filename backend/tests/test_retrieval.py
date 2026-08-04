import numpy as np

from app.models import Chunk, Section, SectionBlock
from app.retrieval import VectorStore


class KeywordEmbedder:
    vocabulary = ("neural", "climate", "protein")

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vector = np.array([lower.count(word) for word in self.vocabulary], dtype="float32")
            if not vector.any():
                vector = np.ones(3, dtype="float32")
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


def test_retrieval_returns_matching_section_and_citation(tmp_path):
    store = VectorStore(tmp_path, KeywordEmbedder())
    store.add(
        [
            Chunk(id="1", paper="study.pdf", page=4, section_name="Methods", text="neural neural representation learning"),
            Chunk(id="2", paper="study.pdf", page=9, section_name="Results", text="climate climate temperature trends"),
        ]
    )

    results = store.search("What does the climate paper say?", k=1)

    assert results[0].paper == "study.pdf"
    assert results[0].page == 9
    assert results[0].section_name == "Results"
    assert "temperature" in results[0].text


def test_index_persists_and_reloads(tmp_path):
    store = VectorStore(tmp_path, KeywordEmbedder())
    store.add([Chunk(id="1", paper="proteins.pdf", page=2, text="protein protein folding")])

    reloaded = VectorStore(tmp_path, KeywordEmbedder())

    assert reloaded.count == 1
    assert reloaded.search("protein", k=1)[0].page == 2


def test_exact_section_lookup_precedes_semantic_retrieval(tmp_path):
    store = VectorStore(tmp_path, KeywordEmbedder())
    store.sections = [
        Section(
            id="methods",
            title="Materials and Methods",
            level=1,
            page_start=3,
            page_end=4,
            blocks=[SectionBlock(page=3, text="The experiment used spectroscopy.")],
        )
    ]

    section = store.find_section("Extract the Materials and Methods section")

    assert section is not None
    assert section.title == "Materials and Methods"
