from pydantic import BaseModel, Field


class PageText(BaseModel):
    paper: str
    page: int
    text: str


class Chunk(BaseModel):
    id: str
    paper: str
    page: int
    section_name: str = "Full document"
    section_level: int = 1
    text: str


class DocumentMetadata(BaseModel):
    filename: str
    title: str
    authors: str
    affiliations: str = ""
    page_count: int
    abstract: str


class SectionBlock(BaseModel):
    page: int
    text: str


class Section(BaseModel):
    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    blocks: list[SectionBlock]

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


class Source(BaseModel):
    paper: str
    page: int
    section_name: str = "Full document"
    text: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str
