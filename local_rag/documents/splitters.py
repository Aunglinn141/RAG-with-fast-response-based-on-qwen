"""LangChain character chunking with exact document/page/paragraph provenance."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.core.config import validate_chunk_settings
from local_rag.core.exceptions import DocumentError
from local_rag.documents.models import ChunkedDocument, Extraction, ReadDocument, location_metadata


class DocumentChunker:
    """Split a reader result without registering, embedding or saving it.

    Size and target overlap count Python characters, not tokens or bytes. Each
    source unit is split independently to keep PDF pages/sections unambiguous.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        validate_chunk_settings(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: ReadDocument) -> ChunkedDocument:
        text, chunks = split_document(
            document.extraction, document.document_id, document.filename,
            "." + document.file_type, self.chunk_size, self.chunk_overlap,
        )
        if not chunks:
            raise DocumentError("no_extractable_text", "文档没有可切分的有效文本。", 422)
        return ChunkedDocument(
            document_id=document.document_id, filename=document.filename,
            file_type=document.file_type, size_bytes=document.size_bytes,
            page_count=document.extraction.page_count, text=text,
            warnings=list(document.extraction.warnings),
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, chunks=chunks,
        )


def split_document(extraction: Extraction, document_id: str, filename: str,
                   extension: str, chunk_size=700, chunk_overlap=100):
    """Compatibility entry point; new callers can pass a ReadDocument to the class."""
    validate_chunk_settings(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "；", "; ", "，", ", ", " ", ""],
        keep_separator="end", add_start_index=True, strip_whitespace=True,
        is_separator_regex=False,
    )
    chunks = []
    text_parts = []
    offset = 0
    for unit_index, unit in enumerate(extraction.units, 1):
        text_parts.append(unit.text)
        for document in splitter.create_documents([unit.text]):
            start = document.metadata["start_index"]
            end = start + len(document.page_content)
            if start < 0 or unit.text[start:end] != document.page_content:
                raise RuntimeError("Cannot map chunk back to its extracted text.")
            paragraphs = [
                {**paragraph, "start_index": offset + paragraph["start_index"],
                 "end_index": offset + paragraph["end_index"]}
                for paragraph in unit.paragraphs
                if paragraph["start_index"] < end and paragraph["end_index"] > start
            ]
            chunk_index = len(chunks) + 1
            chunks.append({
                "chunk_id": f"{document_id}:{chunk_index}",
                "text": document.page_content,
                "metadata": {
                    **location_metadata(unit, paragraphs),
                    "document_id": document_id, "filename": filename,
                    "file_type": extension[1:], "page_number": unit.page_number,
                    "chunk_index": chunk_index, "unit_index": unit_index,
                    "section_title": unit.section_title, "heading_level": unit.heading_level,
                    "paragraph_start": paragraphs[0]["paragraph_index"] if paragraphs else None,
                    "paragraph_end": paragraphs[-1]["paragraph_index"] if paragraphs else None,
                    "paragraphs": paragraphs,
                    "start_index": offset + start, "end_index": offset + end,
                },
            })
        offset += len(unit.text) + 2
    return "\n\n".join(text_parts), chunks
