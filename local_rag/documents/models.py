"""Source-aware text structures independent of parsers and HTTP."""

from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class TextUnit:
    text: str
    page_number: int | None = None
    section_title: str | None = None
    heading_level: int | None = None
    paragraphs: list[dict] = field(default_factory=list)
    source_metadata: dict = field(default_factory=dict)


@dataclass
class Extraction:
    units: list[TextUnit]
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReadDocument:
    """A complete reading result, before chunking, storage or model inference."""

    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    extraction: Extraction

    @property
    def text(self) -> str:
        return "\n\n".join(unit.text for unit in self.extraction.units)

    def to_dict(self) -> dict:
        """Export source units with positions in the cleaned, complete text."""
        units = []
        offset = 0
        for index, unit in enumerate(self.extraction.units, 1):
            paragraphs = [
                {**paragraph, "start_index": offset + paragraph["start_index"],
                 "end_index": offset + paragraph["end_index"]}
                for paragraph in unit.paragraphs
            ]
            units.append({
                "text": unit.text,
                "metadata": {
                    **location_metadata(unit, paragraphs),
                    "document_id": self.document_id, "filename": self.filename,
                    "file_type": self.file_type, "unit_index": index,
                    "page_number": unit.page_number,
                    "section_title": unit.section_title, "heading_level": unit.heading_level,
                    "paragraph_start": paragraphs[0]["paragraph_index"] if paragraphs else None,
                    "paragraph_end": paragraphs[-1]["paragraph_index"] if paragraphs else None,
                    "paragraphs": paragraphs,
                    "start_index": offset, "end_index": offset + len(unit.text),
                },
            })
            offset += len(unit.text) + 2
        text = self.text
        return {
            "document_id": self.document_id, "filename": self.filename,
            "file_type": self.file_type, "size_bytes": self.size_bytes,
            "page_count": self.extraction.page_count, "character_count": len(text),
            "text": text, "warnings": list(self.extraction.warnings), "units": units,
        }


@dataclass
class ChunkedDocument:
    """A source document and its character-bounded chunks, ready for indexing."""

    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    page_count: int | None
    text: str
    warnings: list[str]
    chunk_size: int
    chunk_overlap: int
    chunks: list[dict]

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id, "filename": self.filename,
            "file_type": self.file_type, "size_bytes": self.size_bytes,
            "page_count": self.page_count, "text": self.text,
            "character_count": len(self.text), "warnings": list(self.warnings),
            "chunk_config": {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap},
            "chunk_count": len(self.chunks), "chunks": deepcopy(self.chunks),
        }

def location_metadata(unit: TextUnit, paragraphs: list[dict]) -> dict:
    """Keep table ranges accurate after splitting; a PDF page never becomes a sheet."""
    metadata = dict(unit.source_metadata)
    rows = [p for p in paragraphs if "row_index" in p]
    if rows:
        metadata.update(row_start=min(p["row_index"] for p in rows),
                        row_end=max(p["row_index"] for p in rows),
                        cell_ranges=[p["cell_range"] for p in rows])
    return metadata
