"""Local PDF, DOCX, TXT and Markdown extraction with source locations."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from zipfile import ZipFile

from docx import Document as WordDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from markdown_it import MarkdownIt
from pypdf import PdfReader


from local_rag.core.config import DocumentSettings, IMAGE_EXTENSIONS, TABLE_EXTENSIONS, SUPPORTED_FILE_MESSAGE
from local_rag.core.exceptions import DocumentError
from local_rag.documents.models import Extraction, TextUnit


def clean_text(text: str) -> str:
    """Collapse horizontal whitespace while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = text.replace("\ufeff", "").replace("\x00", "")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _decode_text(data: bytes) -> str:
    if data.startswith((b"%PDF-", b"PK\x03\x04", b"MZ", b"\x89PNG")):
        raise DocumentError("file_type_mismatch", "文件内容不是纯文本。", 415)
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encodings = ["utf-32"]
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    else:
        encodings = ["utf-8-sig", "gb18030"]
    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if any(ord(char) < 32 and char not in "\n\r\t\f" for char in text):
            raise DocumentError("binary_text_file", "文本文件包含二进制控制字符。", 415)
        return text
    raise DocumentError("unsupported_encoding", "无法解码文本，请保存为 UTF-8、带 BOM 的 UTF-16/32 或 GB18030。")


def _paragraph_units(paragraphs, max_characters: int) -> list[TextUnit]:
    units: list[TextUnit] = []
    current = TextUnit("")
    total = 0
    for index, (text, level, style) in enumerate(paragraphs, 1):
        text = clean_text(text)
        if not text:
            continue
        total += len(text) + 2
        if total > max_characters:
            raise DocumentError("extracted_text_too_large", "提取的文本超出字符数量上限。", 413)
        if level is not None:
            if current.text:
                units.append(current)
            current = TextUnit("", section_title=text, heading_level=level)
        start = len(current.text) + (2 if current.text else 0)
        current.text += ("\n\n" if current.text else "") + text
        current.paragraphs.append({
            "paragraph_index": index, "style": style,
            "start_index": start, "end_index": start + len(text),
        })
    if current.text:
        units.append(current)
    return units


def _heading_level(paragraph: Paragraph) -> int | None:
    style = paragraph.style
    if style is not None and style.name == "Title":
        return 0
    # Custom paragraph styles may inherit their outline level from a base style.
    elements = [paragraph._p]
    seen = set()
    while style is not None and style.style_id not in seen:
        seen.add(style.style_id)
        match = re.fullmatch(r"Heading\s*([1-9])", style.name, flags=re.I)
        if match:
            return int(match.group(1))
        elements.append(style.element)
        style = style.base_style
    for element in elements:
        levels = element.xpath("./w:pPr/w:outlineLvl")
        if levels:
            level = int(levels[0].get(qn("w:val")))
            return level + 1 if 0 <= level < 9 else None
    return None


def _word_paragraphs(container):
    for item in container.iter_inner_content():
        if isinstance(item, Paragraph):
            yield item.text, _heading_level(item), item.style.name if item.style else None
        elif isinstance(item, Table):
            seen_cells = set()
            for row in item.rows:
                for cell in row.cells:
                    if cell._tc not in seen_cells:
                        seen_cells.add(cell._tc)
                        yield from _word_paragraphs(cell)


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"br", "p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _markdown_paragraphs(text: str):
    level = None
    for token in MarkdownIt("commonmark").enable("table").parse(text):
        if token.type == "heading_open":
            level = int(token.tag[1:])
        elif token.type == "heading_close":
            level = None
        elif token.type == "inline":
            parts = []
            for child in token.children or []:
                if child.type in {"text", "code_inline", "image"}:
                    parts.append(child.content)
                elif child.type in {"softbreak", "hardbreak"}:
                    parts.append("\n")
            yield "".join(parts), level, "heading" if level else "paragraph"
        elif token.type in {"fence", "code_block"}:
            yield token.content, None, "code"
        elif token.type == "html_block":
            parser = _HTMLText()
            parser.feed(token.content)
            yield "".join(parser.parts), None, "html"


def extract_document(path: Path, extension: str, max_characters=2_000_000, *, settings=None) -> Extraction:
    """Preserve pages, sections and table locations before character chunking."""
    settings = settings or DocumentSettings(max_text_characters=max_characters)
    try:
        if extension in {".txt", ".text", ".log", ".md", ".markdown", ".html", ".htm", ".json"}:
            text = _decode_text(path.read_bytes())
            if len(text) > max_characters:
                raise DocumentError("extracted_text_too_large", "文本超出字符数量上限。", 413)
            if extension in {".html", ".htm"}:
                parser = _HTMLText()
                parser.feed(text)
                text = "".join(parser.parts)
            elif extension == ".json":
                import json
                json.loads(text)
            units = (_paragraph_units(_markdown_paragraphs(text), max_characters)
                     if extension in {".md", ".markdown"} else [TextUnit(clean_text(text))])
            result = Extraction(units)
        elif extension == ".pdf":
            from local_rag.documents.pdf import extract_pdf
            result = extract_pdf(path, settings)
        elif extension in TABLE_EXTENSIONS:
            from local_rag.documents.tables import extract_tables
            result = extract_tables(path, extension, settings)
        elif extension in IMAGE_EXTENSIONS:
            from local_rag.documents.ocr import extract_images
            result = extract_images(path, extension, settings)
        elif extension == ".docx":
            with ZipFile(path) as archive:
                entries = archive.infolist()
                names = set(archive.namelist())
                if not {"[Content_Types].xml", "word/document.xml"} <= names:
                    raise DocumentError("file_type_mismatch", "文件不是有效的 DOCX 文档。", 415)
                if len(entries) > 5000 or sum(item.file_size for item in entries) > 100 * 1024 * 1024:
                    raise DocumentError("docx_too_large", "DOCX 解压后的内容超出处理上限。", 413)
            result = Extraction(_paragraph_units(_word_paragraphs(WordDocument(path)), max_characters))
        else:
            raise DocumentError("unsupported_file_type", SUPPORTED_FILE_MESSAGE, 415)
    except DocumentError:
        raise
    except ImportError as exc:
        raise DocumentError("document_dependency_missing", "缺少文档解析依赖，请安装 requirements.txt。", 503) from exc
    except (OSError, MemoryError):
        raise
    except Exception as exc:
        # Parser-specific failures must not become a successful empty document.
        raise DocumentError("invalid_document", "文档损坏或结构无法解析，请检查后重新上传。") from exc
    result.units = [unit for unit in result.units if unit.text]
    if not result.units:
        message = "未提取到可用文本。"
        if extension == ".pdf":
            message += "PDF 可能为空白页或未识别出文字；扫描件请启用 OCR 并检查图像清晰度。"
        raise DocumentError("no_extractable_text", message)
    return result
