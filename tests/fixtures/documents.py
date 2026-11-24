"""Small real PDF and DOCX fixtures generated without disk or network access."""

from io import BytesIO
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_bytes(pages, password=None):
    """Create text-layer PDF fixtures in memory, including explicit blank pages."""
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    }))
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
            })
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content = DecodedStreamObject()
            content.set_data(f"BT /F1 12 Tf 40 750 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(content)
    if password:
        writer.encrypt(password)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def docx_bytes():
    doc = Document()
    doc.add_heading("Document processing guide", 0)
    doc.add_heading("First section", 1)
    doc.add_paragraph("本地文档处理保留段落信息。" * 100)
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Table key"
    table.cell(0, 1).text = "Table value"
    doc.add_heading("Second section", 2)
    doc.add_paragraph("Second section body.")
    stream = BytesIO()
    doc.save(stream)
    return stream.getvalue()
