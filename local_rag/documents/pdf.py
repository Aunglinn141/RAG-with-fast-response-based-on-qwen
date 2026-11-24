"""Read PDF text first; OCR only the pages that need it, preserving page numbers."""
from pypdf import PdfReader

from local_rag.core.exceptions import DocumentError
from local_rag.documents.models import Extraction, TextUnit
from local_rag.documents.ocr import OCR_WARNING, recognize_pdf_page


def extract_pdf(path, settings):
    from local_rag.documents.extractors import clean_text
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise DocumentError("file_type_mismatch", "扩展名为 PDF，但文件内容不是 PDF。", 415)
        stream.seek(0)
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise DocumentError("encrypted_pdf", "不支持加密 PDF，请先解除密码保护。")
        if len(reader.pages) > 1000:
            raise DocumentError("too_many_pages", "PDF 超出 1000 页上限。", 413)
        result = Extraction([], page_count=len(reader.pages))
        total = ocr_pages = 0
        for number, page in enumerate(reader.pages, 1):
            text = clean_text(page.extract_text() or "")
            metadata = {}
            resources = page.get("/Resources", {})
            if hasattr(resources, "get_object"):
                resources = resources.get_object()
            needs_ocr = (not text and page.get_contents() is not None) or (
                len(text) < 40 and "/XObject" in resources)
            if settings.ocr_mode == "always" or (settings.ocr_mode == "auto" and needs_ocr):
                ocr_pages += 1
                if ocr_pages > settings.ocr_max_pages:
                    raise DocumentError("too_many_ocr_pages", "需要 OCR 的 PDF 页数超出上限，请拆分文件后导入。", 413)
                recognized, source = recognize_pdf_page(path, number - 1, settings)
                if recognized:
                    text, metadata = recognized, source
                elif text:
                    result.warnings.append(f"第 {number} 页 OCR 未识别到文字，保留已有文字层。")
            total += len(text) + 2
            if total > settings.max_text_characters:
                raise DocumentError("extracted_text_too_large", "提取的文本超出字符数量上限。", 413)
            if text:
                result.units.append(TextUnit(text, page_number=number, source_metadata=metadata))
            else:
                result.warnings.append(f"第 {number} 页未提取到文本，可能为空白页或 OCR 未识别出文字。")
        if ocr_pages:
            result.warnings.append(OCR_WARNING)
        return result
