"""Bounded Excel/CSV/TSV reading with real sheet and row provenance."""
import csv
from contextlib import contextmanager
from datetime import date, datetime, time
from io import StringIO
from itertools import zip_longest
from zipfile import BadZipFile, ZipFile

from local_rag.core.exceptions import DocumentError
from local_rag.documents.models import Extraction, TextUnit


def validate_workbook(path):
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if not {"[Content_Types].xml", "xl/workbook.xml"} <= set(archive.namelist()):
                raise DocumentError("file_type_mismatch", "文件不是有效的 Excel 工作簿。", 415)
            if len(entries) > 10000 or sum(e.file_size for e in entries) > 100 * 1024 * 1024:
                raise DocumentError("spreadsheet_too_large", "Excel 解压后的内容超出处理上限。", 413)
    except BadZipFile as exc:
        raise DocumentError("file_type_mismatch", "文件不是有效的 XLSX/XLSM 工作簿。", 415) from exc


def cell_text(value):
    from local_rag.documents.extractors import clean_text
    if value is None:
        return ""
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return clean_text(str(value))


class TableBuilder:
    def __init__(self, settings):
        self.settings = settings
        self.cells = self.characters = 0
        self.result = Extraction([])

    def add_sheet(self, name, index, rows):
        from openpyxl.utils import get_column_letter
        if index > self.settings.spreadsheet_max_sheets:
            raise DocumentError("too_many_sheets", "工作表数量超出处理上限。", 413)
        parts, paragraphs, headers = [], [], None
        offset = 0
        for row_index, values in rows:
            self.cells += len(values)
            if self.cells > self.settings.spreadsheet_max_cells:
                raise DocumentError("spreadsheet_too_large", "表格单元格数量超出处理上限。", 413)
            cells = [(i, cell_text(value)) for i, value in enumerate(values, 1)]
            cells = [(i, value) for i, value in cells if value]
            if not cells:
                continue
            is_header = headers is None
            if is_header:
                headers = dict(cells)
            fields = []
            for column, value in cells:
                address = f"{get_column_letter(column)}{row_index}"
                # Preserve original cell text; a formula without a cache is labeled.
                label = headers.get(column, "") if not is_header else ""
                fields.append(f"{address}" + (f"（{label}）" if label else "") + f"：{value}")
            text = f"工作表 {name}，第 {row_index} 行：" + " | ".join(fields)
            self.characters += len(text) + 2
            if self.characters > self.settings.max_text_characters:
                raise DocumentError("extracted_text_too_large", "表格提取文字超出字符数量上限。", 413)
            start = offset + (2 if parts else 0)
            paragraphs.append({"paragraph_index": row_index, "style": "table_row",
                               "row_index": row_index,
                               "cell_range": f"{get_column_letter(cells[0][0])}{row_index}:{get_column_letter(cells[-1][0])}{row_index}",
                               "start_index": start, "end_index": start + len(text)})
            parts.append(text)
            offset = start + len(text)
        if parts:
            self.result.units.append(TextUnit("\n\n".join(parts), section_title=name,
                                               paragraphs=paragraphs,
                                               source_metadata={"sheet_name": name, "sheet_index": index,
                                                                "extraction_method": "table"}))
        else:
            self.result.warnings.append(f"工作表 {name} 没有可提取的单元格内容。")


def extract_tables(path, extension, settings):
    from local_rag.documents.extractors import _decode_text
    builder = TableBuilder(settings)
    if extension in {".csv", ".tsv"}:
        raw = _decode_text(path.read_bytes())
        if len(raw) > settings.max_text_characters:
            raise DocumentError("extracted_text_too_large", "表格文本超出字符数量上限。", 413)
        # csv handles quoted delimiters and multiline fields; row numbers are record numbers.
        rows = csv.reader(StringIO(raw, newline=""), delimiter="\t" if extension == ".tsv" else ",", strict=True)
        builder.add_sheet("数据", 1, enumerate(rows, 1))
    elif extension == ".xls":
        import xlrd
        with path.open("rb") as stream:
            if stream.read(8) != bytes.fromhex("D0CF11E0A1B11AE1"):
                raise DocumentError("file_type_mismatch", "文件不是有效的 XLS 工作簿。", 415)
        book = xlrd.open_workbook(path, on_demand=True)
        try:
            if book.nsheets > settings.spreadsheet_max_sheets:
                raise DocumentError("too_many_sheets", "工作表数量超出处理上限。", 413)
            for index in range(book.nsheets):
                sheet = book.sheet_by_index(index)
                if builder.cells + sheet.nrows * sheet.ncols > settings.spreadsheet_max_cells:
                    raise DocumentError("spreadsheet_too_large", "表格单元格数量超出处理上限。", 413)
                def rows():
                    for i in range(sheet.nrows):
                        values = []
                        for cell in sheet.row(i):
                            if cell.ctype == xlrd.XL_CELL_DATE:
                                values.append(xlrd.xldate.xldate_as_datetime(cell.value, book.datemode))
                            elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                                values.append(bool(cell.value))
                            elif cell.ctype == xlrd.XL_CELL_ERROR:
                                values.append(xlrd.error_text_from_code.get(cell.value, "#ERROR"))
                            else:
                                values.append(cell.value)
                        yield i + 1, values
                builder.add_sheet(sheet.name, index + 1, rows())
                book.unload_sheet(index)
        finally:
            book.release_resources()
    else:
        from openpyxl import load_workbook
        validate_workbook(path)
        missing_formulas = 0
        with _workbook(path, False) as book, _workbook(path, True) as cached:
            if len(book.worksheets) > settings.spreadsheet_max_sheets:
                raise DocumentError("too_many_sheets", "工作表数量超出处理上限。", 413)
            for index, sheet in enumerate(book.worksheets, 1):
                if (sheet.max_row and sheet.max_column and
                        builder.cells + sheet.max_row * sheet.max_column > settings.spreadsheet_max_cells):
                    raise DocumentError("spreadsheet_too_large", "表格单元格数量超出处理上限。", 413)
                # Do not trust undersized worksheet dimensions supplied by producers.
                sheet.reset_dimensions()
                cached[sheet.title].reset_dimensions()
                def rows():
                    nonlocal missing_formulas
                    for row_index, (original, saved) in enumerate(zip_longest(
                            sheet.iter_rows(), cached[sheet.title].iter_rows(), fillvalue=()), 1):
                        values = []
                        for cell, cache in zip_longest(original, saved):
                            value = cell.value if cell else None
                            if cell and cell.data_type == "f":
                                value = cache.value if cache else None
                                if value is None:
                                    missing_formulas += 1
                                    value = f"[未计算公式：{cell.value}]"
                            values.append(value)
                        yield row_index, values
                builder.add_sheet(sheet.title, index, rows())
        if missing_formulas:
            builder.result.warnings.append(f"{missing_formulas} 个公式没有缓存结果，保留并标注公式；可在 Excel 中计算并保存后重新导入。")
    if extension in {".xls", ".xlsx", ".xlsm"}:
        builder.result.warnings.append("Excel 公式使用文件已保存的结果，本地读取不会执行公式或宏。")
    return builder.result

@contextmanager
def _workbook(path, data_only):
    from openpyxl import load_workbook
    # Own the original handle: read-only XML iterators may remain open after an error.
    with path.open("rb") as source:
        book = load_workbook(source, read_only=True, data_only=data_only, keep_links=False)
        try:
            yield book
        finally:
            book.close()
