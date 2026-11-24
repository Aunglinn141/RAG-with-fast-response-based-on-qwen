"""Small real text-layer and image-only PDFs for reader behavior tests.

These fixtures exercise PDF text mappings, not visual font rendering. Their
ToUnicode table is explicit, so Chinese extraction needs no installed font,
network service or OCR engine.
"""

from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, NameObject,
    NumberObject, TextStringObject,
)


def _serialize(writer):
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def unicode_pdf_bytes(pages):
    """Create pages with two-byte character codes and an explicit Unicode map."""
    writer = PdfWriter()
    characters = sorted(set("".join(pages)))
    if any(ord(character) > 0xFFFF for character in characters):
        raise ValueError("This deliberately small PDF fixture supports BMP characters only")
    mapping = [f"<{ord(character):04X}> <{ord(character):04X}>" for character in characters]
    cmap = DecodedStreamObject()
    cmap.set_data((
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /ReaderFixture-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        + f"{len(mapping)} beginbfchar\n" + "\n".join(mapping)
        + "\nendbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode("ascii"))
    descriptor = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/FontDescriptor"),
        NameObject("/FontName"): NameObject("/ReaderFixture"),
        NameObject("/Flags"): NumberObject(4),
        NameObject("/FontBBox"): ArrayObject([NumberObject(x) for x in (0, -200, 1000, 900)]),
        NameObject("/ItalicAngle"): NumberObject(0),
        NameObject("/Ascent"): NumberObject(900),
        NameObject("/Descent"): NumberObject(-200),
        NameObject("/CapHeight"): NumberObject(700),
        NameObject("/StemV"): NumberObject(80),
    }))
    descendant = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/CIDFontType2"),
        NameObject("/BaseFont"): NameObject("/ReaderFixture"),
        NameObject("/CIDSystemInfo"): DictionaryObject({
            NameObject("/Registry"): TextStringObject("Adobe"),
            NameObject("/Ordering"): TextStringObject("Identity"),
            NameObject("/Supplement"): NumberObject(0),
        }),
        NameObject("/FontDescriptor"): descriptor,
        NameObject("/CIDToGIDMap"): NameObject("/Identity"),
        NameObject("/DW"): NumberObject(1000),
    }))
    font = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type0"),
        NameObject("/BaseFont"): NameObject("/ReaderFixture"),
        NameObject("/Encoding"): NameObject("/Identity-H"),
        NameObject("/DescendantFonts"): ArrayObject([descendant]),
        NameObject("/ToUnicode"): writer._add_object(cmap),
    }))
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if not text:
            continue
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 40 750 Td <{text.encode('utf-16-be').hex()}> Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    return _serialize(writer)


def image_only_pdf_bytes():
    """Return an actual raster-only page, not an empty page mislabeled a scan."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    picture = DecodedStreamObject()
    picture.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Image"),
        NameObject("/Width"): NumberObject(2),
        NameObject("/Height"): NumberObject(2),
        NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
        NameObject("/BitsPerComponent"): NumberObject(8),
    })
    picture.set_data(bytes((0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0)))
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): writer._add_object(picture)}),
    })
    content = DecodedStreamObject()
    content.set_data(b"q 400 0 0 400 40 200 cm /Im0 Do Q")
    page[NameObject("/Contents")] = writer._add_object(content)
    return _serialize(writer)
