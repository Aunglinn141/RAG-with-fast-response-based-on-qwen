"""Offline CPU OCR. Only the explicit setup command copies bundled model files."""
from functools import lru_cache
from contextlib import closing
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
from threading import RLock
from uuid import uuid4

from local_rag.core.exceptions import DocumentError
from local_rag.documents.models import Extraction, TextUnit

MODELS = {
    "Det": ("PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "Cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    "Rec": ("PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
}
_OCR_LOCK = RLock()
OCR_WARNING = "本文件包含本地 OCR 识别文字，复杂排版、手写或模糊文字可能有误，请核对原文件。"


def model_files(directory):
    directory = Path(directory)
    files = {}
    for stage, (name, digest) in MODELS.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise DocumentError("ocr_models_unavailable",
                                "本地 OCR 模型缺失或校验失败，请执行 python -m local_rag.cli.setup_ocr。", 503)
        files[stage] = path
    return files


def prepare_models(directory):
    """Copy the pinned RapidOCR wheel's weights; never download during a request."""
    from filelock import FileLock
    spec = importlib.util.find_spec("rapidocr")
    if spec is None:
        raise DocumentError("ocr_dependency_missing", "请先安装 requirements.txt 中的 OCR 依赖。", 503)
    bundled = model_files(Path(spec.origin).parent / "models")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".setup.lock"), timeout=30):
        for stage, source in bundled.items():
            target = directory / source.name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == MODELS[stage][1]:
                continue
            temporary = directory / (source.name + "." + uuid4().hex + ".tmp")
            try:
                shutil.copyfile(source, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return model_files(directory)


@lru_cache(maxsize=1)
def _engine(directory, threads):
    files = model_files(directory)
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params={
            **{stage + ".model_path": str(path) for stage, path in files.items()},
            "Global.model_root_dir": str(directory), "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.use_dml": False,
        })
    except ImportError as exc:
        raise DocumentError("ocr_dependency_missing", "请安装 requirements.txt 中的本地 OCR 依赖。", 503) from exc


def recognize(image, settings):
    from local_rag.documents.extractors import clean_text
    if settings.ocr_mode == "off":
        raise DocumentError("ocr_disabled", "OCR 已关闭，请设置 DOCUMENT_OCR_MODE=auto 后重试。", 422)
    if image.width * image.height > settings.ocr_max_pixels:
        raise DocumentError("image_too_large", "图片像素数量超出 OCR 处理上限。", 413)
    # Empty pages need neither weights nor model inference.
    with image.convert("L") as grayscale:
        low, high = grayscale.getextrema()
    if low == high:
        return "", {}
    try:
        # RapidOCR updates per-call state; serialize use of the cached CPU engine.
        with _OCR_LOCK:
            result = _engine(str(settings.ocr_model_dir), settings.ocr_threads)(image)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        text = clean_text("\n".join(texts)) if texts is not None else ""
        metadata = {"extraction_method": "ocr", "ocr_engine": "RapidOCR/PP-OCRv6"}
        if scores is not None and len(scores):
            metadata["ocr_confidence"] = round(sum(float(s) for s in scores) / len(scores), 4)
        return text, metadata
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("ocr_failed", "本地 OCR 识别失败，请检查模型、图片和可用内存。", 503) from exc


def recognize_pdf_page(path, index, settings):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise DocumentError("ocr_dependency_missing", "缺少 PDF 渲染依赖，请安装 requirements.txt。", 503) from exc
    # Close every native resource, including a bitmap that backs the PIL adapter.
    with pdfium.PdfDocument(path) as pdf:
        with closing(pdf[index]) as page:
            width, height = page.get_size()
            if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                raise DocumentError("invalid_document", "PDF 页面尺寸无效。")
            scale = min(settings.ocr_dpi / 72, math.sqrt(settings.ocr_max_pixels / (width * height)) * .99)
            with closing(page.render(scale=scale)) as bitmap:
                with bitmap.to_pil() as image:
                    return recognize(image, settings)


def extract_images(path, extension, settings):
    from PIL import Image, ImageOps, UnidentifiedImageError
    expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP",
                ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF"}
    if settings.ocr_mode == "off":
        raise DocumentError("ocr_disabled", "图片需要 OCR，请设置 DOCUMENT_OCR_MODE=auto 后重试。", 422)
    try:
        with Image.open(path) as image:
            if image.format != expected[extension]:
                raise DocumentError("file_type_mismatch", "图片内容与扩展名不匹配。", 415)
            frames = getattr(image, "n_frames", 1)
            if frames > settings.ocr_max_pages:
                raise DocumentError("too_many_ocr_pages", "图片帧数超出 OCR 页数上限。", 413)
            result = Extraction([], page_count=frames, warnings=[OCR_WARNING])
            total = 0
            for index in range(frames):
                image.seek(index)
                if image.width * image.height > settings.ocr_max_pixels:
                    raise DocumentError("image_too_large", "图片像素数量超出 OCR 处理上限。", 413)
                with ImageOps.exif_transpose(image) as oriented:
                    with oriented.convert("RGB") as rgb:
                        text, metadata = recognize(rgb, settings)
                total += len(text) + 2
                if total > settings.max_text_characters:
                    raise DocumentError("extracted_text_too_large", "识别文字超出字符数量上限。", 413)
                if text:
                    result.units.append(TextUnit(text, page_number=index + 1, source_metadata=metadata))
                else:
                    result.warnings.append(f"第 {index + 1} 帧未识别到文字。")
            return result
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise DocumentError("invalid_image", "图片损坏、格式无效或尺寸过大。", 422) from exc
