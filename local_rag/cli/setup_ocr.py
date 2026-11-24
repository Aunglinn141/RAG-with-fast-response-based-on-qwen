"""Prepare project-local OCR weights from the installed wheel, without network."""
import json
import sys

from local_rag.core.config import DocumentSettings
from local_rag.core.exceptions import DocumentError
from local_rag.documents.ocr import prepare_models


def main():
    try:
        files = prepare_models(DocumentSettings.from_env().ocr_model_dir)
        print(json.dumps({"success": True, "models": {k: str(v) for k, v in files.items()},
                          "network_required": False}, ensure_ascii=False, indent=2))
        return 0
    except (DocumentError, OSError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
