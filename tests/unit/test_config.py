"""Guard storage locations against changes in package and working directories."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from local_rag.core.config import DocumentSettings, VectorSettings, PROJECT_ROOT


class ConfigurationTests(unittest.TestCase):
    def test_vector_paths_and_model_configuration(self):
        settings = VectorSettings(chroma_dir=".runtime/test-chroma", registry_path=".runtime/test-registry/db.sqlite3")
        self.assertEqual(settings.chroma_dir, PROJECT_ROOT / ".runtime/test-chroma")
        self.assertEqual(settings.registry_path, PROJECT_ROOT / ".runtime/test-registry/db.sqlite3")
        self.assertEqual(settings.model_name, "bge-m3")
        with self.assertRaises(ValueError):
            VectorSettings(model_name="nomic-embed-text")
        with patch.dict(os.environ, {"RAG_AUTO_INDEX": "false", "OLLAMA_EMBED_MODEL": "bge-m3"}):
            self.assertFalse(VectorSettings.from_env().auto_index)

    def test_default_storage_stays_in_project_when_working_directory_changes(self):
        previous = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT / "tests")
            settings = DocumentSettings()
            self.assertEqual(settings.storage_dir, PROJECT_ROOT / ".runtime/documents")
        finally:
            os.chdir(previous)

    def test_relative_storage_uses_project_root(self):
        previous = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT / "tests")
            settings = DocumentSettings(storage_dir=Path(".runtime/custom-documents"))
            self.assertEqual(settings.storage_dir, PROJECT_ROOT / ".runtime/custom-documents")
        finally:
            os.chdir(previous)

    def test_environment_configuration_is_read_together(self):
        with patch.dict(os.environ, {
            "DOCUMENT_STORAGE_DIR": ".runtime/custom-documents",
            "DOCUMENT_MAX_FILE_BYTES": "2048",
            "DOCUMENT_MAX_TEXT_CHARACTERS": "4000",
            "DOCUMENT_CHUNK_SIZE": "200",
            "DOCUMENT_CHUNK_OVERLAP": "30",
        }):
            settings = DocumentSettings.from_env()
        self.assertEqual(settings.storage_dir, PROJECT_ROOT / ".runtime/custom-documents")
        self.assertEqual((settings.max_file_bytes, settings.max_text_characters), (2048, 4000))
        self.assertEqual((settings.chunk_size, settings.chunk_overlap), (200, 30))

    def test_invalid_environment_fails_early(self):
        with patch.dict(os.environ, {"DOCUMENT_CHUNK_SIZE": "100", "DOCUMENT_CHUNK_OVERLAP": "100"}):
            with self.assertRaises(ValueError):
                DocumentSettings.from_env()


if __name__ == "__main__":
    unittest.main()
