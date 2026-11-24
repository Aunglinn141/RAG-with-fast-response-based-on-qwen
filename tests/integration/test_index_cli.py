"""Check acceptance CLI orchestration with an explicitly synthetic loopback Ollama server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import shutil
import subprocess
import sys
import threading
import unittest
from uuid import uuid4

from local_rag.core.config import PROJECT_ROOT, RUNTIME_DIR


class AcceptanceCommandTests(unittest.TestCase):
    def test_four_cli_phases_use_fresh_processes_and_real_storage(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length)) if length else {}
                if self.path == "/api/show":
                    result = {"capabilities": ["embedding"]}
                elif self.path == "/api/tags":
                    result = {"models": [{"name": "bge-m3:latest", "digest": "synthetic-cli-contract-test"}]}
                elif self.path == "/api/embed":
                    vectors = []
                    for text in body["input"]:
                        vector = [0.0] * 1024
                        vector[0 if "苹果" in text else 1] = 1.0
                        vectors.append(vector)
                    result = {"embeddings": vectors}
                else:
                    result = {"error": "unexpected route"}
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_POST = do_GET

            def log_message(self, *args):
                pass

        root = RUNTIME_DIR / "tests" / ("cli-contract-" + uuid4().hex)
        root.mkdir(parents=True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = {**os.environ, "LOCAL_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                   "OLLAMA_EMBED_MODEL": "bge-m3", "OLLAMA_TIMEOUT": "5", "PYTHONIOENCODING": "utf-8"}
            for phase in ("create", "search", "delete", "verify_deleted"):
                with self.subTest(phase=phase):
                    result = subprocess.run(
                        [sys.executable, "-m", "local_rag.cli.check_index", "--phase", phase, "--root", str(root)],
                        cwd=PROJECT_ROOT, env=env, capture_output=True, encoding="utf-8", timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    data = json.loads(result.stdout)
                    if phase == "create":
                        self.assertEqual(data["model"]["model_version"], "synthetic-cli-contract-test")
                    elif phase == "verify_deleted":
                        self.assertEqual(data["remaining_vectors"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            if root.resolve().parent == (RUNTIME_DIR / "tests").resolve():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
