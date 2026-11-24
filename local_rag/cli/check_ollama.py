"""Run with: python -m local_rag.cli.check_ollama."""

import argparse
import json
import os
from pathlib import Path
import sys

from local_rag.clients.ollama import OllamaClient, OllamaError
from local_rag.core.config import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL


def main():
    parser = argparse.ArgumentParser(description="验证本地 Ollama 问答和 Embedding，无第三方 Python 依赖。")
    parser.add_argument("--url", default=os.getenv("LOCAL_OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--chat-model", default=os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL))
    parser.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--question", default="1 + 1 等于几？请只回答数字。")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, help="将回答和完整向量保存为本地 JSON 文件")
    args = parser.parse_args()

    try:
        client = OllamaClient(args.url, timeout=args.timeout)
        version = client.version()
        print(f"[PASS] Python 已连接 Ollama {version}：{client.base_url}", flush=True)

        answer = client.chat(args.chat_model, args.question)
        if args.question == parser.get_default("question") and answer != "2":
            raise OllamaError(f"简单算术验收失败：预期回答为 2，实际为 {answer!r}。")
        print(f"[PASS] 本地问答模型：{args.chat_model}")
        print(f"问题：{args.question}\n回答：{answer}", flush=True)

        texts = ["Python 可以连接本地 Ollama。", "如何在本机生成文本向量？"]
        vectors = client.embed(args.embed_model, texts)
        print(f"[PASS] Embedding 模型：{args.embed_model}")
        print(f"向量数量：{len(vectors)}；每个向量维度：{len(vectors[0])}")
        print(f"第一个向量前 5 项：{vectors[0][:5]}")
        print("[PASS] 客户端仅连接回环地址，禁用代理和重定向，已检查模型元数据为本地模型。")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "url": client.base_url, "ollama_version": version,
                "chat_model": args.chat_model, "question": args.question, "answer": answer,
                "embedding_model": args.embed_model, "texts": texts,
                "dimension": len(vectors[0]), "embeddings": vectors,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"结果已保存：{args.output.resolve()}")
    except (OllamaError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print("全部验收通过。")
    return 0


def cli():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
