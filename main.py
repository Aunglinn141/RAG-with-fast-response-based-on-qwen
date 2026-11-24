"""Compatibility entry point for the Ollama acceptance command."""

from local_rag.cli.check_ollama import cli


if __name__ == "__main__":
    raise SystemExit(cli())
