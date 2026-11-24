"""Compatibility entry point for: python -m uvicorn app:app."""

from local_rag.api.app import app, create_app

__all__ = ["app", "create_app"]
