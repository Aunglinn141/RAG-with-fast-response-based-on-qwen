"""Validate report destinations before invoking any document or model work."""

from pathlib import Path


def validate_report_path(output):
    path = Path(output).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Benchmark --output must end in .json; the CSV is written beside it")
    return path
