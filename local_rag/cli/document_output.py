"""Shared JSON destination handling for local document commands."""

from pathlib import Path

from local_rag.core.exceptions import DocumentError


def output_destination(source: Path, output: Path | None) -> Path | None:
    """Check destinations before extraction; exclusive creation guards races."""
    if output is None:
        return None
    try:
        destination = output.resolve()
        source_path = source.resolve()
        if destination == source_path or (
            destination.exists() and source_path.exists() and destination.samefile(source_path)
        ):
            raise DocumentError("output_is_source", "JSON 输出路径不能指向原始文档。", 400)
        if output.suffix.lower() != ".json":
            raise DocumentError("invalid_output_path", "输出文件必须使用 .json 扩展名。", 400)
        if output.exists() or output.is_symlink() or destination.exists():
            raise DocumentError("output_exists", "输出文件已存在，请使用新的 JSON 文件名。", 409)
        return destination
    except (OSError, RuntimeError) as exc:
        raise DocumentError("invalid_output_path", "无法访问输出路径，请检查路径及权限。", 400) from exc


def save_report(destination: Path, serialized: str) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Never truncate an existing file, including one created after preflight.
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")
    except FileExistsError as exc:
        raise DocumentError("output_exists", "输出文件已存在，请使用新的 JSON 文件名。", 409) from exc
    except OSError as exc:
        raise DocumentError("output_write_failed", "无法写入 JSON 输出文件，请检查目录权限和可用空间。", 500) from exc
