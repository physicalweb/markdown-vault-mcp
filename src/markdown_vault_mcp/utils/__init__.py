"""Shared pure-function utilities for markdown-vault-mcp."""

from __future__ import annotations

import fnmatch
import posixpath
from typing import TYPE_CHECKING

from markdown_vault_mcp.types import DEFAULT_ATTACHMENT_EXTENSIONS

if TYPE_CHECKING:
    from pathlib import Path
from markdown_vault_mcp.utils.fts import fts_row_to_note_info
from markdown_vault_mcp.utils.links import (
    apply_link_replacement,
    compute_new_raw_target,
)
from markdown_vault_mcp.utils.text import (
    CHAR_SUBS,
    build_position_map,
    find_closest_match,
    normalize_text,
)


def is_path_excluded(path: str, exclude_patterns: list[str] | None) -> bool:
    """Check whether *path* matches any configured exclude pattern.

    Args:
        path: Relative POSIX path string.
        exclude_patterns: Glob patterns to check against.  ``None`` or
            empty means nothing is excluded.

    Returns:
        ``True`` if the path matches any pattern in *exclude_patterns*.
    """
    if not exclude_patterns:
        return False
    return any(fnmatch.fnmatch(path, pat) for pat in exclude_patterns)


def effective_attachment_extensions(
    attachment_extensions: list[str] | None,
) -> frozenset[str]:
    """Return the effective set of allowed attachment extensions.

    Args:
        attachment_extensions: User-configured extension list, or ``None``
            to use the default set.

    Returns:
        Frozenset of lower-case extension strings (without leading dot).
        The special value ``frozenset(["*"])`` means all non-.md files.
    """
    if attachment_extensions is None:
        return DEFAULT_ATTACHMENT_EXTENSIONS
    return frozenset(attachment_extensions)


def validate_path(path: str, source_dir: Path) -> Path:
    """Resolve a relative path and validate it is inside *source_dir*.

    Args:
        path: Relative document path (must end with ``.md``).
        source_dir: Absolute path to the vault root directory.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the path escapes the source directory or does
            not end with ``.md``.
    """
    if not path.endswith(".md"):
        raise ValueError(f"Path must end with '.md': {path}")
    abs_path = (source_dir / path).resolve()
    if not abs_path.is_relative_to(source_dir.resolve()):
        raise ValueError(f"Path traversal detected: {path}")
    return abs_path


def check_write_prefix(path: str, write_prefix: str | None, source_dir: Path) -> None:
    """Refuse a write whose *path* is not under *write_prefix*.

    The one guard behind every mutating operation (write, edit, delete,
    both sides of rename, fetch, uploads).  Reads are not affected.

    Args:
        path: Vault-relative path the caller wants to write.
        write_prefix: Vault-relative folder writes are confined to (e.g.
            ``"docs"``).  ``None`` or empty means no restriction.
        source_dir: Absolute path to the vault root, used to suggest the
            path that was likely meant.

    Raises:
        ValueError: If *path* is not strictly inside *write_prefix*.  The
            message names ``<prefix>/<path>`` when ``<prefix>/<first
            segment>`` exists as a folder.
    """
    if not write_prefix:
        return
    prefix = write_prefix.replace("\\", "/").strip("/")
    norm = posixpath.normpath(path.replace("\\", "/").lstrip("/"))
    prefix_parts = prefix.split("/")
    parts = norm.split("/")
    if parts[: len(prefix_parts)] == prefix_parts and len(parts) > len(prefix_parts):
        return
    msg = f"Writes go only under {prefix}/; '{path}' is outside it."
    if parts[0] not in ("", ".", "..") and (source_dir / prefix / parts[0]).is_dir():
        msg += f" Did you mean {prefix}/{norm}?"
    raise ValueError(msg)


__all__ = [
    "CHAR_SUBS",
    "apply_link_replacement",
    "build_position_map",
    "check_write_prefix",
    "compute_new_raw_target",
    "effective_attachment_extensions",
    "find_closest_match",
    "fts_row_to_note_info",
    "is_path_excluded",
    "normalize_text",
    "validate_path",
]
