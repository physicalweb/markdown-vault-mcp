"""Guards that refuse a malformed write at the act (AGO-853).

Two checks on what a caller sends, each refusing with the rule it applies
and the act that is permitted next (AGO-854):

- :func:`check_placeholder_body` refuses a note body that is empty, only
  headings, or only a placeholder word ("TODO", "lorem ipsum", "...").
  A short note with real content passes.
- :func:`check_added_escapes` refuses a write or edit that adds
  backslash-escaped quotation marks (``\\"``) outside code spans and
  fenced blocks, which is how JSON escaping leaks into markdown.  Escapes
  already in the file, and any inside code, are left alone.

The third malformed write (an edit anchor matching zero or several times)
is refused where the match is made, in
:meth:`~markdown_vault_mcp.managers.document.DocumentManager.edit`.
"""

from __future__ import annotations

import re

# A body that, once headings are set aside, is only one of these (case,
# surrounding punctuation and whitespace ignored) is a placeholder.  Kept
# small and explicit: a word that could be a real one-line note is not here.
_PLACEHOLDER_WORDS = frozenset(
    {
        "todo",
        "tbd",
        "tbc",
        "fixme",
        "placeholder",
        "wip",
        "coming soon",
        "to be written",
        "...",
        "…",
    }
)

_HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FRONTMATTER = re.compile(r"\A---[ \t]*\n.*?\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.S)
# A quotation mark preceded by an odd run of backslashes: ``\"``, ``\\\"``.
_ESCAPED_QUOTE = re.compile(r'(?<!\\)(?:\\\\)*\\"')
_BACKTICKS = re.compile(r"`+")


def placeholder_reason(body: str) -> str | None:
    """Say why *body* is a placeholder, or return ``None`` if it is not.

    Args:
        body: Markdown body as the caller sent it (frontmatter excluded).

    Returns:
        ``"is empty"``, ``"has only headings"``, ``"is only the placeholder
        '<text>'"``, or ``None`` when the body has content.
    """
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return "is empty"
    rest = [ln.strip() for ln in lines if not _HEADING.match(ln)]
    if not rest:
        return "has only headings"
    text = " ".join(rest)
    key = text.lower().strip(" \t.:;!-*_[]()<>").strip()
    if not key or text.strip() in ("...", "…") or key in _PLACEHOLDER_WORDS:
        return f"is only the placeholder {text!r}"
    if key.startswith("lorem ipsum"):
        return "is only placeholder text ('lorem ipsum')"
    return None


def check_placeholder_body(path: str, body: str) -> None:
    """Refuse a note whose body is empty or a placeholder.

    Args:
        path: Vault-relative path being written (for the message).
        body: Markdown body (frontmatter excluded).

    Raises:
        ValueError: If :func:`placeholder_reason` finds the body a
            placeholder.  The message names the rule and the next act.
    """
    reason = placeholder_reason(body)
    if reason is None:
        return
    raise ValueError(
        f"Refused: the body for {path!r} {reason}, so the write would leave "
        "a stub note (rule: a note is written with its content, AGO-853). "
        "Nothing was written. Next: write it again when you have the "
        "content, in one write that carries the body; a one-line note with "
        "real content is fine."
    )


def _strip_inline_code(line: str) -> str:
    """Remove inline code spans (matched backtick runs) from *line*."""
    out: list[str] = []
    pos = 0
    while True:
        m = _BACKTICKS.search(line, pos)
        if m is None:
            out.append(line[pos:])
            return "".join(out)
        run = m.group(0)
        close = re.compile(rf"(?<!`){re.escape(run)}(?!`)").search(line, m.end())
        if close is None:
            out.append(line[pos:])
            return "".join(out)
        out.append(line[pos : m.start()])
        pos = close.end()


def count_stray_escapes(text: str) -> int:
    """Count backslash-escaped quotation marks outside code in *text*.

    A leading YAML frontmatter block, fenced code blocks (`````` ``` `````` or
    ``~~~``) and inline code spans on one line are skipped.

    Args:
        text: Markdown file content or body.

    Returns:
        The number of ``\\"`` sequences in prose.
    """
    text = _FRONTMATTER.sub("", text, count=1)
    count = 0
    fence: str | None = None
    for line in text.splitlines():
        m = _FENCE_OPEN.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
                continue
            count += len(_ESCAPED_QUOTE.findall(_strip_inline_code(line)))
        elif m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
            if not line.strip()[len(m.group(1)) :].strip():
                fence = None
    return count


def check_added_escapes(path: str, before: str, after: str) -> None:
    """Refuse a write that adds backslash-escaped quotes outside code.

    Args:
        path: Vault-relative path being written (for the message).
        before: File content before the write (``""`` for a new file).
        after: File content the write would leave.

    Raises:
        ValueError: If *after* holds more ``\\"`` in prose than *before*.
    """
    added = count_stray_escapes(after) - count_stray_escapes(before)
    if added <= 0:
        return
    marks = "mark" if added == 1 else "marks"
    raise ValueError(
        f"Refused: the text for {path!r} adds {added} backslash-escaped "
        f'quotation {marks} (\\") outside code, which is how JSON escaping '
        "leaks into markdown (rule: prose carries plain quotes, AGO-853). "
        'Nothing was written. Next: send the same text with plain " quotes. '
        "If a backslash really belongs before a quote, put it in a `code "
        "span` or a fenced block, or pass allow_escapes=true."
    )
