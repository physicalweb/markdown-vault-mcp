"""The section and quote rules behind ``note_section`` (Agora Plan 235 step 1).

``note_section`` is a daemon read: the Agora daemon calls it at a resident's
pulse to quote the opening of her own note, dated by git.  This module holds
its pure half, over the file's lines as they are on disk (never the FTS
index, whose section boundaries are the chunker's and which can lag the
file):

- **The section.**  ATX headings outside fenced code.  The heading named by
  its text exactly, else by its opening words (AGO-849's ``_opens_with``,
  after ``_strip_atx_markers``, so a heading opens the same way here as in
  ``read(section=)``).  Several → ``ambiguous``, none → ``no_section``.  It
  runs to the next heading of the same or a higher level (fewer or equal
  ``#``), or the file's end; a deeper heading stays inside it.
- **The quote.**  Whole paragraphs of the body (the heading line excluded),
  in order, while they fit in ``quote_bytes`` UTF-8 bytes.  A first
  paragraph longer than that is cut at its last sentence end inside the
  bound, else its last word, and marked with ``…`` inside the bound.
  ``more_bytes`` is the rest of the body, with leading and trailing
  whitespace trimmed.

The rules match the bench's form of the tool in the agora repo
(``src/tools/bench-mocks/note-section-mock.ts``), whose tests are this
module's reference.  The git half (blame over the quoted lines, the writes
count) is :meth:`markdown_vault_mcp.git.GitWriteStrategy.blame_lines` and
:meth:`~markdown_vault_mcp.git.GitWriteStrategy.count_author_commits`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from markdown_vault_mcp.managers.document import _opens_with, _strip_atx_markers

MARK = "…"

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_ATX_TRAILING = re.compile(r"(^|[ \t]+)#+$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# A sentence end: . ! ? or an ellipsis, then any closing quotes (straight,
# curly, a guillemet) or brackets, followed by whitespace (the mock's rule).
_SENTENCE_END = re.compile(r"[.!?…][\"'”’»)\]]*(?=\s)")  # noqa: RUF001 — curly quotes are meant


@dataclass(frozen=True)
class Heading:
    """One ATX heading: its 0-based line, its level, its text without markers."""

    line: int
    level: int
    text: str


@dataclass(frozen=True)
class Section:
    """A found section: the heading as found, the body's first line, the end (exclusive)."""

    heading: str
    body_start: int
    end: int


@dataclass(frozen=True)
class SectionMiss:
    """Why no section: ``no_section`` or ``ambiguous`` (with the headings that open with the words)."""

    error: str
    headings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quote:
    """The quote, the bytes left in the section, and its body-relative line range (-1 when empty)."""

    quote: str
    more_bytes: int
    first: int
    last: int


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def headings_of(lines: list[str]) -> list[Heading]:
    """Return the ATX headings of *lines*, skipping fenced code.

    A fence opens on a run of three or more backticks or tildes and closes on
    a line that is only a run of the same character at least as long.
    """
    out: list[Heading] = []
    fence: tuple[str, int] | None = None
    for i, line in enumerate(lines):
        f = _FENCE.match(line)
        if f:
            run = f.group(1)
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and line.strip() == run:
                fence = None
            continue
        if fence is not None:
            continue
        m = _ATX.match(line)
        if m:
            text = _ATX_TRAILING.sub("", m.group(2) or "").strip()
            out.append(Heading(line=i, level=len(m.group(1)), text=text))
    return out


def find_section(lines: list[str], heading: str) -> Section | SectionMiss:
    """Return the section *heading* names in *lines*, or why there is none.

    An exact heading wins over one that only opens with the words (as
    ``read(section=)`` does).
    """
    query = _collapse(_strip_atx_markers(heading.strip()))
    if not query:
        return SectionMiss("no_section")
    all_headings = headings_of(lines)
    pick = [h for h in all_headings if _collapse(h.text) == query]
    if not pick:
        pick = [h for h in all_headings if _opens_with(h.text, query)]
    if not pick:
        return SectionMiss("no_section")
    if len(pick) > 1:
        return SectionMiss("ambiguous", tuple(h.text for h in pick))
    h = pick[0]
    end = next(
        (o.line for o in all_headings if o.line > h.line and o.level <= h.level),
        len(lines),
    )
    return Section(heading=h.text, body_start=h.line + 1, end=end)


def _prefix_within(text: str, max_bytes: int) -> str:
    """Return the longest prefix of *text* within *max_bytes* UTF-8 bytes, cut on a code point."""
    used = 0
    for i, ch in enumerate(text):
        used += _bytes(ch)
        if used > max_bytes:
            return text[:i]
    return text


def _paragraphs_of(lines: list[str]) -> list[tuple[int, int]]:
    """Return paragraphs as inclusive line ranges: runs of non-blank lines; a fence is never split."""
    out: list[tuple[int, int]] = []
    start = -1
    fence = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            fence = not fence
        blank = line.strip() == "" and not fence
        if not blank and start < 0:
            start = i
        if blank and start >= 0:
            out.append((start, i - 1))
            start = -1
    if start >= 0:
        out.append((start, len(lines) - 1))
    return out


def cut_quote(body: list[str], bound: int) -> Quote:
    """Return the quote from a section's *body* lines within *bound* UTF-8 bytes."""
    paras = _paragraphs_of(body)
    if not paras:
        return Quote(quote="", more_bytes=0, first=-1, last=-1)
    first = paras[0][0]
    taken = 0
    for _start, end in paras:
        if _bytes("\n".join(body[first : end + 1])) > bound:
            break
        taken += 1
    if taken > 0:
        last = paras[taken - 1][1]
        rest = "\n".join(body[last + 1 :]).strip()
        return Quote(
            quote="\n".join(body[first : last + 1]),
            more_bytes=_bytes(rest),
            first=first,
            last=last,
        )
    # A first paragraph over the bound: its last sentence end inside the
    # bound, else its last word, marked inside the bound.
    para = "\n".join(body[first : paras[0][1] + 1])
    room = _prefix_within(para, max(0, bound - _bytes(MARK)))
    # One character past the room, so an end that falls exactly at the bound
    # is seen as followed by a space.
    probe = para[: len(room) + 1]
    sentences = [m for m in _SENTENCE_END.finditer(probe) if m.end() <= len(room)]
    if sentences:
        head = room[: sentences[-1].end()]
    else:
        nxt = para[len(room) : len(room) + 1]
        if nxt and nxt.isspace():
            word_end = len(room)
        else:
            # \Z, not $: Python's $ also matches before a final newline.
            m = re.search(r"\s\S*\Z", room)
            word_end = m.start() if m else -1
        head = room[:word_end] if word_end > 0 else room
    quote = head.rstrip() + MARK
    rest = (para[len(head) :] + "\n" + "\n".join(body[paras[0][1] + 1 :])).strip()
    return Quote(
        quote=quote,
        more_bytes=_bytes(rest),
        first=first,
        last=first + head.count("\n"),
    )
