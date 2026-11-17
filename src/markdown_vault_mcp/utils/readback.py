"""Edit read-back: the unified diff a write or edit returns (Agora Plan 234 step 1).

A resident who wants to check an edit today has to ``read`` the note again.
``edit`` and ``write`` now return what changed, computed from the text on disk
before and after the act, so a normalized match shows what it really replaced.

The format is git's unified diff (``--- a/<path>`` / ``+++ b/<path>``, hunks
with 3 lines of context, ``\\ No newline at end of file``).  The algorithm is
our own and deliberately small, because the Agora daemon's ``memory_update``
returns the same shape and its tests pin the same bytes on the same inputs
(``src/tools/edit-readback.ts`` in the agora repo):

1. Lines split on ``\\n`` only; a last line without one is marked.
2. The common leading and trailing lines are trimmed.
3. The middle is aligned by a longest-common-subsequence table when it has at
   most :data:`LCS_MAX_CELLS` cells, ties going to the removal; otherwise every
   middle line is shown removed, then every new one added.
4. Within each run of changes, removals come before additions.

A diff longer than :data:`MAX_DIFF_LINES` lines or :data:`MAX_DIFF_BYTES`
bytes is cut at a whole line, whichever bound comes first, and the result
says so.
"""

from __future__ import annotations

from typing import Any

#: Lines of unchanged context around each change.
CONTEXT = 3
#: A diff longer than this many lines is cut (adapter policy, Plan 234 step 1).
MAX_DIFF_LINES = 120
#: ...or than this many UTF-8 bytes (adapter policy, Plan 234 step 1).  A
#: one-word edit on docs/research/** returns about 1 KB and a 30% rewrite
#: about the note's size (median 10 KB, max 34 KB): 16 KB carries the typical
#: rewrite whole and keeps one result far under ``read``'s 256 KB cap (a
#: one-character edit to a 400 KB one-line note returned an 800 KB diff).
MAX_DIFF_BYTES = 16_384
#: Above this many cells the middle is not aligned (one removed/added block).
LCS_MAX_CELLS = 250_000

_Line = tuple[str, bool]  # (text without its newline, ends with a newline)


def _split(text: str) -> list[_Line]:
    if text == "":
        return []
    parts = text.split("\n")
    if parts[-1] == "":
        return [(p, True) for p in parts[:-1]]
    return [(p, True) for p in parts[:-1]] + [(parts[-1], False)]


def _align(a: list[_Line], b: list[_Line]) -> list[str]:
    """Edit ops for the trimmed middle: ``=``, ``-`` or ``+`` per line."""
    n, m = len(a), len(b)
    if n * m > LCS_MAX_CELLS:
        return ["-"] * n + ["+"] * m
    # table[i][j]: the LCS length of a[i:] and b[j:].
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, below = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = below[j + 1] + 1 if a[i] == b[j] else max(below[j], row[j + 1])
    ops: list[str] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            ops.append("=")
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            ops.append("-")
            i += 1
        else:
            ops.append("+")
            j += 1
    return ops + ["-"] * (n - i) + ["+"] * (m - j)


def _removals_first(ops: list[str]) -> list[str]:
    out: list[str] = []
    run: list[str] = []
    for op in [*ops, "="]:
        if op == "=":
            out += sorted(run, key=lambda o: o != "-")
            run = []
            out.append(op)
        else:
            run.append(op)
    return out[:-1]


def _range(start: int, length: int) -> str:
    # GNU diff / git: "N" for one line, "N,0" naming the line before for none.
    if length == 1:
        return str(start + 1)
    if length == 0:
        return f"{start},0"
    return f"{start + 1},{length}"


def unified_diff(path: str, before: str, after: str) -> str:
    """Return the unified diff from *before* to *after*, or ``""`` if equal.

    Args:
        path: The note's vault-relative path, named in both headers.
        before: The text on disk before the act.
        after: The text on disk after it.

    Returns:
        The diff, every line ending in a newline; ``""`` when nothing changed.
    """
    a, b = _split(before), _split(after)
    if a == b:
        return ""
    p = 0
    while p < len(a) and p < len(b) and a[p] == b[p]:
        p += 1
    s = 0
    while s < len(a) - p and s < len(b) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    middle = _removals_first(_align(a[p : len(a) - s], b[p : len(b) - s]))
    ops = ["="] * p + middle + ["="] * s

    # Each op with the index it reads in a and in b.
    rows: list[tuple[str, int, int]] = []
    i = j = 0
    for op in ops:
        rows.append((op, i, j))
        if op != "+":
            i += 1
        if op != "-":
            j += 1

    changes = [k for k, (op, _, _) in enumerate(rows) if op != "="]
    groups: list[list[int]] = [[changes[0], changes[0]]]
    for k in changes[1:]:
        if k - groups[-1][1] - 1 <= 2 * CONTEXT:
            groups[-1][1] = k
        else:
            groups.append([k, k])

    out = [f"--- a/{path}", f"+++ b/{path}"]
    for first, last in groups:
        lo = max(0, first - CONTEXT)
        hi = min(len(rows), last + CONTEXT + 1)
        span = rows[lo:hi]
        a_len = sum(1 for op, _, _ in span if op != "+")
        b_len = sum(1 for op, _, _ in span if op != "-")
        a_start, b_start = rows[lo][1], rows[lo][2]
        out.append(f"@@ -{_range(a_start, a_len)} +{_range(b_start, b_len)} @@")
        for op, ai, bj in span:
            text, newline = a[ai] if op != "+" else b[bj]
            out.append((" " if op == "=" else op) + text)
            if not newline:
                out.append("\\ No newline at end of file")
    return "\n".join(out) + "\n"


def changed_lines(before: str, after: str) -> dict[str, int] | None:
    """The range of the new text that covers every change (1-based).

    It runs from the first changed line to the last, so it may span unchanged
    lines between two changes.  For a change that only removes lines, both are
    the line after the cut, or the line before it when the cut is at the end.
    ``None`` when nothing changed or the new text is empty.
    """
    a, b = _split(before), _split(after)
    if a == b or not b:
        return None
    p = 0
    while p < len(a) and p < len(b) and a[p] == b[p]:
        p += 1
    s = 0
    while s < len(a) - p and s < len(b) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    last = len(b) - s
    if last <= p:  # nothing added: the line after the cut
        line = min(p + 1, len(b))
        return {"first": line, "last": line}
    return {"first": p + 1, "last": last}


def readback(path: str, before: str | None, after: str) -> dict[str, Any]:
    """The read-back fields for one act: ``diff``, ``lines``, ``diff_truncated``.

    Args:
        path: The note's vault-relative path.
        before: The text on disk before the act; ``None`` for a new note,
            which returns its line range (``lines``, from 1) and no diff.
        after: The text on disk after it.

    Returns:
        A dict holding ``lines`` and, over an existing note, ``diff`` (``""``
        when nothing changed) and, for a cut diff, ``diff_truncated``
        (``{"shown": n, "total": n, "reason": "lines" | "bytes"}``, in diff
        lines; *reason* names the bound that cut it).
    """
    if before is None:
        count = len(_split(after))
        return {"lines": {"first": 1, "last": count} if count else None}
    diff = unified_diff(path, before, after)
    out: dict[str, Any] = {"diff": diff, "lines": changed_lines(before, after)}
    rows = diff.split("\n")[:-1]
    shown, size = 0, 0
    for row in rows[:MAX_DIFF_LINES]:
        size += len(row.encode("utf-8")) + 1
        if size > MAX_DIFF_BYTES:
            break
        shown += 1
    if shown < len(rows):
        out["diff"] = "".join(row + "\n" for row in rows[:shown])
        out["diff_truncated"] = {
            "shown": shown,
            "total": len(rows),
            "reason": "lines" if shown == MAX_DIFF_LINES else "bytes",
        }
    return out
