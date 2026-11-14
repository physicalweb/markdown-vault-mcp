"""Tests for ``note_section``, the daemon read behind Agora's own-note line (Plan 235 step 1).

The cases are those Plan 235 step 1 lists, and those of the bench's form of the tool
in the agora repo (``tests/unit/bench-note-section-mock.test.ts``), which is this
tool's reference.  The fixtures are synthetic: shaped to the measures Plan 235's
step 0 took of Ki's circle map (a ``###`` "Current state (run 82, …)" heading after a
"Superseded current-state blocks" heading, its section ending at the next ``##``,
an 864-byte opening paragraph that a 259-byte second one would push past 1,024),
with no resident's text copied.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from markdown_vault_mcp._author_middleware import (
    email_for_participant,
    participant_emails,
    participant_for_email,
)
from markdown_vault_mcp.collection import Collection
from markdown_vault_mcp.git import GitWriteStrategy
from markdown_vault_mcp.note_section import (
    MARK,
    Section,
    SectionMiss,
    cut_quote,
    find_section,
    headings_of,
)
from markdown_vault_mcp.server import make_server

if TYPE_CHECKING:
    from pathlib import Path

KI = ("Ki", "ki@personas.agora.local")
LIN = ("Lín", "lin@personas.agora.local")
KIKI = ("Kiki", "kiki@personas.agora.local")
ARNON = ("Arnon", "arnon@example.com")


def _para(n_bytes: int, word: str = "word") -> str:
    """A paragraph of exactly *n_bytes* ASCII bytes ending in a full stop."""
    text = ""
    while len(text) + len(word) + 1 < n_bytes - 1:
        text += word + " "
    text = text.rstrip()
    return text + "x" * (n_bytes - 1 - len(text)) + "."


OPENING = _para(864, "now")
SECOND = _para(259, "next")

# Shaped to Ki's map at d026f1f46 (step 0): line numbers are 0-based here.
MAP = [
    "# The circle map",  # 0
    "",
    "## 0. The question so far",  # 2
    "",
    "### Superseded current-state blocks, kept whole below",  # 4
    "",
    "### The current-state block as it stood on 10-27",  # 6
    "",
    "## Now",  # 8
    "",
    "### Current state (run 82, 2026-11-01). Read this first",  # 10
    "",
    OPENING,  # 12
    "",
    SECOND,  # 14
    "",
    "#### A deeper heading stays inside",  # 16
    "",
    "```",
    "## not a heading inside a fence",
    "```",
    "",
    "Tail text.",  # 22
    "",
    "## Superseded current-state blocks, kept whole",  # 24
    "old",
]


# ---------------------------------------------------------------------------
# The section rule
# ---------------------------------------------------------------------------


class TestSection:
    def test_opening_words_find_the_current_state_heading_and_its_end(self) -> None:
        s = find_section(MAP, "Current state")
        assert s == Section(
            heading="Current state (run 82, 2026-11-01). Read this first",
            body_start=11,
            end=24,
        )
        # The heading as written in the file names it too.
        assert find_section(MAP, "### Current state") == s

    def test_a_same_level_heading_ends_it_and_a_deeper_one_does_not(self) -> None:
        lines = ["### A", "x", "#### inner", "y", "### B", "z"]
        s = find_section(lines, "A")
        assert isinstance(s, Section)
        assert (s.body_start, s.end) == (1, 4)
        # A higher level ends it as well.
        s2 = find_section(["### A", "x", "## B"], "A")
        assert isinstance(s2, Section)
        assert s2.end == 2

    def test_a_heading_inside_a_fence_is_not_a_heading(self) -> None:
        lines = [
            "## A",
            "```python",
            "## not one",
            "```",
            "x",
            "~~~~",
            "# nor",
            "~~~",
            "~~~~",
            "## B",
        ]
        assert [h.text for h in headings_of(lines)] == ["A", "B"]
        s = find_section(lines, "A")
        assert isinstance(s, Section)
        assert s.end == 9

    def test_misses_are_named(self) -> None:
        assert find_section(MAP, "Current status") == SectionMiss("no_section")
        assert find_section(["## Currently"], "Current") == SectionMiss("no_section")
        assert find_section(MAP, "   ") == SectionMiss("no_section")
        assert find_section(
            ["## Current state one", "## Current state two"], "Current state"
        ) == SectionMiss("ambiguous", ("Current state one", "Current state two"))

    def test_an_exact_heading_wins_over_one_that_only_opens_with_it(self) -> None:
        s = find_section(
            ["## Current state (old)", "## Current state", "x"], "Current state"
        )
        assert isinstance(s, Section)
        assert (s.heading, s.body_start) == ("Current state", 2)

    def test_closing_hashes_are_not_part_of_the_heading(self) -> None:
        assert [h.text for h in headings_of(["## Title ##", "#", "#nospace"])] == [
            "Title",
            "",
        ]


# ---------------------------------------------------------------------------
# The quote
# ---------------------------------------------------------------------------


class TestQuote:
    def test_kis_opening_paragraph_whole_when_the_next_would_pass_1024(self) -> None:
        s = find_section(MAP, "Current state")
        assert isinstance(s, Section)
        q = cut_quote(MAP[s.body_start : s.end], 1024)
        assert q.quote == OPENING
        assert len(q.quote.encode()) == 864
        assert (q.first, q.last) == (1, 1)
        rest = "\n".join(MAP[s.body_start + 2 : s.end]).strip()
        assert q.more_bytes == len(rest.encode())

    def test_whole_paragraphs_while_they_fit(self) -> None:
        p = _para(100)
        body = [p, "", p, "", p]
        q = cut_quote(body, 202)
        assert q.quote == f"{p}\n\n{p}"
        assert (q.first, q.last, q.more_bytes) == (0, 2, 100)
        whole = cut_quote(body, 10_000)
        assert (whole.last, whole.more_bytes) == (4, 0)

    def test_a_first_paragraph_over_the_bound_is_cut_at_a_sentence_end(self) -> None:
        para = "One sentence here. Two sentence here. Three sentence goes on and on past the bound"
        q = cut_quote([para, "", "next"], 50)
        assert q.quote == "One sentence here. Two sentence here." + MARK
        assert len(q.quote.encode()) <= 50
        rest = "Three sentence goes on and on past the bound\nnext"
        assert q.more_bytes == len(rest.encode()) + 1
        assert (q.first, q.last) == (0, 0)

    def test_else_at_the_last_word_and_an_empty_section_quotes_nothing(self) -> None:
        q = cut_quote(["alpha beta gamma delta epsilon"], 17)
        assert q.quote == "alpha beta" + MARK
        assert cut_quote(["", ""], 100).quote == ""
        assert cut_quote(["", ""], 100).first == -1

    def test_a_cut_falls_on_a_code_point_and_the_mark_stays_inside(self) -> None:
        para = "é" * 23  # 46 bytes, no space: no sentence end, no word end
        q = cut_quote([para], 16)
        # 16 - 3 (the mark) leaves 13 bytes: six whole characters, never half of one.
        assert q.quote == "é" * 6 + MARK
        assert len(q.quote.encode()) == 15

    def test_a_cut_paragraph_spanning_lines_reports_its_last_quoted_line(self) -> None:
        body = [
            "Alpha beta gamma",
            "delta epsilon. Zeta eta theta iota kappa lambda mu nu xi",
        ]
        q = cut_quote(body, 40)
        assert q.quote == "Alpha beta gamma\ndelta epsilon." + MARK
        assert q.last == 1
        q2 = cut_quote(body, 20)
        assert q2.quote == "Alpha beta gamma" + MARK
        assert q2.last == 0


# ---------------------------------------------------------------------------
# The git half, through the Collection
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **(env or {})},
    ).stdout


def _commit(repo: Path, who: tuple[str, str], when: str, message: str) -> str:
    """Commit everything as *who* at author and committer time *when*; return the sha."""
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        f"user.name={who[0]}",
        "-c",
        f"user.email={who[1]}",
        "commit",
        "-q",
        "-m",
        message,
        env={"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when},
    )
    return _git(repo, "rev-parse", "HEAD").strip()


def _doc(para1: str, para2: str = "Second.", tail: str = "Tail.") -> str:
    return (
        "\n".join(
            [
                "# Map",
                "",
                "## Current state",
                "",
                para1,
                "",
                para2,
                "",
                "## Later",
                tail,
            ]
        )
        + "\n"
    )


NOTE = "research/circle/map.md"


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    (v / "research" / "circle").mkdir(parents=True)
    (v / "research" / "other").mkdir(parents=True)
    _git(v, "init", "-q")
    return v


def _collection(v: Path) -> Collection:
    return Collection(source_dir=v, git_strategy=GitWriteStrategy())


def _note(col: Collection, quote_bytes: int = 1024, email: str | None = KI[1]) -> dict:
    return col.note_section(
        NOTE,
        "Current state",
        quote_bytes,
        author_email=email,
        program_path="research/circle/",
    )


class TestGit:
    def test_dates_the_quote_by_the_newest_quoted_line_and_counts_her_writes_since(
        self, vault: Path
    ) -> None:
        note = vault / NOTE
        note.write_text(_doc("Plan."))
        _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        note.write_text(_doc("Plan A."))
        _commit(vault, LIN, "2026-11-02T10:00:00Z", "c2")
        note.write_text(_doc("Plan A.", "Second, edited."))
        c3 = _commit(vault, KI, "2026-11-03T10:00:00Z", "c3")
        # Below the section only: does not move the date.
        note.write_text(_doc("Plan A.", "Second, edited.", "Tail, edited."))
        _commit(vault, ARNON, "2026-11-04T10:00:00Z", "c4")
        # Her writes since c3: one under the program; one elsewhere; one by Lín;
        # one by a persona whose email contains hers.
        (vault / "research/circle/ledger.md").write_text("a\n")
        _commit(vault, KI, "2026-11-05T10:00:00Z", "c5")
        (vault / "research/other/x.md").write_text("b\n")
        _commit(vault, KI, "2026-11-06T10:00:00Z", "c6")
        (vault / "research/circle/y.md").write_text("c\n")
        _commit(vault, LIN, "2026-11-07T10:00:00Z", "c7")
        (vault / "research/circle/z.md").write_text("d\n")
        _commit(vault, KIKI, "2026-11-08T10:00:00Z", "c8")

        r = _note(_collection(vault))
        assert r == {
            "heading": "Current state",
            "quote": "Plan A.\n\nSecond, edited.",
            "more_bytes": 0,
            "written": {"at": "2026-11-03T10:00:00Z", "author_email": KI[1], "sha": c3},
            "writes_since": {"count": 1, "path": "research/circle/"},
            "git": "ok",
        }

    def test_an_opening_older_than_lower_lines_reports_the_quoted_lines_date(
        self, vault: Path
    ) -> None:
        note = vault / NOTE
        note.write_text(_doc("Plan."))
        c1 = _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        note.write_text(_doc("Plan.", "Second, newer."))
        c2 = _commit(vault, LIN, "2026-11-02T10:00:00Z", "c2")
        col = _collection(vault)
        whole = _note(col)
        assert whole["written"]["sha"] == c2
        assert whole["written"]["author_email"] == LIN[1]
        cut = _note(col, quote_bytes=16)  # only "Plan." fits
        assert cut["quote"] == "Plan."
        assert cut["written"]["sha"] == c1
        assert cut["written"]["author_email"] == KI[1]

    def test_the_newer_of_two_authors_on_the_quoted_lines_wins_by_author_time(
        self, vault: Path
    ) -> None:
        note = vault / NOTE
        note.write_text(_doc("Plan."))
        _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        note.write_text(_doc("Plan, by Lín."))
        c2 = _commit(vault, LIN, "2026-11-05T10:00:00Z", "c2")
        note.write_text(_doc("Plan, by Lín.", "Second, by Ki."))
        _commit(vault, KI, "2026-11-03T10:00:00Z", "c3 (authored before c2)")
        r = _note(_collection(vault))
        assert r["written"]["sha"] == c2

    def test_a_shallow_clone_answers_no_git_and_still_quotes(
        self, vault: Path, tmp_path: Path
    ) -> None:
        note = vault / NOTE
        note.write_text(_doc("Plan."))
        _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        note.write_text(_doc("Plan.", "Second, edited."))
        _commit(vault, KI, "2026-11-02T10:00:00Z", "c2")
        shallow = tmp_path / "shallow"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", f"file://{vault}", str(shallow)],
            capture_output=True,
            check=True,
        )
        r = _note(_collection(shallow))
        assert r["quote"] == "Plan.\n\nSecond, edited."
        assert (r["written"], r["writes_since"], r["git"]) == (None, None, "no_git")

    def test_an_uncommitted_or_untracked_quoted_line_answers_no_git(
        self, vault: Path
    ) -> None:
        note = vault / NOTE
        note.write_text(_doc("Plan."))
        col = _collection(vault)
        _commit(vault, KI, "2026-11-01T10:00:00Z", "other")  # HEAD exists
        note.write_text(_doc("Plan, not committed."))
        assert _note(col)["git"] == "no_git"
        (vault / "research/circle/new.md").write_text(_doc("Untracked."))
        r = col.note_section(
            "research/circle/new.md",
            "Current state",
            1024,
            author_email=KI[1],
            program_path="",
        )
        assert (r["quote"], r["git"]) == ("Untracked.\n\nSecond.", "no_git")

    def test_no_git_strategy_and_an_empty_section_answer_no_git(
        self, vault: Path
    ) -> None:
        (vault / NOTE).write_text("## Current state\n\n## Later\n")
        _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        r = _note(_collection(vault))
        assert (r["quote"], r["git"], r["more_bytes"]) == ("", "no_git", 0)
        (vault / NOTE).write_text(_doc("Plan."))
        no_git = Collection(source_dir=vault, git_strategy=None)
        assert _note(no_git)["git"] == "no_git"

    def test_a_file_over_the_read_cap_is_still_read(self, vault: Path) -> None:
        big = _doc("Plan.") + ("filler line of the long ledger\n" * 10_000)
        assert len(big.encode()) > 262_144
        (vault / NOTE).write_text(big)
        _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        r = _note(_collection(vault))
        assert (r["quote"], r["git"]) == ("Plan.\n\nSecond.", "ok")

    def test_without_an_email_or_outside_the_vault_nothing_is_counted(
        self, vault: Path
    ) -> None:
        (vault / NOTE).write_text(_doc("Plan."))
        _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        col = _collection(vault)
        assert _note(col, email=None)["writes_since"] is None
        r = col.note_section(
            NOTE,
            "Current state",
            1024,
            author_email=KI[1],
            program_path="../elsewhere/",
        )
        assert (r["git"], r["writes_since"]) == ("ok", None)
        root = col.note_section(
            NOTE, "Current state", 1024, author_email=KI[1], program_path=""
        )
        assert root["writes_since"] == {"count": 0, "path": ""}

    def test_misses(self, vault: Path) -> None:
        (vault / NOTE).write_text(
            "# Map\n\n## Current state one\n\n## Current state two\n"
        )
        col = _collection(vault)
        assert _note(col) == {
            "error": "ambiguous",
            "headings": ["Current state one", "Current state two"],
        }
        assert col.note_section(
            NOTE, "Nothing", 1024, author_email=None, program_path=""
        ) == {"error": "no_section"}
        for path in ("research/missing.md", "../escape.md", "research/circle"):
            assert col.note_section(
                path, "Current state", 1024, author_email=None, program_path=""
            ) == {"error": "unreadable"}
        (vault / "research/bad.md").write_bytes(b"\xff\xfe## Current state\n")
        assert col.note_section(
            "research/bad.md", "Current state", 1024, author_email=None, program_path=""
        ) == {"error": "unreadable"}


class TestGitReadsAnswerNoneNeverRaise:
    def test_outside_a_repository(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        git = GitWriteStrategy()
        assert git.is_shallow(plain) is None
        assert git.blame_newest(plain, plain / "x.md", 1, 1) is None
        assert git.count_author_commits(plain, "abc", KI[1], plain) is None

    def test_a_bad_range_a_bad_sha_and_a_timeout(
        self, vault: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (vault / NOTE).write_text(_doc("Plan."))
        _commit(vault, KI, "2026-11-01T10:00:00Z", "c1")
        git = GitWriteStrategy()
        assert git.blame_newest(vault, vault / NOTE, 0, 1) is None
        assert git.blame_newest(vault, vault / NOTE, 3, 2) is None
        assert git.blame_newest(vault, vault / NOTE, 999, 999) is None
        assert git.count_author_commits(vault, "not-a-sha", KI[1], vault) is None

        def _slow(*_a: object, **_k: object) -> None:
            raise subprocess.TimeoutExpired(cmd="git", timeout=1.0)

        monkeypatch.setattr("markdown_vault_mcp.git.subprocess.run", _slow)
        assert git.is_shallow(vault) is None
        assert git.blame_newest(vault, vault / NOTE, 5, 5) is None


class TestQuoteWordEnd:
    def test_a_room_that_ends_just_before_a_space_keeps_its_last_word(self) -> None:
        assert cut_quote(["alpha beta gamma delta"], 13).quote == "alpha beta" + MARK


# ---------------------------------------------------------------------------
# The attribution mapping
# ---------------------------------------------------------------------------

MAPPING = """\
mappings:
  - auth0_subject: "google-oauth2|abc"
    git_name: "Arnon"
    git_email: "arnon@example.com"
  - participant_id: "ki"
    git_name: "Ki"
    git_email: "ki@personas.agora.local"
  - participant_id: "lin"
    git_name: "Lín"
    git_email: "lin@personas.agora.local"
  - participant_id: "lin:partner"
    git_name: "Lín"
    git_email: "lin@personas.agora.local"
"""


class TestMapping:
    def test_persona_ids_map_both_ways_as_the_middleware_maps_them(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "mapping.yaml"
        path.write_text(MAPPING)
        emails = participant_emails(path)
        assert emails == {
            "ki": KI[1],
            "lin": LIN[1],
            "lin:partner": LIN[1],
        }
        assert email_for_participant(emails, "ki") == KI[1]
        assert email_for_participant(emails, "ki:partner") == KI[1]
        assert email_for_participant(emails, "wren") is None
        assert participant_for_email(emails, LIN[1]) == "lin"  # the bare id first
        assert participant_for_email(emails, "KI@personas.agora.local") == "ki"
        assert (
            participant_for_email(emails, ARNON[1]) is None
        )  # a person, not a persona
        assert participant_emails(None) == {}


# ---------------------------------------------------------------------------
# The tool, on the wire
# ---------------------------------------------------------------------------


@pytest.fixture
def _tool_env(vault: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    note = vault / NOTE
    note.write_text(_doc("Plan."))
    _commit(vault, ARNON, "2026-11-01T10:00:00Z", "c1")
    note.write_text(_doc("Plan, by Ki."))
    sha = _commit(vault, KI, "2026-11-02T10:00:00Z", "c2")
    (vault / "research/circle/ledger.md").write_text("a\n")
    _commit(vault, KI, "2026-11-03T10:00:00Z", "c3")
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(MAPPING)
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_SOURCE_DIR", str(vault))
    monkeypatch.setenv("MARKDOWN_VAULT_MCP_GIT_AUTHOR_MAPPING", str(mapping))
    for var in (
        "MARKDOWN_VAULT_MCP_READ_ONLY",
        "MARKDOWN_VAULT_MCP_INDEX_PATH",
        "MARKDOWN_VAULT_MCP_EMBEDDINGS_PATH",
        "MARKDOWN_VAULT_MCP_STATE_PATH",
        "MARKDOWN_VAULT_MCP_EXCLUDE",
        "MARKDOWN_VAULT_MCP_WRITE_PREFIX",
        "MARKDOWN_VAULT_MCP_BEARER_TOKEN",
        "MARKDOWN_VAULT_MCP_AUTH_MODE",
    ):
        monkeypatch.delenv(var, raising=False)
    return sha


ARGS = {
    "path": NOTE,
    "heading": "Current state",
    "quote_bytes": 1024,
    "author": "ki:partner",
    "program_path": "research/circle/",
}


class TestTool:
    async def test_the_wire_contract_the_daemon_parses(self, _tool_env: str) -> None:
        server = make_server()
        async with Client(server) as client:
            result = await client.call_tool("note_section", ARGS)
            lin = await client.call_tool("note_section", {**ARGS, "author": "lin"})
        # The daemon joins the result's text blocks and parses one JSON object.
        body = json.loads("".join(b.text for b in result.content if hasattr(b, "text")))
        assert body == {
            "heading": "Current state",
            "quote": "Plan, by Ki.\n\nSecond.",
            "more_bytes": 0,
            "written": {"at": "2026-11-02T10:00:00Z", "author": "ki", "sha": _tool_env},
            "writes_since": {"count": 1, "path": "research/circle/"},
            "git": "ok",
        }
        assert result.structured_content == body
        # Lín's count is her own: 0 since Ki's commit.
        assert lin.structured_content is not None
        assert lin.structured_content["writes_since"] == {
            "count": 0,
            "path": "research/circle/",
        }

    async def test_an_author_no_persona_maps_to_reads_null(
        self, _tool_env: str, vault: Path
    ) -> None:
        (vault / NOTE).write_text(_doc("Plan, by Arnon."))
        _commit(vault, ARNON, "2026-11-04T10:00:00Z", "c4")
        server = make_server()
        async with Client(server) as client:
            r = await client.call_tool("note_section", {**ARGS, "author": "wren"})
        assert r.structured_content is not None
        assert r.structured_content["written"]["author"] is None
        assert r.structured_content["writes_since"] is None  # wren has no email
        assert r.structured_content["git"] == "ok"

    async def test_a_miss_and_a_bad_bound(self, _tool_env: str) -> None:
        server = make_server()
        async with Client(server) as client:
            miss = await client.call_tool("note_section", {**ARGS, "heading": "Nope"})
            assert miss.structured_content == {"error": "no_section"}
            with pytest.raises(ToolError, match="quote_bytes"):
                await client.call_tool("note_section", {**ARGS, "quote_bytes": 3})

    async def test_listed_read_only_and_said_to_be_a_daemon_read(
        self, _tool_env: str
    ) -> None:
        server = make_server()
        async with Client(server) as client:
            tools = {t.name: t for t in await client.list_tools()}
        tool = tools["note_section"]
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.description is not None
        assert tool.description.startswith("A daemon read for the Agora arrival line")
        # The injected collection and context are not parameters on the wire.
        assert sorted(tool.inputSchema["properties"]) == [
            "author",
            "heading",
            "path",
            "program_path",
            "quote_bytes",
        ]
