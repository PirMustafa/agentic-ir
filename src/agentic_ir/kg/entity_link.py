"""Surface form to entity: normalisation, aliases, and the alias trie.

This module is the only place that decides *what a node is called*, and both
the offline builder and the online navigator go through it. That shared path is
load-bearing: if the builder folded ``Arthur's Magazine`` one way and the
navigator folded the question's mention another way, the graph would be
perfectly correct and would never link to anything.

Design notes
------------
**Normalisation** follows ``docs/architecture.md`` 3.3 exactly -- NFKD, lower
case, strip diacritics, strip parenthetical disambiguators, collapse
whitespace. Two consequences are deliberate and worth stating rather than
hiding:

* ``Jimmy Butler (basketball)`` and ``Jimmy Butler (singer)`` collapse onto one
  node. The graph is a *mention* graph over page titles, not an ontology; the
  builder counts these merges and reports them so the loss is visible.
* Titles made entirely of punctuation (``!!!`` is a real HotpotQA page) keep
  their raw form as an id but produce no matchable tokens, so they simply never
  acquire edges. Deterministic, and better than inventing a token for them.

**Matching is token-level, not substring.** A substring scan would link the
page ``Ohio`` from the word ``Ohioan`` and would make edge counts depend on
punctuation. Tokens are folded individually so that character offsets into the
*raw* sentence survive folding -- the relation labeller in
:mod:`agentic_ir.kg.build` needs those offsets to quote the original sentence.

**No Aho-Corasick dependency.** ``pyahocorasick`` is not in
``requirements.txt`` and adding a compiled dependency for one offline script is
not worth it. A nested-dict token trie with greedy longest match costs a few
hundred nanoseconds per token position and runs the whole 66k-passage corpus in
minutes.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import lru_cache
from typing import NamedTuple

__all__ = [
    "AliasMatch",
    "AliasTable",
    "MIN_SINGLE_TOKEN_ALIAS_CHARS",
    "STOP_ALIASES",
    "Token",
    "alias_variants",
    "fold",
    "fold_token",
    "normalise_entity",
    "strip_parentheticals",
    "tokenize",
]

# A parenthetical disambiguator, e.g. "(band)", "(1997 film)". Applied
# repeatedly because titles like "Foo (bar) (baz)" occur.
_PAREN_RE = re.compile(r"\s*\([^()]*\)")

# One raw token, apostrophes included so that "Arthur's" stays a single unit
# and folds to "arthurs" rather than "arthur" + "s".
_RAW_TOKEN_RE = re.compile(r"[^\W_]+(?:['’ʼ`´][^\W_]+)*", re.UNICODE)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_APOSTROPHES = dict.fromkeys(map(ord, "'’ʼ`´"), None)

# Latin letters NFKD leaves alone because they are atomic code points rather
# than base + combining mark. Without these, a stroked-L spelling and its plain
# transcription would be two different entities.
_TRANSLIT = str.maketrans(
    {
        "ł": "l", "Ł": "l",      # l with stroke
        "ø": "o", "Ø": "o",      # o with stroke
        "đ": "d", "Đ": "d",      # d with stroke
        "ß": "ss",                    # sharp s
        "æ": "ae", "Æ": "ae",
        "œ": "oe", "Œ": "oe",
        "þ": "th", "Þ": "th",
        "ð": "d", "Ð": "d",
        "ı": "i",                     # dotless i
    }
)

#: Single-token aliases that are too common to be informative. These are all
#: real page titles in a Wikipedia-derived corpus, and admitting them would
#: attach a hub edge to essentially every passage while carrying no signal.
#: Kept deliberately small -- "Time", "Mercury" and friends are *not* here,
#: because the out-degree weight (architecture 3.3) is the principled way to
#: discount hubs, and hard-coding a topical stoplist would not be.
STOP_ALIASES = frozenset(
    """
    a an the and or but if then than that this these those there here
    of in on at to for from by with as is was were are be been being am
    it its he she they them his her their we you who whom whose
    what when where why how all any both each more most other some such
    no nor not only own same so too very can will just should now
    """.split()  # noqa: SIM905 -- a curated word list is reviewable as prose
)

#: Single-token aliases shorter than this are dropped. Two-letter pages match
#: far more noise than signal.
MIN_SINGLE_TOKEN_ALIAS_CHARS = 3


class Token(NamedTuple):
    """One token of a raw string, with its folded form and character span.

    ``start``/``end`` index the *raw* text, which is what lets the relation
    labeller quote the original sentence rather than a folded one.
    """

    norm: str
    raw: str
    start: int
    end: int


class AliasMatch(NamedTuple):
    """A longest-match alias occurrence inside a scanned string."""

    alias: str
    entity_ids: tuple[str, ...]
    start: int          # character offset into the raw text
    end: int
    token_start: int    # index into the token list returned by tokenize()
    token_end: int      # exclusive


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1 << 20)
def fold_token(raw: str) -> str:
    """Fold one raw token to its matching key.

    NFKD, drop combining marks, transliterate the atomic Latin letters NFKD
    cannot decompose, delete apostrophes, lower case, and concatenate whatever
    word characters remain. Returns ``""`` for a token with no word content.

    Cached because the corpus contains millions of token occurrences drawn from
    a much smaller vocabulary; the cache turns the dominant cost of the build
    into a dictionary lookup.
    """
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.translate(_TRANSLIT).translate(_APOSTROPHES).lower()
    return "".join(_WORD_RE.findall(stripped))


def tokenize(text: str) -> tuple[Token, ...]:
    """Tokenise ``text``, keeping raw character offsets alongside folded forms.

    Tokens whose folded form is empty (pure punctuation, non-Latin scripts) are
    dropped: they can never participate in a match, and keeping them would put
    holes in the token index space the relation labeller walks.
    """
    out: list[Token] = []
    for m in _RAW_TOKEN_RE.finditer(text):
        raw = m.group(0)
        norm = fold_token(raw)
        if norm:
            out.append(Token(norm, raw, m.start(), m.end()))
    return tuple(out)


def fold(text: str) -> str:
    """Space-joined folded tokens of ``text``. The canonical matching key."""
    return " ".join(t.norm for t in tokenize(text))


def strip_parentheticals(title: str) -> str:
    """Remove ``(disambiguator)`` groups, innermost first, until stable."""
    prev = title
    while True:
        nxt = _PAREN_RE.sub("", prev)
        if nxt == prev:
            return nxt.strip()
        prev = nxt


def normalise_entity(title: str) -> str:
    """The canonical ``entity_id`` of a page title.

    Falls back to the raw title (whitespace-collapsed, lower-cased) when
    folding yields nothing, so that a page such as ``!!!`` still gets a stable,
    unique id instead of colliding with every other punctuation-only title on
    the empty string.
    """
    bare = strip_parentheticals(title)
    # Wikipedia sometimes stores personal names surname-first (``DiCaprio,
    # Leonardo``), while prose overwhelmingly uses the natural order. Keep
    # their canonical ids aligned as well as their aliases. Restrict this to
    # short name-like halves so titles such as ``John II, Prince of ...`` are
    # not rearranged into invented noun phrases.
    if bare.count(",") == 1:
        head, _, tail = bare.partition(",")
        if head and tail and len(head.split()) <= 2 and len(tail.split()) <= 2:
            bare = f"{tail.strip()} {head.strip()}"
    folded = fold(bare)
    if folded:
        return folded
    folded = fold(title)
    if folded:
        return folded
    return " ".join(title.split()).lower()


def alias_variants(title: str) -> tuple[str, ...]:
    """Surface forms an occurrence of ``title`` may take in running text.

    Per architecture 3.3: the surface title, the parenthetical-free variant,
    and the comma-inverted variant (``"Lyon, John" -> "John Lyon"``). The
    inversion applies only to a single-comma title; multi-comma titles are
    almost always genuine noun phrases ("John II, Prince of Anhalt-Zerbst")
    where inverting invents a name that never occurs in text.
    """
    variants: list[str] = []

    def _push(value: str) -> None:
        value = " ".join(value.split()).strip()
        if value and value not in variants:
            variants.append(value)

    _push(title)
    bare = strip_parentheticals(title)
    _push(bare)
    for candidate in (title, bare):
        if candidate.count(",") == 1:
            head, _, tail = candidate.partition(",")
            head, tail = head.strip(), tail.strip()
            if head and tail:
                _push(f"{tail} {head}")
    return tuple(variants)


# ---------------------------------------------------------------------------
# Alias trie
# ---------------------------------------------------------------------------

_TERMINAL = None  # sentinel trie key; never collides with a str token


class AliasTable:
    """A token trie supporting greedy longest-match alias lookup.

    ``entity_linker: alias_match`` in the shipped config means this class *is*
    the entity linker: zero LLM calls, and a lookup cost independent of corpus
    size. Ambiguity is preserved rather than resolved -- an alias maps to every
    entity that claims it, sorted -- and the caller decides. In practice
    ambiguity is rare here, because parenthetical stripping has already merged
    the disambiguated pages onto a single node.
    """

    __slots__ = ("_alias_ids", "_max_len", "_root")

    def __init__(self) -> None:
        self._root: dict = {}
        self._alias_ids: dict[str, tuple[str, ...]] = {}
        self._max_len = 0

    # -- construction ------------------------------------------------------

    def add(self, surface: str, entity_id: str) -> bool:
        """Register ``surface`` as an alias of ``entity_id``.

        Returns ``False`` when the alias is rejected (empty after folding, or a
        single stop-word / too-short / all-digit token), so that the builder
        can count and report rejections instead of losing them silently.
        """
        tokens = tuple(t.norm for t in tokenize(surface))
        if not tokens:
            return False
        if len(tokens) == 1:
            tok = tokens[0]
            if tok in STOP_ALIASES or len(tok) < MIN_SINGLE_TOKEN_ALIAS_CHARS or tok.isdigit():
                return False
        key = " ".join(tokens)
        current = self._alias_ids.get(key, ())
        if entity_id not in current:
            self._alias_ids[key] = tuple(sorted((*current, entity_id)))
        node = self._root
        for tok in tokens:
            node = node.setdefault(tok, {})
        node[_TERMINAL] = key
        self._max_len = max(self._max_len, len(tokens))
        return True

    @classmethod
    def build(cls, aliases: Iterable[tuple[str, str]]) -> AliasTable:
        """Build from ``(surface, entity_id)`` pairs."""
        table = cls()
        for surface, entity_id in aliases:
            table.add(surface, entity_id)
        return table

    @classmethod
    def from_entities(cls, entities: Iterable[Mapping[str, object]]) -> AliasTable:
        """Build from node records carrying ``entity_id`` and ``aliases``."""
        table = cls()
        for node in entities:
            entity_id = str(node["entity_id"])
            surfaces = node.get("aliases") or ()
            for surface in surfaces:  # type: ignore[union-attr]
                table.add(str(surface), entity_id)
        return table

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._alias_ids)

    def __contains__(self, surface: object) -> bool:
        return isinstance(surface, str) and fold(surface) in self._alias_ids

    @property
    def max_alias_tokens(self) -> int:
        return self._max_len

    def aliases(self) -> tuple[str, ...]:
        """Every registered alias key, sorted. Deterministic by construction."""
        return tuple(sorted(self._alias_ids))

    def lookup(self, surface: str) -> tuple[str, ...]:
        """Entity ids claiming exactly this surface form (after folding)."""
        return self._alias_ids.get(fold(surface), ())

    # -- scanning ----------------------------------------------------------

    def match_tokens(self, tokens: Sequence[Token]) -> tuple[AliasMatch, ...]:
        """Greedy longest, non-overlapping matches over pre-tokenised text.

        Left to right; at each position the longest alias wins and the scan
        resumes after it. Longest-match is what keeps ``University of Chicago``
        from being linked as ``Chicago``.
        """
        out: list[AliasMatch] = []
        i = 0
        n = len(tokens)
        while i < n:
            node: dict | None = self._root
            best_key: str | None = None
            best_end = i
            j = i
            while j < n:
                node = node.get(tokens[j].norm)  # type: ignore[union-attr]
                if node is None:
                    break
                j += 1
                terminal = node.get(_TERMINAL)
                if terminal is not None:
                    best_key = terminal
                    best_end = j
            if best_key is None:
                i += 1
                continue
            out.append(
                AliasMatch(
                    alias=best_key,
                    entity_ids=self._alias_ids[best_key],
                    start=tokens[i].start,
                    end=tokens[best_end - 1].end,
                    token_start=i,
                    token_end=best_end,
                )
            )
            i = best_end
        return tuple(out)

    def find_all(self, text: str) -> tuple[AliasMatch, ...]:
        """Greedy longest, non-overlapping matches inside ``text``."""
        return self.match_tokens(tokenize(text))

    def iter_entity_ids(self, text: str) -> Iterator[str]:
        """Every entity id mentioned in ``text``, in first-occurrence order."""
        seen: set[str] = set()
        for match in self.find_all(text):
            for entity_id in match.entity_ids:
                if entity_id not in seen:
                    seen.add(entity_id)
                    yield entity_id
