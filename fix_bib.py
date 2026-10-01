"""
Rule-based BibTeX cleaner:
  1. Removes unnecessary fields (abstract, keywords, file, urldate, etc.)
  2. Normalises venue names: well-known AI/ML/NLP venues become
     "Full Name (ACRONYM)"; other proceedings lose their year / edition number
  3. Capitalises content words in booktitle and protects capitals in title
  4. Normalises arXiv entries, upgrades them to their published version
     (Crossref), resolves OpenReview venues and fills in missing URLs

No LLM involved -- purely deterministic string matching.
"""

import argparse
import html
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── Rate limiting ────────────────────────────────────────────────────────


class RateLimiter:
    """Thread-safe rate limiter: at most `rate` requests per second."""

    def __init__(self, rate: float = 1.0):
        self._min_interval = 1.0 / rate
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()


# Crossref's "polite pool" allows ~50 req/s for clients that identify themselves
# via a User-Agent containing a mailto:; we stay well under that.
CROSSREF_RATE_LIMITER = RateLimiter(rate=5.0)
OPENREVIEW_RATE_LIMITER = RateLimiter(rate=1.0)
# Contact address for Crossref's polite pool; set via --mailto or this env var.
_CROSSREF_MAILTO: str | None = os.environ.get("CROSSREF_MAILTO") or None


def _crossref_headers() -> dict[str, str]:
    if _CROSSREF_MAILTO:
        return {"User-Agent": f"fix_bib (mailto:{_CROSSREF_MAILTO})"}
    return {"User-Agent": "fix_bib"}


# ── Progress bar ─────────────────────────────────────────────────────────


class ProgressBar:
    """
    Thread-safe progress bar on stderr for the API lookups. Draws nothing when
    stderr is not a terminal, so redirected logs stay clean.
    """

    WIDTH = 30

    def __init__(self, total: int, label: str):
        self.total = total
        self.label = label
        self.done = 0
        self.enabled = total > 0 and sys.stderr.isatty()
        self._lock = threading.Lock()
        self._width = 0  # length of the line currently drawn
        if self.enabled:
            self._draw()

    def _draw(self):
        filled = self.WIDTH * self.done // self.total
        bar = "#" * filled + "." * (self.WIDTH - filled)
        line = f"  {self.label} [{bar}] {self.done}/{self.total}"
        sys.stderr.write("\r" + line)
        sys.stderr.flush()
        self._width = len(line)

    def _erase(self):
        sys.stderr.write("\r" + " " * self._width + "\r")
        sys.stderr.flush()

    def advance(self):
        with self._lock:
            self.done += 1
            if self.enabled:
                self._draw()

    def message(self, text: str):
        """Print `text` on its own line without garbling the bar."""
        with self._lock:
            if self.enabled:
                self._erase()
            print(text, file=sys.stderr)
            if self.enabled:
                self._draw()

    def close(self):
        with self._lock:
            if self.enabled:
                self._erase()
                self.enabled = False


_PROGRESS: ProgressBar | None = None  # the bar currently on screen, if any


def _warn(text: str):
    """Print a warning to stderr, working around the progress bar if one is up."""
    bar = _PROGRESS
    if bar is not None:
        bar.message(text)
    else:
        print(text, file=sys.stderr)


# ── API result cache ─────────────────────────────────────────────────────


class APICache:
    """Persistent JSON cache for API results, namespaced by API endpoint."""

    NAMESPACES = ("crossref_search", "openreview")

    def __init__(self, path: Path | None, load: bool = True):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, dict] = {ns: {} for ns in self.NAMESPACES}
        if load and path and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                for ns in self.NAMESPACES:
                    if isinstance(loaded.get(ns), dict):
                        self.data[ns] = loaded[ns]
            except Exception as e:
                print(f"  cache: failed to load {path}: {e}", file=sys.stderr)

    def has(self, namespace: str, key: str) -> bool:
        return key in self.data[namespace]

    def get(self, namespace: str, key: str):
        return self.data[namespace].get(key)

    def set(self, namespace: str, key: str, value) -> None:
        with self._lock:
            self.data[namespace][key] = value

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            payload = json.dumps(self.data, indent=2)
        # Write to a temp file first so an interrupted save can't corrupt the cache
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)


_CACHE: APICache | None = None  # initialised in main()

# ── Fields to KEEP (everything else is dropped) ─────────────────────────
KEEP_FIELDS = {
    "title",
    "author",
    "year",
    "booktitle",
    "journal",
    "volume",
    "number",
    "pages",
    "publisher",
    "series",
    "eprint",
    "primaryclass",
    "archiveprefix",
    "url",
    "howpublished",
    "school",
    "institution",
    "isbn",
    "issn",
    "note",
    "chapter",
    "edition",
    "eprinttype",
}

# ── Conference acronym rules ─────────────────────────────────────────────
# Each rule: (pattern_to_match_in_booktitle, acronym)
# Patterns are matched case-insensitively against the plain text of the
# booktitle (with {{...}} braces stripped and "&" spelled out as "and" for
# matching purposes).
# Order matters: first match wins.
CONFERENCE_RULES = [
    # --- ACL ---
    (r"annual meeting of the association for computational linguistics", "ACL"),
    (r"annual meeting of the acl\b", "ACL"),
    # --- NAACL ---
    (
        r"north american chapter of the association for computational linguistics",
        "NAACL",
    ),
    (
        r"nations of the americas chapter of the association for computational linguistics",
        "NAACL",
    ),
    # --- EMNLP ---
    (r"conference on empirical methods in natural language processing", "EMNLP"),
    # --- EACL ---
    (r"european chapter of the association for computational linguistics", "EACL"),
    # --- AACL ---
    (r"asia-pacific chapter of the association for computational linguistics", "AACL"),
    # --- COLING (and the joint LREC-COLING 2024) ---
    (
        r"international conference on computational linguistics, language resources and evaluation",
        "LREC-COLING",
    ),
    (r"international conference on computational linguistics", "COLING"),
    # --- CoNLL ---
    (r"conference on computational natural language learning", "CoNLL"),
    # --- Findings of ACL/EMNLP/NAACL/EACL/AACL ---
    # \b keeps "EACL"/"NAACL"/"AACL" from being read as "ACL".
    (
        r"findings of the association for computational linguistics.*?\bemnlp\b",
        "Findings of EMNLP",
    ),
    (
        r"findings of the association for computational linguistics.*?\bnaacl\b",
        "Findings of NAACL",
    ),
    (
        r"findings of the association for computational linguistics.*?\beacl\b",
        "Findings of EACL",
    ),
    (
        r"findings of the association for computational linguistics.*?\baacl\b",
        "Findings of AACL",
    ),
    (
        r"findings of the association for computational linguistics.*?\bacl\b",
        "Findings of ACL",
    ),
    # --- NeurIPS ---
    (
        r"neural information processing systems.*datasets and benchmarks",
        "NeurIPS D\\&B",
    ),
    (r"\b(neurips|nips)\b.*datasets and benchmarks", "NeurIPS D\\&B"),
    (r"conference on neural information processing systems", "NeurIPS"),
    (r"advances in neural information processing systems", "NeurIPS"),
    (r"^neural information processing systems$", "NeurIPS"),
    # --- ICML ---
    (r"international conference on machine learning", "ICML"),
    # --- ICLR ---
    (r"international conference on learning representations", "ICLR"),
    # --- AAAI ---
    (r"aaai conference on artificial intelligence(?! and)", "AAAI"),
    # --- IJCAI ---
    (r"international joint conference on artificial intelligence", "IJCAI"),
    (r"\bijcai\b", "IJCAI"),
    # --- CVPR ---
    (r"conference on computer vision and pattern recognition", "CVPR"),
    # --- ICCV ---
    (r"international conference on computer vision(?! and)", "ICCV"),
    # --- ECCV ---
    (r"european conference on computer vision", "ECCV"),
    (r"computer vision\W+eccv\b", "ECCV"),
    # --- SIGIR ---
    (r"research and development in information retrieval", "SIGIR"),
    # --- KDD ---
    (r"\bsigkdd\b.*knowledge discovery", "KDD"),
    # --- CIKM ---
    (r"international conference on information and knowledge management", "CIKM"),
    # --- WWW / The Web Conference ---
    (r"\bweb conference\b", "WWW"),
    (r"international conference on world wide web", "WWW"),
    (r"world wide web conference", "WWW"),
    # --- LREC ---
    (r"conference on language resources and evaluation", "LREC"),
    (r"language resources and evaluation conference", "LREC"),
    # --- SIGdial ---
    (r"sigdial meeting on discourse and dialogue", "SIGdial"),
    (r"special interest group on discourse and dialogue", "SIGdial"),
    # --- AISTATS ---
    (r"international conference on artificial intelligence and statistics", "AISTATS"),
    # --- CLeaR ---
    (r"conference on causal learning and reasoning", "CLeaR"),
    # --- COLM ---
    (r"conference on language modeling", "COLM"),
    # --- TACL (journal) ---
    (r"transactions of the association for computational linguistics", "TACL"),
    # --- JMLR (journal) ---
    (r"journal of machine learning research", "JMLR"),
    (r"j\.\s*mach\.\s*learn\.\s*res\.", "JMLR"),
    # --- TMLR (journal) ---
    (r"transactions on machine learning research", "TMLR"),
    # --- BlackboxNLP workshop ---
    (r"blackboxnlp workshop", "BlackboxNLP"),
    # --- Semantic Computing ---
    (r"international conference on semantic computing", "ICSC"),
]

# Known terms that should be preserved in their canonical form (case-insensitive match)
CANONICAL_TERMS = {
    "t-sne": "t-SNE",
}

# Acronym → canonical full venue name. A booktitle / journal recognised as one
# of these venues (by CONFERENCE_RULES, an ACL Anthology ID, or because it is
# JUST the acronym, e.g. `{ICLR}` / `{ICLR 2024}`) is rewritten to
# "Full Name (ACRONYM)" -- no year, edition number or "Proceedings of the" --
# so every paper from the same venue gets an identical venue string.
# Acronyms in CONFERENCE_RULES but not listed here only get the acronym appended.
VENUE_NAMES = {
    "ACL": "Annual Meeting of the Association for Computational Linguistics",
    "NAACL": "Conference of the North American Chapter of the Association for Computational Linguistics",
    "EMNLP": "Conference on Empirical Methods in Natural Language Processing",
    "EACL": "Conference of the European Chapter of the Association for Computational Linguistics",
    "AACL": "Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    "COLING": "International Conference on Computational Linguistics",
    "LREC-COLING": "Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
    "CoNLL": "Conference on Computational Natural Language Learning",
    "Findings of ACL": "Findings of the Association for Computational Linguistics: ACL",
    "Findings of EMNLP": "Findings of the Association for Computational Linguistics: EMNLP",
    "Findings of NAACL": "Findings of the Association for Computational Linguistics: NAACL",
    "Findings of EACL": "Findings of the Association for Computational Linguistics: EACL",
    "Findings of AACL": "Findings of the Association for Computational Linguistics: AACL",
    "NeurIPS": "Advances in Neural Information Processing Systems",
    "NeurIPS D\\&B": "Neural Information Processing Systems Track on Datasets and Benchmarks",
    "ICML": "International Conference on Machine Learning",
    "ICLR": "International Conference on Learning Representations",
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "IJCAI": "International Joint Conference on Artificial Intelligence",
    "CVPR": "Conference on Computer Vision and Pattern Recognition",
    "ICCV": "International Conference on Computer Vision",
    "ECCV": "European Conference on Computer Vision",
    "SIGIR": "ACM SIGIR Conference on Research and Development in Information Retrieval",
    "KDD": "ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
    "CIKM": "ACM International Conference on Information and Knowledge Management",
    "WWW": "The Web Conference",
    "LREC": "International Conference on Language Resources and Evaluation",
    "SIGdial": "Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    "AISTATS": "International Conference on Artificial Intelligence and Statistics",
    "CLeaR": "Conference on Causal Learning and Reasoning",
    "COLM": "Conference on Language Modeling",
    "ICSC": "International Conference on Semantic Computing",
    "TACL": "Transactions of the Association for Computational Linguistics",
    "JMLR": "Journal of Machine Learning Research",
    "TMLR": "Transactions on Machine Learning Research",
}

# Old or alternative acronyms, mapped to the acronym used in VENUE_NAMES.
VENUE_ALIASES = {
    "NIPS": "NeurIPS",
}

# Venues published as journals (→ @article / journal when resolved from OpenReview).
JOURNAL_ACRONYMS = {"TACL", "JMLR", "TMLR"}

# Non-main tracks of a known conference. The track is kept in the canonical
# name as "Full Name: Track (ACRONYM)".
VENUE_TRACKS = [
    (r"\bdemonstrations?\b|\bdemos?\b", "System Demonstrations"),
    (r"industry track", "Industry Track"),
    (r"student research workshop", "Student Research Workshop"),
    (r"workshop track", "Workshop Track"),
    (r"tutorial", "Tutorial Abstracts"),
]

# Events that merely take place at / alongside a known conference. A venue
# string mentioning one of these (and no track above) is NOT rewritten to the
# conference's canonical name.
_SUB_EVENT_RE = re.compile(
    r"\b(workshops?|symposium|tutorials?|shared task|companion|adjunct|satellite)\b",
    re.IGNORECASE,
)

# Venue strings that name an event (as opposed to a book or a journal); only
# these get their years and edition numbers stripped.
_EVENT_RE = re.compile(
    r"\b(proceedings|proc\.|conference|workshop|symposium|meeting|colloquium|congress|summit)\b",
    re.IGNORECASE,
)

# Words that should NOT be capitalised (unless they start a title)
FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "nor",
    "for",
    "yet",
    "so",
    "in",
    "on",
    "at",
    "to",
    "by",
    "of",
    "up",
    "as",
    "if",
    "from",
    "with",
    "into",
    "over",
    "upon",
    "than",
    "via",
}


# ── BibTeX parser (brace-aware, preserves formatting) ────────────────────

# Non-record entry types: @string{...}, @comment{...}, @preamble{...}
# These don't have a cite key + fields structure and are not parsed as entries.
_NON_ENTRY_TYPES = {"string", "comment", "preamble"}
# ...of which these are copied to the output verbatim (entries may rely on them).
_PASSTHROUGH_TYPES = {"string", "preamble"}

_BLOCK_START_RE = re.compile(r"@(\w+)\s*\{")
_FIELD_SEPARATOR_RE = re.compile(r"[\s,]*")
# Field names may contain more than \w (e.g. BibDesk's `date-added`, `bdsk-url-1`)
_FIELD_NAME_RE = re.compile(r"([^\s=,{}\"#]+)\s*=\s*")
_BARE_VALUE_RE = re.compile(r"[\w.-]+")


def _iter_blocks(text: str):
    """Yield (block_type, start, brace_start, end) for each @type{...} block."""
    i = 0
    while True:
        at_pos = text.find("@", i)
        if at_pos == -1:
            return
        m = _BLOCK_START_RE.match(text, at_pos)
        if not m:
            i = at_pos + 1
            continue
        brace_start = m.end() - 1  # position of the opening {
        # Find matching closing brace
        depth = 1
        j = brace_start + 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth > 0:
            line = text.count("\n", 0, at_pos) + 1
            print(
                f"  warning: unbalanced braces in the entry at line {line}; "
                "it swallowed the rest of the file",
                file=sys.stderr,
            )
        yield m.group(1), at_pos, brace_start, j
        i = j


def parse_bib(text: str) -> list[dict]:
    """
    Parse a .bib file into a list of entry dicts.
    Each dict has keys:
      - entry_type: e.g. "inproceedings"
      - cite_key: e.g. "wei2021finetuned"
      - fields: list of (field_name, field_value) preserving order
      - raw: the original raw text of the entry
    @string / @comment / @preamble blocks are skipped.
    """
    entries = []
    for entry_type, start, brace_start, end in _iter_blocks(text):
        if entry_type.lower() in _NON_ENTRY_TYPES:
            continue
        # Extract cite key and fields from inside the braces
        inner = text[brace_start + 1 : end - 1]
        # The cite key is everything up to the first comma
        comma_pos = inner.find(",")
        if comma_pos == -1:
            cite_key = inner.strip()
            fields_text = ""
        else:
            cite_key = inner[:comma_pos].strip()
            fields_text = inner[comma_pos + 1 :]
        entries.append(
            {
                "entry_type": entry_type,
                "cite_key": cite_key,
                "fields": parse_fields(fields_text, cite_key),
                "raw": text[start:end],
            }
        )
    return entries


def extract_passthrough_blocks(text: str) -> list[str]:
    """Raw text of the @string / @preamble blocks, in file order."""
    return [
        text[start:end]
        for block_type, start, _, end in _iter_blocks(text)
        if block_type.lower() in _PASSTHROUGH_TYPES
    ]


def parse_fields(text: str, cite_key: str = "") -> list[tuple[str, str]]:
    """
    Parse BibTeX fields from the text after the cite key.
    Returns list of (field_name, raw_value) tuples.
    raw_value includes the braces/quotes.
    """
    fields = []
    i = 0
    while i < len(text):
        # Skip whitespace/commas
        i = _FIELD_SEPARATOR_RE.match(text, i).end()
        if i >= len(text):
            break
        # Match field name and =
        m = _FIELD_NAME_RE.match(text, i)
        if not m:
            snippet = " ".join(text[i : i + 40].split())
            print(
                f"  warning: {cite_key}: can't parse fields from {snippet!r} on; "
                "the rest of this entry is dropped",
                file=sys.stderr,
            )
            break
        field_name = m.group(1).lower()
        i = m.end()
        if i >= len(text):
            break
        # Parse the value
        value, end_pos = parse_field_value(text, i)
        fields.append((field_name, value))
        i = end_pos
    return fields


def parse_field_value(text: str, start: int) -> tuple[str, int]:
    """
    Parse a BibTeX field value starting at position `start`.
    Handles {braced values}, "quoted values", and bare words/numbers.
    Also handles concatenation with #.
    Returns (value_string, end_position).
    """
    parts = []
    i = start
    while i < len(text):
        # Skip whitespace
        while i < len(text) and text[i] in " \t\n\r":
            i += 1
        if i >= len(text):
            break
        if text[i] == "{":
            # Brace-delimited value
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            parts.append(text[i:j])
            i = j
        elif text[i] == '"':
            # Quote-delimited value
            j = i + 1
            while j < len(text) and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            if j < len(text):
                j += 1  # skip closing quote
            parts.append(text[i:j])
            i = j
        else:
            # Bare word/number (e.g., month names, numbers)
            m = _BARE_VALUE_RE.match(text, i)
            if m:
                parts.append(m.group(0))
                i = m.end()
            else:
                break
        # Check for # concatenation
        while i < len(text) and text[i] in " \t\n\r":
            i += 1
        if i < len(text) and text[i] == "#":
            parts.append(" # ")
            i += 1
        else:
            break
    return "".join(parts), i


def strip_braces(s: str) -> str:
    """Remove the outermost layer of {} or "" from a string, if present."""
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1]
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def strip_all_braces(s: str) -> str:
    """Remove ALL {{ and }} to get plain text for matching."""
    return s.replace("{{", "").replace("}}", "").replace("{", "").replace("}", "")


_CONTROL_SEQ_RE = re.compile(r"\\(?:[a-zA-Z]+|.)", re.DOTALL)


def _find_closing_brace(s: str, start: int) -> int:
    """Index of the } matching the { at `start`, or -1 if it is never closed."""
    depth = 0
    i = start
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_math_end(s: str, start: int) -> int:
    """Index of the unescaped $ closing the $ at `start`, or -1 if there is none."""
    i = start + 1
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "$":
            return i
        i += 1
    return -1


def _is_literal(value: str) -> bool:
    """
    True if a raw field value is a single {braced} or "quoted" string, i.e. not
    a number, an @string macro or a `#` concatenation (which can't be rewritten
    as text).
    """
    v = value.strip()
    if v.startswith("{"):
        return _find_closing_brace(v, 0) == len(v) - 1
    return len(v) >= 2 and v[0] == '"' and v[-1] == '"' and '"' not in v[1:-1]


def strip_protective_braces(s: str) -> str:
    """
    Remove case-protecting braces ({Word}, {{Two Words}}) while keeping the
    braces LaTeX needs: command arguments (\\textsc{x}), groups that start with
    a command ({\\"o}, {\\em x}) and everything inside $math$.
    """
    out = []
    i = 0
    after_command = False  # previous token was a command or one of its arguments
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = _CONTROL_SEQ_RE.match(s, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                after_command = True
                continue
        elif c == "$":
            end = _find_math_end(s, i)
            if end != -1:
                out.append(s[i : end + 1])
                i = end + 1
                after_command = False
                continue
        elif c == "{":
            end = _find_closing_brace(s, i)
            if end != -1:
                content = s[i + 1 : end]
                if after_command:
                    # Argument of the preceding command (stays True for \frac{a}{b})
                    out.append("{" + strip_protective_braces(content) + "}")
                else:
                    if (
                        content.startswith("{")
                        and _find_closing_brace(content, 0) == len(content) - 1
                    ):
                        # {{...}}: protection around the whole group
                        out.append(strip_protective_braces(content[1:-1]))
                    elif content.startswith("\\"):
                        out.append("{" + strip_protective_braces(content) + "}")
                    else:
                        out.append(strip_protective_braces(content))
                i = end + 1
                continue
        out.append(c)
        i += 1
        after_command = False
    return "".join(out)


def _split_top_level_words(s: str) -> list[str]:
    """
    Split on whitespace that is outside braces and $math$. Whitespace runs are
    kept as their own tokens, so "".join(tokens) == s.
    """
    tokens = []
    buf = []
    depth = 0
    in_math = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            buf.append(s[i : i + 2])
            i += 2
            continue
        if c == "$":
            if in_math:
                in_math = False
            elif _find_math_end(s, i) != -1:
                in_math = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        elif c.isspace() and depth == 0 and not in_math:
            if buf:
                tokens.append("".join(buf))
                buf = []
            j = i
            while j < len(s) and s[j].isspace():
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        buf.append(c)
        i += 1
    if buf:
        tokens.append("".join(buf))
    return tokens


def _latex_to_plain(s: str) -> str:
    """Approximate plain text of a LaTeX string (for matching and API queries)."""
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)  # \textsc, \emph, ...
    s = re.sub(r"\\(.)", r"\1", s)  # \& -> &, \'e -> 'e
    return s.replace("{", "").replace("}", "").replace("$", "")


# ── Capitalisation logic ─────────────────────────────────────────────────


def capitalise_content_words(booktitle: str) -> str:
    """
    Capitalise content words in a booktitle string.
    Words already wrapped in {{...}} are left untouched.
    Function words (articles, prepositions, conjunctions) are lowercased
    unless they are the first word of the title or of a subtitle (i.e. they
    follow a colon or a dash).

    This operates on the inner value (with outer braces already stripped).
    """
    # Tokenize into segments: either {{...}} blocks or plain text
    tokens = re.split(r"(\{\{.*?\}\})", booktitle)
    result_tokens = []
    word_index = 0  # track position across all words
    prev_word = ""  # the word before the current one

    for token in tokens:
        if token.startswith("{{") and token.endswith("}}"):
            # Protected block -- count words inside for position tracking
            inner_words = token[2:-2].split()
            word_index += len(inner_words)
            if inner_words:
                prev_word = inner_words[-1]
            result_tokens.append(token)
        else:
            # Plain text -- capitalise content words
            words = re.split(r"(\s+)", token)  # preserve whitespace
            new_words = []
            for w in words:
                if not w.strip():
                    new_words.append(w)
                    continue
                starts_phrase = (
                    word_index == 0
                    or prev_word.endswith(":")
                    or re.fullmatch(r"[-–—]+", prev_word) is not None
                )
                prev_word = w
                # Check if it's a punctuation-only token
                if re.match(r"^[^a-zA-Z]+$", w):
                    new_words.append(w)
                    word_index += 1
                    continue
                # Skip words starting with a digit (e.g. "55th", "7th", "2024")
                # and LaTeX commands (e.g. "\textit{x}", "{\'e}cole")
                if w[0].isdigit() or "\\" in w:
                    new_words.append(w)
                    word_index += 1
                    continue
                # Separate leading/trailing punctuation
                m = re.match(r"^([^a-zA-Z]*)(.*?)([^a-zA-Z]*)$", w)
                prefix, core, suffix = m.group(1), m.group(2), m.group(3)
                if not core:
                    new_words.append(w)
                    word_index += 1
                    continue
                if starts_phrase:
                    # First word (of the title or a subtitle): always capitalise
                    core = core[0].upper() + core[1:]
                elif core.lower() in FUNCTION_WORDS:
                    core = core.lower()
                else:
                    core = core[0].upper() + core[1:]
                new_words.append(prefix + core + suffix)
                word_index += 1
            result_tokens.append("".join(new_words))
    return "".join(result_tokens)


def fix_canonical_terms(text: str) -> str:
    """Replace known terms with their canonical forms (case-insensitive)."""
    for pattern, canonical in CANONICAL_TERMS.items():
        text = re.sub(re.escape(pattern), canonical, text, flags=re.IGNORECASE)
    return text


def protect_capitals(title: str) -> str:
    """
    Wrap words containing uppercase letters in {{...}} to prevent LaTeX
    from lowercasing them. Expects existing protection to have been removed
    with strip_protective_braces(); a word is whitespace-delimited text outside
    braces and $math$, so LaTeX commands and their arguments stay in one piece.
    """
    result = []
    for token in _split_top_level_words(title):
        if not token.isspace() and re.search(r"[A-Z]", token):
            result.append("{{" + token + "}}")
        else:
            result.append(token)
    return "".join(result)


# ── Venue normalisation ──────────────────────────────────────────────────


# ACL Anthology venue ids (post-2020 IDs such as "2022.naacl-main.42") and
# letter prefixes (pre-2020 IDs such as "N19-1419") → acronym in VENUE_NAMES.
# Workshops have their own ids / the W prefix and are intentionally absent:
# their booktitles are too varied to auto-rewrite.
_ANTHOLOGY_VENUE_IDS = {
    "acl": "ACL",
    "naacl": "NAACL",
    "emnlp": "EMNLP",
    "eacl": "EACL",
    "aacl": "AACL",
    "coling": "COLING",
    "conll": "CoNLL",
    "lrec": "LREC",
}
_ANTHOLOGY_LETTER_PREFIXES = {
    "P": "ACL",
    "N": "NAACL",
    "D": "EMNLP",
    "E": "EACL",
    "C": "COLING",
    "K": "CoNLL",
    "L": "LREC",
}
# Volume names in post-2020 IDs: the main conference...
_ANTHOLOGY_MAIN_VOLUME_RE = re.compile(r"^(main|long|short|papers|\d+)$")
# ...and its non-main tracks (values are track names from VENUE_TRACKS).
_ANTHOLOGY_TRACK_VOLUMES = {
    "demo": "System Demonstrations",
    "demos": "System Demonstrations",
    "industry": "Industry Track",
    "srw": "Student Research Workshop",
    "tutorial": "Tutorial Abstracts",
    "tutorials": "Tutorial Abstracts",
}


def infer_acl_anthology_venue(entry: dict) -> tuple[str, str | None] | None:
    """
    If the entry has a DOI like 10.18653/v1/<id> or a URL on aclanthology.org,
    return (acronym, track) inferred from the Anthology ID; track is None for
    the main conference. The inference is authoritative -- it overrides
    whatever the booktitle field says.

    Returns None when the ID alone doesn't settle the venue: workshops, unknown
    volume names, and pre-2020 volumes other than 1 (whose meaning -- short
    papers, demos, SRW, co-located workshops -- changed from year to year).
    """
    anth_id: str | None = None

    doi = _get_field(entry, "doi")
    if doi:
        m = re.match(r"10\.18653/v1/(.+)", strip_braces(doi).strip())
        if m:
            anth_id = m.group(1).strip().rstrip("/")

    if anth_id is None:
        url = _get_field(entry, "url")
        if url:
            m = re.search(r"aclanthology\.org/([\w.\-/]+)", strip_braces(url))
            if m:
                anth_id = m.group(1).strip().rstrip("/")
                anth_id = re.sub(r"\.(pdf|bib)$", "", anth_id)

    if not anth_id:
        return None

    # Post-2020 format: 2021.findings-emnlp.10, 2022.naacl-main.42, 2020.acl-demos.14
    m = re.match(r"^\d{4}\.findings-([a-z]+)", anth_id, re.IGNORECASE)
    if m:
        acronym = "Findings of " + _ANTHOLOGY_VENUE_IDS.get(m.group(1).lower(), "")
        return (acronym, None) if acronym in VENUE_NAMES else None
    m = re.match(r"^(\d{4})\.([a-z]+)-([a-z0-9]+)", anth_id, re.IGNORECASE)
    if m:
        acronym = _ANTHOLOGY_VENUE_IDS.get(m.group(2).lower())
        if not acronym:
            return None
        if acronym == "LREC" and m.group(1) == "2024":
            acronym = "LREC-COLING"  # held jointly that year, filed under lrec
        volume = m.group(3).lower()
        if _ANTHOLOGY_MAIN_VOLUME_RE.match(volume):
            return (acronym, None)
        track = _ANTHOLOGY_TRACK_VOLUMES.get(volume)
        return (acronym, track) if track else None

    # Pre-2020 format: N19-1419 (volume 1 is always the main conference)
    m = re.match(r"^([A-Za-z])\d\d-1\d{3}\b", anth_id)
    if m:
        acronym = _ANTHOLOGY_LETTER_PREFIXES.get(m.group(1).upper())
        return (acronym, None) if acronym else None
    return None


_ACRONYM_LOOKUP = {a.lower(): a for a in VENUE_NAMES if a.isalpha()}
_ACRONYM_LOOKUP.update({alias.lower(): a for alias, a in VENUE_ALIASES.items()})


def _format_canonical(acronym: str, track: str | None = None) -> str:
    """ "Full Name (ACRONYM)", or "Full Name: Track (ACRONYM)"."""
    name = VENUE_NAMES[acronym]
    if track:
        name += ": " + track
    return f"{name} ({acronym})"


def expand_bare_acronym(plain: str) -> str | None:
    """
    If `plain` is just a known venue acronym (case-insensitive, optionally
    followed by a year), return its canonical "Full Name (ACRONYM)" form;
    else None.

    Examples:
      "ICLR"      -> "International Conference on Learning Representations (ICLR)"
      "NIPS'17"   -> "Advances in Neural Information Processing Systems (NeurIPS)"
      "TMLR 2024" -> "Transactions on Machine Learning Research (TMLR)"
    """
    m = re.match(r"^([A-Za-z]+)(?:\s*['’]\d{2}|\s+\d{4})?$", plain.strip())
    if not m:
        return None
    acronym = _ACRONYM_LOOKUP.get(m.group(1).lower())
    return _format_canonical(acronym) if acronym else None


def get_acronym(field_value: str) -> str | None:
    """
    Given a booktitle or journal field value (inner text, braces stripped once),
    determine which known venue it names.
    Returns the acronym string or None.
    """
    plain = strip_all_braces(field_value).strip()
    plain = re.sub(r"\s*\\?&\s*", " and ", plain)
    for pattern, acronym in CONFERENCE_RULES:
        if re.search(pattern, plain, re.IGNORECASE):
            return acronym
    return None


def already_has_acronym(field_value: str, acronym: str) -> bool:
    """Check if the booktitle/journal already contains the acronym in parens."""
    plain = strip_all_braces(field_value).strip()
    # Check for (ACRONYM) anywhere (e.g. already appended, or inline like "(EMNLP)")
    # Also check for variants like "(EMNLP-IJCNLP)" or "(Coling 2008)"
    return bool(
        re.search(
            r"\([^)]*" + re.escape(acronym) + r"[^)]*\)", plain, re.IGNORECASE
        )
    )


def _detect_track(plain: str) -> str | None:
    """The non-main track (see VENUE_TRACKS) a venue string mentions, if any."""
    for pattern, track in VENUE_TRACKS:
        if re.search(pattern, plain, re.IGNORECASE):
            return track
    return None


def canonical_venue(
    text: str, anth_venue: tuple[str, str | None] | None = None
) -> str | None:
    """
    Canonical "Full Name (ACRONYM)" form of a booktitle / journal value that
    names a venue in VENUE_NAMES, else None. `anth_venue` is the result of
    infer_acl_anthology_venue() and, when given, wins over the text.
    """
    if anth_venue:
        return _format_canonical(*anth_venue)
    plain = strip_all_braces(text).strip()
    expanded = expand_bare_acronym(plain)
    if expanded:
        return expanded
    acronym = get_acronym(plain)
    if acronym not in VENUE_NAMES:
        return None
    track = _detect_track(plain)
    if track is None and _SUB_EVENT_RE.search(plain):
        # A workshop etc. held at the conference, not the conference itself
        return None
    return _format_canonical(acronym, track)


_YEAR = r"(?:19|20)\d{2}"
_SPELLED_ORDINAL = (
    r"(?:(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s])?"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)"
    r"|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth"
    r"|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth)"
)
_EDITION_RULES = [
    # IEEE-style "ICASSP 2024 - 2024 IEEE International Conference ... (ICASSP)"
    (re.compile(rf"^([A-Z][A-Za-z-]*)\s+{_YEAR}\s+-\s+(?=.*\(\1\))"), ""),
    # "CVPR'06", "WWW '22"
    (re.compile(r"(?<=[A-Za-z])\s?['’]\d{2}\b"), ""),
    # "SemEval-2022", "ICLR 2017", leading "2025 ", ", 1994"
    (re.compile(rf"(?:(?<=[A-Za-z*])[\s\-–—]+|\b){_YEAR}\b"), ""),
    # "(IWP2005)"
    (re.compile(rf"\(([A-Za-z*-]+?){_YEAR}\)"), r"(\1)"),
]
_ORDINAL_RULES = [
    # "12th", "22nd" (but not "21st Century")
    (re.compile(r"\b\d+(?:st|nd|rd|th)\b(?!\s+century)\s*", re.IGNORECASE), ""),
    # "Seventh", "Thirty-Eighth" -- only at the start or after "the", so that
    # e.g. "Workshop on Second Language Acquisition" is left alone
    (
        re.compile(
            rf"(?:^|(?<=\bthe\s)){_SPELLED_ORDINAL}\b(?!-)\s+", re.IGNORECASE
        ),
        "",
    ),
]
_EDITION_TIDY_RULES = [
    (re.compile(r"\(\s*\)"), ""),  # parens emptied by the rules above
    (re.compile(r"\(\s+"), "("),
    (re.compile(r"[\s\-–—]+\)"), ")"),
    (re.compile(r"\s+([,:;])"), r"\1"),
    (re.compile(r",(?:\s*,)+"), ","),
    (re.compile(r"\s{2,}"), " "),
    (re.compile(r"^[\s,:;\-–—]+|[\s,:;\-–—]+$"), ""),
]


def strip_venue_editions(text: str, ordinals: bool = True) -> str:
    """
    Remove what distinguishes one edition of an event from the next -- years
    and (if `ordinals`) edition numbers -- from a venue name.

    Examples:
      "Proceedings of the 12th Workshop on Multiword Expressions"
          -> "Proceedings of the Workshop on Multiword Expressions"
      "2025 China Automation Congress (CAC)" -> "China Automation Congress (CAC)"
      "... on Semantic Evaluation (SemEval-2022)" -> "... on Semantic Evaluation (SemEval)"
    """
    rules = _EDITION_RULES + (_ORDINAL_RULES if ordinals else []) + _EDITION_TIDY_RULES
    result = text
    for pattern, replacement in rules:
        result = pattern.sub(replacement, result)
    # Don't reduce a venue that is nothing but a year / number to an empty string
    return result or text


# ── Field manipulation helpers ───────────────────────────────────────────

ARXIV_SPECIFIC_FIELDS = {
    "eprint",
    "primaryclass",
    "archiveprefix",
    "eprinttype",
    "number",
}


_ARXIV_VENUE_RE = re.compile(
    r"^(arxiv(\.org)?(\s+e-?prints?)?|corr|arxiv\s+preprint\b.*|arxiv:.*)$",
    re.IGNORECASE,
)


def _is_arxiv_venue(value: str) -> bool:
    """
    True for a journal value that only says "this is on arXiv": arXiv,
    arXiv.org, CoRR, "arXiv preprint arXiv:2005.14165", "arXiv:2005.14165 [cs]".
    """
    return bool(_ARXIV_VENUE_RE.match(strip_all_braces(strip_braces(value)).strip()))


def _strip_arxiv_fields(
    fields: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Drop fields that only make sense for arxiv preprints:
      - eprint / primaryclass / archiveprefix / eprinttype / number
      - publisher = {arXiv}
      - journal = {arXiv}, {arXiv:...}, {CoRR}, ... (see _is_arxiv_venue)
      - volume = {abs/...}
    """
    out = []
    for fn, fv in fields:
        if fn in ARXIV_SPECIFIC_FIELDS:
            continue
        if fn == "publisher" and strip_braces(fv).strip().lower() == "arxiv":
            continue
        if fn == "journal" and _is_arxiv_venue(fv):
            continue
        if fn == "volume" and strip_braces(fv).strip().startswith("abs/"):
            continue
        out.append((fn, fv))
    return out


def _insert_venue_field(
    fields: list[tuple[str, str]], venue: str, venue_field: str
) -> list[tuple[str, str]]:
    """
    Insert {venue_field} = {venue} after the title/author/year block.
    No-op if a field with that name already exists.
    """
    if any(fn == venue_field for fn, _ in fields):
        return list(fields)
    insert_pos = 0
    for i, (fn, _) in enumerate(fields):
        if fn in ("title", "author", "year"):
            insert_pos = i + 1
    new_fields = list(fields)
    new_fields.insert(insert_pos, (venue_field, "{" + venue + "}"))
    return new_fields


def _set_field(
    fields: list[tuple[str, str]], name: str, raw_value: str
) -> list[tuple[str, str]]:
    """Replace the existing `name` field with `raw_value`, or append it."""
    out = []
    replaced = False
    for fn, fv in fields:
        if fn == name:
            if not replaced:
                out.append((name, raw_value))
            replaced = True
        else:
            out.append((fn, fv))
    if not replaced:
        out.append((name, raw_value))
    return out


def _set_url_field(
    fields: list[tuple[str, str]], url: str
) -> list[tuple[str, str]]:
    """Replace the existing url field with `url`, or append it."""
    return _set_field(fields, "url", "{" + url + "}")


def _set_year_field(
    fields: list[tuple[str, str]], year: int
) -> list[tuple[str, str]]:
    """Set the year field, keeping the entry's `year = 2020` / `{2020}` style."""
    old = next((fv for fn, fv in fields if fn == "year"), None)
    bare = old is not None and not old.lstrip().startswith(("{", '"'))
    return _set_field(fields, "year", str(year) if bare else "{" + str(year) + "}")


def _bibtex_escape(text: str) -> str:
    """Escape the LaTeX special characters that show up in API-provided names."""
    return re.sub(r"(?<!\\)([&%#_])", r"\\\1", text)


# ── Main processing ──────────────────────────────────────────────────────


# series = {WWW '22}, {NIPS'17}, {ICML 2024}: a venue + year tag that only
# repeats (with a year) what the booktitle already says.
_SERIES_VENUE_YEAR_RE = re.compile(r"^[A-Za-z*&-]+\s*(?:['’]\d{2}|(?:19|20)\d{2})$")

# Fields whose text is rewritten (as opposed to kept or dropped as a whole).
_REWRITTEN_FIELDS = {"author", "title", "booktitle", "journal"}


def process_entry(entry: dict) -> dict:
    """Process a single BibTeX entry: remove fields, normalise venues, capitalise."""
    new_fields = []
    doi_url = None
    has_url = False
    has_author = any(fn == "author" for fn, _ in entry["fields"])
    is_proceedings = entry["entry_type"].lower() in ("inproceedings", "conference")

    # Pre-pass: if the entry has an ACL Anthology DOI/URL, that's authoritative
    # for the venue — override whatever the booktitle field happens to say.
    anth_venue = infer_acl_anthology_venue(entry)

    for field_name, field_value in entry["fields"]:
        # Collect DOI to convert to url later
        if field_name == "doi":
            inner = strip_braces(field_value).strip()
            # ACL Anthology DOIs map directly to aclanthology.org URLs
            acl_m = re.match(r"10\.18653/v1/(.+)", inner)
            if acl_m:
                doi_url = "https://aclanthology.org/" + acl_m.group(1)
            elif inner.startswith("http"):
                doi_url = inner
            else:
                doi_url = "https://doi.org/" + inner
            continue

        # Track whether entry already has a url field
        if field_name == "url":
            has_url = True

        # An editor is only needed when there is no author to cite the work by
        if field_name == "editor" and not has_author:
            new_fields.append((field_name, field_value))
            continue

        # Skip unwanted fields
        if field_name not in KEEP_FIELDS:
            continue

        # Skip publisher = {Association for Computational Linguistics}
        if field_name == "publisher":
            inner_pub = strip_braces(field_value).strip()
            if inner_pub == "Association for Computational Linguistics":
                continue

        if field_name == "series" and _SERIES_VENUE_YEAR_RE.match(
            strip_all_braces(strip_braces(field_value)).strip()
        ):
            continue

        # @string macros and `#` concatenations can't be rewritten as text
        if field_name in _REWRITTEN_FIELDS and not _is_literal(field_value):
            new_fields.append((field_name, field_value))
            continue

        # Clip author list at 25
        if field_name == "author":
            inner = strip_braces(field_value)
            authors = re.split(r"\s+and\s+", inner.strip())
            if len(authors) > 25:
                inner = " and ".join(authors[:25]) + " and others"
                field_value = "{" + inner + "}"
            new_fields.append((field_name, field_value))
            continue

        # Process title: protect capitalised words with {{}}
        if field_name == "title":
            # Strip the existing protection, then fix canonical terms, then re-protect
            inner = strip_protective_braces(strip_braces(field_value))
            inner = fix_canonical_terms(inner)
            inner = protect_capitals(inner)
            new_fields.append((field_name, "{" + inner + "}"))
            continue

        # Process booktitle: normalise the venue name
        if field_name == "booktitle":
            inner = strip_protective_braces(strip_braces(field_value))
            # Remove volume/paper-type annotations like
            #   (Volume 1: Long Papers), (Volume 2: Short Papers),
            #   , Volume 1 (Long and Short Papers)
            inner = re.sub(
                r"\s*[,(]\s*\{*Volume\}*\s*\d+\s*[:(]\s*\{*\w+\}*\s*(\{*and\}*\s*\{*\w+\}*\s*)?\{*Papers?\}*\s*\)*",
                "",
                inner,
                flags=re.IGNORECASE,
            )
            #   , Companion Volume: Short Papers
            inner = re.sub(
                r",?\s*Companion Volume:\s*Short Papers", "", inner, flags=re.IGNORECASE
            )
            # Known venue -> "Full Name (ACRONYM)". An ACL Anthology DOI/URL
            # overrides whatever the booktitle currently says (e.g. a bare
            # "ACL" tag on a NAACL paper).
            canonical = canonical_venue(inner, anth_venue)
            if canonical:
                inner = canonical
            else:
                # Any other event: drop the year / edition number, capitalise
                # content words, and append the acronym of a known venue it
                # mentions (e.g. a workshop held at that venue)
                if is_proceedings or _EVENT_RE.search(inner):
                    inner = strip_venue_editions(inner)
                inner = capitalise_content_words(inner)
                acronym = get_acronym(inner)
                if acronym and not already_has_acronym(inner, acronym):
                    inner += " (" + acronym + ")"
            new_fields.append((field_name, "{" + inner + "}"))

        # Process journal: same venue normalisation, but no capitalisation
        # changes, and only proceedings filed under `journal` lose their year.
        elif field_name == "journal":
            inner = strip_braces(field_value)
            # journal = {arXiv ...} is left for the arxiv normalisation step
            if not _is_arxiv_venue(inner):
                canonical = canonical_venue(inner)
                if canonical:
                    inner = canonical
                else:
                    if _EVENT_RE.search(inner):
                        inner = strip_venue_editions(inner, ordinals=False)
                    acronym = get_acronym(inner)
                    if acronym and not already_has_acronym(inner, acronym):
                        inner += " (" + acronym + ")"
                field_value = "{" + inner + "}"
            new_fields.append((field_name, field_value))

        else:
            new_fields.append((field_name, field_value))

    # Add url from DOI if entry doesn't already have a url field
    if doi_url and not has_url:
        new_fields.append(("url", "{" + doi_url + "}"))

    return {
        "entry_type": entry["entry_type"],
        "cite_key": entry["cite_key"],
        "fields": new_fields,
    }


def format_entry(entry: dict) -> str:
    """Format a processed entry back to BibTeX string."""
    lines = [f"@{entry['entry_type']}{{{entry['cite_key']},"]
    for field_name, field_value in entry["fields"]:
        lines.append(f"  {field_name} = {field_value},")
    lines.append("}")
    return "\n".join(lines)


# ── Arxiv detection & normalisation ──────────────────────────────────────

def _get_field(entry: dict, name: str) -> str | None:
    """Get raw field value from an entry, or None."""
    for fn, fv in entry["fields"]:
        if fn == name:
            return fv
    return None


# New-style (2502.16147) or old-style (hep-th/9901001, math.GT/0309136) arxiv ID,
# with an optional version suffix
_ARXIV_ID = r"(?:\d{4}\.\d{4,6}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?"
_ARXIV_URL_RE = re.compile(
    rf"arxiv\.org/(?:abs|pdf)/({_ARXIV_ID})|10\.48550/arxiv\.({_ARXIV_ID})",
    re.IGNORECASE,
)
_ARXIV_PREFIXED_ID_RE = re.compile(rf"arxiv:\s*({_ARXIV_ID})", re.IGNORECASE)
# Fields that can hold the ID of an entry already known to be an arxiv paper
_ARXIV_ID_SOURCES = [
    ("journal", _ARXIV_PREFIXED_ID_RE),
    ("url", _ARXIV_URL_RE),
    ("howpublished", _ARXIV_URL_RE),
    ("volume", re.compile(rf"^abs/({_ARXIV_ID})", re.IGNORECASE)),
    ("number", _ARXIV_PREFIXED_ID_RE),
]


def detect_arxiv_id(entry: dict) -> str | None:
    """
    Detect whether an entry is an arxiv paper and return the arxiv ID.
    Handles all known formats:
      A. eprint = {2502.16147} (+ archiveprefix=arXiv)
      B. journal={ArXiv} / {arXiv.org} / {CoRR}, or publisher={arXiv}, with
         the ID in url, howpublished, volume={abs/...} or number={arXiv:...}
      C. journal={arXiv:XXXX.XXXXX [cs]} / {arXiv preprint arXiv:XXXX.XXXXX}
    Excludes JSTOR eprints (eprinttype=jstor).
    Returns the arxiv ID string (e.g. '2502.16147') or None.
    """
    # Exclude JSTOR eprints
    eprinttype = _get_field(entry, "eprinttype")
    if eprinttype and "jstor" in strip_braces(eprinttype).lower():
        return None

    # Check eprint field (Format A)
    eprint = _get_field(entry, "eprint")
    if eprint:
        eid = re.sub(r"^arxiv:\s*", "", strip_braces(eprint).strip(), flags=re.IGNORECASE)
        if re.fullmatch(_ARXIV_ID, eid, re.IGNORECASE):
            return eid

    # Formats B and C: something has to say "this is an arxiv paper" before an
    # arxiv link elsewhere in the entry is taken to be its ID
    journal = _get_field(entry, "journal")
    publisher = _get_field(entry, "publisher")
    is_arxiv = (journal is not None and _is_arxiv_venue(journal)) or (
        publisher is not None and strip_braces(publisher).strip().lower() == "arxiv"
    )
    if not is_arxiv:
        return None
    for name, pattern in _ARXIV_ID_SOURCES:
        value = _get_field(entry, name)
        if value:
            m = pattern.search(strip_all_braces(strip_braces(value)).strip())
            if m:
                return next(g for g in m.groups() if g)
    return None


def is_published_venue_entry(entry: dict) -> bool:
    """Check if an entry already has a real (non-arxiv) venue."""
    for fn, fv in entry["fields"]:
        if fn == "booktitle":
            return True
        if fn == "journal" and not _is_arxiv_venue(fv):
            return True
    return False


def normalize_arxiv_entry(entry: dict, arxiv_id: str) -> dict:
    """
    Normalize an arxiv entry to a canonical format:
      @misc with eprint, archiveprefix, url fields.
    Removes arxiv-style journal fields. Skips entries with real venues.
    """
    if is_published_venue_entry(entry):
        return entry

    new_fields = []
    has_eprint = False
    has_archiveprefix = False
    has_url = False

    for fn, fv in entry["fields"]:
        # Remove arxiv-style journal field
        if fn == "journal" and _is_arxiv_venue(fv):
            continue
        # Remove volume if it's arxiv-style (e.g. abs/XXXX.XXXXX)
        if fn == "volume":
            inner = strip_braces(fv).strip()
            if inner.startswith("abs/"):
                continue
        # Remove howpublished = {https://arxiv.org/abs/...} (becomes the url)
        if fn == "howpublished" and _ARXIV_URL_RE.search(fv):
            continue
        # Track what we have
        if fn == "eprint":
            has_eprint = True
            # Ensure the eprint value is just the ID
            new_fields.append(("eprint", "{" + arxiv_id + "}"))
            continue
        if fn == "archiveprefix":
            has_archiveprefix = True
            new_fields.append(("archiveprefix", "{arXiv}"))
            continue
        if fn == "url":
            has_url = True
            new_fields.append(("url", "{https://arxiv.org/abs/" + arxiv_id + "}"))
            continue
        new_fields.append((fn, fv))

    # Add missing canonical fields
    if not has_eprint:
        new_fields.append(("eprint", "{" + arxiv_id + "}"))
    if not has_archiveprefix:
        new_fields.append(("archiveprefix", "{arXiv}"))
    if not has_url:
        new_fields.append(("url", "{https://arxiv.org/abs/" + arxiv_id + "}"))

    return {
        "entry_type": "misc",
        "cite_key": entry["cite_key"],
        "fields": new_fields,
    }


# ── HTTP ─────────────────────────────────────────────────────────────────


# Exponential backoff for HTTP 429: 5s, 10s, 20s (3 retries beyond the first try).
_HTTP_429_BACKOFF = (5, 10, 20)


def _http_get_json(
    url: str,
    rate_limiter: RateLimiter,
    label: str,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> dict | None:
    """
    GET `url` and parse the JSON body. Retries on HTTP 429 with exponential
    backoff (5s, 10s, 20s). Other errors return None immediately. The rate
    limiter gates every attempt, so retries don't burst past the limit.
    """
    last_err: Exception | None = None
    for attempt in range(1 + len(_HTTP_429_BACKOFF)):
        rate_limiter.wait()
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 and attempt < len(_HTTP_429_BACKOFF):
                time.sleep(_HTTP_429_BACKOFF[attempt])
                continue
            break
        except Exception as e:
            last_err = e
            break
    _warn(f"    lookup failed for {label}: {last_err}")
    return None


def _parallel_map(fn, jobs: list, label: str) -> list:
    """
    Run fn(job) for every job in a thread pool, showing a progress bar.
    Returns the results in job order.
    """
    global _PROGRESS
    bar = ProgressBar(len(jobs), label)

    def run(job):
        try:
            return fn(job)
        finally:
            bar.advance()

    _PROGRESS = bar
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(run, jobs))
    finally:
        bar.close()
        _PROGRESS = None


# ── Crossref API ─────────────────────────────────────────────────────────

# Crossref `type` values that count as a real publication of an arxiv paper.
# Anything else (notably `posted-content` for preprints, or `book`/`monograph`
# for unrelated full-book matches by title) is rejected.
_CROSSREF_PUBLISHED_TYPES = {
    "journal-article",
    "proceedings-article",
    "book-chapter",
}

# Crossref `type` values that are never the work a bib entry cites (checked
# when filling in a missing URL): containers, datasets, preprints, ...
_CROSSREF_NON_WORK_TYPES = {
    "journal",
    "journal-volume",
    "journal-issue",
    "proceedings",
    "proceedings-series",
    "book-series",
    "book-set",
    "dataset",
    "database",
    "component",
    "posted-content",
    "peer-review",
    "grant",
    "standard",
    "other",
}

# Preprint servers that Crossref lists as the container of a "published" work.
_PREPRINT_VENUE_RE = re.compile(
    r"\b(arxiv|corr|ssrn|biorxiv|medrxiv|chemrxiv|techrxiv|psyarxiv"
    r"|research square|preprints?|zenodo|osf)\b",
    re.IGNORECASE,
)

# How many hits to ask Crossref for; the right one isn't always ranked first.
_CROSSREF_ROWS = 5
# A lookup that found nothing usable is repeated after this long (the paper
# may have been published in the meantime). Usable results are kept for good.
_CROSSREF_RECHECK_SECONDS = 30 * 24 * 3600


def _fold(s: str) -> str:
    """Lowercase and strip accents."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def _normalize_title_for_match(s: str) -> str:
    """Reduce a title to its lowercase ASCII letters and digits."""
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return re.sub(r"[^a-z0-9]", "", _fold(_latex_to_plain(s)))


def _titles_match(query: str, returned: str) -> bool:
    """Check if two titles match (ignoring case, punctuation, spacing and accents)."""
    q = _normalize_title_for_match(query)
    return bool(q) and q == _normalize_title_for_match(returned)


def _entry_title(entry: dict) -> str | None:
    """Plain-text title of an entry (for API queries), or None."""
    raw = _get_field(entry, "title")
    if not raw:
        return None
    return " ".join(_latex_to_plain(strip_braces(raw)).split()) or None


def _entry_year(entry: dict) -> int | None:
    m = re.search(r"\d{4}", _get_field(entry, "year") or "")
    return int(m.group(0)) if m else None


def _arxiv_id_year(arxiv_id: str) -> int | None:
    """Year a paper was first posted, from its arxiv ID (2305.09145 -> 2023)."""
    m = re.search(r"(\d{2})\d{2}(?:\.\d{4,6}|\d{3})", arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy < 91 else 1900 + yy


def _surname_key(name: str) -> str:
    """Reduce a surname to lowercase ASCII letters, for comparison."""
    return re.sub(r"[^a-z]", "", _fold(_latex_to_plain(name)))


# "Gibbs Jr.", "King, III"
_NAME_SUFFIX_RE = re.compile(r"[\s,]+(?:jr|sr|ii|iii|iv)\.?$", re.IGNORECASE)


def _first_surname(entry: dict) -> str | None:
    """Comparable surname of the first author (or editor, if there is no author)."""
    raw = _get_field(entry, "author") or _get_field(entry, "editor")
    if not raw:
        return None
    first = re.split(r"\s+and\s+", strip_braces(raw).strip())[0].strip()
    first = _NAME_SUFFIX_RE.sub("", first)
    if not first:
        return None
    if first.startswith("{") and first.endswith("}"):
        surname = first  # corporate author, e.g. {R Core Team}
    elif "," in first:
        surname = first.split(",")[0]  # "Last, First" / "Last Jr., First"
    else:
        surname = first.split()[-1]  # "First Last"
    return _surname_key(_NAME_SUFFIX_RE.sub("", surname.strip())) or None


def _author_matches(entry: dict, item: dict) -> bool:
    """
    Check that the entry's first author (or editor) is among the authors and
    editors of a Crossref item. This is what tells apart two different works
    that happen to share a title.
    """
    surname = _first_surname(entry)
    if not surname:
        return False
    for role in ("author", "editor"):
        for person in item.get(role) or []:
            name = (person.get("family") or person.get("name") or "").strip()
            other = _surname_key(_NAME_SUFFIX_RE.sub("", name))
            if not other:
                continue
            if other == surname:
                return True
            # "Maaten" vs "van der Maaten"
            shorter, longer = sorted((other, surname), key=len)
            if len(shorter) >= 4 and longer.endswith(shorter):
                return True
    return False


def _crossref_year(item: dict) -> int | None:
    parts = (item.get("issued") or {}).get("date-parts") or []
    try:
        return int(parts[0][0])
    except (IndexError, TypeError, ValueError):
        return None


def _crossref_title_matches(title: str, item: dict) -> bool:
    """Check `title` against a Crossref item's title, with and without subtitle."""
    item_title = (item.get("title") or [""])[0]
    if _titles_match(title, item_title):
        return True
    subtitle = (item.get("subtitle") or [""])[0]
    return bool(subtitle) and _titles_match(title, item_title + " " + subtitle)


def _slim_crossref_item(item: dict) -> dict:
    """Keep only what the script uses from a Crossref item (this is what gets cached)."""
    slim = {
        key: item[key]
        for key in ("DOI", "URL", "title", "subtitle", "container-title", "type")
        if item.get(key)
    }
    if _crossref_year(item):
        slim["issued"] = {"date-parts": [[_crossref_year(item)]]}
    for role in ("author", "editor"):
        people = [
            {key: person[key]}
            for person in item.get(role) or []
            for key in ("family", "name")
            if person.get(key)
        ]
        if people:
            slim[role] = people
    return slim


def crossref_lookup(title: str, pick):
    """
    Search Crossref for works titled `title` and return pick(items), where
    `items` are the hits whose title matches and `pick` chooses among them
    (returning None if none will do).

    Hits are cached by normalized title. A cached result that pick() accepts
    is final; one it rejects is looked up again once it is older than
    _CROSSREF_RECHECK_SECONDS. Failed lookups are never cached.
    """
    cache_key = _normalize_title_for_match(title)
    if not cache_key:
        return None
    if _CACHE is not None:
        cached = _CACHE.get("crossref_search", cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("items"), list):
            picked = pick(cached["items"])
            age = time.time() - cached.get("ts", 0)
            if picked is not None or age < _CROSSREF_RECHECK_SECONDS:
                return picked

    params = urllib.parse.urlencode(
        {
            "query.bibliographic": title,
            "rows": _CROSSREF_ROWS,
            "select": "DOI,URL,title,subtitle,author,editor,container-title,type,issued",
        }
    )
    url = f"https://api.crossref.org/works?{params}"
    data = _http_get_json(
        url,
        CROSSREF_RATE_LIMITER,
        title[:60],
        timeout=15,
        headers=_crossref_headers(),
    )
    if data is None:
        return None

    items = [
        _slim_crossref_item(item)
        for item in data.get("message", {}).get("items", [])
        if _crossref_title_matches(title, item)
    ]
    if _CACHE is not None:
        _CACHE.set("crossref_search", cache_key, {"ts": int(time.time()), "items": items})
    return pick(items)


def _venue_is_real(venue: str) -> bool:
    """Check if a venue string represents a real publication (not a preprint server)."""
    return bool(venue and venue.strip()) and not _PREPRINT_VENUE_RE.search(venue)


def _crossref_url_for(item: dict) -> str | None:
    """Pick a canonical URL from a Crossref work item."""
    doi = item.get("DOI")
    if doi:
        acl_m = re.match(r"10\.18653/v1/(.+)", doi)
        if acl_m:
            return "https://aclanthology.org/" + acl_m.group(1)
        return "https://doi.org/" + doi
    return item.get("URL")


def _crossref_venue(item: dict) -> tuple[str, str]:
    """Return (venue_name, kind) where kind is 'journal' or 'conference'."""
    container = [html.unescape(c) for c in item.get("container-title") or [] if c]
    if not container:
        venue_name = ""
    elif item.get("type") == "book-chapter":
        # Book chapters list the series first ("Lecture Notes in Computer
        # Science") and the book itself last
        venue_name = container[-1]
    else:
        venue_name = container[0]
    kind = "journal" if item.get("type") == "journal-article" else "conference"
    return venue_name, kind


def _pick_published(entry: dict, arxiv_id: str, items: list[dict]) -> dict | None:
    """
    The Crossref hit that is the published version of an arxiv entry: a real
    publication, by the same first author, not older than the preprint.
    """
    preprint_year = _arxiv_id_year(arxiv_id) or _entry_year(entry)
    candidates = []
    for item in items:
        if item.get("type") not in _CROSSREF_PUBLISHED_TYPES:
            continue
        if not _venue_is_real(_crossref_venue(item)[0]):
            continue
        if not _author_matches(entry, item):
            continue
        year = _crossref_year(item)
        if year is None or (preprint_year and year < preprint_year - 1):
            continue
        candidates.append(item)
    # Several hits can qualify (e.g. the conference paper and a reprint
    # elsewhere): prefer a venue we know over Crossref's ranking
    for item in candidates:
        if canonical_venue(_crossref_venue(item)[0]):
            return item
    return candidates[0] if candidates else None


# Print and online-first editions of the same work can be dated this far apart.
_URL_YEAR_TOLERANCE = 2


def _pick_url(entry: dict, items: list[dict]) -> str | None:
    """
    URL of the Crossref hit that is this entry: a citable work by the same
    first author (or editor), dated within _URL_YEAR_TOLERANCE years of the
    entry's year.
    """
    entry_year = _entry_year(entry)
    for item in items:
        if item.get("type") in _CROSSREF_NON_WORK_TYPES:
            continue
        if not _author_matches(entry, item):
            continue
        year = _crossref_year(item)
        if entry_year and year and abs(year - entry_year) > _URL_YEAR_TOLERANCE:
            continue
        url = _crossref_url_for(item)
        if url:
            return url
    return None


def upgrade_arxiv_to_published(entry: dict, cr_item: dict) -> dict | None:
    """
    If the Crossref item is a real publication (not a preprint or unrelated
    book), upgrade the arxiv entry to @inproceedings/@article with the venue,
    year and URL of the published version.
    """
    if cr_item.get("type") not in _CROSSREF_PUBLISHED_TYPES:
        return None

    venue_name, kind = _crossref_venue(cr_item)
    if not _venue_is_real(venue_name):
        return None

    is_journal = kind == "journal"
    new_entry_type = "article" if is_journal else "inproceedings"
    venue_field = "journal" if is_journal else "booktitle"

    fields = _strip_arxiv_fields(entry["fields"])
    # Cite the year of the published version, not of the preprint
    year = _crossref_year(cr_item)
    if year:
        fields = _set_year_field(fields, year)
    fields = _insert_venue_field(fields, _bibtex_escape(venue_name), venue_field)

    proceedings_url = _crossref_url_for(cr_item)
    if proceedings_url:
        fields = _set_url_field(fields, proceedings_url)

    return {
        "entry_type": new_entry_type,
        "cite_key": entry["cite_key"],
        "fields": fields,
    }


def check_arxiv_publications(entries: list[dict]) -> list[dict]:
    """
    For each arxiv entry (no real venue), query Crossref by title in parallel.
    If Crossref has a published version of the paper, upgrade the entry.
    """
    todo = []  # (index, title, arxiv_id)
    for i, entry in enumerate(entries):
        if is_published_venue_entry(entry):
            continue
        arxiv_id = detect_arxiv_id(entry)
        if not arxiv_id:
            continue
        title = _entry_title(entry)
        if not title:
            continue
        todo.append((i, title, arxiv_id))

    result = list(entries)
    if not todo:
        print("  0 arxiv entries upgraded to published venues")
        return result

    def lookup(job):
        i, title, arxiv_id = job
        return crossref_lookup(
            title, lambda items: _pick_published(entries[i], arxiv_id, items)
        )

    items = _parallel_map(lookup, todo, "Crossref")

    upgraded = 0
    for (i, _, _), item in zip(todo, items):
        if not item:
            continue
        upgraded_entry = upgrade_arxiv_to_published(entries[i], item)
        if not upgraded_entry:
            continue
        # Re-process so acronyms/capitalization apply
        upgraded_entry = process_entry(upgraded_entry)
        result[i] = upgraded_entry
        venue_field = (
            "booktitle"
            if upgraded_entry["entry_type"] == "inproceedings"
            else "journal"
        )
        venue_val = ""
        for fn, fv in upgraded_entry["fields"]:
            if fn == venue_field:
                venue_val = strip_all_braces(fv)
                break
        print(f"  Upgraded: {entries[i]['cite_key']} -> {venue_val}")
        upgraded += 1

    print(f"  {upgraded} arxiv entries upgraded to published venues")
    return result


def resolve_missing_urls(entries: list[dict]) -> list[dict]:
    """
    For entries that still lack a url field, try Crossref by title.
    Operates on entry dicts (parallel) rather than formatted strings.
    """
    todo = []  # (index, title)
    for i, entry in enumerate(entries):
        has_url = any(fn == "url" for fn, _ in entry["fields"])
        if has_url:
            continue
        has_venue = any(
            fn in ("booktitle", "journal") for fn, _ in entry["fields"]
        )
        etype = entry["entry_type"].lower()
        if not has_venue and etype in ("book", "misc", "phdthesis", "techreport"):
            continue
        title = _entry_title(entry)
        if not title:
            continue
        todo.append((i, title))

    result = list(entries)
    print(f"  {len(todo)} entries need URL lookup...")
    if not todo:
        print("  Resolved: 0 via Crossref")
        return result

    def lookup(job):
        i, title = job
        return crossref_lookup(title, lambda items: _pick_url(entries[i], items))

    urls = _parallel_map(lookup, todo, "Crossref")

    resolved = 0
    for (i, title), found_url in zip(todo, urls):
        if not found_url:
            continue
        entry = result[i]
        result[i] = {
            "entry_type": entry["entry_type"],
            "cite_key": entry["cite_key"],
            "fields": _set_url_field(entry["fields"], found_url),
        }
        resolved += 1
        print(f"  Crossref: {title[:60]}...")

    print(f"  Resolved: {resolved} via Crossref")
    return result


# ── OpenReview API integration ───────────────────────────────────────────


def extract_openreview_id(url: str) -> str | None:
    """Extract the forum ID from an OpenReview URL."""
    m = re.search(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def query_openreview(forum_id: str) -> dict | None:
    """Query the OpenReview API for a note by forum ID.

    Cached only on success; failures are retried on the next run.
    """
    if _CACHE is not None and _CACHE.has("openreview", forum_id):
        return _CACHE.get("openreview", forum_id)

    url = f"https://api2.openreview.net/notes?id={forum_id}"
    data = _http_get_json(url, OPENREVIEW_RATE_LIMITER, forum_id, timeout=15)
    if data is None:
        return None
    notes = data.get("notes", [])
    if not notes:
        return None
    result = notes[0]

    if _CACHE is not None:
        _CACHE.set("openreview", forum_id, result)
    return result


def _openreview_get_venue(note: dict) -> str | None:
    """Extract venue string from an OpenReview note."""
    content = note.get("content", {})
    # API v2 format: content fields are {value: ...}
    for key in ("venue", "_bibtex", "venueid"):
        val = content.get(key)
        if isinstance(val, dict):
            val = val.get("value", "")
        if val and isinstance(val, str):
            if key == "_bibtex":
                # Try to extract booktitle from the bibtex
                m = re.search(r"booktitle\s*=\s*\{(.+?)\}", val)
                if m:
                    return m.group(1)
                continue
            if key == "venueid":
                # "ICLR.cc/2024/Conference" -> "ICLR 2024"; other ids aren't names
                m = re.match(r"^(\w+)\.cc/(\d{4})/Conference$", val)
                if m:
                    return f"{m.group(1)} {m.group(2)}"
                continue
            return val
    return None


_OPENREVIEW_STATUS_PREFIX_RE = re.compile(
    r"^(?:accepted|published)\s+(?:by|to|at|in)\s+", re.IGNORECASE
)
_OPENREVIEW_PRESENTATION_RE = re.compile(
    r"[\s(]+(?:poster|oral|spotlight|talk|notable[\s-]top[\s-]\d+%)\)?$", re.IGNORECASE
)


def _parse_openreview_venue(venue: str) -> tuple[str, int | None]:
    """
    Split an OpenReview venue string into (venue name, year), dropping the
    acceptance status and presentation type:
      "ICLR 2024 poster"  -> ("ICLR 2024", 2024)
      "Accepted by TMLR"  -> ("TMLR", None)
    The year is left in the name; process_entry() normalises it away.
    """
    name = _OPENREVIEW_STATUS_PREFIX_RE.sub("", venue.strip())
    name = _OPENREVIEW_PRESENTATION_RE.sub("", name).strip()
    m = re.search(rf"\b{_YEAR}\b", name)
    return name, int(m.group(0)) if m else None


def _convert_howpublished_to_url(
    fields: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Replace a howpublished={...openreview.net...} field with a url field
    (or just drop it, if the entry already has a url).
    """
    has_url = any(fn == "url" for fn, _ in fields)
    out = []
    for fn, fv in fields:
        if fn == "howpublished" and "openreview.net" in fv:
            m = re.search(r"(?:https?://)?(openreview\.net/[^\s{}]+)", fv)
            if m and not has_url:
                out.append(("url", "{https://" + m.group(1) + "}"))
                has_url = True
        else:
            out.append((fn, fv))
    return out


def resolve_openreview_entries(entries: list[dict]) -> list[dict]:
    """
    For entries with OpenReview URLs, query the API (in parallel) to resolve
    venue info. If accepted at a venue, upgrade the entry.
    """
    todo = []  # (index, forum_id)
    for i, entry in enumerate(entries):
        if is_published_venue_entry(entry):
            continue
        or_url = None
        for fn, fv in entry["fields"]:
            if fn in ("url", "howpublished"):
                inner = strip_braces(fv)
                if "openreview.net" in inner:
                    or_url = inner
                    break
        if not or_url:
            continue
        forum_id = extract_openreview_id(or_url)
        if not forum_id:
            continue
        todo.append((i, forum_id))

    result = list(entries)
    if not todo:
        print("  0 entries resolved via OpenReview")
        return result

    notes = _parallel_map(lambda x: query_openreview(x[1]), todo, "OpenReview")

    resolved = 0
    for (i, _), note in zip(todo, notes):
        if not note:
            continue
        venue = _openreview_get_venue(note)
        if not venue:
            continue
        venue_lower = venue.lower()
        if any(w in venue_lower for w in ("reject", "withdraw", "desk")):
            continue
        if venue_lower.startswith("submitted to"):
            continue

        # "ICLR 2024 poster" -> "ICLR 2024" + 2024
        venue, year = _parse_openreview_venue(venue)
        if not venue:
            continue
        venue_lower = venue.lower()
        is_journal = any(
            j in venue_lower for j in ("journal", "transactions", "j. mach. learn.")
        ) or venue.split()[0].upper() in JOURNAL_ACRONYMS
        new_entry_type = "article" if is_journal else "inproceedings"
        venue_field = "journal" if is_journal else "booktitle"

        entry = entries[i]
        fields = _strip_arxiv_fields(entry["fields"])
        fields = _convert_howpublished_to_url(fields)
        if year:
            fields = _set_year_field(fields, year)
        fields = _insert_venue_field(fields, _bibtex_escape(venue), venue_field)

        upgraded = {
            "entry_type": new_entry_type,
            "cite_key": entry["cite_key"],
            "fields": fields,
        }
        # Re-process so acronyms/capitalization apply
        upgraded = process_entry(upgraded)
        result[i] = upgraded

        venue_display = strip_all_braces(venue)
        print(f"  OpenReview: {entry['cite_key']} -> {venue_display}")
        resolved += 1

    print(f"  {resolved} entries resolved via OpenReview")
    return result


def _rebiber_url(entry: dict, rebiber_entry: dict) -> str | None:
    """
    The URL rebiber found for `entry`, or None if its match can't be trusted.

    rebiber matches titles with digits and punctuation removed, so
    "SemEval-2016 Task 12: Clinical TempEval" picks up the entry of
    "SemEval-2015 Task 6: Clinical TempEval"; it also matches a different
    publication that merely shares the title.
    """
    url_raw = _get_field(rebiber_entry, "url")
    if not url_raw:
        return None
    url = strip_braces(url_raw).strip()
    if url == strip_braces(_get_field(entry, "url") or "").strip():
        return url  # the entry's own url, passed through
    if not _titles_match(_entry_title(entry) or "", _entry_title(rebiber_entry) or ""):
        return None
    if "arxiv.org" in url.lower():
        # A preprint's year may differ, but the link must be a real arxiv ID
        # (rebiber sometimes builds one out of volume/number)
        return url if _ARXIV_URL_RE.search(url) else None
    year, rebiber_year = _entry_year(entry), _entry_year(rebiber_entry)
    if year and rebiber_year and abs(year - rebiber_year) > 1:
        return None
    return url


def run_rebiber(input_path: Path, entries: list[dict]) -> dict[str, str]:
    """
    Run rebiber on the original bib file to get canonical URLs for `entries`
    (the parsed content of that file).
    Returns a dict mapping cite_key -> url.
    """
    try:
        import io

        import rebiber
        from rebiber import construct_bib_db, load_bib_file, normalize_bib
    except ImportError:
        print("  rebiber not installed, skipping URL enrichment", file=sys.stderr)
        return {}

    pkg_dir = os.path.dirname(rebiber.__file__)
    bib_list_file = os.path.join(pkg_dir, "bib_list.txt")

    # Suppress rebiber's verbose loading output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        bib_db = construct_bib_db(bib_list_file, start_dir=pkg_dir + os.sep)
    finally:
        sys.stdout = old_stdout

    # rebiber is optional enrichment: if it chokes on the file, carry on without
    tmp_path = input_path.with_stem(input_path.stem + "_rebiber_tmp")
    try:
        all_bib_entries = load_bib_file(str(input_path))

        # Write rebiber output to a temp file
        normalize_bib(bib_db, all_bib_entries, str(tmp_path))

        # Parse rebiber output to extract URLs per cite key
        rebiber_text = tmp_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  rebiber failed ({e}), skipping URL enrichment", file=sys.stderr)
        return {}
    finally:
        tmp_path.unlink(missing_ok=True)

    originals = {entry["cite_key"]: entry for entry in entries}
    url_map = {}
    rejected = 0
    for rebiber_entry in parse_bib(rebiber_text):
        original = originals.get(rebiber_entry["cite_key"])
        if original is None or _get_field(rebiber_entry, "url") is None:
            continue
        url = _rebiber_url(original, rebiber_entry)
        if url:
            url_map[rebiber_entry["cite_key"]] = url
        else:
            rejected += 1

    print(
        f"  rebiber: found {len(url_map)} canonical URLs"
        f" ({rejected} rejected: title or year doesn't match)"
    )
    return url_map


def main():
    parser = argparse.ArgumentParser(description="Rule-based BibTeX cleaner")
    parser.add_argument(
        "input",
        nargs="?",
        help="Input .bib file (default: main.bib next to script)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Ignore the API cache and re-query everything",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip all network API calls (Crossref + OpenReview); "
        "still runs rebiber, field cleanup, and arxiv normalisation",
    )
    parser.add_argument(
        "--mailto",
        help="Contact e-mail sent to Crossref to get its faster 'polite pool' "
        "(default: the CROSSREF_MAILTO environment variable)",
    )
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path(__file__).parent / "main.bib"

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = input_path.with_stem(input_path.stem + "_cleaned")
    cache_path = input_path.with_suffix(".cache.json")

    global _CACHE, _CROSSREF_MAILTO
    _CACHE = APICache(cache_path, load=not args.force)
    if args.mailto:
        _CROSSREF_MAILTO = args.mailto

    text = input_path.read_text(encoding="utf-8")
    entries = parse_bib(text)
    passthrough = extract_passthrough_blocks(text)

    # Step 1: Run rebiber to get canonical URLs
    print("Running rebiber for URL enrichment...")
    rebiber_urls = run_rebiber(input_path, entries)

    # Step 2: Process entries with our rules
    processed = []
    for entry in entries:
        p = process_entry(entry)
        # Apply rebiber canonical URLs early so arxiv detection can use them
        if p["cite_key"] in rebiber_urls:
            p["fields"] = _set_url_field(p["fields"], rebiber_urls[p["cite_key"]])
        processed.append(p)

    # Step 3: Normalize arxiv entries to canonical format
    print("Normalizing arxiv entries...")
    for i, entry in enumerate(processed):
        arxiv_id = detect_arxiv_id(entry)
        if arxiv_id and not is_published_venue_entry(entry):
            processed[i] = normalize_arxiv_entry(entry, arxiv_id)
        elif not arxiv_id and not is_published_venue_entry(entry):
            # Handle entries with journal={ArXiv} but no extractable ID:
            # convert to @misc and drop the arxiv journal field
            journal = _get_field(entry, "journal")
            if journal and _is_arxiv_venue(journal):
                new_fields = [(fn, fv) for fn, fv in entry["fields"] if fn != "journal"]
                processed[i] = {
                    "entry_type": "misc",
                    "cite_key": entry["cite_key"],
                    "fields": new_fields,
                }

    if args.offline:
        print("Offline mode: skipping Crossref + OpenReview lookups")
    else:
        if not _CROSSREF_MAILTO:
            print(
                "  note: no --mailto / CROSSREF_MAILTO set; "
                "using Crossref's slower public pool"
            )
        try:
            # Step 4: Check if arxiv papers are actually published at a venue
            print("Checking arxiv papers for published venues via Crossref...")
            processed = check_arxiv_publications(processed)

            # Step 5: Resolve OpenReview entries
            print("Resolving OpenReview entries...")
            processed = resolve_openreview_entries(processed)

            # Step 6: For entries still missing a URL, try Crossref title search
            print("Resolving missing URLs via Crossref title search...")
            processed = resolve_missing_urls(processed)
        finally:
            # Step 7: Persist cache (also when interrupted, so that the
            # lookups done so far aren't repeated on the next run)
            try:
                _CACHE.save()
            except Exception as e:
                print(f"  cache: failed to save: {e}", file=sys.stderr)

    # Step 8: Format and write
    formatted = passthrough + [format_entry(p) for p in processed]
    output_path.write_text("\n\n".join(formatted) + "\n", encoding="utf-8")
    print(f"Processed {len(entries)} entries -> {output_path}")


if __name__ == "__main__":
    main()
