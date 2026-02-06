"""
Rule-based BibTeX cleaner:
  1. Removes unnecessary fields (abstract, keywords, file, urldate, etc.)
  2. Appends conference acronyms to booktitle for well-known AI/ML/NLP venues
  3. Capitalises content words in booktitle (preserving existing {{...}} braces)

No LLM involved -- purely deterministic string matching.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
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


S2_RATE_LIMITER = RateLimiter(rate=1.0)  # 1 request per second
OPENREVIEW_RATE_LIMITER = RateLimiter(rate=1.0)

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
# Each rule: (pattern_to_match_in_booktitle, acronym_to_append)
# Patterns are matched case-insensitively against the plain text of the
# booktitle (with {{...}} braces stripped for matching purposes).
# Order matters: first match wins.
CONFERENCE_RULES = [
    # --- ACL ---
    (r"annual meeting of the association for computational linguistics", "ACL"),
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
    # --- COLING ---
    (r"international conference on computational linguistics", "COLING"),
    (r"joint international conference on computational linguistics", "COLING"),
    # --- Findings of ACL/EMNLP/NAACL ---
    # These are handled specially: we detect "Findings of ... ACL/EMNLP/NAACL"
    # and append the appropriate acronym only if it isn't already there.
    (
        r"findings of the association for computational linguistics.*?emnlp",
        "Findings of EMNLP",
    ),
    (
        r"findings of the association for computational linguistics.*?naacl",
        "Findings of NAACL",
    ),
    (
        r"findings of the association for computational linguistics.*?acl",
        "Findings of ACL",
    ),
    # --- NeurIPS ---
    (r"conference on neural information processing systems", "NeurIPS"),
    (
        r"neural information processing systems track on datasets and benchmarks",
        "NeurIPS D\\&B",
    ),
    (r"advances in neural information processing systems", "NeurIPS"),
    (r"^neural information processing systems$", "NeurIPS"),
    # --- ICML ---
    (r"international conference on machine learning", "ICML"),
    # --- ICLR ---
    (r"international conference on learning representations", "ICLR"),
    # --- AAAI ---
    (r"aaai conference on artificial intelligence", "AAAI"),
    # --- IJCAI ---
    (r"international joint conference on artificial intelligence", "IJCAI"),
    (r"\bijcai\b", "IJCAI"),
    # --- CVPR ---
    (r"conference on computer vision and pattern recognition", "CVPR"),
    # --- ICCV ---
    (r"international conference on computer vision(?! and)", "ICCV"),
    # --- ECCV ---
    (r"european conference on computer vision", "ECCV"),
    # --- SIGIR ---
    (r"acm .* information retrieval", "SIGIR"),
    # --- KDD ---
    (r"acm .* knowledge discovery and data mining", "KDD"),
    # --- CIKM ---
    (r"acm international conference on information and knowledge management", "CIKM"),
    # --- WWW / The Web Conference ---
    (r"the web conference", "WWW"),
    # --- LREC ---
    (r"language resources and evaluation", "LREC"),
    # --- SIGdial ---
    (r"sigdial meeting on discourse and dialogue", "SIGdial"),
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
    # --- BlackboxNLP workshop ---
    (r"blackboxnlp workshop", "BlackboxNLP"),
    # --- Semantic Computing ---
    (r"international conference on semantic computing", "ICSC"),
]

# Known terms that should be preserved in their canonical form (case-insensitive match)
CANONICAL_TERMS = {
    "t-sne": "t-SNE",
}

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


def parse_bib(text: str) -> list[dict]:
    """
    Parse a .bib file into a list of entry dicts.
    Each dict has keys:
      - entry_type: e.g. "inproceedings"
      - cite_key: e.g. "wei2021finetuned"
      - fields: list of (field_name, field_value) preserving order
      - raw: the original raw text of the entry
    """
    entries = []
    i = 0
    while i < len(text):
        # Find the next @
        at_pos = text.find("@", i)
        if at_pos == -1:
            break
        # Parse entry type
        m = re.match(r"@(\w+)\s*\{", text[at_pos:])
        if not m:
            i = at_pos + 1
            continue
        entry_type = m.group(1)
        brace_start = at_pos + m.end() - 1  # position of the opening {
        # Find matching closing brace
        depth = 1
        j = brace_start + 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        raw_entry = text[at_pos:j]
        # Extract cite key and fields from inside the braces
        inner = text[brace_start + 1 : j - 1]
        # The cite key is everything up to the first comma
        comma_pos = inner.find(",")
        if comma_pos == -1:
            cite_key = inner.strip()
            fields_text = ""
        else:
            cite_key = inner[:comma_pos].strip()
            fields_text = inner[comma_pos + 1 :]
        # Parse fields
        fields = parse_fields(fields_text)
        entries.append(
            {
                "entry_type": entry_type,
                "cite_key": cite_key,
                "fields": fields,
                "raw": raw_entry,
            }
        )
        i = j
    return entries


def parse_fields(text: str) -> list[tuple[str, str]]:
    """
    Parse BibTeX fields from the text after the cite key.
    Returns list of (field_name, raw_value) tuples.
    raw_value includes the braces/quotes.
    """
    fields = []
    i = 0
    while i < len(text):
        # Skip whitespace/commas
        m = re.match(r"[\s,]*", text[i:])
        if m:
            i += m.end()
        if i >= len(text):
            break
        # Match field name and =
        m = re.match(r"(\w+)\s*=\s*", text[i:])
        if not m:
            break
        field_name = m.group(1).lower()
        i += m.end()
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
            m = re.match(r"[\w.-]+", text[i:])
            if m:
                parts.append(m.group(0))
                i += m.end()
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
    """Remove the outermost layer of {} from a string, if present."""
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1]
    return s


def strip_all_braces(s: str) -> str:
    """Remove ALL {{ and }} to get plain text for matching."""
    return s.replace("{{", "").replace("}}", "").replace("{", "").replace("}", "")


# ── Capitalisation logic ─────────────────────────────────────────────────


def capitalise_content_words(booktitle: str) -> str:
    """
    Capitalise content words in a booktitle string.
    Words already wrapped in {{...}} are left untouched.
    Function words (articles, prepositions, conjunctions) are lowercased
    unless they are the first word.

    This operates on the inner value (with outer braces already stripped).
    """
    # Tokenize into segments: either {{...}} blocks or plain text
    tokens = re.split(r"(\{\{.*?\}\})", booktitle)
    result_tokens = []
    word_index = 0  # track position across all words

    for token in tokens:
        if token.startswith("{{") and token.endswith("}}"):
            # Protected block -- count words inside for position tracking
            inner_words = token[2:-2].split()
            word_index += len(inner_words)
            result_tokens.append(token)
        else:
            # Plain text -- capitalise content words
            words = re.split(r"(\s+)", token)  # preserve whitespace
            new_words = []
            for w in words:
                if not w.strip():
                    new_words.append(w)
                    continue
                # Check if it's a punctuation-only token
                if re.match(r"^[^a-zA-Z]+$", w):
                    new_words.append(w)
                    word_index += 1
                    continue
                # Skip words starting with a digit (e.g. "55th", "7th", "2024")
                if w[0].isdigit():
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
                if word_index == 0:
                    # First word: always capitalise
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
    from lowercasing them. Words already inside {{...}} are left alone.
    Also handles LaTeX commands like {\\textbar} by leaving them untouched.
    """
    # Tokenize into: {{...}} blocks, {...} blocks, or plain text
    tokens = re.split(r"(\{\{.*?\}\}|\{[^{}]*\})", title)
    result = []
    for token in tokens:
        if (token.startswith("{{") and token.endswith("}}")) or (
            token.startswith("{") and token.endswith("}")
        ):
            # Already protected
            result.append(token)
        else:
            # Plain text: wrap each word that has uppercase letters
            words = re.split(r"(\s+)", token)
            new_words = []
            for w in words:
                if not w.strip():
                    new_words.append(w)
                elif re.search(r"[A-Z]", w):
                    new_words.append("{{" + w + "}}")
                else:
                    new_words.append(w)
            result.append("".join(new_words))
    return "".join(result)


# ── Acronym appending ────────────────────────────────────────────────────


def get_acronym(field_value: str) -> str | None:
    """
    Given a booktitle or journal field value (inner text, braces stripped once),
    determine if a conference acronym should be appended.
    Returns the acronym string or None.
    """
    plain = strip_all_braces(field_value).strip()
    for pattern, acronym in CONFERENCE_RULES:
        if re.search(pattern, plain, re.IGNORECASE):
            return acronym
    return None


def already_has_acronym(field_value: str, acronym: str) -> bool:
    """Check if the booktitle/journal already contains the acronym in parens."""
    plain = strip_all_braces(field_value).strip()
    # Check for (ACRONYM) anywhere (e.g. already appended, or inline like "(EMNLP)")
    # Also check for variants like "(EMNLP-IJCNLP)" containing the acronym
    return bool(re.search(r"\([^)]*" + re.escape(acronym) + r"[^)]*\)", plain))


def append_acronym(field_raw: str, acronym: str) -> str:
    """
    Append ' (ACRONYM)' to a brace-delimited field value.
    E.g., '{International Conference ...}' -> '{International Conference ... (ICLR)}'
    """
    # The raw value is like {Some Title}
    # We need to insert before the final }
    field_raw = field_raw.rstrip()
    if field_raw.endswith("}"):
        return field_raw[:-1] + " (" + acronym + ")}"
    return field_raw


# ── Main processing ──────────────────────────────────────────────────────


def process_entry(entry: dict) -> dict:
    """Process a single BibTeX entry: remove fields, add acronyms, capitalise."""
    new_fields = []
    doi_url = None
    has_url = False

    for field_name, field_value in entry["fields"]:
        # Collect DOI to convert to url later
        if field_name == "doi":
            inner = strip_braces(field_value)
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

        # Skip unwanted fields
        if field_name not in KEEP_FIELDS:
            continue

        # Skip publisher = {Association for Computational Linguistics}
        if field_name == "publisher":
            inner_pub = strip_braces(field_value).strip()
            if inner_pub == "Association for Computational Linguistics":
                continue

        # Process title: protect capitalised words with {{}}
        if field_name == "title":
            inner = strip_braces(field_value)
            # Strip all inner braces, then fix canonical terms, then re-protect
            inner = strip_all_braces(inner)
            inner = fix_canonical_terms(inner)
            inner = protect_capitals(inner)
            new_fields.append((field_name, "{" + inner + "}"))
            continue

        # Process booktitle: clean up, capitalise + add acronym
        if field_name == "booktitle":
            inner = strip_braces(field_value)
            # Remove volume/paper-type annotations like
            #   (Volume 1: Long Papers), (Volume 2: Short Papers),
            #   , Volume 1 (Long and Short Papers)
            inner = re.sub(
                r"\s*[,(]\s*\{*Volume\}*\s*\d+\s*[:(]\s*\{*\w+\}*\s*(\{*and\}*\s*\{*\w+\}*\s*)?\{*Papers?\}*\s*\)*",
                "",
                inner,
                flags=re.IGNORECASE,
            )
            # Capitalise content words
            inner = capitalise_content_words(inner)
            # Check for acronym
            acronym = get_acronym(inner)
            rebuilt = "{" + inner + "}"
            if acronym and not already_has_acronym(inner, acronym):
                rebuilt = append_acronym(rebuilt, acronym)
            new_fields.append((field_name, rebuilt))

        # Process journal: add acronym (no capitalisation changes)
        elif field_name == "journal":
            inner = strip_braces(field_value)
            acronym = get_acronym(inner)
            if acronym and not already_has_acronym(inner, acronym):
                field_value = append_acronym(field_value, acronym)
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

ARXIV_FIELDS = {"eprint", "primaryclass", "archiveprefix", "eprinttype"}


def _get_field(entry: dict, name: str) -> str | None:
    """Get raw field value from an entry, or None."""
    for fn, fv in entry["fields"]:
        if fn == name:
            return fv
    return None


def detect_arxiv_id(entry: dict) -> str | None:
    """
    Detect whether an entry is an arxiv paper and return the arxiv ID.
    Handles all known formats:
      A. @misc with eprint + archiveprefix=arXiv
      B. @article with journal={ArXiv}
      C. @article with journal={arXiv:XXXX.XXXXX [cs]}
    Excludes JSTOR eprints (eprinttype=jstor).
    Returns the arxiv ID string (e.g. '2502.16147') or None.
    """
    # Exclude JSTOR eprints
    eprinttype = _get_field(entry, "eprinttype")
    if eprinttype and "jstor" in strip_braces(eprinttype).lower():
        return None

    # Check eprint field (Formats A and some C)
    eprint = _get_field(entry, "eprint")
    if eprint:
        eid = strip_braces(eprint).strip()
        # Validate it looks like an arxiv ID (YYMM.NNNNN or category/NNNNNNN)
        if re.match(r"^\d{4}\.\d{4,6}(v\d+)?$", eid) or re.match(r"^[a-z-]+/\d+$", eid):
            return eid

    # Check journal field for arxiv patterns (Formats B and C)
    journal = _get_field(entry, "journal")
    if journal:
        inner = strip_braces(journal).strip()
        if inner.lower() == "arxiv":
            # Format B: journal = {ArXiv} -- need to find ID elsewhere
            # Try the url field
            url = _get_field(entry, "url")
            if url:
                m = re.search(r"arxiv\.org/abs/(\S+)", strip_braces(url))
                if m:
                    return m.group(1)
            # Try the volume field (e.g. volume = {abs/2505.01325})
            volume = _get_field(entry, "volume")
            if volume:
                m = re.match(r"abs/(\d{4}\.\d{4,6})", strip_braces(volume))
                if m:
                    return m.group(1)
            return None
        # Format C: journal = {arXiv:2005.14165 [cs]}
        m = re.match(r"arXiv:(\d{4}\.\d{4,6})", inner)
        if m:
            return m.group(1)

    return None


def is_published_venue_entry(entry: dict) -> bool:
    """Check if an entry already has a real (non-arxiv) venue."""
    for fn, fv in entry["fields"]:
        if fn == "booktitle":
            return True
        if fn == "journal":
            inner = strip_braces(fv).strip().lower()
            if inner not in ("arxiv",) and not inner.startswith("arxiv:"):
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
        if fn == "journal":
            inner = strip_braces(fv).strip().lower()
            if inner == "arxiv" or inner.startswith("arxiv:"):
                continue
        # Remove volume if it's arxiv-style (e.g. abs/XXXX.XXXXX)
        if fn == "volume":
            inner = strip_braces(fv).strip()
            if inner.startswith("abs/"):
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


# ── Semantic Scholar: check if arxiv papers are published ────────────────


def query_s2_by_arxiv_id(arxiv_id: str, retries: int = 2) -> dict | None:
    """Query Semantic Scholar for a paper by arxiv ID. Returns API response or None."""
    for attempt in range(1 + retries):
        S2_RATE_LIMITER.wait()
        try:
            fields = "title,venue,publicationVenue,externalIds,year,url"
            url = f"https://api.semanticscholar.org/graph/v1/paper/arxiv:{arxiv_id}?fields={fields}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(3)
                continue
            print(f"    S2 lookup failed for {arxiv_id}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"    S2 lookup failed for {arxiv_id}: {e}", file=sys.stderr)
            return None
    return None


def _venue_is_real(venue: str) -> bool:
    """Check if a venue string represents a real publication (not arxiv)."""
    if not venue:
        return False
    v = venue.strip().lower()
    return v not in ("", "arxiv", "arxiv.org", "arxiv.org e-print archive")


def _build_proceedings_url(s2_data: dict) -> str | None:
    """Extract the best URL from S2 data for a published paper."""
    ext = s2_data.get("externalIds", {})
    if ext.get("DOI"):
        doi = ext["DOI"]
        acl_m = re.match(r"10\.18653/v1/(.+)", doi)
        if acl_m:
            return "https://aclanthology.org/" + acl_m.group(1)
        return "https://doi.org/" + doi
    if s2_data.get("url"):
        return s2_data["url"]
    return None


def upgrade_arxiv_to_published(entry: dict, s2_data: dict) -> dict | None:
    """
    If S2 data shows the paper was published at a real venue, upgrade the entry.
    Returns the upgraded entry dict, or None if not published.
    """
    venue_name = ""
    venue_type = None

    pub_venue = s2_data.get("publicationVenue") or {}
    if pub_venue.get("name"):
        venue_name = pub_venue["name"]
        venue_type = pub_venue.get("type", "").lower()
    elif _venue_is_real(s2_data.get("venue", "")):
        venue_name = s2_data["venue"]
    else:
        return None

    if not _venue_is_real(venue_name):
        return None

    # Determine entry type
    is_journal = venue_type == "journal" if venue_type else False
    new_entry_type = "article" if is_journal else "inproceedings"
    venue_field = "journal" if is_journal else "booktitle"

    # Build new fields: keep title, author, year, pages; drop arxiv-specific ones
    arxiv_specific = {"eprint", "primaryclass", "archiveprefix", "eprinttype", "number"}
    new_fields = []
    has_venue_field = False

    for fn, fv in entry["fields"]:
        if fn in arxiv_specific:
            continue
        if fn == "publisher" and strip_braces(fv).strip().lower() == "arxiv":
            continue
        if fn == "journal":
            inner = strip_braces(fv).strip().lower()
            if inner == "arxiv" or inner.startswith("arxiv:"):
                continue
        if fn == "volume":
            inner = strip_braces(fv).strip()
            if inner.startswith("abs/"):
                continue
        if fn == venue_field:
            has_venue_field = True
        if fn == "url":
            # Replace with proceedings URL
            proceedings_url = _build_proceedings_url(s2_data)
            if proceedings_url:
                new_fields.append(("url", "{" + proceedings_url + "}"))
            else:
                new_fields.append((fn, fv))
            continue
        new_fields.append((fn, fv))

    if not has_venue_field:
        # Insert venue field after title (or author, or at position 2)
        insert_pos = 0
        for i, (fn, _) in enumerate(new_fields):
            if fn in ("title", "author", "year"):
                insert_pos = i + 1
        new_fields.insert(insert_pos, (venue_field, "{" + venue_name + "}"))

    # Add URL if not present
    has_url = any(fn == "url" for fn, _ in new_fields)
    if not has_url:
        proceedings_url = _build_proceedings_url(s2_data)
        if proceedings_url:
            new_fields.append(("url", "{" + proceedings_url + "}"))

    return {
        "entry_type": new_entry_type,
        "cite_key": entry["cite_key"],
        "fields": new_fields,
    }


def check_arxiv_publications(entries: list[dict]) -> list[dict]:
    """
    For each arxiv entry, query S2 to check if it's been published.
    If so, upgrade it. Returns the (possibly modified) list of entries.
    """
    result = list(entries)
    upgraded = 0

    for i, entry in enumerate(entries):
        arxiv_id = detect_arxiv_id(entry)
        if not arxiv_id:
            continue
        if is_published_venue_entry(entry):
            continue

        s2_data = query_s2_by_arxiv_id(arxiv_id)
        if not s2_data:
            continue

        upgraded_entry = upgrade_arxiv_to_published(entry, s2_data)
        if upgraded_entry:
            # Re-process through process_entry so acronyms/capitalization apply
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
            print(f"  Upgraded: {entry['cite_key']} -> {venue_val}")
            upgraded += 1

    print(f"  {upgraded} arxiv entries upgraded to published venues")
    return result


# ── OpenReview API integration ───────────────────────────────────────────


def extract_openreview_id(url: str) -> str | None:
    """Extract the forum ID from an OpenReview URL."""
    m = re.search(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def query_openreview(forum_id: str) -> dict | None:
    """Query the OpenReview API for a note by forum ID."""
    OPENREVIEW_RATE_LIMITER.wait()
    try:
        url = f"https://api2.openreview.net/notes?id={forum_id}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            notes = data.get("notes", [])
            if notes:
                return notes[0]
    except Exception as e:
        print(f"    OpenReview lookup failed for {forum_id}: {e}", file=sys.stderr)
    return None


def _openreview_get_venue(note: dict) -> str | None:
    """Extract venue string from an OpenReview note."""
    content = note.get("content", {})
    # API v2 format: content fields are {value: ...}
    for key in ("venue", "venueid", "_bibtex"):
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
            return val
    return None


def resolve_openreview_entries(entries: list[dict]) -> list[dict]:
    """
    For entries with OpenReview URLs, query the API to resolve venue info.
    If paper was accepted at a venue, upgrade the entry.
    """
    result = list(entries)
    resolved = 0

    for i, entry in enumerate(entries):
        # Find OpenReview URL in url or howpublished fields
        or_url = None
        for fn, fv in entry["fields"]:
            if fn in ("url", "howpublished"):
                inner = strip_braces(fv)
                if "openreview.net" in inner:
                    or_url = inner
                    break
        if not or_url:
            continue

        # Skip if entry already has a proper venue
        if is_published_venue_entry(entry):
            continue

        forum_id = extract_openreview_id(or_url)
        if not forum_id:
            continue

        note = query_openreview(forum_id)
        if not note:
            continue

        venue = _openreview_get_venue(note)
        if not venue:
            continue

        # Skip if venue indicates rejection or withdrawal
        venue_lower = venue.lower()
        if any(w in venue_lower for w in ("reject", "withdraw", "desk")):
            continue
        # Skip if venue is just "Submitted to ..."
        if venue_lower.startswith("submitted to"):
            continue

        # Determine entry type from venue
        is_journal = any(
            j in venue_lower for j in ("journal", "transactions", "j. mach. learn.")
        )
        new_entry_type = "article" if is_journal else "inproceedings"
        venue_field = "journal" if is_journal else "booktitle"

        # Build upgraded entry: remove howpublished, arxiv fields; add venue
        arxiv_specific = {
            "eprint",
            "primaryclass",
            "archiveprefix",
            "eprinttype",
            "number",
        }
        new_fields = []
        has_venue_field = False
        has_url = False

        for fn, fv in entry["fields"]:
            if fn in arxiv_specific:
                continue
            if fn == "publisher" and strip_braces(fv).strip().lower() == "arxiv":
                continue
            if fn == "howpublished":
                # Convert to url if it was the OpenReview link
                inner = strip_braces(fv)
                if "openreview.net" in inner:
                    new_fields.append(("url", fv))
                    has_url = True
                    continue
                new_fields.append((fn, fv))
                continue
            if fn == venue_field:
                has_venue_field = True
            if fn == "url":
                has_url = True
            new_fields.append((fn, fv))

        if not has_venue_field:
            insert_pos = 0
            for j, (fn, _) in enumerate(new_fields):
                if fn in ("title", "author", "year"):
                    insert_pos = j + 1
            new_fields.insert(insert_pos, (venue_field, "{" + venue + "}"))

        upgraded = {
            "entry_type": new_entry_type,
            "cite_key": entry["cite_key"],
            "fields": new_fields,
        }
        # Re-process so acronyms/capitalization apply
        upgraded = process_entry(upgraded)
        result[i] = upgraded

        venue_display = strip_all_braces(venue)
        print(f"  OpenReview: {entry['cite_key']} -> {venue_display}")
        resolved += 1

    print(f"  {resolved} entries resolved via OpenReview")
    return result


def run_rebiber(input_path: Path) -> dict[str, str]:
    """
    Run rebiber on the original bib file to get canonical URLs.
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

    all_bib_entries = load_bib_file(str(input_path))

    # Write rebiber output to a temp file
    tmp_path = input_path.with_stem(input_path.stem + "_rebiber_tmp")
    normalize_bib(bib_db, all_bib_entries, str(tmp_path))

    # Parse rebiber output to extract URLs per cite key
    rebiber_text = tmp_path.read_text(encoding="utf-8")
    rebiber_entries = parse_bib(rebiber_text)
    url_map = {}
    for entry in rebiber_entries:
        for field_name, field_value in entry["fields"]:
            if field_name == "url":
                url_map[entry["cite_key"]] = strip_braces(field_value)
                break
    tmp_path.unlink(missing_ok=True)

    print(f"  rebiber: found {len(url_map)} canonical URLs")
    return url_map


def main():
    if len(sys.argv) < 2:
        input_path = Path(__file__).parent / "main.bib"
    else:
        input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = input_path.with_stem(input_path.stem + "_cleaned")

    # Step 1: Run rebiber to get canonical URLs
    print("Running rebiber for URL enrichment...")
    rebiber_urls = run_rebiber(input_path)

    # Step 2: Parse and process entries with our rules
    text = input_path.read_text(encoding="utf-8")
    entries = parse_bib(text)

    processed = []
    for entry in entries:
        p = process_entry(entry)
        # Apply rebiber canonical URLs early so arxiv detection can use them
        if p["cite_key"] in rebiber_urls:
            rebiber_url = rebiber_urls[p["cite_key"]]
            new_fields = []
            replaced = False
            for fname, fval in p["fields"]:
                if fname == "url":
                    new_fields.append(("url", "{" + rebiber_url + "}"))
                    replaced = True
                else:
                    new_fields.append((fname, fval))
            if not replaced:
                new_fields.append(("url", "{" + rebiber_url + "}"))
            p["fields"] = new_fields
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
            if journal and strip_braces(journal).strip().lower() == "arxiv":
                new_fields = [(fn, fv) for fn, fv in entry["fields"] if fn != "journal"]
                processed[i] = {
                    "entry_type": "misc",
                    "cite_key": entry["cite_key"],
                    "fields": new_fields,
                }

    # Step 4: Check if arxiv papers are actually published at a venue
    print("Checking arxiv papers for published venues via Semantic Scholar...")
    processed = check_arxiv_publications(processed)

    # Step 5: Resolve OpenReview entries
    print("Resolving OpenReview entries...")
    processed = resolve_openreview_entries(processed)

    # Step 6: Format entries to strings
    formatted = [format_entry(p) for p in processed]

    # Step 7: For entries still missing a URL, try Semantic Scholar title search
    print("Resolving missing URLs via Semantic Scholar title search...")
    formatted = resolve_missing_urls(formatted)

    output_path.write_text("\n\n".join(formatted) + "\n", encoding="utf-8")
    print(f"Processed {len(entries)} entries -> {output_path}")


def _s2_title_search(title: str, retries: int = 2) -> str | None:
    """Search Semantic Scholar by title and return the best URL, with retry on 429."""
    for attempt in range(1 + retries):
        S2_RATE_LIMITER.wait()
        try:
            params = urllib.parse.urlencode(
                {"query": title, "limit": 1, "fields": "externalIds,url,title"}
            )
            req = urllib.request.Request(
                f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("data"):
                    paper = data["data"][0]
                    if _titles_match(title, paper.get("title", "")):
                        ext = paper.get("externalIds", {})
                        if ext.get("DOI"):
                            doi = ext["DOI"]
                            acl_m = re.match(r"10\.18653/v1/(.+)", doi)
                            if acl_m:
                                return "https://aclanthology.org/" + acl_m.group(1)
                            return "https://doi.org/" + doi
                        if ext.get("ArXiv"):
                            return "https://arxiv.org/abs/" + ext["ArXiv"]
                        if paper.get("url"):
                            return paper["url"]
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(3)
                continue
            return None
        except Exception:
            return None
    return None


def resolve_missing_urls(formatted_entries: list[str]) -> list[str]:
    """
    For entries that still lack a url field, try Semantic Scholar API (title search).
    Uses rate-limited sequential requests.
    """
    # Collect entries needing lookup
    needs_lookup = []  # (index, title)
    result = list(formatted_entries)

    for i, entry_str in enumerate(formatted_entries):
        has_url = bool(re.search(r"^\s+url\s*=", entry_str, re.MULTILINE))
        entry_type_m = re.match(r"@(\w+)\{", entry_str)
        entry_type = entry_type_m.group(1).lower() if entry_type_m else ""
        has_venue = bool(
            re.search(r"^\s+(booktitle|journal)\s*=", entry_str, re.MULTILINE)
        )
        if has_url or (
            not has_venue and entry_type in ("book", "misc", "phdthesis", "techreport")
        ):
            continue
        title_match = re.search(r"title\s*=\s*\{(.+?)\},?\s*$", entry_str, re.MULTILINE)
        if not title_match:
            continue
        title = strip_all_braces(title_match.group(1)).strip()
        needs_lookup.append((i, title))

    print(f"  {len(needs_lookup)} entries need URL lookup...")

    resolved = 0
    for idx, title in needs_lookup:
        found_url = _s2_title_search(title)
        if found_url:
            # Insert url field before the closing }
            entry_str = result[idx].rstrip()
            if entry_str.endswith("}"):
                entry_str = entry_str[:-1].rstrip()
            result[idx] = entry_str + f"\n  url = {{{found_url}}},\n}}"
            resolved += 1
            print(f"  S2: {title[:60]}...")

    print(f"  Resolved: {resolved} via Semantic Scholar")
    return result


def _titles_match(query: str, returned: str) -> bool:
    """Check if two titles match (case-insensitive, ignoring punctuation)."""

    def normalize(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    return normalize(query) == normalize(returned)


if __name__ == "__main__":
    main()
