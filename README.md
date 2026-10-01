# bib_fixer

A rule-based BibTeX cleaner that normalizes formatting, resolves missing URLs, and upgrades arxiv preprints to their published versions. Everything is deterministic string matching — no LLM touches the bibliography.

## Features

- **Field cleanup**: Removes unnecessary fields (abstract, keywords, file, urldate, etc.). `editor` is kept only for entries without an `author`; `series` tags that just repeat the venue and year (`WWW '22`) are dropped
- **Author clipping**: Clips author lists longer than 25 to the first 25 authors plus "and others"
- **Venue normalization** (never includes a year):
  - Well-known AI/ML/NLP venues are rewritten to a single canonical form, `Full Name (ACRONYM)`, e.g. every ICLR variant becomes `International Conference on Learning Representations (ICLR)`. This covers full names, bare acronyms (`ICLR 2024`), and ACL Anthology DOIs/URLs, which override whatever the `booktitle` says
  - Non-main tracks are kept: `Annual Meeting of the Association for Computational Linguistics: System Demonstrations (ACL)`
  - Other proceedings lose their year and edition number but are otherwise left alone: `Proceedings of the 12th Workshop on Multiword Expressions` → `Proceedings of the Workshop on Multiword Expressions`
  - Book titles (`@incollection` with a non-event `booktitle`) and ordinary journal names keep their years and numbers
- **Title capitalization**: Capitalizes content words in `booktitle` and protects capitalized words in `title` with braces. LaTeX commands, accents and math in titles are preserved
- **Arxiv normalization**: Detects arxiv papers across the common export formats (`eprint`, `journal = {arXiv}` / `{arXiv.org}` / `{CoRR}` / `{arXiv preprint arXiv:...}`, `publisher = {arXiv}`) and converts them to a canonical `@misc` with `eprint`, `archiveprefix`, and `url` fields
- **Arxiv-to-published upgrade**: Searches Crossref by title; if a published version exists, the entry becomes `@inproceedings`/`@article` with the venue, year and URL of the published version. A hit only counts if the title matches, the first author matches, it is a real publication (not a preprint server), and it is not older than the preprint
- **OpenReview resolution**: Resolves OpenReview URLs to the accepted venue (and year) via the OpenReview API
- **URL enrichment**: Uses [rebiber](https://github.com/yuchenlin/rebiber) for canonical URLs, then fills remaining gaps from Crossref (same title + first author/editor + year within one year)
- **Caching**: API results are cached in `<input>.cache.json`. Lookups that found nothing usable are retried after 30 days; use `-f` to ignore the cache
- **Rate limiting**: Crossref at 5 requests/second, OpenReview at 1 request/second, with automatic retry on HTTP 429

## Requirements

- Python 3.12+
- [rebiber](https://github.com/yuchenlin/rebiber) (optional, for canonical URL enrichment)

```bash
pip install rebiber
```

No other external dependencies are needed — the script uses only the Python standard library plus optionally rebiber.

## Usage

```bash
# Clean main.bib (default), outputs main_cleaned.bib
python fix_bib.py

# Clean a specific file
python fix_bib.py path/to/references.bib

# Ignore the API cache and re-query everything
python fix_bib.py -f

# No network: field cleanup, venue/title normalization, rebiber and arxiv normalization only
python fix_bib.py --offline

# Identify yourself to Crossref to get its faster "polite pool"
python fix_bib.py --mailto you@example.com   # or set CROSSREF_MAILTO
```

The output file is written alongside the input with a `_cleaned` suffix; the input file is never modified. `@string` and `@preamble` blocks are copied to the top of the output.

## Pipeline

The script processes entries through these steps in order:

1. **Rebiber** — extract canonical URLs from rebiber's database
2. **Parse & process** — parse BibTeX, remove unwanted fields, clip long author lists, normalize venues, fix capitalization, apply rebiber URLs
3. **Normalize arxiv** — convert all arxiv entry variants to canonical `@misc` format
4. **Upgrade arxiv to published** — search Crossref by title; if a verified published version exists, convert to `@inproceedings`/`@article`
5. **Resolve OpenReview** — query OpenReview API for entries with OpenReview URLs; upgrade to proper venue entries
6. **Resolve missing URLs** — for entries still without a URL, search Crossref by title
7. **Save cache & write** — persist API results and render entries to `<input>_cleaned.bib`

Steps 4–6 are skipped with `--offline`.

## Adding a venue

Venues live in two tables at the top of `fix_bib.py`:

- `CONFERENCE_RULES`: regex → acronym (first match wins)
- `VENUE_NAMES`: acronym → canonical full name. An acronym that has a rule but no entry here only gets `(ACRONYM)` appended instead of being rewritten

## License

MIT
