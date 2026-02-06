# bib_fixer

A rule-based BibTeX cleaner that normalizes formatting, resolves missing URLs, and upgrades arxiv preprints to their published versions.

## Features

- **Field cleanup**: Removes unnecessary fields (abstract, keywords, file, urldate, etc.)
- **Conference acronyms**: Appends standard acronyms (NeurIPS, ICML, ACL, EMNLP, etc.) to `booktitle` for well-known AI/ML/NLP venues
- **Title capitalization**: Capitalizes content words in `booktitle` and protects proper nouns/acronyms in `title` with braces
- **Arxiv normalization**: Detects arxiv papers across four common formats and converts them to a canonical `@misc` format with `eprint`, `archiveprefix`, and `url` fields
- **Arxiv-to-published upgrade**: Checks Semantic Scholar to see if arxiv preprints have been published at a venue, and upgrades them to `@inproceedings`/`@article` entries
- **OpenReview resolution**: Resolves OpenReview URLs to proper venue information via the OpenReview API
- **URL enrichment**: Uses [rebiber](https://github.com/yuchenlin/rebiber) for canonical URLs, then fills remaining gaps with Semantic Scholar title search
- **Rate limiting**: All API calls (Semantic Scholar, OpenReview) are rate-limited to 1 request/second with automatic retry on HTTP 429

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
```

The output file is written alongside the input with a `_cleaned` suffix.

## Pipeline

The script processes entries through these steps in order:

1. **Rebiber** — extract canonical URLs from rebiber's database
2. **Parse & process** — parse BibTeX, remove unwanted fields, fix capitalization, append acronyms, apply rebiber URLs
3. **Normalize arxiv** — convert all arxiv entry variants to canonical `@misc` format
4. **Upgrade arxiv to published** — query Semantic Scholar by arxiv ID; if published, convert to `@inproceedings`/`@article`
5. **Resolve OpenReview** — query OpenReview API for entries with OpenReview URLs; upgrade to proper venue entries
6. **Format** — render entries to BibTeX strings
7. **Resolve missing URLs** — for entries still without a URL, search Semantic Scholar by title

## License

MIT
