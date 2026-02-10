# bib_fixer

A rule-based BibTeX cleaner that normalizes formatting, resolves missing URLs, and upgrades arxiv preprints to their published versions.

## Features

- **Field cleanup**: Removes unnecessary fields (abstract, keywords, file, urldate, etc.) and drops `publisher`, `series`, and `isbn` from `@inproceedings` entries
- **Author clipping**: Clips author lists longer than 25 to the first 25 authors plus "and others"
- **Booktitle cleaning**: Strips "Proceedings of the", ordinals (numeric and spelled-out), standalone years, and colon-separated suffixes from booktitles
- **Conference acronyms**: Appends standard acronyms (NeurIPS, ICML, ICLR, ACL, EMNLP, COLM, TMLR, etc.) to `booktitle`/`journal` for well-known AI/ML/NLP venues
- **Title capitalization**: Capitalizes content words in `booktitle` and protects proper nouns/acronyms in `title` with braces
- **Arxiv normalization**: Detects arxiv papers across four common formats and converts them to a canonical `@misc` format with `eprint`, `archiveprefix`, and `url` fields
- **Arxiv-to-published upgrade**: Checks Semantic Scholar to see if arxiv preprints have been published at a venue, and upgrades them to `@inproceedings`/`@article` entries
- **OpenReview resolution**: Resolves OpenReview URLs to proper venue information via the OpenReview API
- **URL enrichment**: Uses [rebiber](https://github.com/yuchenlin/rebiber) for canonical URLs, then fills remaining gaps with Semantic Scholar title search
- **Incremental processing**: Caches results from previous runs; only new/changed entries are re-processed (use `-f` to force a full re-run)
- **Changelog**: Writes a markdown changelog (`<input>_updates.md`) summarizing venue upgrades and added URLs
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
# Clean main.bib (default), outputs main_cleaned.bib + main_updates.md
python fix_bib.py

# Clean a specific file
python fix_bib.py path/to/references.bib

# Force re-processing of all entries (ignore cache)
python fix_bib.py -f
```

The output file is written alongside the input with a `_cleaned` suffix. A changelog is written with a `_updates.md` suffix.

## Pipeline

The script processes entries through these steps in order:

1. **Rebiber** — extract canonical URLs from rebiber's database
2. **Cache check** — load already-processed entries from a previous output file; skip entries that haven't changed
3. **Parse & process** — parse BibTeX, remove unwanted fields, clip long author lists, clean booktitles, fix capitalization, append acronyms, apply rebiber URLs
4. **Normalize arxiv** — convert all arxiv entry variants to canonical `@misc` format
5. **Upgrade arxiv to published** — query Semantic Scholar by arxiv ID; if published, convert to `@inproceedings`/`@article`
6. **Resolve OpenReview** — query OpenReview API for entries with OpenReview URLs; upgrade to proper venue entries
7. **Format** — render entries to BibTeX strings
8. **Resolve missing URLs** — for entries still without a URL, search Semantic Scholar by title
9. **Write changelog** — output a markdown summary of all venue upgrades and URL additions

## License

MIT
