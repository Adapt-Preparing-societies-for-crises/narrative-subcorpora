# Python API Reference

This document covers every class and function available in the `narrative_subcorpora` package. All examples assume you have already installed the package (`pip install -e .`).

```python
from narrative_subcorpora import Corpus, Subcorpus, Event, ingest
from narrative_subcorpora import selection_report, score_distribution
```


## Event

An event is a historical occurrence with a date and a list of seed terms. Events are defined in a JSON file (see `events.json` for examples).

### Load a single event

```python
from narrative_subcorpora import Event

event = Event.from_json("events.json", "spaanse_griep")

print(event.label)       # "spaanse_griep"
print(event.full_name)   # "Spaanse Griep"
print(event.start_date)  # datetime.date(1918, 7, 1)
print(event.terms)       # ["spaans", "spaanse", "griep", ...]
print(len(event.terms))  # number of seed terms
```

### Load all events

```python
events = Event.load_all("events.json")

for ev in events:
    print(ev.label, ev.full_name, ev.start_date)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `label` | `str` | Short identifier, e.g. `"spaanse_griep"` |
| `full_name` | `str` | Human-readable name, e.g. `"Spaanse Griep"` |
| `start_date` | `datetime.date` | Start date of the event |
| `terms` | `list[str]` | Seed terms associated with the event |


## Ingest

Convert spreadsheet data (CSV, TSV, or Excel) to a parquet file.

```python
from narrative_subcorpora import ingest

ingest(
    "my_data.csv",
    "my_corpus.parquet",
    text_col="ocr",           # name of the text column in your file
    date_col="date",          # name of the date column
    do_clean=True,            # collapse whitespace, strip text
    language="nl",            # language hint (for future NLP cleaning)
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `str` or `Path` | required | Input file path (`.csv`, `.tsv`, `.xls`, `.xlsx`) |
| `output` | `str` or `Path` | required | Output parquet file path |
| `text_col` | `str` | `"text"` | Name of the column containing text |
| `date_col` | `str` | `"date"` | Name of the column containing dates |
| `do_clean` | `bool` | `False` | Apply basic text cleaning |
| `language` | `str` | `"nl"` | Language hint for cleaning |

Returns the output `Path`.


## Corpus

A parquet-backed text corpus. This is your starting point for querying.

### Create a Corpus

```python
from narrative_subcorpora import Corpus

corpus = Corpus(
    "my_corpus.parquet",
    text_col="ocr",    # name of the text column in your parquet file
    date_col="date",   # name of the date column
)
```

If the column names you specify do not exist in the file, you get a clear error listing the available columns:

```
KeyError: "text_col='text' not found. Available columns: ['date', 'ocr', 'source', 'page']"
```

### Time-window methods

All time-window methods return a `Subcorpus` object.

#### `.after(event, months=6)`

Select texts from the event start date up to N months later.

```python
sub = corpus.after(event, months=6)
```

#### `.before(event, months=1)`

Select texts from N months before the event up to the event start date.

```python
sub = corpus.before(event, months=3)
```

#### `.around(event, months_before=1, months_after=6)`

Select texts in a window around the event.

```python
sub = corpus.around(event, months_before=2, months_after=12)
```

#### `.between(start, end)`

Select texts between two dates. Accepts date strings or `datetime.date` objects.

```python
sub = corpus.between("1918-01-01", "1919-12-31")
```

### Describe a corpus

```python
info = corpus.describe()
print(info)
# {
#     "path": "my_corpus.parquet",
#     "num_rows": 150000,
#     "num_columns": 5,
#     "columns": [
#         {"name": "date", "type": "timestamp[ns]"},
#         {"name": "ocr", "type": "string"},
#         ...
#     ]
# }
```


## Subcorpus

A filtered slice of a corpus. Returned by time-window methods on `Corpus`. Supports chained scoring, filtering, and export. All scoring methods return the same `Subcorpus` object, so you can chain them.

### Scoring methods

Every scoring method adds a new column to the data and returns `self` so you can keep chaining. All metadata columns from the original parquet file are preserved throughout.

#### `.score(terms, freq_weight=1.0, density_weight=0.0, col="score")`

The default scoring method. Computes a weighted average of term coverage and term density.

- **Term coverage**: fraction of seed terms found at least once (0 to 1).
- **Term density**: total seed-term hits divided by word count.

With default weights, this is pure term coverage.

```python
sub = corpus.after(event, months=6).score(terms=event.terms)

# Mix coverage and density equally
sub = corpus.after(event, months=6).score(
    terms=event.terms,
    freq_weight=1.0,
    density_weight=1.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `terms` | `list[str]` | required | Seed terms to search for |
| `freq_weight` | `float` | `1.0` | Weight for term coverage component |
| `density_weight` | `float` | `0.0` | Weight for term density component |
| `col` | `str` | `"score"` | Name of the output column |

#### `.score_tfidf(terms, col="score_tfidf")`

TF-IDF scoring. Rewards terms that appear frequently in a text but are rare across the subcorpus. Automatically computes document frequencies from the current subcorpus.

```python
sub = corpus.after(event, months=6).score_tfidf(terms=event.terms)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `terms` | `list[str]` | required | Seed terms |
| `col` | `str` | `"score_tfidf"` | Name of the output column |

#### `.score_bm25(terms, k1=1.5, b=0.75, col="score_bm25")`

BM25 ranking. The standard search-engine formula. Like TF-IDF but with length normalisation — long texts do not automatically score higher.

```python
sub = corpus.after(event, months=6).score_bm25(terms=event.terms)

# Tune parameters
sub = corpus.after(event, months=6).score_bm25(
    terms=event.terms,
    k1=1.2,   # lower = term frequency saturates faster
    b=0.5,    # lower = less length normalisation
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `terms` | `list[str]` | required | Seed terms |
| `k1` | `float` | `1.5` | Term-frequency saturation (higher = slower saturation) |
| `b` | `float` | `0.75` | Length normalisation (0 = none, 1 = full) |
| `col` | `str` | `"score_bm25"` | Name of the output column |

#### `.score_cluster(terms, window=50, col="score_cluster")`

Proximity-based scoring. Slides a window across the text and finds the window with the highest concentration of distinct seed terms. Returns that count divided by the total number of seed terms (0 to 1).

Good for finding focused passages about the event inside long texts.

```python
sub = corpus.after(event, months=6).score_cluster(terms=event.terms)

# Smaller window = stricter proximity requirement
sub = corpus.after(event, months=6).score_cluster(terms=event.terms, window=20)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `terms` | `list[str]` | required | Seed terms |
| `window` | `int` | `50` | Window size in words |
| `col` | `str` | `"score_cluster"` | Name of the output column |

#### `.score_weighted(term_weights, col="score_weighted")`

Score using per-term importance weights. Assign higher weights to rare or highly diagnostic terms and lower weights to common ones.

```python
weights = {
    "spaanse": 2.0,    # very specific to the event
    "griep": 2.0,
    "epidemie": 1.5,
    "koorts": 0.5,     # common, could appear in other contexts
    "ziekte": 0.5,
}
sub = corpus.after(event, months=6).score_weighted(weights)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `term_weights` | `dict[str, float]` | required | Term-to-weight mapping |
| `col` | `str` | `"score_weighted"` | Name of the output column |

### Combining multiple scores

You can chain multiple scoring methods. Each adds its own column.

```python
sub = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)                     # -> "score"
    .score_tfidf(terms=event.terms)               # -> "score_tfidf"
    .score_bm25(terms=event.terms)                # -> "score_bm25"
    .score_cluster(terms=event.terms, window=30)  # -> "score_cluster"
    .score_weighted(my_weights)                   # -> "score_weighted"
)

df = sub.to_dataframe()
# df now has all original columns + all five score columns
```

### Filtering

#### `.above(threshold, col="score")`

Keep only rows where a score column is at or above a threshold.

```python
# Filter on the default "score" column
sub = sub.above(0.1)

# Filter on a specific score column
sub = sub.above(5.0, col="score_bm25")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | required | Minimum score to keep |
| `col` | `str` | `"score"` | Which score column to filter on |

### Export

All export methods preserve every column — your original metadata, the text, dates, and any score columns you added.

#### `.to_dataframe()`

Return the data as a pandas DataFrame.

```python
df = sub.to_dataframe()
```

#### `.to_parquet(path)`

Write to a parquet file. Good for further programmatic processing.

```python
sub.to_parquet("output.parquet")
```

#### `.to_csv(path)`

Write to a CSV file. Good for opening in Excel or Google Sheets.

```python
sub.to_csv("output.csv")
```

### Info

```python
len(sub)    # number of texts
print(sub)  # "Subcorpus(1234 texts)"
```


## Diagnostics

Two visualisation functions for inspecting your subcorpus selection.

### Selection report

Produces a two-panel figure:
1. **Daily bar chart** — selected vs excluded articles per day since the event.
2. **Cumulative curve** — total articles in window vs selected over time.

```python
from narrative_subcorpora import Corpus, Event, selection_report

corpus = Corpus("my_corpus.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "spaanse_griep")

subcorpus = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)
    .above(0.1)
)

# Show interactively
fig = selection_report(corpus, event, subcorpus, months=6)
fig.show()

# Or save to a file
selection_report(corpus, event, subcorpus, months=6, output="report.png")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `corpus` | `Corpus` | required | The full corpus |
| `event` | `Event` | required | The event used for windowing |
| `subcorpus` | `Subcorpus` | required | The filtered subcorpus |
| `months` | `int` | `6` | Window size (must match what you used for `.after()`) |
| `output` | `str`, `Path`, or `None` | `None` | Save path (`.png`, `.pdf`, `.svg`). `None` = don't save |

Returns a `matplotlib.figure.Figure`.

### Score distribution

Histogram of all score values with an optional threshold line.

```python
from narrative_subcorpora import Corpus, Event, score_distribution

corpus = Corpus("my_corpus.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "spaanse_griep")

scored = corpus.after(event, months=6).score(terms=event.terms)

# Show distribution with threshold line
fig = score_distribution(scored, threshold=0.1)
fig.show()

# For a different score column
scored = scored.score_bm25(terms=event.terms)
fig = score_distribution(scored, score_col="score_bm25", threshold=5.0)
fig.show()

# Save to file
score_distribution(scored, output="scores.png")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `subcorpus` | `Subcorpus` | required | A subcorpus with a score column |
| `score_col` | `str` | `"score"` | Which score column to plot |
| `bins` | `int` | `40` | Number of histogram bins |
| `threshold` | `float` or `None` | `None` | Draw a vertical line at this value |
| `output` | `str`, `Path`, or `None` | `None` | Save path. `None` = don't save |

Returns a `matplotlib.figure.Figure`.


## Low-level scoring functions

These are the individual scoring functions used internally by `Subcorpus`. You can call them directly on individual texts if needed.

```python
from narrative_subcorpora.score import (
    term_frequency_score,
    term_density_score,
    combined_score,
    tfidf_score,
    bm25_score,
    build_doc_frequencies,
    term_cluster_score,
    weighted_term_score,
)
```

### `term_frequency_score(text, terms)`

Fraction of unique seed terms found in the text. Returns 0 to 1.

```python
term_frequency_score("De griep epidemie was hevig", ["griep", "epidemie", "koorts"])
# 0.667 — 2 out of 3 terms found
```

### `term_density_score(text, terms)`

Total seed-term hits divided by word count.

```python
term_density_score("De griep epidemie was hevig", ["griep", "epidemie"])
# 0.4 — 2 hits in 5 words
```

### `combined_score(text, terms, freq_weight=1.0, density_weight=0.0)`

Weighted average of frequency and density scores.

### `tfidf_score(text, terms, doc_frequencies, n_docs)`

TF-IDF score for a single text. Requires pre-computed document frequencies.

```python
texts = ["De griep epidemie was hevig", "Het weer is mooi", "Koorts en griep"]
terms = ["griep", "epidemie", "koorts"]
doc_freq = build_doc_frequencies(texts, terms)

tfidf_score(texts[0], terms, doc_freq, len(texts))
```

### `build_doc_frequencies(texts, terms)`

Count how many documents each term appears in. Returns a `dict[str, int]`. Use this as input for `tfidf_score` and `bm25_score`.

### `bm25_score(text, terms, doc_frequencies, n_docs, avgdl, k1=1.5, b=0.75)`

BM25 score for a single text. Requires pre-computed document frequencies and average document length.

```python
texts = ["De griep epidemie was hevig", "Het weer is mooi", "Koorts en griep"]
terms = ["griep", "epidemie", "koorts"]
doc_freq = build_doc_frequencies(texts, terms)
avgdl = sum(len(t.split()) for t in texts) / len(texts)

bm25_score(texts[0], terms, doc_freq, len(texts), avgdl)
```

### `term_cluster_score(text, terms, window=50)`

Maximum concentration of distinct seed terms in any single word-window. Returns 0 to 1.

### `weighted_term_score(text, term_weights)`

Weighted coverage score. `term_weights` is a `dict[str, float]` mapping terms to importance weights.


## Complete examples

### Basic workflow

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "spaanse_griep")

subcorpus = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)
    .above(0.1)
)

subcorpus.to_csv("spaanse_griep_subcorpus.csv")
print(f"Exported {len(subcorpus)} articles")
```

### Compare scoring methods

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "spaanse_griep")

sub = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)
    .score_tfidf(terms=event.terms)
    .score_bm25(terms=event.terms)
    .score_cluster(terms=event.terms)
)

df = sub.to_dataframe()
score_cols = ["score", "score_tfidf", "score_bm25", "score_cluster"]
print(df[score_cols].describe())
```

### Diagnostics workflow

```python
from narrative_subcorpora import Corpus, Event, selection_report, score_distribution

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "spaanse_griep")

# Score without filtering first
scored = corpus.after(event, months=6).score(terms=event.terms)

# Look at the score distribution to choose a threshold
score_distribution(scored, threshold=0.1, output="score_dist.png")

# Apply the threshold
filtered = scored.above(0.1)

# See how many articles were kept per day
selection_report(corpus, event, filtered, months=6, output="selection.png")

print(f"Kept {len(filtered)} of {len(scored)} articles")
```

### Multiple events

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
events = Event.load_all("events.json")

for event in events:
    sub = (
        corpus
        .after(event, months=6)
        .score(terms=event.terms)
        .above(0.1)
    )
    sub.to_parquet(f"{event.label}_subcorpus.parquet")
    print(f"{event.label}: {len(sub)} articles")
```
