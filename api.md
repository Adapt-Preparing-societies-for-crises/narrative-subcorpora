# Python API Reference

This document covers every class and function available in the `narrative_subcorpora` package. All examples assume you have already installed the package (`pip install -e .`).

```python
from narrative_subcorpora import Corpus, Subcorpus, Event, ingest
from narrative_subcorpora import selection_report, score_distribution, group_score_distribution
```


## Event

An event is a historical occurrence with a date and a list of seed terms. Events are defined in a JSON file (see `events.json` for examples).

### Load a single event

```python
from narrative_subcorpora import Event

event = Event.from_json("events.json", "watersnood")

print(event.label)        # "watersnood"
print(event.full_name)    # "Watersnoodramp 1953"
print(event.start_date)   # datetime.date(1953, 2, 1)
print(event.terms)        # ["watersnoodramp", "watersnood", ...]
print(event.term_groups)  # {"location": [...], "event_type": [...], ...}
print(len(event.terms))   # number of seed terms
```

### Load all events

```python
events = Event.load_all("events.json")

for ev in events:
    print(ev.label, ev.full_name, ev.start_date, list(ev.term_groups))
```

### Properties

| Property | Type | Description |
|---|---|---|
| `label` | `str` | Short identifier, e.g. `"watersnood"` |
| `full_name` | `str` | Human-readable name, e.g. `"Watersnoodramp 1953"` |
| `start_date` | `datetime.date` | Start date of the event |
| `terms` | `list[str]` | Flat list of all seed terms (for backward-compatible scoring) |
| `term_groups` | `dict[str, list[str]]` | Named groups of seed terms for grouped scoring. Empty dict if not defined. |

### JSON format

A minimal event entry with term groups:

```json
{
  "label": "watersnood",
  "full name": "Watersnoodramp 1953",
  "start_date": "01-02-1953",
  "terms": ["zeeland", "overstroming", "dijkbreuk", "slachtoffers", ...],
  "term_groups": {
    "location":   ["zeeland", "walcheren", "tholen", "beveland"],
    "event_type": ["overstroming", "dijkbreuk", "stormvloed", "watersnood"],
    "cause":      ["springtij", "noordwesterstorm", "springvloed"],
    "impact":     ["slachtoffers", "doden", "evacuatie", "noodhulp"]
  }
}
```

The `terms` field is still required for backward compatibility. `term_groups` is optional; events without it still work with all scoring methods except `score_grouped`.


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
sub = corpus.between("1953-01-01", "1953-12-31")
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

#### `.score_grouped(term_groups, weights=None, combine="geometric", col="score_grouped")`

Score texts using named term groups combined into a single score. This is the smartest filtering approach when you have terms of different semantic types (e.g. location names, event type words, impact words).

For each group, a term-coverage score (0 to 1) is computed independently. Per-group scores are written to columns named `score_{group_name}`. The groups are then combined into a single score using the chosen strategy.

```python
# Using groups from the event definition
event = Event.from_json("events.json", "watersnood")
sub = (
    corpus
    .after(event, months=6)
    .score_grouped(event.term_groups)   # writes score_location, score_event_type, ...
    .above(0.0, col="score_grouped")    # keep only texts with evidence in every group
)

# Inline group definition
groups = {
    "location":   ["zeeland", "walcheren", "tholen"],
    "event_type": ["overstroming", "dijkbreuk", "stormvloed"],
}
sub = corpus.after(event, months=6).score_grouped(groups)

# Custom weights with weighted_sum strategy
sub = corpus.after(event, months=6).score_grouped(
    event.term_groups,
    weights={"location": 1.0, "event_type": 2.0, "impact": 1.5},
    combine="weighted_sum",
)

# Custom combination function
sub = corpus.after(event, months=6).score_grouped(
    event.term_groups,
    combine=lambda scores: scores.get("location", 0) * scores.get("event_type", 0),
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `term_groups` | `dict[str, list[str]]` | required | Named groups of seed terms |
| `weights` | `dict[str, float]` or `None` | `None` | Per-group weights for `"weighted_sum"` strategy |
| `combine` | `str` or callable | `"geometric"` | Combination strategy (see table below) |
| `col` | `str` | `"score_grouped"` | Name of the combined output column |

**Combination strategies:**

| Strategy | Formula | Semantics |
|---|---|---|
| `"geometric"` *(default)* | geometric mean | Requires evidence in every group; zero in any group → zero combined |
| `"weighted_sum"` | weighted average | Strong groups compensate for weak ones |
| `"min"` | minimum across groups | Combined score equals the weakest group |
| `"product"` | product of all scores | Harsher than geometric; degrades quickly with many groups |
| callable | `f(dict) -> float` | Fully custom logic |

**Output columns written:**

- `score_{group_name}` for each group in `term_groups`
- `score_grouped` (or the value of `col`) for the combined score

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
    .score_grouped(event.term_groups)             # -> "score_grouped", "score_location", ...
)

df = sub.to_dataframe()
# df now has all original columns + all score columns
```

### Filtering

#### `.above(threshold, col="score")`

Keep only rows where a score column is at or above a threshold.

```python
# Filter on the default "score" column
sub = sub.above(0.1)

# Filter on the grouped score
sub = sub.above(0.05, col="score_grouped")

# Filter on a specific group
sub = sub.above(0.0, col="score_location")  # must mention at least one location term
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | required | Minimum score to keep |
| `col` | `str` | `"score"` | Which score column to filter on |

#### `.below(threshold, col="score_outlier")`

Keep only rows where a score column is at or below a threshold. The complement of `.above()` — useful for removing outliers.

```python
# Remove texts with high outlier scores
sub = sub.below(0.5, col="score_outlier")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | required | Maximum score to keep |
| `col` | `str` | `"score_outlier"` | Which score column to filter on |

### Embedding-based scoring

These methods use sentence-transformers to compute dense vector representations of your texts and score them based on semantic content rather than keyword matching. Requires the optional `embeddings` dependency:

```bash
pip install narrative-subcorpora[embeddings]
```

#### `.embed(model="all-MiniLM-L6-v2", batch_size=64, show_progress=True)`

Compute embeddings for all texts in the subcorpus. This must be called before `.score_outlier()` or `.score_similarity()`. The embeddings are stored internally on the subcorpus object.

```python
sub = corpus.after(event, months=6).embed("all-MiniLM-L6-v2")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformers model name |
| `batch_size` | `int` | `64` | Batch size for encoding |
| `show_progress` | `bool` | `True` | Show a progress bar |

Some recommended models:

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | Fast | Good | Good default, English-focused |
| `paraphrase-multilingual-MiniLM-L12-v2` | Medium | Good | Multilingual, good for Dutch |
| `all-mpnet-base-v2` | Slower | Best | Highest quality, English-focused |

#### `.score_outlier(method="centroid", k=5, col="score_outlier")`

Add an outlier score column. Higher scores mean the text is more unlike the rest of the subcorpus. Call `.embed()` first.

Two methods are available:

- **centroid** — cosine distance from the subcorpus centroid (the "average" text). Fast and simple. Finds texts that are globally different from the majority.
- **knn** — average cosine distance to the *k* nearest neighbours. Finds texts in sparse regions, even if they are not far from the centroid.

```python
# Centroid method (default)
sub = sub.score_outlier()

# kNN method
sub = sub.score_outlier(method="knn", k=10, col="score_outlier_knn")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"centroid"` | `"centroid"` or `"knn"` |
| `k` | `int` | `5` | Number of neighbours (knn method only) |
| `col` | `str` | `"score_outlier"` | Name of the output column |

#### `.score_similarity(terms, model=None, batch_size=64, col="score_similarity")`

Score each text by its embedding similarity to the seed terms. Embeds the seed terms, averages them into a single vector, and computes cosine similarity to each text. Call `.embed()` first.

Values range from -1 to 1, where 1 means the text is semantically very similar to the seed terms.

```python
sub = sub.score_similarity(terms=event.terms)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `terms` | `list[str]` | required | Seed terms to compare against |
| `model` | `str` or `None` | `None` | Model for embedding the terms. If `None`, reuses the model from `.embed()` |
| `batch_size` | `int` | `64` | Batch size for encoding terms |
| `col` | `str` | `"score_similarity"` | Name of the output column |

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

Three visualisation functions for inspecting your subcorpus selection.

### Selection report

Produces a two-panel figure:
1. **Daily bar chart** — selected vs excluded articles per day since the event.
2. **Cumulative curve** — total articles in window vs selected over time.

```python
from narrative_subcorpora import Corpus, Event, selection_report

corpus = Corpus("my_corpus.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

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
event = Event.from_json("events.json", "watersnood")

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

### Group score distribution *(new)*

Side-by-side histograms of per-group scores and the combined score. Call `.score_grouped()` first.

```python
from narrative_subcorpora import Corpus, Event, group_score_distribution

corpus = Corpus("my_corpus.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

scored = corpus.after(event, months=6).score_grouped(event.term_groups)

fig = group_score_distribution(
    scored,
    group_cols=["score_location", "score_event_type", "score_cause", "score_impact"],
    threshold=0.05,
)
fig.show()

# Auto-detect group columns (finds all score_ columns not in the known global list)
fig = group_score_distribution(scored)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `subcorpus` | `Subcorpus` | required | A subcorpus scored with `.score_grouped()` |
| `group_cols` | `list[str]` or `None` | `None` | Per-group column names to show. Auto-detected if `None`. |
| `combined_col` | `str` | `"score_grouped"` | Name of the combined score column |
| `bins` | `int` | `30` | Number of histogram bins per panel |
| `threshold` | `float` or `None` | `None` | Threshold line on the combined-score panel |
| `output` | `str`, `Path`, or `None` | `None` | Save path. `None` = don't save |

Returns a `matplotlib.figure.Figure`.


## CLI diagnostics

You can also run diagnostics from the command line without writing Python.

### Save a selection report to a file

```bash
nsc diagnose --corpus my_corpus.parquet --events events.json \
    --event spaanse_griep --window 6m --min-score 0.1 \
    --text-col ocr -o report.png
```

### Display interactively

```bash
nsc diagnose --corpus my_corpus.parquet --events events.json \
    --event spaanse_griep --window 6m --min-score 0.1 \
    --text-col ocr
```

If no interactive display is available (e.g. on a server), the figure is saved to `diagnose_report.png` automatically.

### Options

| Option | Default | Description |
|---|---|---|
| `--corpus` | required | Path to the parquet corpus file |
| `--events` | required | Path to the events JSON file |
| `--event` | required | Event label (e.g. `spaanse_griep`) |
| `--window` | `6m` | Time window after event (e.g. `6m` for 6 months, `12m` for a year) |
| `--min-score` | `0.0` | Minimum score. Articles below this are counted as "excluded" |
| `--text-col` | `text` | Name of the text column in your parquet file |
| `--date-col` | `date` | Name of the date column |
| `-o` | none | Output file path (`.png`, `.pdf`, `.svg`). Omit to display interactively |


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
    group_term_scores,
    combine_group_scores,
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

### `term_cluster_score(text, terms, window=50)`

Maximum concentration of distinct seed terms in any single word-window. Returns 0 to 1.

### `weighted_term_score(text, term_weights)`

Weighted coverage score. `term_weights` is a `dict[str, float]` mapping terms to importance weights.

### `group_term_scores(text, term_groups)` *(new)*

Compute term-coverage scores for each named group independently.

```python
text = "De watersnood in Zeeland was een nationale ramp. Dijkbreuk bij Walcheren."
groups = {
    "location":   ["zeeland", "walcheren", "tholen"],
    "event_type": ["watersnood", "dijkbreuk", "overstroming"],
}
scores = group_term_scores(text, groups)
# {"location": 0.667, "event_type": 0.667}
```

Returns `dict[str, float]` with one entry per group.

### `combine_group_scores(scores, weights=None, combine="geometric")` *(new)*

Combine a dict of per-group scores into a single value.

```python
from narrative_subcorpora.score import combine_group_scores

scores = {"location": 0.5, "event_type": 0.8, "impact": 0.2}

combine_group_scores(scores)                          # geometric: ~0.415
combine_group_scores(scores, combine="weighted_sum")  # equal-weight mean: 0.5
combine_group_scores(scores, combine="min")           # 0.2
combine_group_scores(scores, combine="product")       # 0.08

# Custom weights with weighted_sum
combine_group_scores(
    scores,
    weights={"location": 1.0, "event_type": 2.0, "impact": 1.5},
    combine="weighted_sum",
)  # ~0.578

# Custom function
combine_group_scores(scores, combine=lambda s: s["location"] * s["event_type"])
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scores` | `dict[str, float]` | required | Per-group scores |
| `weights` | `dict[str, float]` or `None` | `None` | Per-group weights (used only by `"weighted_sum"`) |
| `combine` | `str` or callable | `"geometric"` | Combination strategy |


## Low-level embedding functions

These are the building blocks used internally by `Subcorpus.embed()`, `.score_outlier()`, and `.score_similarity()`. You can use them directly for custom workflows.

```python
from narrative_subcorpora.embed import (
    embed_texts,
    centroid_distance_scores,
    knn_distance_scores,
    seed_similarity_scores,
    outlier_scores,
)
```

### `embed_texts(texts, model, batch_size=64, show_progress=True)`

Encode a list of texts into an embedding matrix. Returns a numpy array of shape `(n, embedding_dim)`.

### `centroid_distance_scores(embeddings)`

Cosine distance from each embedding to the centroid. Returns values in [0, 2].

### `knn_distance_scores(embeddings, k=5)`

Average cosine distance to the *k* nearest neighbours. Returns values in [0, 2].

### `seed_similarity_scores(embeddings, seed_embedding)`

Cosine similarity between each text embedding and a seed embedding. Returns values in [-1, 1].

### `outlier_scores(texts, model, method="centroid", k=5, ...)`

Convenience function: embed texts and compute outlier scores in one call.


## Complete examples

### Basic workflow

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

subcorpus = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)
    .above(0.1)
)

subcorpus.to_csv("watersnood_subcorpus.csv")
print(f"Exported {len(subcorpus)} articles")
```

### Grouped scoring workflow

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

subcorpus = (
    corpus
    .after(event, months=6)
    .score_grouped(event.term_groups)       # score per group + combined
    .above(0.0, col="score_grouped")        # keep texts with evidence in all groups
)

df = subcorpus.to_dataframe()
score_cols = [c for c in df.columns if c.startswith("score_")]
print(df[score_cols].describe())
```

### Compare combination strategies

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

window = corpus.after(event, months=6)

# Compute grouped scores under four different strategies
for strategy in ["geometric", "weighted_sum", "min", "product"]:
    sub = window.score_grouped(
        event.term_groups,
        combine=strategy,
        col=f"score_{strategy}",
    )

df = sub.to_dataframe()
strategy_cols = [f"score_{s}" for s in ["geometric", "weighted_sum", "min", "product"]]
print(df[strategy_cols].describe())
```

### Compare scoring methods

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

sub = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)
    .score_tfidf(terms=event.terms)
    .score_bm25(terms=event.terms)
    .score_cluster(terms=event.terms)
    .score_grouped(event.term_groups)
)

df = sub.to_dataframe()
score_cols = ["score", "score_tfidf", "score_bm25", "score_cluster", "score_grouped"]
print(df[score_cols].describe())
```

### Diagnostics workflow

```python
from narrative_subcorpora import (
    Corpus, Event, selection_report,
    score_distribution, group_score_distribution,
)

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

# Score without filtering first
scored = corpus.after(event, months=6).score_grouped(event.term_groups)

# Per-group score distributions — see which group is the binding constraint
group_score_distribution(
    scored,
    group_cols=["score_location", "score_event_type", "score_cause", "score_impact"],
    threshold=0.05,
    output="group_distributions.png",
)

# Standard distribution of the combined score
score_distribution(scored, score_col="score_grouped", threshold=0.05,
                   output="combined_dist.png")

# Apply threshold and inspect temporal selection
filtered = scored.above(0.05, col="score_grouped")
selection_report(corpus, event, filtered, months=6, output="selection.png")

print(f"Kept {len(filtered)} of {len(scored)} articles")
```

### Multiple events

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
events = Event.load_all("events.json")

for event in events:
    if event.term_groups:
        sub = (
            corpus
            .after(event, months=6)
            .score_grouped(event.term_groups)
            .above(0.0, col="score_grouped")
        )
    else:
        sub = (
            corpus
            .after(event, months=6)
            .score(terms=event.terms)
            .above(0.1)
        )
    sub.to_parquet(f"{event.label}_subcorpus.parquet")
    print(f"{event.label}: {len(sub)} articles")
```

### Embedding-based outlier removal

Use embeddings to find and remove articles that passed keyword filtering but are semantically off-topic.

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

sub = (
    corpus
    .after(event, months=6)
    .score_grouped(event.term_groups)             # grouped keyword filter
    .above(0.0, col="score_grouped")              # must have evidence in every group
    .embed("paraphrase-multilingual-MiniLM-L12-v2")
    .score_outlier(method="centroid")
    .below(0.5, col="score_outlier")              # remove semantic outliers
)

sub.to_csv("clean_subcorpus.csv")
print(f"Kept {len(sub)} articles after grouped + outlier filtering")
```

### Combine keyword and semantic scoring

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("newspapers.parquet", text_col="ocr", date_col="date")
event = Event.from_json("events.json", "watersnood")

sub = (
    corpus
    .after(event, months=6)
    .score_grouped(event.term_groups)             # grouped keyword coverage
    .above(0.0, col="score_grouped")              # loose grouped filter first
    .embed("paraphrase-multilingual-MiniLM-L12-v2")
    .score_similarity(terms=event.terms)          # semantic similarity to seed terms
    .score_outlier()                              # outlier detection
)

df = sub.to_dataframe()

# Sort by semantic similarity — most relevant first
df = df.sort_values("score_similarity", ascending=False)
print(df[["date", "score_grouped", "score_similarity", "score_outlier"]].head(20))
```


---

## ActiveLearner

`ActiveLearner` wraps a `Subcorpus` and iteratively refines document selection by asking the researcher to label a small batch of documents as *relevant* or *not relevant*. A logistic regression classifier is retrained on those labels and used to rank remaining documents by uncertainty, so each annotation round gives the most information per label.

Requires the `active-learning` optional extras:

```
pip install narrative-subcorpora[active-learning]
```

### Import

```python
from narrative_subcorpora import ActiveLearner
```

### Constructor

```python
ActiveLearner(
    subcorpus,
    *,
    features="scores",      # "scores" | "tfidf" | "embeddings"
    seed_terms=None,        # restrict tfidf vocabulary to these terms
    cold_start_col=None,    # column to use for cold-start ranking
    score_col="score_al",   # output column name
    random_state=42,
)
```

| Parameter | Description |
|---|---|
| `subcorpus` | A `Subcorpus` object (the candidate pool) |
| `features` | How to build the feature matrix (see below) |
| `seed_terms` | If `features="tfidf"`, restrict vocabulary to these terms |
| `cold_start_col` | Score column to use for initial ranking (auto-detected if None) |
| `score_col` | Name of the output column written by `to_subcorpus()` |
| `random_state` | Random seed for reproducibility |

### Feature modes

| `features=` | Description |
|---|---|
| `"scores"` *(default)* | Uses existing score columns on the subcorpus. Fast; requires no extra computation. |
| `"tfidf"` | Builds a TF-IDF matrix over the raw text. Slower; works without pre-computed scores. |
| `"embeddings"` | Uses the embedding matrix stored by `.embed()`. Highest quality; call `.embed()` first. |

### Methods

#### `.label(idx, relevant)` → `ActiveLearner`

Label a single document.

```python
al.label(0, True)   # row 0 is relevant
al.label(7, False)  # row 7 is not relevant
```

#### `.label_batch(labels)` → `ActiveLearner`

Label multiple documents at once.

```python
al.label_batch({0: True, 1: False, 3: True, 5: False})
```

#### `.retrain()` → `ActiveLearner`

Fit the logistic regression classifier on current labels and update all document scores.
Requires at least one positive and one negative label.

```python
al.retrain()
```

#### `.next_batch(n=10)` → `list[int]`

Return the indices of the next *n* documents to label.

- **Cold start** (before training): returns top-scored + random mix.
- **After training**: returns the documents with the highest uncertainty (closest to 0.5 probability).

```python
batch = al.next_batch(n=5)
for idx in batch:
    print(idx, al._df.at[idx, "ocr"][:200])
```

#### `.annotate(n=10, *, auto_retrain=True)` → `None`

Interactive annotation session. Shows ipywidgets cards in Jupyter (Relevant / Not relevant / Skip / Stop buttons); falls back to `input()` prompts in a terminal.

```python
al.annotate(n=10)          # label 10 documents interactively
al.annotate(n=20)          # continue with 20 more
```

When `auto_retrain=True` (the default), the classifier is retrained automatically after each batch, provided there is at least one label of each class.

#### `.status()` → `None`

Print a summary of labelling progress, rank stability (Spearman correlation between previous and current scores), and cross-validated F1.

```python
al.status()
# Labelled: 20 / 4352  (+12 relevant, -8 irrelevant)
# Rank stability (Spearman vs previous fit): 0.912
# CV F1 (k=3): 0.731 ± 0.045
```

#### `.to_subcorpus()` → `Subcorpus`

Return a `Subcorpus` with a `score_al` column containing classifier probabilities (or cold-start scores if the classifier has not been trained).

```python
result = al.to_subcorpus()
result.to_csv("al_ranked.csv")
```

### Complete example

```python
from narrative_subcorpora import Corpus, Event, ActiveLearner

corpus = Corpus("news.parquet", text_col="ocr", date_col="date")
event  = Event.from_json("events.json", "watersnood")

# Build candidate pool
sub = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)
    .score_grouped(event.term_groups)
)

# Create learner using existing score columns as features
al = ActiveLearner(sub, features="scores")

# Interactive annotation (ipywidgets in Jupyter, text in terminal)
al.annotate(n=15)

# Check model quality
al.status()

# Export with al scores — sort by score_al descending
result = al.to_subcorpus()
df = result.to_dataframe().sort_values("score_al", ascending=False)
df.to_csv("al_subcorpus.csv", index=False)
```

### Simulated labelling (scripting / testing)

```python
al = ActiveLearner(sub, features="scores")

# Simulate two rounds of labelling + retraining
batch1 = al.next_batch(n=10)
labels1 = {idx: True if al._df.at[idx, "score"] > 0.2 else False
           for idx in batch1}
al.label_batch(labels1).retrain()

batch2 = al.next_batch(n=10)
labels2 = {idx: True if al._df.at[idx, "score_grouped"] > 0.1 else False
           for idx in batch2}
al.label_batch(labels2).retrain()

al.status()
result = al.to_subcorpus()
```
