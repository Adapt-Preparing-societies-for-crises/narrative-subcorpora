# Documentation

This tool helps researchers build subcorpora of historical texts around specific events. For example, you could collect all newspaper articles published in the six months after the Spanish Flu outbreak and filter them to only keep texts that are actually about the epidemic.

The two main tasks are:

1. **Ingest** -- convert your spreadsheet data (CSV, TSV, or Excel) into a fast, queryable format (parquet).
2. **Extract** -- select texts from a time window around an event and score them for relevance using seed terms.

Everything can be done from the command line. No Python knowledge is required.


## Installation

Open a terminal in the project folder and run:

```
pip install -e .
```

This installs the `nsc` command. You can verify it works by running:

```
nsc --help
```


## Your data

Your input data should be a spreadsheet file (CSV, TSV, or Excel) with at least two columns:

- A **text** column containing the texts (articles, paragraphs, etc.)
- A **date** column containing the publication date of each text

The column names can be anything -- you tell the tool what they are called.


## Step 1: Ingest your data

Convert your spreadsheet into a parquet file. Parquet is a compact file format that allows fast filtering and querying, even for millions of rows.

```
nsc ingest my_data.csv -o my_corpus.parquet --text-col text --date-col date
```

Replace `my_data.csv` with the path to your file. Replace `text` and `date` with the actual names of your columns.

If you want basic text cleaning (collapsing extra whitespace), add `--clean`:

```
nsc ingest my_data.csv -o my_corpus.parquet --text-col text --date-col date --clean
```

To check that everything worked, run:

```
nsc describe my_corpus.parquet
```

This prints the number of rows and the column names and types.


## Step 2: Define your events

Events are stored in `events.json`. Each event has:

- **label** -- a short identifier (no spaces), e.g. `spaanse_griep`
- **full name** -- a human-readable name, e.g. `Spaanse Griep`
- **start_date** -- the start date in DD-MM-YYYY format, e.g. `01-07-1918`
- **terms** -- a list of seed terms that indicate the text is about this event

The file already contains several events. To see them:

```
nsc events events.json
```

To add your own event, open `events.json` in any text editor and add a new entry following the same pattern. Make sure the file remains valid JSON (watch your commas and brackets).


## Step 3: Extract a subcorpus

This is where the filtering happens. The tool:

1. Selects all texts published within a time window after the event.
2. Scores each text based on how many of the seed terms it contains.
3. Keeps only texts that score above a threshold you set.

```
nsc extract --corpus my_corpus.parquet --events events.json --event spaanse_griep --window 6m --min-score 0.1 -o subcorpus.parquet
```

What the options mean:

| Option | What it does |
|---|---|
| `--corpus` | Path to your parquet file from step 1 |
| `--events` | Path to the events JSON file |
| `--event` | The label of the event you want |
| `--window` | How many months after the event to include (e.g. `6m` for 6 months) |
| `--min-score` | Minimum relevance score to keep a text (0 to 1). Higher = stricter filtering |
| `-o` | Where to save the result. Use `.parquet` or `.csv` as the file extension |

The output file contains all the original columns plus a `score` column.


## Understanding the scores

The tool offers several ways to measure how relevant a text is to your event. The default method used by `nsc extract` is **term coverage**, but the others are available through the Python API (see below).

### Term coverage (default)

The fraction of seed terms that appear at least once in the text. A score of 0.1 means 10% of the seed terms were found. A score of 0.5 means half of them were found. This is what `nsc extract --min-score` uses.

### Term density

The total number of seed-term hits divided by the number of words in the text. Measures how concentrated the event language is. A short text mentioning many seed terms scores higher than a long text with the same number of mentions.

### TF-IDF

A classic information-retrieval measure. It rewards terms that appear frequently in a specific text but are rare across the rest of the corpus. This helps surface texts where event language stands out, rather than texts that happen to share common words with the seed list.

### BM25

The standard search-engine ranking formula. Similar to TF-IDF but also accounts for document length -- a long article does not automatically score higher than a short one just because it has more words. Uses two tuneable parameters (k1 and b) that control how quickly term frequency saturates and how much length matters.

### Term clustering

Measures how tightly seed terms cluster together within a text. A sliding window (default: 50 words) moves across the text, and the score reflects the best concentration of distinct seed terms found in any single window. Useful for detecting focused passages about the event inside otherwise unrelated texts.

### Weighted terms

Lets you assign different importance to different seed terms. For example, a highly specific term like "spaanse" might get weight 2.0, while a generic term like "ziekte" gets 0.5. The score is the sum of weights for terms found, divided by the total possible weight.

### Which method to use

- **Start with term coverage** (the default). It is simple and interpretable.
- If you get too many false positives (irrelevant texts slipping through), try **TF-IDF** or **BM25** -- they downweight common terms automatically.
- If your texts are long and you suspect the event is only discussed in a small section, try **term clustering**.
- If you know some terms are much more diagnostic than others, use **weighted terms**.

You can also compute multiple scores at once and compare them side by side in a spreadsheet.

### Choosing a threshold

What threshold to use depends on your research question. A good starting point for term coverage:

- **0.05** -- very loose, includes many texts with only tangential mentions
- **0.1** -- moderate, a reasonable default for exploratory work
- **0.3** -- strict, keeps only texts with substantial coverage of event terms

You can always export with a low threshold first, inspect the results in a spreadsheet, and then re-run with a higher threshold. For TF-IDF and BM25 the scores are not bounded between 0 and 1, so inspect the distribution first before setting a cutoff.


## Diagnostics

To visualise how your filtering performs, use the `diagnose` command:

```
nsc diagnose --corpus my_corpus.parquet --events events.json --event spaanse_griep --window 6m --min-score 0.1 -o report.png
```

This produces a figure with two panels:

1. A daily bar chart showing how many articles were selected vs excluded on each day since the event.
2. A cumulative curve comparing the total number of articles in the time window to the number that passed the filter.

Leave out `-o report.png` to display the figure interactively instead of saving it.


## Python API

For users comfortable with Python, the package provides a fluent (chainable) API. This is useful for notebooks, custom scoring, or combining multiple scoring methods.

```python
from narrative_subcorpora import Corpus, Event

corpus = Corpus("my_corpus.parquet", text_col="text", date_col="date")
event = Event.from_json("events.json", "spaanse_griep")

# Basic workflow -- same as the CLI
subcorpus = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)
    .above(0.1)
)
subcorpus.to_csv("output.csv")

# Multiple scoring methods at once
subcorpus = (
    corpus
    .after(event, months=6)
    .score(terms=event.terms)                    # term coverage  -> "score"
    .score_tfidf(terms=event.terms)              # TF-IDF         -> "score_tfidf"
    .score_bm25(terms=event.terms)               # BM25           -> "score_bm25"
    .score_cluster(terms=event.terms, window=50) # clustering     -> "score_cluster"
    .score_weighted({"spaanse": 2, "griep": 2, "ziekte": 0.5})  # -> "score_weighted"
    .above(0.1, col="score_tfidf")               # filter on any score column
)

# Other time windows
corpus.before(event, months=1)
corpus.around(event, months_before=1, months_after=6)
corpus.between("1918-01-01", "1919-01-01")

# Inspect as a DataFrame
df = subcorpus.to_dataframe()
```


## Output formats

- `.parquet` -- compact, fast to re-query. Use this if you plan to do further processing.
- `.csv` -- opens in Excel, Google Sheets, or any spreadsheet program. Use this if you want to read and annotate the results manually.


## Quick reference

| Command | What it does |
|---|---|
| `nsc ingest data.csv -o corpus.parquet --text-col text --date-col date` | Convert a spreadsheet to parquet |
| `nsc describe corpus.parquet` | Show file info (rows, columns) |
| `nsc events events.json` | List available events |
| `nsc extract --corpus corpus.parquet --events events.json --event spaanse_griep --window 6m --min-score 0.1 -o out.csv` | Extract a subcorpus |
| `nsc diagnose --corpus corpus.parquet --events events.json --event spaanse_griep --window 6m -o report.png` | Visualise selection diagnostics |


## Troubleshooting

**"command not found: nsc"** -- Make sure you ran `pip install -e .` in the project folder and that the correct Python environment is active.

**"Event 'xyz' not found"** -- Check the exact label with `nsc events events.json`. Labels are case-sensitive.

**Empty output** -- Try lowering `--min-score` or increasing `--window`. Your time window might not overlap with any texts, or the threshold might be too strict.

**Wrong dates** -- Dates in events.json use DD-MM-YYYY format (day first). Dates in your spreadsheet are parsed automatically, but if they look wrong after ingest, check the original format.
