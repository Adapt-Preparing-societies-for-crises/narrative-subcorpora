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


## Understanding the score

The score measures what fraction of the seed terms appear in a text. A score of 0.1 means 10% of the seed terms were found. A score of 0.5 means half of them were found.

What threshold to use depends on your research question. A good starting point:

- **0.05** -- very loose, includes many texts with only tangential mentions
- **0.1** -- moderate, a reasonable default for exploratory work
- **0.3** -- strict, keeps only texts with substantial coverage of event terms

You can always export with a low threshold first, inspect the results in a spreadsheet, and then re-run with a higher threshold.


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


## Troubleshooting

**"command not found: nsc"** -- Make sure you ran `pip install -e .` in the project folder and that the correct Python environment is active.

**"Event 'xyz' not found"** -- Check the exact label with `nsc events events.json`. Labels are case-sensitive.

**Empty output** -- Try lowering `--min-score` or increasing `--window`. Your time window might not overlap with any texts, or the threshold might be too strict.

**Wrong dates** -- Dates in events.json use DD-MM-YYYY format (day first). Dates in your spreadsheet are parsed automatically, but if they look wrong after ingest, check the original format.
