# Repository Contents

This repo contains code to construct subcorpora of historical text data around particular events. This is done by:
- assembling sources into parquet files
- preprocessing text data into 1) ngrams 2) and document embeddings
- identifying relevant texts based on keywords
- narrowing keyword-based selection based on additional (embedding-based) metrics
