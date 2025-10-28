# WebDataCommons Processing

This project provides tools for discovering and enriching CSV files from WebDataCommons datasets.

## Table of Contents

- [Setup Instructions](#setup-instructions)
  - [1. Run the Setup Script](#1-run-the-setup-script)
  - [2. Install Requirements](#2-install-requirements)
- [Naive Enrichment Method](#naive-enrichment-method)
  - [How the `find_join_keys` Function Works](#how-the-find_join_keys-function-works)
    - [What is a Join Key?](#what-is-a-join-key)
    - [Basic Usage](#basic-usage)
    - [How It Works](#how-it-works)
    - [Example 1: Single Column Match](#example-1-single-column-match)
    - [Example 2: Composite Keys](#example-2-composite-keys)
    - [Example 3: Different Column Names](#example-3-different-column-names)
    - [Parameters](#parameters)
  - [How the `enrich_directory` Function Works](#how-the-enrich_directory-function-works)
    - [What Does It Do?](#what-does-it-do)
    - [Visual Example](#visual-example)
    - [Basic Usage](#basic-usage-1)
    - [How Column Selection Works](#how-column-selection-works)
    - [Parameters](#parameters-1)
    - [The Enrichment Process in Detail](#the-enrichment-process-in-detail)
    - [Example: Column Selection Process](#example-column-selection-process)
    - [Example: Column Rejection Process](#example-column-rejection-process)
- [Semantic Enrichment Method](#semantic-enrichment-method)
  - [How Semantic Column Selection Works](#how-semantic-column-selection-works)
    - [Overview](#overview)
    - [Semantic Relevance Scoring](#semantic-relevance-scoring)
    - [Semantic Plausibility Checking](#semantic-plausibility-checking)
    - [Column Selection Process](#column-selection-process)
    - [Usage Example](#usage-example)
- [Method Comparison](#method-comparison)
  - [Quick Reference Table](#quick-reference-table)
  - [Detailed Comparison](#detailed-comparison)
  - [When to Use Each Method](#when-to-use-each-method)
- [Running the Enrichment](#running-the-enrichment)
  - [Basic Usage](#basic-usage-2)
  - [Advanced Options](#advanced-options)

---

## Setup Instructions

### 1. Run the Setup Script

First, run the setup script to download and prepare the sample data:

```bash
python setup.py
```

This will:
- Download the WebDataCommons sample10 dataset
- Extract and organize the CSV files into folders by prefix
- Copy all CSV files into a `csvs/` directory

### 2. Install Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The requirements include:
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `sentence-transformers` - Optional: for semantic embeddings (SBERT)
- `scikit-learn` - TF-IDF vectorization fallback

---

## Naive Enrichment Method

The naive enrichment method uses keyword-based matching and statistical checks to identify relevant columns to add from other CSV files. It relies on value overlap for join key discovery and hardcoded keyword lists for relevance scoring.

### How the `find_join_keys` Function Works

Both enrichment methods use the same join key discovery mechanism, which is detailed in this section.

#### What is a Join Key?

A **join key** is a column (or set of columns) that exists in both datasets and contains matching values, allowing you to combine data from different sources. Think of it like a unique identifier that links rows from one table to rows in another table.

For example:
- If you have population data for countries and another dataset with GDP for countries, the `country` column is the join key that lets you merge these datasets.
- If you have multiple years of data, you might need both `country` AND `year` to uniquely identify each row - this is called a **composite key**.

When you join two tables on a key, you're essentially combining data where the key values match. This is especially useful when you have data spread across multiple CSV files that share common identifiers.

#### Basic Usage

```python
import pandas as pd
from modules.naive_enrich import find_join_keys

base_df = pd.read_csv("csvs/base_file.csv")
other_df = pd.read_csv("csvs/other_file.csv")

keys = find_join_keys(base_df, other_df)
```

#### How It Works

The function searches for matching columns between two dataframes based on value overlap. It returns potential join keys with their overlap score.

##### Example 1: Single Column Match

If two CSVs share a common column like `country` with overlapping values:

```python
# Base CSV has: country, population, GDP
# Other CSV has: country, inflation_rate, unemployment

keys = find_join_keys(base_df, other_df)
# Returns: [
#   {
#     "base_keys": ["country"],
#     "other_keys": ["country"],
#     "overlap": 0.95
#   }
# ]
```

This means 95% of the `country` values overlap between both files, making it a good join key.

##### Example 2: Composite Keys

The function also searches for multi-column join keys:

```python
# Base CSV has: country, year, population
# Other CSV has: country, year, gdp_per_capita

keys = find_join_keys(base_df, other_df, max_key_len=2)
# Returns: [
#   {
#     "base_keys": ["country", "year"],
#     "other_keys": ["country", "year"],
#     "overlap": 0.88
#   }
# ]
```

Here, the combination of `country` and `year` creates a composite key that matches 88% of the rows.

##### Example 3: Different Column Names

The function can match columns even when they have different names:

```python
# Base CSV has: nation_name, population
# Other CSV has: country_name, gdp

keys = find_join_keys(base_df, other_df)
# Returns: [
#   {
#     "base_keys": ["nation_name"],
#     "other_keys": ["country_name"],
#     "overlap": 0.82
#   }
# ]
```

Despite different column names (`nation_name` vs `country_name`), if 82% of their values overlap, it's detected as a potential join key.

#### Parameters

- `base`: The primary dataframe
- `other`: The dataframe to join with
- `max_key_len`: Maximum number of columns for composite keys (default: 2)
- `min_overlap`: Minimum overlap ratio to consider (default: 0.8, i.e., 80%)

### How the `enrich_directory` Function Works

#### What Does It Do?

The `enrich_directory` function automatically combines data from multiple CSV files into a single enriched dataset. It:

1. **Finds matching files** - Scans all CSVs in a directory
2. **Discovers join keys** - Uses `find_join_keys` to identify how to link files
3. **Selects best columns** - Picks the most relevant columns to add to your base dataset
4. **Creates enriched output** - Produces one CSV with all the combined data

#### Visual Example

Imagine you have these files:

**Base CSV (`countries.csv`):**
```
country        population
USA            328000000
China          1390000000
Japan          126000000
```

**File 1 (`gdp_data.csv`):**
```
nation         gdp_millions    inflation_rate
USA            21000000       2.3
China          14000000       1.8
Japan          5000000        0.5
```

**File 2 (`health_data.csv`):**
```
country_name   life_expectancy   hospital_beds
USA            78.9              2.9
China          77.0              4.3
Japan          84.4              13.1
```

#### The Function Automatically:

1. **Detects join keys**: 
   - Matches `country` (base) ↔ `nation` (File 1)
   - Matches `country` (base) ↔ `country_name` (File 2)

2. **Selects relevant columns**:
   - From File 1: `gdp_millions`, `inflation_rate`
   - From File 2: `life_expectancy`, `hospital_beds`

3. **Creates enriched output**:
```
country        population    gdp_millions    inflation_rate    life_expectancy    hospital_beds
USA            328000000     21000000        2.3                78.9               2.9
China          1390000000    14000000        1.8                77.0               4.3
Japan          126000000     5000000         0.5                84.4               13.1
```

#### Basic Usage

```python
from modules.naive_enrich import enrich_directory

enriched_csv, log_file = enrich_directory(
    base_csv="csvs/base_countries.csv",
    dir_path="csvs",
    out_csv="enriched_countries.csv",
    out_log="provenance.json"
)
```

#### How Column Selection Works

The naive function scores each potential column based on:

- **Coverage** (40%): How complete is the data?
- **Relevance** (30%): Does the column name suggest useful data?
- **Novelty** (30%): Does it add new information not already in base?

Only the **top scoring columns** are added (default: top 50 globally).

The **relevance scoring** uses hardcoded keyword matching:

```python
POS = ["gdp", "inflation", "mortality", "fertility", "life_expectancy",
       "internet", "electricity", "education", "literacy", "unemployment",
       "growth", "co2", "emission", "poverty", "access", "health", 
       "income", "exports", "imports"]
NEG = ["note", "comment", "description", "footnote", "source"]
```

#### Parameters

- `base_csv`: Path to your main/base CSV file
- `dir_path`: Directory containing CSV files to search
- `out_csv`: Output file path (default: "enriched.csv")
- `out_log`: Provenance/log file path (default: "provenance.json")
- `explicit_base_keys`: Specify known join keys, or `None` for auto-detection
- `top_k_global`: Number of best columns to add (default: 50)
- `min_overlap`: Minimum join key overlap (default: 0.8)

#### The Enrichment Process in Detail

Here's how the `enrich_directory` function processes your data:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: INITIALIZE                                                       │
│   base = pd.read_csv(base_csv)                                          │
│   files = scan_directory(dir_path)  # Get all CSV files                  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: FOR EACH SOURCE FILE                                             │
│   ┌────────────────────────────────────────────────┐                   │
│   │ file1.csv → find_join_keys() → join_keys_1      │                   │
│   │               ↓                                 │                   │
│   │            select_enrichment_columns()          │                   │
│   │              ↓                                   │                   │
│   │           [col1: score=0.85, col2: score=0.72] │                   │
│   └────────────────────────────────────────────────┘                   │
│   ┌────────────────────────────────────────────────┐                   │
│   │ file2.csv → find_join_keys() → join_keys_2      │                   │
│   │               ↓                                 │                   │
│   │            select_enrichment_columns()          │                   │
│   │              ↓                                   │                   │
│   │           [col3: score=0.91, col4: score=0.68] │                   │
│   └────────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: GLOBAL RANKING                                                   │
│   ALL_CANDIDATES = [col1: 0.85, col2: 0.72, col3: 0.91,                │
│                     col4: 0.68, col5: 0.79]                             │
│   sort by score → [col3: 0.91, col1: 0.85, col5: 0.79,                │
│                     col2: 0.72, col4: 0.68]                             │
│   keep TOP_K (default: 50)                                              │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: BUILD ENRICHED DATASET                                           │
│   enriched = base.copy()                                                 │
│   for each selected column:                                             │
│     1. Read source file (only needed columns)                           │
│     2. Rename other_keys → base_keys for alignment                      │
│     3. pd.merge() to add column to enriched dataframe                    │
│     4. Handle column name conflicts with suffixes                       │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: SAVE OUTPUTS                                                     │
│   enriched.to_csv(out_csv)                                              │
│   json.dump(selections, provenance.json)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Example: Column Selection Process

Let's say `select_enrichment_columns()` evaluates a column called `gdp_per_capita`:

```
Candidate: gdp_per_capita
├─ Coverage Check (40% weight)
│  ├─ 95% of rows have data ✓
│  └─ Coverage score: 0.95
├─ Relevance Check (30% weight)
│  ├─ Keywords found: "gdp", "per_capita" ✓
│  └─ Relevance score: 0.8
└─ Novelty Check (30% weight)
   ├─ Correlation with base columns: 0.15 (low)
   └─ Novelty score: 0.85 (adds unique info)

Final Score = 0.4×0.95 + 0.3×0.8 + 0.3×0.85 = 0.875
Result: KEPT (high score, provides valuable new data)
```

#### Example: Column Rejection Process

Now for a column called `data_note`:

```
Candidate: data_note
├─ Coverage Check (40% weight)
│  ├─ 60% of rows have data ✗ (below 70% threshold)
│  └─ FAILED
└─ STOP (rejected due to low coverage)

Final Score: Not calculated
Result: REJECTED (doesn't meet quality standards)
```

---

## Semantic Enrichment Method

The semantic enrichment method uses semantic embeddings (SBERT/TF-IDF) and unit-aware plausibility checks to identify relevant columns. Instead of hardcoded keyword lists, it understands the *meaning* of column names and values through vector representations.

### How Semantic Column Selection Works

#### Overview

The semantic approach replaces the naive `_relevance_score()` and `_plausible()` functions with:

1. **`semantic_relevance()`**: Uses embeddings or TF-IDF to compute semantic similarity between the base table's column names (as a query) and candidate columns from other tables
2. **`semantic_plausibility()`**: Performs unit-aware statistical checks (e.g., percentage ranges 0-100, index ranges, per-capita ranges)

Both methods use the **same join key discovery** (`find_join_keys()` from `modules/naive_enrich.py`).

#### Semantic Relevance Scoring

The function `semantic_relevance(query, col_name, series, context)`:

1. **Builds a semantic query** from base table column names
   - Example: Base has columns `["country", "population", "gdp"]` → query = `"country population gdp"`
   
2. **Creates a representation** of the candidate column:
   - Column name
   - Sample values (first 6 non-null)
   - Optional context (filename, table identifier)
   - Format: `"{col_name} | samples: {val1, val2, ...} | context: {context}"`

3. **Computes similarity** using one of two methods:
   
   **Primary: SBERT embeddings** (if `sentence-transformers` installed)
   - Uses `all-MiniLM-L6-v2` model
   - Converts text to 384-dimensional vectors
   - Normalized embeddings → cosine similarity naturally in [0,1] range
   - Understands semantic relationships (e.g., "GDP" ≈ "gross domestic product")
   
   **Fallback: TF-IDF cosine** (if embeddings unavailable)
   - Uses n-grams (1-grams and 2-grams)
   - Computes TF-IDF vectors
   - Cosine similarity on sparse vectors
   - Falls back to token overlap if sklearn unavailable

#### Semantic Plausibility Checking

The function `semantic_plausibility(col_name, series)` performs unit-aware validation:

1. **Detects unit hints** from column name patterns:
   ```python
   "percent" → regex: "(percent|percentage|%|_pct|rate)"
   "per_1000" → regex: "per\s*1000|/1000"
   "usd" → regex: "(usd|us$|$|ppp|currency)"
   "index" → regex: "(index|score|rating)"
   "per_capita" → regex: "per[_ ]?capita|pc"
   ```

2. **Validates ranges** based on detected units:
   - **Percent**: `min >= -1, max <= 100.5`
   - **Per 1000**: `min >= 0, max <= 2000`
   - **USD**: `min >= -1e5, max < 1e15`
   - **Index**: `min >= -100, max <= 10000`
   - **Generic**: `min >= -1e6, max <= 1e15`

3. **Checks uniqueness**:
   - Rejects near-unique integer columns (potential IDs)
   - Allows high uniqueness for "index" columns

4. **Checks variability**:
   - Requires `std > 0` (non-constant values)

5. **Returns normalized score**: `score / num_checks` in [0,1]

#### Column Selection Process

The `select_enrichment_columns_semantic()` function uses a **4-part scoring system**:

```python
score = 0.4*cov + 0.3*rel + 0.2*plaus + 0.1*nov
```

Where:
- **Coverage (40%)**: Same as naive method - data completeness
- **Semantic Relevance (30%)**: Semantic similarity to base column names
- **Semantic Plausibility (20%)**: Unit-aware value validation
- **Novelty (10%)**: Same as naive method - adds unique information

**Filtering thresholds:**
- Coverage ≥ 70%
- Plausibility ≥ 30%
- Semantic relevance ≥ 10%
- Novelty ≥ 10%
- Not an ID-like column

#### Usage Example

```python
from modules.semantic_enrich import enrich_directory_semantic

enriched_csv, log = enrich_directory_semantic(
    base_csv="csvs/query_table.csv",
    dir_path="csvs",
    out_csv="enriched_semantic.csv",
    out_log="provenance_semantic.json",
    explicit_base_keys=None,   # Auto-detect join keys
    top_k_global=50,           # Add top 50 columns
    min_overlap=0.8            # Require 80% overlap
)
```

**How it differs from naive:**

Given a base table with columns `["country", "population", "year"]`:

1. **Query is built**: `"country population year"`
2. A candidate column `"gdp_per_capita"` with values `[52000, 48000, 51000, ...]`:

   **Semantic relevance**:
   - SBERT embeddings: `semantic_relevance("country population year", "gdp_per_capita", series, None)`
   - Even though column name differs, embedding similarity might be 0.75 (high relevance to economic indicators)
   
   **Semantic plausibility**:
   - Detects "per_capita" unit hint
   - Validates: values are in reasonable GDP range (48k-52k)
   - Uniqueness check: Not an ID
   - Variability check: std > 0
   - Plausibility score: 0.85
   
3. **Final score**: `0.4×0.95 + 0.3×0.75 + 0.2×0.85 + 0.1×0.90 = 0.865`

---

## Method Comparison

### Quick Reference Table

| Feature | Naive Method | Semantic Method |
|---------|-------------|-----------------|
| **Column Selection** | Keyword matching | Semantic embeddings (SBERT/TF-IDF) |
| **Relevance Scorer** | `_relevance_score()` - hardcoded POS/NEG tokens | `semantic_relevance()` - vector similarity |
| **Plausibility Checker** | `_plausible()` - basic range checks | `semantic_plausibility()` - unit-aware stats |
| **Query Construction** | Not used | Column names from base table |
| **Join Key Discovery** | ✅ Value overlap | ✅ Same (value overlap) |
| **Dependencies** | numpy, pandas | numpy, pandas, sentence-transformers, scikit-learn |
| **Performance** | Fast | Slower (embeddings computation) |
| **Robustness** | Domain-specific keywords required | General-purpose semantic understanding |

### Detailed Comparison

#### 1. **Relevance Scoring**

**Naive:**
- Hardcoded list of positive keywords: `["gdp", "inflation", "mortality", ...]`
- Hardcoded list of negative keywords: `["note", "comment", ...]`
- Score formula: `0.2*pos_count - 0.2*neg_count + 0.2*feature_bonus`
- Pros: Fast, predictable
- Cons: Limited to predefined domain, doesn't understand synonyms (e.g., "gdp" vs "gross domestic product")

**Semantic:**
- Builds query from base column names
- Uses SBERT embeddings to understand meaning
- Handles synonyms and related concepts automatically
- Pros: General-purpose, understands semantic relationships
- Cons: Slower, requires model downloads (few hundred MB)

#### 2. **Plausibility Checking**

**Naive:**
- Generic range checks (e.g., `x.min() >= -1e5, x.max() <= 1e15`)
- Percentage detection: checks for keywords "rate", "pct", "%", "percentage"
- No unit-aware validation
- Example: A column named "gdp_pct" must have values 0-100

**Semantic:**
- Detects units via regex patterns
- Validates ranges based on detected units
- Checks: percent (0-100), per_1000 (0-2000), usd (reasonable range), index, per_capita
- More sophisticated statistical checks
- Example: Detects "per_capita" hint and validates GDP ranges (expects thousands to millions range)

#### 3. **Scoring Weights**

**Naive:**
```python
score = 0.4*cov + 0.3*rel + 0.3*nov
```
- Coverage: 40%
- Keyword relevance: 30%
- Novelty: 30%

**Semantic:**
```python
score = 0.4*cov + 0.3*rel + 0.2*plaus + 0.1*nov
```
- Coverage: 40%
- Semantic relevance: 30%
- Semantic plausibility: 20%
- Novelty: 10%

#### 4. **Performance**

**Naive:**
- Constant-time keyword matching
- Fast execution
- Suitable for real-time or batch processing of large datasets

**Semantic:**
- SBERT model loading: ~2-5 seconds first time
- Embedding computation: ~0.01-0.1s per column pair
- TF-IDF fallback: similar speed to naive
- Suitable for offline processing, quality-focused use cases

#### 5. **Use Case Examples**

**Naive: Best for...**
- Specific domains with known terminology (e.g., economics: GDP, inflation, unemployment)
- Speed-critical applications
- Environments without GPU/high RAM
- Datasets with consistent, predictable column naming

**Semantic: Best for...**
- General-purpose enrichment across unknown domains
- Datasets with varied terminology or languages
- Quality-focused scenarios where computational time is acceptable
- Research applications requiring nuanced understanding

### When to Use Each Method

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| Economic/financial data with standardized terms | **Naive** | Keyword lists work well for domain-specific terms |
| Unknown data domains, varied terminology | **Semantic** | Embeddings understand context and synonyms |
| Real-time or interactive enrichment | **Naive** | Fast execution, minimal dependencies |
| Offline batch processing, quality > speed | **Semantic** | Better understanding of column relationships |
| Limited computational resources | **Naive** | Lower memory and CPU requirements |
| High-quality, research-grade enrichment | **Semantic** | More sophisticated plausibility checks |
| Consistent naming conventions | **Naive** | Keyword matching is sufficient |
| Inconsistent naming, abbreviations, synonyms | **Semantic** | Handles variations better |

---

## Running the Enrichment

### Basic Usage

The `main.py` script provides a unified interface for both enrichment methods:

```bash
# Naive enrichment (default)
python main.py --method naive --input csvs/query.csv --dir csvs

# Semantic enrichment
python main.py --method semantic --input csvs/query.csv --dir csvs
```

### Advanced Options

Full command-line interface:

```bash
python main.py \
  --method semantic \           # naive or semantic
  --input csvs/query.csv \      # Input CSV to enrich
  --dir csvs \                  # Directory with source CSVs
  --topk 50 \                   # Number of best columns to add
  --overlap 0.8 \               # Minimum join key overlap ratio
  --keys "country,year"          # Explicit join keys (optional)
```

**Output files:**
- `enriched_naive.csv` or `enriched_semantic.csv` - The enriched dataset
- `provenance_naive.json` or `provenance_semantic.json` - Track which columns came from which files

### Example Workflow

```bash
# 1. Download and setup data
python setup.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run naive enrichment
python main.py --method naive \
  --input csvs/26054757_1_2067817572683155548.csv \
  --dir csvs \
  --topk 30

# 4. Run semantic enrichment (alternative)
python main.py --method semantic \
  --input csvs/26054757_1_2067817572683155548.csv \
  --dir csvs \
  --topk 30

# 5. Compare results
# Check enriched_naive.csv vs enriched_semantic.csv
```

Both methods use the same join-key discovery but differ in how they select which columns to add from the source files. The semantic method provides better generalization across different data domains, while the naive method offers faster execution for domain-specific use cases.