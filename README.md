# WebDataCommons Processing

This project provides tools for discovering and enriching CSV files from WebDataCommons datasets.

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

## How the `find_join_keys` Function Works

### What is a Join Key?

A **join key** is a column (or set of columns) that exists in both datasets and contains matching values, allowing you to combine data from different sources. Think of it like a unique identifier that links rows from one table to rows in another table.

For example:
- If you have population data for countries and another dataset with GDP for countries, the `country` column is the join key that lets you merge these datasets.
- If you have multiple years of data, you might need both `country` AND `year` to uniquely identify each row - this is called a **composite key**.

When you join two tables on a key, you're essentially combining data where the key values match. This is especially useful when you have data spread across multiple CSV files that share common identifiers.

### Basic Usage

```python
import pandas as pd
from enrich import find_join_keys

base_df = pd.read_csv("csvs/base_file.csv")
other_df = pd.read_csv("csvs/other_file.csv")

keys = find_join_keys(base_df, other_df)
```

### How It Works

The function searches for matching columns between two dataframes based on value overlap. It returns potential join keys with their overlap score.

#### Example 1: Single Column Match

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

#### Example 2: Composite Keys

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

#### Example 3: Different Column Names

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

### Parameters

- `base`: The primary dataframe
- `other`: The dataframe to join with
- `max_key_len`: Maximum number of columns for composite keys (default: 2)
- `min_overlap`: Minimum overlap ratio to consider (default: 0.8, i.e., 80%)

### Example from main.py

The `main.py` script demonstrates finding join keys across multiple CSV files:

```python
from enrich import find_join_keys
import pandas as pd

base_df = pd.read_csv("csvs/8188057_0_4575467631524475515.csv")

for csv_file in csv_files:
    other_df = pd.read_csv(f"csvs/{csv_file}")
    keys = find_join_keys(base_df, other_df, max_key_len=2, min_overlap=0.8)
    if keys:
        print(f"{csv_file}: {keys}")
    else:
        print(f"{csv_file}: No matching keys found")
```

This processes each CSV file and prints any matching join keys found.

## How the `enrich_directory` Function Works

### What Does It Do?

The `enrich_directory` function automatically combines data from multiple CSV files into a single enriched dataset. It:

1. **Finds matching files** - Scans all CSVs in a directory
2. **Discovers join keys** - Uses `find_join_keys` to identify how to link files
3. **Selects best columns** - Picks the most relevant columns to add to your base dataset
4. **Creates enriched output** - Produces one CSV with all the combined data

### Visual Example

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

### The Function Automatically:

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

### Basic Usage

```python
from enrich import enrich_directory

enriched_csv, log_file = enrich_directory(
    base_csv="csvs/base_countries.csv",
    dir_path="csvs",
    out_csv="enriched_countries.csv",
    out_log="provenance.json"
)
```

### How Column Selection Works

The function scores each potential column based on:

- **Coverage** (40%): How complete is the data?
- **Relevance** (30%): Does the column name suggest useful data?
- **Novelty** (30%): Does it add new information not already in base?

Only the **top scoring columns** are added (default: top 50 globally).

### Parameters

- `base_csv`: Path to your main/base CSV file
- `dir_path`: Directory containing CSV files to search
- `out_csv`: Output file path (default: "enriched.csv")
- `out_log`: Provenance/log file path (default: "provenance.json")
- `explicit_base_keys`: Specify known join keys, or `None` for auto-detection
- `top_k_global`: Number of best columns to add (default: 50)
- `min_overlap`: Minimum join key overlap (default: 0.8)

### Complete Example

```python
from enrich import enrich_directory

# Enrich a base CSV with data from all other CSVs in the directory
enriched_csv, log = enrich_directory(
    base_csv="csvs/26054757_1_2067817572683155548.csv",
    dir_path="csvs",
    out_csv="enriched2.csv",
    out_log="provenance.json",
    explicit_base_keys=None,   # Auto-detect join keys
    top_k_global=50,           # Add top 50 columns
    min_overlap=0.8            # Require 80% overlap
)

print(f"Enriched data saved to: {enriched_csv}")
print(f"Provenance log saved to: {log}")
```

The `provenance.json` file tracks which columns came from which source files, so you always know where your data originated.

## The Enrichment Process in Detail

### Visual Pipeline

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
│   ┌────────────────────────────────────────────────┐                   │
│   │ file3.csv → find_join_keys() → join_keys_3      │                   │
│   │               ↓                                 │                   │
│   │            select_enrichment_columns()          │                   │
│   │              ↓                                   │                   │
│   │           [col5: score=0.79]                    │                   │
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

### Detailed Step Breakdown

#### **Step 1: Initialization**
```python
# enrich_directory() line 195
base = pd.read_csv(base_csv)
```
- Reads the base CSV file into a DataFrame
- Scans the directory for all CSV files (excluding the base file)

#### **Step 2: Discovery Loop** (Lines 201-223)

For each CSV file in the directory:

**2a. Discover Join Keys** (line 213)
```python
keys_map = find_join_keys(base, df, max_key_len=2, min_overlap=min_overlap)
```
- Uses `_value_overlap_ratio()` to calculate how much values overlap between columns
- Tries single-column and composite key matches
- Returns `[{"base_keys": [...], "other_keys": [...], "overlap": 0.95}]`

**2b. Select Enrichment Columns** (line 220)
```python
picks = select_enrichment_columns(base, df, bk, ok)
```

This function uses a **3-stage filtering and scoring** process:

**Filter 1: Quality Checks** (Lines 177-181 in enrich.py)
- Coverage threshold: At least 70% of values must be present
- Not an ID field: Rejects columns that look like identifiers
- Plausibility check: Values must be reasonable (not absurd outliers)

**Filter 2: Relevance Scoring** (Lines 60-69 in enrich.py)
```python
def _relevance_score(col_name: str) -> float:
    POS = ["gdp", "inflation", "mortality", "life_expectancy", ...]
    NEG = ["note", "comment", "description", ...]
    # Returns score based on relevant keywords in column name
```
- Positive keywords boost score (e.g., "gdp", "life_expectancy")
- Negative keywords reduce score (e.g., "note", "comment")

**Filter 3: Novelty Check** (Lines 184-185 in enrich.py)
```python
nov = 1.0 - _max_abs_corr(s, base_num)
```
- Compares new column with existing numeric columns in base
- High correlation = low novelty (column is redundant)
- Only keeps columns that add new information

**Final Score** (Line 186)
```python
score = 0.4*cov + 0.3*rel + 0.3*nov
```
- 40% coverage (data completeness)
- 30% relevance (usefulness based on name)
- 30% novelty (adds unique information)

**2c. Collect Selections**
```python
selections.append({"file": f, "base_keys": bk, "other_keys": ok,
                   "col": p["col"], "score": p["score"]})
```
- Builds a list of all potential columns to add with their metadata

#### **Step 3: Global Ranking** (Lines 226-227)
```python
selections.sort(key=lambda x: x["score"], reverse=True)
keep = selections[:top_k_global]
```
- Sorts ALL columns from ALL files by their scores
- Keeps only the top K columns globally (default: 50)

#### **Step 4: Perform Joins** (Lines 230-241)

For each selected column, the function:

**4a. Read Only Needed Data** (Line 232)
```python
df = pd.read_csv(s["file"], usecols=[...])
```
- Loads only the join keys and target column (memory efficient)

**4b. Align Column Names** (Line 233)
```python
slim = df.rename(columns={ok: bk for ok, bk in zip(s["other_keys"], s["base_keys"])})
```
- Renames columns so they match the base file's key names
- Example: `nation` → `country`

**4c. Perform Merge** (Line 236)
```python
enriched = pd.merge(enriched, to_merge, on=s["base_keys"], how="left", ...)
```
- Uses pandas merge to add the new column
- Left join preserves all base rows
- Adds suffix if column name conflicts (e.g., `gdp__file1`)

#### **Step 5: Save Results** (Lines 243-246)
```python
enriched.to_csv(out_csv, index=False)
json.dump(keep, f, indent=2)
```
- Writes the enriched CSV with all added columns
- Creates provenance file tracking every column's source

### Example: Column Selection Process

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

### Example: Column Rejection Process

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

This detailed process ensures only high-quality, relevant, and novel data makes it into your enriched dataset!
