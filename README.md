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
