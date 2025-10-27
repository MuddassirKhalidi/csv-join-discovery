from enrich import find_join_keys, enrich_directory, select_enrichment_columns
import pandas as pd

# 1) Auto-detect join keys between base CSV and all other CSVs
base_df = pd.read_csv("csvs/8188057_0_4575467631524475515.csv")
import os

csv_files = [f for f in os.listdir("csvs") if f.endswith(".csv") and f != "8188057_0_4575467631524475515.csv"]

print(f"Base file: 8188057_0_4575467631524475515.csv\n")
for csv_file in csv_files:
    other_df = pd.read_csv(f"csvs/{csv_file}")
    keys = find_join_keys(base_df, other_df, max_key_len=2, min_overlap=0.8)
    if keys:
        print(f"{csv_file}: {keys}")
    else:
        print(f"{csv_file}: No matching keys found")

# 2) End-to-end enrichment over a directory
# enriched_csv, log_json = enrich_directory(
#     base_csv="csvs/26054757_1_2067817572683155548.csv",
#     dir_path="csvs",
#     out_csv="enriched2.csv",
#     out_log="provenance.json",
#     explicit_base_keys=None,   # or e.g. ['country_iso3','year'] if you know them
#     top_k_global=50,
#     min_overlap=0.8
# )
