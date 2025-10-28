"""
Main entry point for CSV enrichment using either naive or semantic approaches.

Usage:
    python main.py --method naive
    python main.py --method semantic
    
Default: naive
"""
import argparse
import pandas as pd
from modules.naive_enrich import enrich_directory, find_join_keys
from modules.semantic_enrich import enrich_directory_semantic

def main():
    parser = argparse.ArgumentParser(description="Enrich a query CSV using directory of CSV files")
    parser.add_argument("--method", choices=["naive", "semantic"], default="naive",
                       help="Enrichment method: naive (keyword-based) or semantic (embedding-based)")
    parser.add_argument("--input", default="csvs/26054757_1_2067817572683155548.csv",
                       help="Input CSV file to enrich (query table)")
    parser.add_argument("--dir", default="csvs",
                       help="Directory containing CSV files to enrich from")
    parser.add_argument("--topk", type=int, default=50,
                       help="Top K columns to add globally")
    parser.add_argument("--overlap", type=float, default=0.8,
                       help="Minimum value overlap for join keys")
    parser.add_argument("--keys", default=None,
                       help="Explicit join keys (comma-separated), e.g. 'country_iso3,year'")
    
    args = parser.parse_args()
    
    # Parse explicit keys if provided
    explicit_keys = args.keys.split(",") if args.keys else None
    
    print(f"Method: {args.method}")
    print(f"Input CSV: {args.input}")
    print(f"Source directory: {args.dir}")
    print(f"Top-K: {args.topk}")
    print(f"Min overlap: {args.overlap}")
    print(f"Explicit keys: {explicit_keys}\n")
    
    # Choose method
    if args.method == "naive":
        out_csv = "enriched_naive.csv"
        out_log = "provenance_naive.json"
        print("Running NAIVE enrichment...")
        enriched_csv, log_json = enrich_directory(
            base_csv=args.input,
            dir_path=args.dir,
            out_csv=out_csv,
            out_log=out_log,
            explicit_base_keys=explicit_keys,
            top_k_global=args.topk,
            min_overlap=args.overlap
        )
    else:  # semantic
        out_csv = "enriched_semantic.csv"
        out_log = "provenance_semantic.json"
        print("Running SEMANTIC enrichment...")
        enriched_csv, log_json = enrich_directory_semantic(
            base_csv=args.input,
            dir_path=args.dir,
            out_csv=out_csv,
            out_log=out_log,
            explicit_base_keys=explicit_keys,
            top_k_global=args.topk,
            min_overlap=args.overlap
        )
    
    print(f"\n✓ Enrichment complete!")
    print(f"  Output CSV: {enriched_csv}")
    print(f"  Provenance log: {log_json}")
    
    # Print summary
    import json
    with open(log_json, 'r') as f:
        log_data = json.load(f)
    print(f"  Added {len(log_data)} columns")

if __name__ == "__main__":
    main()

