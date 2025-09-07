import pandas as pd
import sys

def inspect_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        print("--- Data Inspection ---")
        print(f"File: {file_path}")
        print("\n--- DataFrame Info ---")
        df.info()
        print("\n--- DataFrame Head ---")
        print(df.head())
        print("\n--- DataFrame Tail ---")
        print(df.tail())
        print("\n--- Null Value Counts ---")
        print(df.isnull().sum())
        print("--- End Data Inspection ---")
    except Exception as e:
        print(f"Error inspecting data file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_data.py <file_path>", file=sys.stderr)
        sys.exit(1)
    inspect_data(sys.argv[1])
