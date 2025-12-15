import pandas as pd
from pathlib import Path

# Path to your data folder
DATA_FOLDER = Path(__file__).parent.parent.parent / "dev" / "python" / "convergence-study" / "meem-vs-capytaine-data" / "csv_data"

def inspect():
    if not DATA_FOLDER.exists():
        print(f"Error: Folder not found: {DATA_FOLDER}")
        return

    csv_files = sorted(list(DATA_FOLDER.glob("*.csv")))
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    # Inspect the first file
    target_file = csv_files[0]
    print(f"--- Inspecting: {target_file.name} ---")
    
    try:
        df = pd.read_csv(target_file)
        print(f"Row Count: {len(df)}")
        print("\nColumns found:")
        for col in df.columns:
            print(f"  - {col}")
            
    except Exception as e:
        print(f"Failed to read file: {e}")

if __name__ == "__main__":
    inspect()