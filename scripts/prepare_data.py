"""Prepare Data4Cyber dataset.

Usage:
    python scripts/prepare_data.py data4cyber_dataset.zip
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_NAME = "data4cyber_dataset"


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def extract_dataset(zip_path: Path) -> Path:
    """Extract Data4Cyber ZIP archive."""

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found:\n{zip_path}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    extract_path = RAW_DIR / DATASET_NAME
    extract_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extracting Data4Cyber dataset...")
    print("=" * 70)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_path)

    print(f"Extracted to:\n{extract_path}")

    return extract_path


# ---------------------------------------------------------
# Find scenario datasets
# ---------------------------------------------------------

def find_scenario_files(dataset_root: Path):
    """Find all scenario dataset.csv files."""

    dataset_files = sorted(dataset_root.rglob("dataset.csv"))

    if not dataset_files:
        raise FileNotFoundError(
            f"No dataset.csv files found inside:\n{dataset_root}"
        )

    print("\nScenario datasets found:")

    for file in dataset_files:
        print(f"  - {file}")

    return dataset_files


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

def load_scenarios(dataset_files):

    frames = []

    print("\n" + "=" * 70)
    print("Loading scenario datasets...")
    print("=" * 70)

    for file in dataset_files:

        scenario_name = file.parent.name

        print(f"\nLoading: {scenario_name}")

        try:
            df = pd.read_csv(file)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read:\n{file}\n\nError: {exc}"
            ) from exc

        if df.empty:
            print("  WARNING: Dataset is empty. Skipping.")
            continue

        # Preserve original data and add scenario identifier
        df["scenario"] = scenario_name

        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns):,}")

        frames.append(df)

    if not frames:
        raise ValueError("No valid scenario datasets were loaded.")

    print("\nCombining datasets...")

    combined = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
        sort=False
    )

    print(f"Combined rows: {len(combined):,}")
    print(f"Combined columns: {len(combined.columns):,}")

    return combined


# ---------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------

def clean_dataset(df):

    print("\n" + "=" * 70)
    print("Cleaning dataset...")
    print("=" * 70)

    # Replace infinite values
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # Remove completely duplicated rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing duplicate rows: {duplicate_count:,}")
        df = df.drop_duplicates()

    # Remove completely empty columns
    empty_columns = [
        column
        for column in df.columns
        if df[column].isna().all()
    ]

    if empty_columns:
        print(
            f"Removing completely empty columns: "
            f"{len(empty_columns)}"
        )

        df = df.drop(columns=empty_columns)

    # Keep column names clean
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ---------------------------------------------------------
# Save processed dataset
# ---------------------------------------------------------

def save_processed_dataset(df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DIR /
        "combined_dataset.csv"
    )

    print("\n" + "=" * 70)
    print("Saving processed dataset...")
    print("=" * 70)

    df.to_csv(
        output_file,
        index=False
    )

    print(f"\nProcessed dataset saved to:")
    print(output_file)

    print(f"\nFinal shape:")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    return output_file


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "python scripts/prepare_data.py "
            "data4cyber_dataset.zip"
        )

    zip_path = Path(sys.argv[1]).resolve()

    # 1. Extract
    dataset_root = extract_dataset(zip_path)

    # 2. Find all scenario datasets
    dataset_files = find_scenario_files(dataset_root)

    # 3. Load and combine
    combined_df = load_scenarios(dataset_files)

    # 4. Clean
    processed_df = clean_dataset(combined_df)

    # 5. Save
    output_file = save_processed_dataset(processed_df)

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nOutput:")
    print(output_file)


if __name__ == "__main__":
    main()