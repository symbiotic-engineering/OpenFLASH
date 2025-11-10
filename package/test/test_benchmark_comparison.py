import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---

# The test will look for CSVs in this folder relative to the test file
# Updated path to match your project structure
DATA_FOLDER = Path(__file__).parent.parent.parent / "dev" / "python" / "convergence-study" / "meem-vs-capytaine-data" / "csv_data"

# The benchmark "ground truth" we are testing against
BENCHMARK_PREFIX = "Capytaine"

# The code we are testing
TEST_PREFIX = "pyMEEM"

# Relative tolerance for comparison (e.g., 5e-2 = 5%).
# Comparing different numerical codes requires some tolerance.
# Adjust this value if the test is too strict or too loose.
RELATIVE_TOLERANCE = 5e-2 

# --- End Configuration ---


# Skip all tests in this file if the data folder doesn't exist
if not DATA_FOLDER.exists():
    pytest.skip(f"Data folder '{DATA_FOLDER}' not found. Skipping benchmark comparison tests.", allow_module_level=True)

# Find all CSV files in the data folder
CSV_FILES = sorted(DATA_FOLDER.glob("*.csv"))
CSV_FILE_IDS = [csv.name for csv in CSV_FILES] # Used for clear test naming

@pytest.mark.parametrize("csv_path", CSV_FILES, ids=CSV_FILE_IDS)
def test_pyMEEM_vs_benchmark(csv_path):
    """
    Compares pyMEEM results against a benchmark (e.g., Capytaine) 
    from a CSV file.
    
    This test works by:
    1. Loading the CSV.
    2. Finding all data series for TEST_PREFIX (e.g., "pyMEEM_Mu").
    3. Finding the corresponding BENCHMARK_PREFIX (e.g., "Capytaine_Mu").
    4. Interpolating the benchmark Y-values to match the test's X-values.
    5. Asserting the test Y-values are all close to the interpolated
       benchmark Y-values within a relative tolerance.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        pytest.fail(f"Failed to read CSV file {csv_path.name}: {e}")

    # Find all available data series "stems" (e.g., "pyMEEM_Mu", "Capytaine_Lambda")
    all_stems = set(col[:-2] for col in df.columns if col.endswith('_x'))
    
    # Find the stems we want to test
    test_stems = {s for s in all_stems if s.startswith(TEST_PREFIX)}
    
    if not test_stems:
        pytest.skip(f"No '{TEST_PREFIX}' data found in {csv_path.name}")

    # For each "pyMEEM" series, find its "Capytaine" counterpart and compare
    for test_stem in test_stems:
        # e.g., test_stem = "pyMEEM_Mu_Body1"
        # benchmark_stem = "Capytaine_Mu_Body1"
        benchmark_stem = test_stem.replace(TEST_PREFIX, BENCHMARK_PREFIX)

        if benchmark_stem not in all_stems:
            print(f"\nWarning: No benchmark data '{benchmark_stem}' for test '{test_stem}' in {csv_path.name}. Skipping this comparison.")
            continue

        # 1. Get Benchmark Data (Ground Truth)
        x_bench = df[benchmark_stem + '_x'].dropna()
        y_bench = df[benchmark_stem + '_y'].dropna()

        # 2. Get Test Data (What we are testing)
        x_test = df[test_stem + '_x'].dropna()
        y_test = df[test_stem + '_y'].dropna()

        if x_test.empty or x_bench.empty:
            pytest.fail(f"Data series '{test_stem}' or '{benchmark_stem}' in {csv_path.name} is empty.")

        # 3. Interpolate Benchmark data to match the Test x-points
        # We find the expected y-values from the benchmark at the exact
        # x-values that pyMEEM produced.
        try:
            y_expected = np.interp(x_test, x_bench, y_bench)
        except Exception as e:
            pytest.fail(f"Failed to interpolate benchmark data for '{benchmark_stem}' in {csv_path.name}: {e}")

        # 4. Assert that the actual test values are close to the expected benchmark values
        try:
            np.testing.assert_allclose(
                y_test, 
                y_expected, 
                rtol=RELATIVE_TOLERANCE,
                err_msg=f"Failed comparison for '{test_stem}' vs '{benchmark_stem}'"
            )
        except AssertionError as e:
            # Make the error message more helpful for debugging
            pytest.fail(f"Comparison failed in {csv_path.name} for {test_stem}:\n{e}")

