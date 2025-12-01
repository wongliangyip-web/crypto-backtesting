import os
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from src.helper.BacktestHelper import BacktestHelper


FOLDER_PATH = r"C:\Users\User\Desktop\backtesting\interval_1h-20250830T115325Z-1\interval_1h\train data"
WINDOW_LIST = np.round(np.arange(10, 100, 5), 1)
THRESHOLD_LIST = np.round(np.arange(0, 2, 0.25), 2)
ANNUALIZATION_FACTOR = 365 * 24 # 1h data
SHARPE_THRESHOLD = 1.5
CALMAR_THRESHOLD = 2.0
OUT_FILE = r"C:\Users\User\Desktop\backtesting\interval_1h-20250830T115325Z-1\interval_1h\best_sharpe_file.txt"
# ---------------------------------------------


def process_file(file_path):
    """Run backtest for 1 file."""
    try:
        df = pd.read_csv(file_path)
        helper = BacktestHelper(df, 'zScore', ANNUALIZATION_FACTOR)
        best_sharpe, best_calmar = helper.findBestSharpe(WINDOW_LIST, THRESHOLD_LIST, 0)
        return os.path.basename(file_path), best_sharpe, best_calmar
    except Exception as e:
        return os.path.basename(file_path), f"ERROR: {e}", 0


def main():
    start_time = time.perf_counter()

    # All CSV files
    files = [
        os.path.join(FOLDER_PATH, f)
        for f in os.listdir(FOLDER_PATH)
        if f.endswith(".csv")
    ]

    print(f"Found {len(files)} CSV files. Starting multiprocessing...\n")

    # Use all CPU cores
    pool = Pool(cpu_count())

    # Run in parallel
    results = pool.map(process_file, files)

    pool.close()
    pool.join()

    # Filter only Sharpe >= threshold AND Calmar >= threshold
    good_files = [
        filename for filename, sharpe, calmar in results
        if isinstance(sharpe, (int, float)) and sharpe >= SHARPE_THRESHOLD and calmar >= CALMAR_THRESHOLD
    ]

    # Save to text file
    with open(OUT_FILE, "w") as f:
        for filename in good_files:
            f.write(filename + "\n")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("\n=== DONE ===")
    print(f"Files with Sharpe >= {SHARPE_THRESHOLD} and Calmar >= {CALMAR_THRESHOLD}: {len(good_files)}")
    print(f"Saved to: {OUT_FILE}\n")

    if elapsed_time > 60:
        mins = int(elapsed_time // 60)
        secs = int(elapsed_time % 60)
        print(f"Total Execution Time: {mins}m {secs}s")
    else:
        print(f"Total Execution Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
