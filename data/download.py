"""
data/download.py
================
Auto-downloads the PSM (Pooled Server Metrics) dataset from the official
eBay/RANSynCoders GitHub repository.

PSM is a standard multivariate time-series anomaly detection benchmark
with 25 server KPI features and a labeled test set.

Source:  https://github.com/eBay/RANSynCoders/tree/main/data
Paper:   "Practical Approach to Asynchronous Multivariate Time Series
          Anomaly Detection and Localization" (eBay, 2021)

File sizes (approximate):
  train.csv      ~61 MB
  test.csv       ~41 MB
  test_label.csv ~0.9 MB
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR,
    PSM_TRAIN_URL, PSM_TRAIN_FILE,
    PSM_TEST_URL,  PSM_TEST_FILE,
    PSM_LABEL_URL, PSM_LABEL_FILE,
)


def _download_file(url: str, dest: str, name: str) -> None:
    """Download a file using requests with streaming and a progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"  [skip] {name} already exists ({os.path.getsize(dest)//1024:,} KB)")
        return

    print(f"  Downloading {name} …")

    try:
        import requests
        with requests.get(url, stream=True, timeout=120,
                          allow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(f"\r    {pct:5.1f}%  ({downloaded//1024:,} / {total//1024:,} KB)",
                                  end="", flush=True)
            print(f"\r    Done — {downloaded//1024:,} KB saved to {dest}")

    except Exception as exc:
        # Clean up partial download
        if os.path.exists(dest):
            os.remove(dest)
        print(f"\n  [ERROR] Could not download {name}: {exc}")
        _print_manual_instructions(url, dest)
        raise


def _print_manual_instructions(url: str, dest: str) -> None:
    """Helpful fallback message if network is unavailable."""
    print("\n" + "=" * 65)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 65)
    print(f"  URL      : {url}")
    print(f"  Save to  : {dest}")
    print()
    print("  Option 1 (curl):")
    print(f'    curl -L -o "{dest}" "{url}"')
    print()
    print("  Option 2 (wget):")
    print(f'    wget -O "{dest}" "{url}"')
    print()
    print("  Option 3 (browser):")
    print(f"    Navigate to: {url}")
    print("    and download manually into data/raw/")
    print("=" * 65 + "\n")


def download_psm() -> None:
    """Download all three PSM CSV files (train, test, test_label)."""
    print("\n=== Downloading PSM Dataset ===")
    print("  Source: eBay/RANSynCoders (GitHub)")
    print("  Note: files are 60–100 MB each, please be patient.\n")

    files = [
        (PSM_LABEL_URL, PSM_LABEL_FILE, "test_label.csv"),  # smallest first
        (PSM_TEST_URL,  PSM_TEST_FILE,  "test.csv"),
        (PSM_TRAIN_URL, PSM_TRAIN_FILE, "train.csv"),
    ]
    for url, dest, name in files:
        _download_file(url, dest, name)

    print("\n=== PSM Download Complete ===\n")


if __name__ == "__main__":
    download_psm()
