"""Download economic time‑series data from FRED.

This script fetches one or more data series from the Federal Reserve Bank of
St. Louis FRED API and saves each series as a CSV file.  It reads your FRED
API key from the project’s ``settings.yaml`` configuration file and allows
you to specify which series to download via a mapping defined in the script.

Usage
-----
Run the script from the root of your project repository:

```
python download_fred_data.py
```

It will read the API key under ``data.fred_api_key`` in
``configs/settings.yaml`` and download the default series defined in
``SERIES_TO_DOWNLOAD``.  You can modify ``SERIES_TO_DOWNLOAD`` to include
additional FRED series IDs and their desired output filenames.

Requirements
------------
The script depends only on the standard library and ``requests`` and
``pandas``; install them if they are not already available in your
environment:

```
pip install requests pandas pyyaml
```

Notes
-----
FRED limits the number of API requests per key per time period.  If you
download many series frequently, consider caching the results or
respecting the FRED terms of service.  See the FRED API documentation
for more details on allowable usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
import yaml


# -----------------------------------------------------------------------------
# Configuration
#
# Map FRED series identifiers to output CSV filenames.  Feel free to
# customise this dictionary with additional series and output paths.  The
# default entry fetches the CBOE Volatility Index (VIX) closing values,
# which is available from FRED under the series ID "VIXCLS".
SERIES_TO_DOWNLOAD: Dict[str, str] = {
    # FRED series ID : output CSV path (relative to repository root)
    "VIXCLS": "data/external/vix.csv",
    # Add additional series here as needed, e.g.:
    # "DGS10": "data/external/10yr_treasury_yield.csv",
}


def load_api_key(settings_path: Path) -> str:
    """Load the FRED API key from the given YAML settings file.

    Parameters
    ----------
    settings_path : Path
        Path to the ``settings.yaml`` configuration file.

    Returns
    -------
    str
        The FRED API key.

    Raises
    ------
    KeyError
        If the key cannot be found in the configuration.
    """
    with settings_path.open("r") as f:
        config = yaml.safe_load(f)
    try:
        # Expect the API key under data.fred_api_key
        api_key: str = config["data"]["fred_api_key"]
    except Exception as exc:
        raise KeyError(
            "No FRED API key found in settings.yaml; please add "
            "`fred_api_key` under the `data` section."
        ) from exc
    if not api_key:
        raise ValueError("FRED API key in settings.yaml is empty.")
    return api_key


def fetch_series(series_id: str, api_key: str) -> pd.DataFrame:
    """Fetch observations for a single FRED series.

    Parameters
    ----------
    series_id : str
        The FRED series identifier (e.g., ``"VIXCLS"``).
    api_key : str
        Your FRED API key.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``date`` and ``value``.

    Raises
    ------
    RuntimeError
        If the HTTP request fails or returns an unexpected response.
    """
    base_url = (
        "https://api.stlouisfed.org/fred/series/observations"
    )
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to reach FRED API for series {series_id}: {exc}"
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(
            f"FRED API returned status {response.status_code} for series {series_id}: "
            f"{response.text}"
        )
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not decode JSON for series {series_id}: {exc}"
        ) from exc
    observations = data.get("observations")
    if observations is None:
        raise RuntimeError(
            f"Unexpected response structure for series {series_id}: {data}"
        )
    df = pd.DataFrame(observations)
    # Keep only date and value columns and convert to numeric where possible
    df = df[["date", "value"]]
    df.rename(columns={"value": series_id}, inplace=True)
    # Convert value column to float; missing values are represented by '.'
    df[series_id] = pd.to_numeric(df[series_id].replace(".", pd.NA), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_series(df: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame of observations to CSV.

    Ensures that the parent directory exists before writing.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to save.  Assumes there is a ``date`` column.
    output_path : Path
        The file to write to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download one or more FRED series defined in this script and save "
            "them to CSV files."
        )
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("configs/settings.yaml"),
        help=(
            "Path to settings.yaml containing your FRED API key (default: "
            "configs/settings.yaml)"
        ),
    )
    parser.add_argument(
        "--series",
        nargs="*",
        metavar="SERIES_ID=OUTPUT.csv",
        help=(
            "Override the default list of series to download.  Provide one or "
            "more mappings of the form SERIES_ID=output_path.csv.  If not "
            "specified, the SERIES_TO_DOWNLOAD dictionary defined in this "
            "script is used."
        ),
    )
    args = parser.parse_args()
    try:
        api_key = load_api_key(args.settings)
    except Exception as exc:
        print(f"Error loading API key: {exc}", file=sys.stderr)
        return 1
    # Determine which series to download
    series_map: Dict[str, str] = SERIES_TO_DOWNLOAD.copy()
    if args.series:
        series_map.clear()
        for item in args.series:
            if "=" not in item:
                print(
                    f"Invalid --series entry '{item}'; expected format SERIES_ID=output.csv",
                    file=sys.stderr,
                )
                return 1
            series_id, out_file = item.split("=", 1)
            series_map[series_id.strip()] = out_file.strip()
    for series_id, out_path in series_map.items():
        try:
            df = fetch_series(series_id, api_key)
        except Exception as exc:
            print(f"Failed to fetch {series_id}: {exc}", file=sys.stderr)
            continue
        output_path = Path(out_path)
        try:
            save_series(df, output_path)
        except Exception as exc:
            print(f"Failed to save {series_id} to {out_path}: {exc}", file=sys.stderr)
            continue
        print(f"Downloaded series {series_id} to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())