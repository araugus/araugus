"""Download all SGS series used in the project into data/raw (JSON cache) and data/panel.csv."""

from __future__ import annotations

import argparse
import logging

from bcb_forecast.data import PROJECT_ROOT, SERIES, SGSClient, build_panel, download_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = download_all(SGSClient(), refresh=args.refresh)
    panel = build_panel(raw)
    out = PROJECT_ROOT / "data" / "panel.csv"
    panel.to_csv(out)
    print(f"Panel: {panel.shape[0]} months x {panel.shape[1]} columns "
          f"({panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}) -> {out}")
    for col, spec in SERIES.items():
        last = panel[col].last_valid_index()
        print(f"  SGS {spec.code:>6}  {col:<22} last={last:%Y-%m}  {spec.name}")


if __name__ == "__main__":
    main()
