"""Download the Hillstrom MineThatData e-mail experiment CSV into data/ (checksum-verified)."""

from uplift_campaign.data import RAW_FILE, download

if __name__ == "__main__":
    path = download(RAW_FILE)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
