"""Download the Hindi subset of IndicVoices-R (parquet shards with embedded 48 kHz audio).

Requires accepting the dataset license on Hugging Face and `huggingface-cli login`.
"""
import argparse

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/indicvoices_r")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    path = snapshot_download(
        "ai4bharat/indicvoices_r",
        repo_type="dataset",
        allow_patterns=["Hindi/*.parquet"],
        local_dir=args.out,
        max_workers=args.workers,
    )
    print("downloaded to", path)


if __name__ == "__main__":
    main()
