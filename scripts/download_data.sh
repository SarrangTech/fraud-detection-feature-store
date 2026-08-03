#!/usr/bin/env bash
# Downloads the Kaggle Credit Card Fraud Detection dataset into data/raw/creditcard.csv.
# Requires: pip install kaggle, and ~/.kaggle/kaggle.json with your API credentials.
# See https://www.kaggle.com/docs/api
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: kaggle CLI not found. Install it with: pip install kaggle" >&2
  exit 1
fi

mkdir -p data/raw

if [ -f data/raw/creditcard.csv ]; then
  echo "data/raw/creditcard.csv already exists, skipping download."
  exit 0
fi

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

if [ ! -f data/raw/creditcard.csv ]; then
  echo "error: download finished but data/raw/creditcard.csv is missing." >&2
  exit 1
fi

echo "Downloaded data/raw/creditcard.csv ($(wc -l < data/raw/creditcard.csv) lines)"
