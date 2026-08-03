# Dataset

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions made by European cardholders in September 2013, 492 labeled as fraud).

The raw CSV (~150MB) is **not committed to this repo**. Download it yourself:

```bash
make download-data
```

which runs `scripts/download_data.sh` and expects the Kaggle CLI to be installed and
authenticated (`pip install kaggle`, `~/.kaggle/kaggle.json` with your API token — see
https://www.kaggle.com/docs/api for how to generate one). The file is written to
`data/raw/creditcard.csv`.

## Schema

| Column      | Type    | Description                                                  |
|-------------|---------|----------------------------------------------------------------|
| `Time`      | float   | Seconds elapsed between this transaction and the first in the dataset |
| `V1`...`V28`| float   | PCA-anonymized transaction features (original features cannot be published for confidentiality) |
| `Amount`    | float   | Transaction amount                                             |
| `Class`     | int     | 1 = fraud, 0 = legitimate (492 / 284,807 = 0.173% positive rate) |

## Known limitation: no customer identifier

The dataset is anonymized at the *transaction* level and has **no account/customer/card
identifier**, which is a problem for a feature store keyed on `user_id` with per-user
velocity and spend-pattern features. This repo synthesizes one: at ingestion time
(`streaming/producer.py`), each row is deterministically hashed into a fixed pool of
`SIMULATED_USER_POOL_SIZE` (default 5,000) simulated user accounts, seeded so the
assignment is reproducible across runs. This is a standard, documented workaround used
in fraud-detection demos built on this dataset — see `docs/architecture.md` for the
full rationale and its implications for interpreting per-user features.
