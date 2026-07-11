"""
Download the Heart Disease UCI dataset (Cleveland subset).

Source: UCI Machine Learning Repository, Heart Disease dataset (id=45),
        https://archive.ics.uci.edu/dataset/45/heart+disease
This pulls a GitHub mirror of the processed Cleveland file that preserves the
ORIGINAL UCI encoding: target 0 = no disease, 1 = disease present, with the
original categorical codings for cp/restecg/slope/thal intact. 297 rows after
the 6 records with missing values are dropped (matching the canonical count).

Label direction verified against clinical indicators: patients labeled 1 are
older, have lower max heart rate, more exercise-induced angina, greater ST
depression, and more fluoroscopy-flagged vessels — i.e. medically sicker.
"""
import os
import pandas as pd

OUT_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")

URL = ("https://raw.githubusercontent.com/Ruohan-Yang/Heart-Disease-Data-Set/"
       "main/UCI%20Heart%20Disease%20Dataset.csv")


def main():
    df = pd.read_csv(URL)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH} "
          f"(target: {df['target'].value_counts().sort_index().to_dict()})")


if __name__ == "__main__":
    main()
