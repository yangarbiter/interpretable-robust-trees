"""
data from: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
"""

import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("./bank-additional-full.csv", sep=";")
    categorical_features = [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week",
        "poutcome", "y",
    ]

    df = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]
    df.to_csv("./proc-bank_data.csv", index=False)


if __name__ == "__main__":
    main()