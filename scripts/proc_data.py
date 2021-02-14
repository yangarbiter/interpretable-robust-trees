import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("../rsep_explain/datasets/datasets/car_evaluation/data.csv")
    df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]
    df.to_csv("../rsep_explain/datasets/risk_dsets/careval_data.csv", index=False)

    df = pd.read_csv("../rsep_explain/datasets/datasets/fico/fico-binary.csv")
    df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]
    df.to_csv("../rsep_explain/datasets/risk_dsets/ficobin_data.csv", index=False)

    df = pd.read_csv("../rsep_explain/datasets/datasets/compas/binned.csv")
    df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]
    df.to_csv("../rsep_explain/datasets/risk_dsets/compasbin_data.csv", index=False)


if __name__ == "__main__":
    main()
