import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file


def proc_svmlight(data_path, categorical_features, output_path):
    X, y = load_svmlight_file(data_path)
    X = X.todense()
    y = ((y + 1) // 2).reshape(len(y), 1)
    data = np.concatenate((y, X), axis=1)
    df = pd.DataFrame(data)
    if categorical_features is not None:
        df = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    df.to_csv(output_path, index=False)

def main():
    proc_svmlight("./diabetes", None, "./proc-diabetes_data.csv")
    proc_svmlight("./ionosphere_scale", None, "./proc-ionosphere_data.csv")


if __name__ == "__main__":
    main()
