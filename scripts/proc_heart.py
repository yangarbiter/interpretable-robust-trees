"""
data url: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart

Attribute Information:


Attribute Information:
------------------------
-- 1. age
-- 2. sex
-- 3. chest pain type (4 values)
-- 4. resting blood pressure
-- 5. serum cholesterol in mg/dl
-- 6. fasting blood sugar > 120 mg/dl
-- 7. resting electrocardiographic results (values 0,1,2)
-- 8. maximum heart rate achieved
-- 9. exercise induced angina
-- 10. oldpeak = ST depression induced by exercise relative to rest
-- 11. the slope of the peak exercise ST segment
-- 12. number of major vessels (0-3) colored by flourosopy
-- 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

Attributes types
-----------------

Real: 1,4,5,8,10,12
Ordered:11,
Binary: 2,6,9
Nominal:7,3,13

Variable to be predicted
------------------------
Absence (1) or presence (2) of heart disease
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

def main():
    categorical_features = [3, 7, 13]
    X, y = load_svmlight_file("./heart.svmlight")
    X = X.todense()
    y = ((y + 1) // 2).reshape(len(y), 1)
    data = np.concatenate((y, X), axis=1)
    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    df.to_csv("./proc-heart_data.csv", index=False)




if __name__ == "__main__":
    main()