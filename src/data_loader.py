import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath="data/adult.csv", rare_threshold=1000):
    """
    Loads Adult dataset, cleans missing values, groups rare categories in 'race',
    and returns scaled train/test splits.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Clean missing values
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Target Encoding (Income)
    df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    # Sensitive attribute processing (Race) as per notebook
    df["race"] = df["race"].str.strip()
    race_counts = df["race"].value_counts()
    
    # Grouping rare races into 'Other' to ensure robust fairness metrics
    df["race_grouped"] = df["race"].apply(
        lambda x: x if race_counts[x] >= rare_threshold else "Other"
    )
    sensitive_feature = df["race_grouped"]

    # Drop target and original race to avoid leakage/redundancy
    # We keep 'race_grouped' in X for dummy encoding if desired, 
    # but usually, we separate it as the sensitive feature 's'
    X = df.drop(columns=["income", "race", "race_grouped"])
    y = df["income"]

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # One-hot encoding for remaining categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Stratified Split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_encoded, y, sensitive_feature,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test, s_train, s_test