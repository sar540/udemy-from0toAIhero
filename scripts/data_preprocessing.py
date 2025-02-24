import pandas as pd
from sklearn.impute import SimpleImputer

def load_dataset(filename):
    """Load CSV dataset."""
    return pd.read_csv(filename)

def handle_missing_values(df, strategy='mean'):
    """Fill missing values using mean, median, or mode."""
    imputer = SimpleImputer(strategy=strategy)
    df.iloc[:, :] = imputer.fit_transform(df)
    return df

def encode_categorical_columns(df):
    """Convert categorical columns into numerical using one-hot encoding."""
    return pd.get_dummies(df)

if __name__ == "__main__":
    # Example Usage
    df = load_dataset("../datasets/titanic.csv")
    df = handle_missing_values(df)
    df = encode_categorical_columns(df)
    print(df.head())
