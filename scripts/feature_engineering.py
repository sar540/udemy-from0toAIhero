import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_new_features(df):
    """Add custom features like family size in Titanic dataset."""
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def scale_features(df):
    """Standardize numerical features."""
    scaler = StandardScaler()
    df.iloc[:, :] = scaler.fit_transform(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv("../datasets/titanic.csv")
    df = add_new_features(df)
    df = scale_features(df)
    print(df.head())
