import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(dataset_path):
    """Train and save a classification model."""
    df = pd.read_csv(dataset_path)
    
    # Assuming last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Save model
    with open("../models/classification_model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    print(f"Model trained with accuracy: {model.score(X_test, y_test) * 100:.2f}%")

if __name__ == "__main__":
    train_model("../datasets/titanic.csv")
