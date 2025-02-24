import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path, dataset_path):
    """Evaluate trained model using test dataset."""
    df = pd.read_csv(dataset_path)
    
    X = df.iloc[:, :-1]
    y_true = df.iloc[:, -1]

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    y_pred = model.predict(X)
    
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model("../models/classification_model.pkl", "../datasets/titanic.csv")
