from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("../models/classification_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    """API Endpoint to make predictions."""
    data = request.json["features"]
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
