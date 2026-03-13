import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "Iris ML Model API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]

    prediction = model.predict([features])[0]

    class_names = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    return jsonify({
        "prediction": int(prediction),
        "species": class_names[prediction]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)