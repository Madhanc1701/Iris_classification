# import os
# import joblib
# import numpy as np
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# model = joblib.load("model/model.pkl")

# @app.route("/")
# def home():
#     return "Iris ML Model API is running"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()

#     features = [
#         data["sepal_length"],
#         data["sepal_width"],
#         data["petal_length"],
#         data["petal_width"]
#     ]

#     prediction = model.predict([features])[0]

#     class_names = {
#         0: "setosa",
#         1: "versicolor",
#         2: "virginica"
#     }

#     return jsonify({
#         "prediction": int(prediction),
#         "species": class_names[prediction]
#     })

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)

import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
model = joblib.load(MODEL_PATH)

class_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            features = [[sepal_length, sepal_width, petal_length, petal_width]]
            prediction = model.predict(features)[0]
            species = class_names[int(prediction)]

            prediction_text = f"Predicted Flower: {species}"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]]

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": int(prediction),
        "species": class_names[int(prediction)]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)