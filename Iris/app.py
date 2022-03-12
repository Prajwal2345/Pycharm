import pickle

import numpy as np
from flask import Flask, render_template, request
import model
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text="The flower species is {}".format(prediction))


if __name__ == "main":
    app.run(debug=True)
