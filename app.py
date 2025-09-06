from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# load model
model = joblib.load("disease_model.joblib")

@app.route("/")
def home():
    return "Disease Detector API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run()
