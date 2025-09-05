from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

# Root endpoint
@app.route("/")
def home():
    return jsonify({"message": "Disease Detector API is running!"})

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Example: expecting input like {"features": [1, 2, 3, 4]}
        features = data.get("features")
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
