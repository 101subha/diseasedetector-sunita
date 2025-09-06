from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "Disease Detector API is running!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        features = data.get("features") or []
        # Run your ML model here
        result = "Positive" if sum(map(float, features)) > 2 else "Negative"
        return jsonify({"prediction": result})
    else:
        # Simple HTML form for browser testing
        return '''
            <form method="POST">
                Features (comma separated): <input name="features">
                <input type="submit">
            </form>
        '''
