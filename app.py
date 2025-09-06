# app.py
import os, logging, joblib, numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("disease-api")

app = Flask(__name__)

# ---- Load model + encoders (make sure files exist in repo root) ----
MODEL_F = "disease_model.joblib"
MLB_F   = "symptom_encoder.joblib"
LE_F    = "label_encoder.joblib"

def load_artifacts():
    for f in (MODEL_F, MLB_F, LE_F):
        if not os.path.exists(f):
            log.error(f"Missing file: {f} (upload to repo root)")
            raise FileNotFoundError(f"{f} not found")
    model = joblib.load(MODEL_F)
    mlb = joblib.load(MLB_F)
    le = joblib.load(LE_F)
    log.info("Loaded model and encoders.")
    return model, mlb, le

try:
    model, mlb, le = load_artifacts()
except Exception as e:
    # If startup fails, keep app running but requests will return error details
    model = mlb = le = None
    log.exception("Failed to load artifacts: %s", e)

# ---- Helpers ----
def normalize_symptoms(raw_list):
    """Normalize strings: lower, strip, dedupe and return list"""
    out = []
    seen = set()
    for s in raw_list:
        if not isinstance(s, str):
            continue
        t = s.strip().lower()
        if not t:
            continue
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def filter_known(symptoms):
    known = set([str(x).lower() for x in list(mlb.classes_)]) if mlb is not None else set()
    valid = [s for s in symptoms if s in known]
    unknown = [s for s in symptoms if s not in known]
    return valid, unknown

# ---- Routes ----
@app.route("/")
def home():
    return "Disease Detector API is running!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if model is None or mlb is None or le is None:
        return jsonify({"error": "Model or encoders not loaded on server. Check logs."}), 500

    # 1) Read symptoms from different possible inputs
    symptoms = []
    if request.method == "POST":
        if request.is_json:
            body = request.get_json(silent=True) or {}
            # Accept either {"symptoms": ["fever","cough"]} or {"features": [...]}
            symptoms = body.get("symptoms") or body.get("features") or []
        else:
            # Form submission (from browser)
            s = request.form.get("symptoms")
            if s:
                symptoms = [x.strip() for x in s.split(",") if x.strip()]
            else:
                symptoms = request.form.getlist("symptoms") or []
    else:  # GET
        q = request.args.get("symptoms")
        if q:
            symptoms = [x.strip() for x in q.split(",") if x.strip()]
        else:
            # show a simple form to test in browser
            return """
                <h3>Predict — enter comma-separated symptoms (browser test)</h3>
                <form method="POST">
                  <input name="symptoms" style="width:400px" placeholder="fever,cough,headache"><br>
                  <input type="submit" value="Predict">
                </form>
            """

    # 2) Normalize and filter
    symptoms = normalize_symptoms(symptoms if isinstance(symptoms, (list,tuple)) else [symptoms])
    valid, unknown = filter_known(symptoms)
    if len(valid) == 0:
        return jsonify({"error": "No valid/known symptoms provided", "unknown_symptoms": unknown}), 400

    # 3) Encode and predict
    try:
        Xq = mlb.transform([valid])  # shape (1, n_features)
    except Exception as e:
        log.exception("Encoding error: %s", e)
        return jsonify({"error": "Error encoding symptoms", "detail": str(e)}), 500

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xq)[0]
            topk = np.argsort(probs)[::-1][:5]
            results = [{"disease": str(le.inverse_transform([int(i)])[0]), "probability": float(probs[i])} for i in topk]
        else:
            pred = model.predict(Xq)[0]
            results = [{"disease": str(le.inverse_transform([int(pred)])[0]), "probability": None}]
    except Exception as e:
        log.exception("Model prediction error: %s", e)
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    return jsonify({
        "input_symptoms": symptoms,
        "used_symptoms": valid,
        "unknown_symptoms": unknown,
        "predictions": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
