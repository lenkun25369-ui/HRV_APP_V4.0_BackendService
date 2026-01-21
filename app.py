# app.py
from flask import Flask, render_template, jsonify
from demo_oauth import get_access_token
from demo_fhir import fetch_patient_observation
from analysis_app import run_analysis

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_backend():
    # === 1. Backend Services OAuth ===
    token = get_access_token()

    # === 2. 存取 FHIR（或 mock）===
    patient_id = "680280"
    patient_data = fetch_patient_observation(token, patient_id)

    # === 3. 分析 ===
    result = run_analysis(patient_data)

    return jsonify({
        "patient_id": patient_id,
        "risk": result["risk"]
    })


if __name__ == "__main__":
    app.run()
