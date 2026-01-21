# app.py
from flask import Flask, render_template, jsonify
import requests

from analysis_app import run_analysis

FHIR_OBSERVATION_URL = (
    "https://thas.mohw.gov.tw/v/r4/fhir/Observation/686714"
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_backend():
    # === 1. 直接抓 Observation JSON ===
    r = requests.get(FHIR_OBSERVATION_URL, timeout=30)
    r.raise_for_status()
    observation = r.json()

    # === 2. 從 Observation 取 ECG waveform ===
    sampled = observation["valueSampledData"]
    ecg_data = sampled["data"]  # space-separated string

    # 轉成 list of float
    ecg_signal = [float(x) for x in ecg_data.split()]

    # === 3. 組成 patient_data（等同新文字文件 28） ===
    patient_data = {
        "patient_id": observation["subject"]["reference"],
        "ecg_signal": ecg_signal,
        "sampling_rate": sampled.get("frequency")
    }

    # === 4. 丟進你原本的分析 pipeline ===
    result = run_analysis(patient_data)

    return jsonify({
        "patient_id": patient_data["patient_id"],
        "risk": result["risk"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

