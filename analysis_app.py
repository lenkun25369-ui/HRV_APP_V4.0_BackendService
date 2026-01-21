# analysis_app.py
import json
import os
import tempfile
import subprocess
import pandas as pd
from shock_rate import predict_shock


def run_analysis(patient_data: dict) -> dict:
    """
    patient_data: 等價於「新文字文件 (28).txt」內容
    """
    with tempfile.TemporaryDirectory() as td:
        patient_json = os.path.join(td, "patient.json")
        ecg_csv = os.path.join(td, "ecg.csv")
        hrv_csv = os.path.join(td, "hrv.csv")

        # === 寫入 patient data ===
        with open(patient_json, "w") as f:
            json.dump(patient_data, f)

        # === Parse ECG ===
        proc = subprocess.run(
            ["python", "parse_fhir_ecg_to_csv.py", patient_json, ecg_csv],
            capture_output=True,
            text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)

        # === Generate HRV ===
        proc = subprocess.run(
            ["python", "generate_HRV_10_features.py", ecg_csv, hrv_csv],
            capture_output=True,
            text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)

        # === Predict Shock ===
        preds = predict_shock(hrv_csv)
        risk = float(preds[0])

    return {
        "risk": risk
    }
