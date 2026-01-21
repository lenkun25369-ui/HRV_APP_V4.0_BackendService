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
    try:
        # === 1. 直接抓完整 FHIR Observation JSON ===
        r = requests.get(FHIR_OBSERVATION_URL, timeout=30)
        r.raise_for_status()
        observation = r.json()

        # === 2. ❗ 不要拆解、不重組，直接送進原本 pipeline ===
        result = run_analysis(observation)

        # === 3. 回傳結果 ===
        return jsonify({
            "patient_id": observation.get("subject", {}).get("reference", "unknown"),
            "risk": result["risk"]
        })

    except Exception as e:
        # 這行一定會出現在 Render logs，方便 debug
        print("ERROR in /run:", repr(e))
        return jsonify({"error": str(e)}), 500


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    # 本機測試用；Render 會用 gunicorn，不會跑到這行
    app.run(host="0.0.0.0", port=5000, debug=False)
