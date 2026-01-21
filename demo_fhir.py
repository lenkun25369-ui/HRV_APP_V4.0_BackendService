import numpy as np

def fetch_patient_observation(access_token, patient_id):
    ecg = np.random.normal(0, 1, 125 * 300).tolist()
    return {
        "patient_id": patient_id,
        "ecg_signal": ecg
    }
