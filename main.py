from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="DisasterSense AI Service")


# ==========================================================
# LOAD TRAINED AI MODEL
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "urgency_model.pkl"
)

model = joblib.load(MODEL_PATH)


# ==========================================================
# REQUEST FORMAT
# ==========================================================

class ReportRequest(BaseModel):
    report_text: str


# ==========================================================
# HOME / HEALTH CHECK
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "DisasterSense AI Service is running"
    }


# ==========================================================
# AI PREDICTION
# ==========================================================

@app.post("/predict")
def predict(request: ReportRequest):

    # Get prediction from trained model
    prediction = model.predict([
        request.report_text
    ])[0]

    return {
        "report_text": request.report_text,
        "urgency_level": prediction
    }