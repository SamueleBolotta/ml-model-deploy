# =============================================================================
# FASTAPI SERVER
# =============================================================================

import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib

# Caricamento modello
MODEL_PATH = Path("api_models") / "api_model_complete.pkl"
try:
    model_package = joblib.load(MODEL_PATH)
    MODEL = model_package['model']
    METADATA = model_package['metadata']
    FEATURE_NAMES = model_package['feature_names']
    print("✅ Modello FastAPI caricato con successo")
except Exception as e:
    print(f"❌ Errore caricamento modello FastAPI: {e}")
    MODEL = None

# Pydantic models
class Features(BaseModel):
    features: List[float]

# Creazione app FastAPI
app = FastAPI(
    title="ML Prediction API - FastAPI",
    description="API per predizioni ML con validazione automatica",
    version="2.0.0"
)

start_time = time.time()

@app.get("/health")
async def health_check():
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    return {
        'status': 'healthy',
        'service': 'ML Prediction API - FastAPI',
        'model_loaded': MODEL is not None,
        'model_version': METADATA['version'] if MODEL else None,
        'uptime_seconds': round(time.time() - start_time, 2),
        'memory_usage_mb': round(memory_usage, 2),
        'timestamp': datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(features: Features):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modello non disponibile")
    
    features_array = np.array(features.features).reshape(1, -1)
    prediction = MODEL.predict(features_array)[0]
    probabilities = MODEL.predict_proba(features_array)[0]
    
    return {
        'prediction': int(prediction),
        'confidence': float(max(probabilities)),
        'probabilities': {
            'class_0': float(probabilities[0]),
            'class_1': float(probabilities[1])
        },
        'timestamp': datetime.now().isoformat()
    }

if __name__ == '__main__':
    print("\n🚀 Avvio FastAPI server...")
    print("URL: http://localhost:8000")
    print("Documentazione: http://localhost:8000/docs")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server fermato")
