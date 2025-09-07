# =============================================================================
# FLASK API SERVER
# =============================================================================

import json
import time
import uuid
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import psutil
from flask import Flask, request, jsonify
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('flask_api')

# Statistiche per monitoring
class APIStats:
    def __init__(self):
        self.requests_count = 0
        self.errors_count = 0
        self.total_processing_time = 0
        self.start_time = time.time()
    
    def add_request(self, processing_time, is_error=False):
        self.requests_count += 1
        if is_error:
            self.errors_count += 1
        else:
            self.total_processing_time += processing_time
    
    def get_stats(self):
        uptime = time.time() - self.start_time
        successful_requests = self.requests_count - self.errors_count
        avg_response_time = (self.total_processing_time / max(1, successful_requests))
        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self.requests_count,
            'successful_requests': successful_requests,
            'error_requests': self.errors_count,
            'error_rate': round(self.errors_count / max(1, self.requests_count), 3),
            'avg_response_time_ms': round(avg_response_time * 1000, 2)
        }

# Creazione app Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
api_stats = APIStats()

# Caricamento modello globale
MODEL_PATH = Path("api_models") / "api_model_complete.pkl"
try:
    model_package = joblib.load(MODEL_PATH)
    MODEL = model_package['model']
    METADATA = model_package['metadata']
    FEATURE_NAMES = model_package['feature_names']
    print("✅ Modello Flask caricato con successo")
except Exception as e:
    print(f"❌ Errore caricamento modello Flask: {e}")
    MODEL = None

@app.route('/health', methods=['GET'])
def health_check():
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    return jsonify({
        'status': 'healthy',
        'service': 'ML Prediction API - Flask',
        'model_loaded': MODEL is not None,
        'model_version': METADATA['version'] if MODEL else None,
        'uptime_seconds': round(time.time() - api_stats.start_time, 2),
        'memory_usage_mb': round(memory_usage, 2),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Modello non disponibile'}), 500
    
    data = request.get_json()
    if 'features' not in data:
        return jsonify({'error': 'Campo features mancante'}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    prediction = MODEL.predict(features)[0]
    probabilities = MODEL.predict_proba(features)[0]
    
    return jsonify({
        'prediction': int(prediction),
        'confidence': float(max(probabilities)),
        'probabilities': {
            'class_0': float(probabilities[0]),
            'class_1': float(probabilities[1])
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n🧪 Avvio Flask server...")
    print("URL: http://localhost:5000")
    print("\nPer testare:")
    print("curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{\"features\":[0.5,-1.2,2.1,-0.8,1.5,-2.1,0.3,1.8,-0.9,0.7]}'")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Flask server fermato")
