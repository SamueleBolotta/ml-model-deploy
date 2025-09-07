# =============================================================================
# WORKSHOP: API RESTFUL CON FLASK E FASTAPI - ESECUZIONE AUTOMATICA
# Serving di modelli attraverso interfacce web moderne
# =============================================================================

import json
import time
import threading
import requests
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging
import os
import sys
from pathlib import Path
import subprocess
from multiprocessing import Process
import signal
import psutil
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("⚠️  FastAPI non disponibile - installare con: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

def main():
    """Funzione principale del workshop - esecuzione automatica"""
    
    # Crea directory per i modelli
    models_dir = Path("api_models")
    models_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("WORKSHOP: API RESTFUL CON FLASK E FASTAPI - ESECUZIONE AUTOMATICA")
    print("="*80)

    # =============================================================================
    # STEP 1: PREPARAZIONE MODELLO E DATI
    # =============================================================================

    print("\n1. PREPARAZIONE MODELLO E DATI")
    print("-" * 50)

    # Creazione dataset realistico
    np.random.seed(42)
    X, y = make_classification(
        n_samples=5000, 
        n_features=10, 
        n_informative=8,
        n_redundant=1,
        n_clusters_per_class=1,
        flip_y=0.05,  # Introduce 5% di rumore per realismo
        random_state=42, 
        n_classes=2
    )

    # Split corretto train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TODO: Validare la qualità del dataset per deployment in produzione
    # Verificare bilanciamento classi, dimensioni dataset e split corretto
    raise NotImplementedError("Implementare validazione qualità dataset per API")

    # Training pipeline con parametri realistici
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nTraining del modello...")
    start_training = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_training

    # Accuracy su train e test
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)

    print(f"Training completato in {training_time:.2f} secondi")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting gap: {(train_accuracy - test_accuracy):.4f}")

    # Salvataggio modello con metadata completa
    import sklearn
    model_metadata = {
        'model_type': 'RandomForest_Pipeline',
        'version': '1.0.0',
        'training_samples': len(X_train),
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'features_count': X_train.shape[1],
        'created_at': datetime.now().isoformat(),
        'sklearn_version': sklearn.__version__,
        'python_version': sys.version.split()[0]
    }

    # TODO: Preparare package completo per deployment con campioni di test
    # Includere modello, metadata e samples rappresentativi per validation
    raise NotImplementedError("Implementare packaging modello per deployment API")

    model_path = models_dir / 'api_model_complete.pkl'
    joblib.dump(model_package, model_path, compress=3)
    print(f"Modello salvato in: {model_path}")

    # Feature names per validazione
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    print(f"Feature attese: {feature_names}")

    # =============================================================================
    # STEP 2: CREAZIONE SERVER FILES
    # =============================================================================
    
    print("\n2. CREAZIONE FILE SERVER")
    print("-" * 50)
    
    create_flask_server(models_dir)
    if FASTAPI_AVAILABLE:
        create_fastapi_server(models_dir)
    create_test_client(models_dir)
    
    # =============================================================================
    # STEP 3: DEMO PREDIZIONI
    # =============================================================================
    
    demo_prediction(pipeline, X_test, y_test)
    
    # =============================================================================
    # STEP 4: INFORMAZIONI MODELLO
    # =============================================================================
    
    show_model_info(model_package, model_path)
    
    # =============================================================================
    # STEP 5: ISTRUZIONI D'USO
    # =============================================================================
    
    show_usage_instructions(models_dir)
    
    # =============================================================================
    # STEP 6: AVVIO AUTOMATICO SERVER (opzionale)
    # =============================================================================
    
    auto_start_servers(models_dir)
    
    # =============================================================================
    # STEP 7: RIEPILOGO FINALE
    # =============================================================================
    
    print_final_summary(train_accuracy, test_accuracy, training_time)

def create_flask_server(models_dir):
    """Crea file server Flask separato"""
    
    flask_file = models_dir / "flask_server.py"
    
    # Scriviamo il codice Flask in un file separato
    flask_code = """# =============================================================================
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
    print("\\n🧪 Avvio Flask server...")
    print("URL: http://localhost:5000")
    print("\\nPer testare:")
    print("curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{\\\"features\\\":[0.5,-1.2,2.1,-0.8,1.5,-2.1,0.3,1.8,-0.9,0.7]}'")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\\n🛑 Flask server fermato")
"""
    
    with open(flask_file, 'w', encoding='utf-8') as f:
        f.write(flask_code)
    print(f"✅ Flask server creato: {flask_file}")

def create_fastapi_server(models_dir):
    """Crea file server FastAPI separato"""
    if not FASTAPI_AVAILABLE:
        return
    
    fastapi_file = models_dir / "fastapi_server.py"
    
    fastapi_code = """# =============================================================================
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
    print("\\n🚀 Avvio FastAPI server...")
    print("URL: http://localhost:8000")
    print("Documentazione: http://localhost:8000/docs")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\\n🛑 FastAPI server fermato")
"""
    
    with open(fastapi_file, 'w', encoding='utf-8') as f:
        f.write(fastapi_code)
    print(f"✅ FastAPI server creato: {fastapi_file}")

def create_test_client(models_dir):
    """Crea client di test per le API"""
    
    test_file = models_dir / "test_client.py"
    
    test_code = """# =============================================================================
# CLIENT DI TEST PER LE API
# =============================================================================

import requests
import json
import time

def test_api(base_url, api_name):
    print(f"\\n🔬 Testing {api_name} API...")
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"  ✅ Health check: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return
    
    # Test prediction
    test_data = {
        "features": [0.5, -1.2, 2.1, -0.8, 1.5, -2.1, 0.3, 1.8, -0.9, 0.7]
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict", 
            json=test_data, 
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        else:
            print(f"  ❌ Prediction failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Prediction error: {e}")

def main():
    print("🧪 TEST AUTOMATICO DELLE API")
    print("=" * 50)
    
    # Test Flask
    test_api("http://localhost:5000", "Flask")
    
    # Test FastAPI  
    test_api("http://localhost:8000", "FastAPI")
    
    print("\\n💡 Per avviare i server:")
    print("  python api_models/flask_server.py")
    print("  python api_models/fastapi_server.py")

if __name__ == '__main__':
    main()
"""
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)
    print(f"✅ Test client creato: {test_file}")

def demo_prediction(model, X_test, y_test):
    """Dimostra predizioni usando campioni di test reali"""
    print("\n3. DEMO PREDIZIONI CON CAMPIONI REALI")
    print("-" * 50)
    
    # Usa i primi 3 campioni di test
    test_samples = X_test[:3]
    true_labels = y_test[:3]
    
    print("Predizioni su campioni di test reali:\n")
    
    # TODO: Implementare simulazione endpoint /predict con formato JSON
    # Processare campioni e restituire predizioni nel formato API standard
    raise NotImplementedError("Implementare simulazione endpoint predizione API")
    
def show_model_info(model_package, model_path):
    """Mostra informazioni dettagliate sul modello"""
    print("\n4. INFORMAZIONI MODELLO")
    print("-" * 50)
    
    metadata = model_package['metadata']
    print(f"Tipo modello: {metadata['model_type']}")
    print(f"Versione: {metadata['version']}")
    print(f"Test accuracy: {metadata['test_accuracy']:.4f}")
    print(f"Campioni training: {metadata['training_samples']}")
    print(f"Numero features: {metadata['features_count']}")
    print(f"Tempo training: {metadata['training_time']:.2f}s")
    print(f"Creato: {metadata['created_at']}")
    
    file_size = model_path.stat().st_size / 1024 / 1024
    print(f"Dimensione file: {file_size:.2f} MB")

def show_usage_instructions(models_dir):
    """Mostra istruzioni per l'uso"""
    print("\n5. ISTRUZIONI PER L'USO")
    print("-" * 50)
    
    print("\n📋 COMANDI PER AVVIARE I SERVER:")
    print(f"  Flask:   python {models_dir}/flask_server.py")
    print(f"  FastAPI: python {models_dir}/fastapi_server.py")
    
    print("\n🔗 URL DISPONIBILI:")
    print("  Flask:   http://localhost:5000")
    print("  FastAPI: http://localhost:8000")
    print("  Docs:    http://localhost:8000/docs")
    
    print("\n🧪 TEST:")
    print(f"  python {models_dir}/test_client.py")
    
    print("\n📡 ESEMPIO CURL:")
    print("curl -X POST http://localhost:5000/predict \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"features\":[0.5,-1.2,2.1,-0.8,1.5,-2.1,0.3,1.8,-0.9,0.7]}'")

def auto_start_servers(models_dir):
    """Opzione per avvio automatico server"""
    print("\n6. AVVIO AUTOMATICO SERVER")
    print("-" * 50)
    
    # TODO: Verificare disponibilità porte e preparare ambiente per deployment
    # Controllare che le porte 5000/8000 siano libere e i file server esistano
    raise NotImplementedError("Implementare verifica ambiente deployment e disponibilità porte")


def print_final_summary(train_accuracy, test_accuracy, training_time):
    """Stampa riepilogo finale"""
    print("\n7. RIEPILOGO FINALE")
    print("-" * 50)
    
    print("✅ Dataset creato e modello trainato")
    print("✅ API Flask implementata")
    if FASTAPI_AVAILABLE:
        print("✅ API FastAPI implementata con validazione Pydantic")
        print("✅ Documentazione automatica disponibile")
    else:
        print("⚠️  FastAPI non disponibile (opzionale)")
    print("✅ Sistema di test automatico implementato")
    print("✅ File server creati e pronti all'uso")
    
    print(f"\n📊 METRICHE MODELLO:")
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Overfitting Gap: {(train_accuracy - test_accuracy):.4f}")
    print(f"   Training Time: {training_time:.2f}s")
    
    print(f"\n🎯 PROSSIMI PASSI:")
    print("1. Avvia i server in terminali separati")
    print("2. Testa le API con il client automatico")
    print("3. Visita la documentazione FastAPI su /docs")
    print("4. Prova le API con curl o Postman")
    
    print(f"\n🎉 Workshop completato con successo!")

if __name__ == "__main__":
    main()
