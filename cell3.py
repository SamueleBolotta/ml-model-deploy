#!/usr/bin/env python3
"""
Docker ML Setup 
"""

import os
import subprocess
from pathlib import Path
import joblib
import numpy as np
import pkg_resources
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def run_cmd(cmd, cwd=None):
    """Esegue comando e restituisce output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr, result.returncode

# TODO: Implementare rilevamento automatico versioni per reproducibilità container
# Garantire compatibilità tra ambiente locale e Docker container
raise NotImplementedError("Implementare rilevamento versioni per reproducibilità Docker")

# TODO: Implementare validazione completa ambiente Docker
# Verificare installazione, daemon, permessi e spazio disco disponibile
raise NotImplementedError("Implementare validazione ambiente Docker per deployment")

def create_project(work_dir):
    """Crea progetto completo"""
    print(f"\n📦 Creo progetto in: {work_dir}")
    work_dir.mkdir(exist_ok=True)
    
    # Ottieni versioni locali
    local_versions = get_local_versions()
    
    # 1. MODELLO ML
    print("🤖 Addestro modello...")
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    model_package = {
        'model': model,
        'metadata': {'accuracy': accuracy, 'n_features': 10},
        'test_sample': X_test[0].tolist()
    }
    joblib.dump(model_package, work_dir / "model.pkl")
    print(f"✅ Modello salvato (accuracy: {accuracy:.3f})")
    
    # 2. DOCKERFILE - USA PYTHON 3.11
    dockerfile = '''FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py model.pkl ./
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:5000/health
CMD ["python", "app.py"]
'''
    
    # 3. REQUIREMENTS - USA VERSIONI LOCALI
    requirements = f'''flask=={local_versions['flask']}
scikit-learn=={local_versions['scikit-learn']}
numpy=={local_versions['numpy']}
joblib=={local_versions['joblib']}
'''
    
    # 4. FLASK APP
    app_code = '''import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Carica modello
try:
    MODEL_PACKAGE = joblib.load('model.pkl')
    MODEL = MODEL_PACKAGE['model']
    METADATA = MODEL_PACKAGE['metadata']
    print(f"✅ Modello caricato: {METADATA}")
except Exception as e:
    print(f"❌ Errore caricamento: {e}")
    MODEL = None
    METADATA = {}

@app.route('/health')
def health():
    if MODEL is None:
        return jsonify({'status': 'unhealthy'}), 503
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'metadata': METADATA,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not available'}), 503
    
    try:
        data = request.get_json()
        features = data['features']
        
        if len(features) != 10:
            return jsonify({'error': 'Expected 10 features'}), 400
        
        X = np.array(features).reshape(1, -1)
        prediction = MODEL.predict(X)[0]
        probabilities = MODEL.predict_proba(X)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(max(probabilities)),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    return jsonify({
        'name': 'ML API Service',
        'model_info': METADATA,
        'endpoints': ['/health', '/predict', '/info']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
    
    # 5. DOCKER COMPOSE
    compose = '''services:
  ml-api:
    build: .
    ports:
      - "5000:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
'''
    
    # 6. SCRIPT BUILD
    build_script = '''#!/bin/bash
set -e
echo "🐳 Building Docker image..."
docker build -t ml-model:latest .
echo "✅ Build completato!"

echo "🧪 Testing container..."
docker run --rm -d --name ml-test -p 5001:5000 ml-model:latest
sleep 15

if curl -f http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Health check OK"
else
    echo "❌ Health check failed"
fi

if curl -X POST http://localhost:5001/predict \\
    -H "Content-Type: application/json" \\
    -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}' > /dev/null 2>&1; then
    echo "✅ Prediction test OK"
else
    echo "❌ Prediction test failed"
fi

docker stop ml-test
echo "🎉 Test completati!"
'''
    
    # 7. SCRIPT TEST
    test_script = '''#!/bin/bash
BASE_URL=${1:-http://localhost:5000}
echo "🧪 Testing API at $BASE_URL"

echo "Testing /health..."
curl -s $BASE_URL/health | python -m json.tool

echo -e "\\nTesting /predict..."
curl -s -X POST $BASE_URL/predict \\
    -H "Content-Type: application/json" \\
    -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}' | python -m json.tool

echo -e "\\nTesting /info..."
curl -s $BASE_URL/info | python -m json.tool
'''
    
    # Scrivi tutti i file
    files = {
        'Dockerfile': dockerfile,
        'requirements.txt': requirements,
        'app.py': app_code,
        'docker-compose.yml': compose,
        'build.sh': build_script,
        'test.sh': test_script
    }
    
    for filename, content in files.items():
        filepath = work_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Rendi eseguibili gli script
        if filename.endswith('.sh'):
            os.chmod(filepath, 0o755)
    
    print("✅ File creati:")
    for filename in files.keys():
        print(f"   - {filename}")

def build_and_test(work_dir):
    """Build e test reali"""
    print("\n🐳 BUILD DOCKER IMAGE")
    print("-" * 40)
    
    # Build
    print("Building image...")
    stdout, stderr, code = run_cmd("docker build -t ml-model:latest .", cwd=work_dir)
    if code != 0:
        print(f"❌ Build fallito:\n{stderr}")
        return False
    print("✅ Build completato!")
    
    # Test container
    print("\n🧪 TEST CONTAINER")
    print("-" * 40)
    
    # Avvia container di test
    print("Avvio container di test...")
    run_cmd("docker stop ml-test 2>/dev/null || true")
    run_cmd("docker rm ml-test 2>/dev/null || true")
    
    stdout, stderr, code = run_cmd("docker run -d --name ml-test -p 5001:5000 ml-model:latest")
    if code != 0:
        print(f"❌ Avvio fallito:\n{stderr}")
        return False
    
    print("Attendo avvio servizio...")
    import time
    time.sleep(10)
    
    # TODO: Implementare suite di test automatizzati per container ML
    # Testare tutti gli endpoints, performance, health checks e error handling
    raise NotImplementedError("Implementare sistema testing automatizzato container")

    # Cleanup
    run_cmd("docker stop ml-test")
    run_cmd("docker rm ml-test")
    
    return True

def show_usage(work_dir):
    """Mostra comandi di utilizzo"""
    print(f"\n🚀 PROGETTO PRONTO!")
    print("=" * 50)
    print(f"Directory: {work_dir.absolute()}")
    print(f"\nComandi principali:")
    print(f"cd {work_dir}")
    print(f"./build.sh              # Build e test automatico")
    print(f"docker-compose up -d    # Avvia servizio")
    print(f"./test.sh               # Test API")
    print(f"docker-compose down     # Stop servizio")
    print(f"\nEndpoint disponibili:")
    print(f"http://localhost:5000/health   - Health check")
    print(f"http://localhost:5000/predict  - Predizioni ML") 
    print(f"http://localhost:5000/info     - Info servizio")
    
    print(f"\nEsempio test manuale:")
    print(f'''curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}}\'''')

def main():
    """Funzione principale"""
    print("🐳 DOCKER ML SETUP - VERSIONE REALE")
    print("=" * 50)
    
    # Verifica Docker
    if not check_docker():
        return
    
    # Crea progetto
    work_dir = Path("ml_docker_real")
    create_project(work_dir)
    
    # Build e test
    if build_and_test(work_dir):
        show_usage(work_dir)
    else:
        print("❌ Setup fallito. Controlla i log sopra.")

if __name__ == "__main__":
    main()
