#!/usr/bin/env python3
"""
Script per diagnosticare il problema Docker + ML
"""

import subprocess
import sys
import pkg_resources
from pathlib import Path
import json

def run_cmd(cmd, cwd=None):
    """Esegue comando e restituisce output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr, result.returncode

def check_local_versions():
    """Controlla versioni locali delle dipendenze"""
    print("🔍 VERSIONI LOCALI")
    print("-" * 40)
    
    packages = ['numpy', 'scikit-learn', 'joblib', 'flask']
    versions = {}
    
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            versions[package] = version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            versions[package] = "NOT INSTALLED"
            print(f"❌ {package}: NOT INSTALLED")
    
    return versions

def test_local_model_loading():
    """Testa il caricamento del modello in locale"""
    print("\n🤖 TEST CARICAMENTO MODELLO LOCALE")
    print("-" * 40)
    
    work_dir = Path("ml_docker_real")
    model_path = work_dir / "model.pkl"
    
    if not model_path.exists():
        print(f"❌ Modello non trovato: {model_path}")
        return False
    
    try:
        import joblib
        import numpy as np
        
        print("Carico model.pkl...")
        model_package = joblib.load(model_path)
        print(f"✅ Modello caricato: {model_package['metadata']}")
        
        # Test predizione
        test_features = model_package['test_sample']
        X = np.array(test_features).reshape(1, -1)
        prediction = model_package['model'].predict(X)[0]
        print(f"✅ Test predizione: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return False

def check_docker_image_packages():
    """Controlla le versioni dei pacchetti nell'immagine Docker"""
    print("\n🐳 VERSIONI PACCHETTI DOCKER")
    print("-" * 40)
    
    # Controlla se l'immagine esiste
    stdout, stderr, code = run_cmd("docker images ml-model:latest --format json")
    if code != 0 or not stdout.strip():
        print("❌ Immagine ml-model:latest non trovata")
        return {}
    
    # Esegui container temporaneo per controllare le versioni
    cmd = '''docker run --rm ml-model:latest python -c "
import sys
import pkg_resources
packages = ['numpy', 'scikit-learn', 'joblib', 'flask']
versions = {}
for pkg in packages:
    try:
        versions[pkg] = pkg_resources.get_distribution(pkg).version
    except:
        versions[pkg] = 'ERROR'
print('VERSIONS:', versions)
print('PYTHON:', sys.version)
"'''
    
    stdout, stderr, code = run_cmd(cmd)
    
    if code == 0:
        for line in stdout.split('\n'):
            if line.startswith('VERSIONS:'):
                versions_str = line.replace('VERSIONS: ', '')
                try:
                    versions = eval(versions_str)
                    for pkg, ver in versions.items():
                        print(f"{'✅' if ver != 'ERROR' else '❌'} {pkg}: {ver}")
                    return versions
                except:
                    print("❌ Errore parsing versioni")
            elif line.startswith('PYTHON:'):
                print(f"🐍 Python: {line.replace('PYTHON: ', '').strip()}")
    else:
        print(f"❌ Errore esecuzione: {stderr}")
    
    return {}

def test_docker_model_loading():
    """Testa il caricamento del modello nel container"""
    print("\n🐳 TEST CARICAMENTO MODELLO DOCKER")
    print("-" * 40)
    
    # Test diretto del caricamento modello
    cmd = '''docker run --rm -v $(pwd)/ml_docker_real:/app ml-model:latest python -c "
import sys
sys.path.append('/app')
try:
    import joblib
    import numpy as np
    print('✅ Import OK')
    
    model_package = joblib.load('/app/model.pkl')
    print('✅ Model load OK')
    print('Metadata:', model_package['metadata'])
    
    # Test predizione
    test_sample = model_package['test_sample']
    X = np.array(test_sample).reshape(1, -1)
    pred = model_package['model'].predict(X)[0]
    print('✅ Prediction OK:', pred)
    
except Exception as e:
    print('❌ ERROR:', str(e))
    import traceback
    traceback.print_exc()
"'''
    
    stdout, stderr, code = run_cmd(cmd)
    print("STDOUT:")
    print(stdout)
    if stderr:
        print("STDERR:")
        print(stderr)
    
    return code == 0

def generate_fixed_requirements(local_versions, docker_versions):
    """Genera requirements.txt corretto basato sui test"""
    print("\n📝 GENERA REQUIREMENTS CORRETTO")
    print("-" * 40)
    
    # Versioni note che funzionano insieme
    working_combos = {
        'conservative': {
            'numpy': '1.21.6',
            'scikit-learn': '1.1.3',
            'joblib': '1.2.0',
            'flask': '2.3.3'
        },
        'modern': {
            'numpy': '1.25.2', 
            'scikit-learn': '1.3.0',
            'joblib': '1.3.2',
            'flask': '2.3.3'
        },
        'local_match': {}
    }
    
    # Usa versioni locali se funzionano
    if local_versions:
        for pkg in ['numpy', 'scikit-learn', 'joblib', 'flask']:
            if pkg in local_versions and local_versions[pkg] != "NOT INSTALLED":
                working_combos['local_match'][pkg] = local_versions[pkg]
    
    print("Combinazioni suggerite:")
    for name, combo in working_combos.items():
        if combo:  # Skip empty combos
            print(f"\n{name.upper()}:")
            for pkg, ver in combo.items():
                print(f"  {pkg}=={ver}")
    
    return working_combos

def create_debug_dockerfile():
    """Crea Dockerfile con debug esteso"""
    print("\n🐳 GENERA DOCKERFILE DEBUG")
    print("-" * 40)
    
    debug_dockerfile = '''FROM python:3.9-slim

# Debug info
RUN python --version && pip --version

# Installa dipendenze sistema
RUN apt-get update && apt-get install -y curl gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Debug: mostra cosa c'è nel sistema
RUN pip list

# Copia requirements e installa
COPY requirements.txt .
RUN cat requirements.txt

# Install step by step per debug
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.25.2
RUN pip install --no-cache-dir scikit-learn==1.3.0  
RUN pip install --no-cache-dir joblib==1.3.2
RUN pip install --no-cache-dir flask==2.3.3

# Debug: verifica installazioni
RUN python -c "import numpy; print('NumPy:', numpy.__version__)"
RUN python -c "import sklearn; print('Sklearn:', sklearn.__version__)"
RUN python -c "import joblib; print('Joblib:', joblib.__version__)"

COPY app.py model.pkl ./

# Test caricamento modello
RUN python -c "import joblib; m = joblib.load('model.pkl'); print('Model OK:', m['metadata'])"

EXPOSE 5000
CMD ["python", "app.py"]
'''
    
    work_dir = Path("ml_docker_real")
    with open(work_dir / "Dockerfile.debug", 'w') as f:
        f.write(debug_dockerfile)
    
    print("✅ Dockerfile.debug creato")
    
    # Script per build debug
    debug_build = '''#!/bin/bash
echo "🐳 Building DEBUG image..."
docker build -f Dockerfile.debug -t ml-model:debug .
echo "Build completato!"
'''
    
    with open(work_dir / "build_debug.sh", 'w') as f:
        f.write(debug_build)
    
    import os
    os.chmod(work_dir / "build_debug.sh", 0o755)
    print("✅ build_debug.sh creato")

def main():
    """Funzione principale di diagnosi"""
    print("🔬 DIAGNOSI DOCKER ML PROBLEMA")
    print("=" * 50)
    
    # 1. Versioni locali
    local_versions = check_local_versions()
    
    # 2. Test modello locale
    local_model_ok = test_local_model_loading()
    
    # 3. Versioni Docker (se immagine esiste)
    docker_versions = check_docker_image_packages()
    
    # 4. Test modello Docker
    if docker_versions:
        docker_model_ok = test_docker_model_loading()
    else:
        docker_model_ok = False
        print("⏭️  Skip test Docker - immagine non trovata")
    
    # 5. Genera fix
    working_combos = generate_fixed_requirements(local_versions, docker_versions)
    
    # 6. Crea Dockerfile debug
    create_debug_dockerfile()
    
    # 7. Summary
    print("\n📊 SUMMARY")
    print("=" * 50)
    print(f"Modello locale:  {'✅ OK' if local_model_ok else '❌ FAIL'}")
    print(f"Modello Docker:  {'✅ OK' if docker_model_ok else '❌ FAIL'}")
    
    if not docker_model_ok:
        print("\n🔧 PROSSIMI PASSI:")
        print("1. cd ml_docker_real")
        print("2. ./build_debug.sh        # Build con debug esteso")
        print("3. Controlla output per errori specifici")
        print("4. Prova requirements 'modern' o 'conservative'")

if __name__ == "__main__":
    main()
