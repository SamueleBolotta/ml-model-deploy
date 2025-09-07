# =============================================================================
# CLIENT DI TEST PER LE API
# =============================================================================

import requests
import json
import time

def test_api(base_url, api_name):
    print(f"\n🔬 Testing {api_name} API...")
    
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
    
    print("\n💡 Per avviare i server:")
    print("  python api_models/flask_server.py")
    print("  python api_models/fastapi_server.py")

if __name__ == '__main__':
    main()
