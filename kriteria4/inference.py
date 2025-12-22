import requests
import json

# Prometheus Exporter endpoint (untuk tracking metrics)
MODEL_URL = "http://localhost:8001/predict"

# Sample data untuk testing
# Columns: Unnamed: 0, fixed_acidity, residual_sugar, alcohol, density
sample_data = {
    "dataframe_split": {
        "columns": ["Unnamed: 0", "fixed_acidity", "residual_sugar", "alcohol", "density"],
        "data": [
            [0, 9.3, 6.4, 13.6, 1.0005],   # Sample 1
            [1, 11.2, 2.0, 14.0, 0.9912],  # Sample 2
            [2, 11.6, 0.9, 8.2, 0.9935],   # Sample 3
        ]
    }
}

def predict():
    """Send prediction request to MLflow model server"""
    try:
        print("="*50)
        print("🚀 Sending prediction request...")
        print("="*50)
        print(f"URL: {MODEL_URL}")
        print(f"Data: {json.dumps(sample_data, indent=2)}")
        print("="*50)
        
        response = requests.post(
            MODEL_URL,
            headers={"Content-Type": "application/json"},
            json=sample_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction successful!")
            print("="*50)
            print(f"Response: {json.dumps(result, indent=2)}")
            print("="*50)
            return result
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error!")
        print("Make sure the model server is running:")
        print("  docker run -d -p 8000:8000 -e GUNICORN_CMD_ARGS=\"--bind=0.0.0.0:8000\" itsam77/kriteria3-model:latest")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    predict()

