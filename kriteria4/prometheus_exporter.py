from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')

# Metrik untuk model invocations
INVOCATIONS_TOTAL = Counter('model_invocations_total', 'Total model invocations')
INVOCATIONS_SUCCESS = Counter('model_invocations_success', 'Successful model invocations')
INVOCATIONS_FAILED = Counter('model_invocations_failed', 'Failed model invocations')

# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

# Endpoint Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint proxy ke model MLflow
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()
    INVOCATIONS_TOTAL.inc()  # Track total invocations

    # MLflow model serving endpoint (port 8000)
    api_url = "http://localhost:8000/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        if response.status_code == 200:
            INVOCATIONS_SUCCESS.inc()  # Track successful invocations
        else:
            INVOCATIONS_FAILED.inc()  # Track failed invocations
        
        return jsonify(response.json()), response.status_code
    except Exception as e:
        INVOCATIONS_FAILED.inc()  # Track failed invocations
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)