# prometheus_exporter.py
from prometheus_client import start_http_server, Gauge
import psutil
import time
import random

# 3 METRICS
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
request_count = Gauge('request_count', 'Number of requests')

if __name__ == '__main__':
    start_http_server(8000)
    print("✅ Prometheus exporter running on port 8000")
    print("📊 Metrics: http://localhost:8000/metrics")
    
    counter = 0
    while True:
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
        request_count.set(counter % 100)
        counter += 1
        time.sleep(5)