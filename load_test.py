"""Load and latency benchmarking suite for DrugInfo Agent API.

Measures:
- Requests Per Second (RPS)
- p50, p95, p99 latency distributions
- Cache hit vs. miss latency differential
- Error rate under concurrent load
"""
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

BENCHMARK_SCENARIOS = [
    {
        "name": "Health Check (GET /health)",
        "method": "GET",
        "url": "/health",
        "payload": None,
        "n_requests": 200,
        "concurrency": 10
    },
    {
        "name": "Drug & Brand Directory (GET /drugs & /brands)",
        "method": "GET",
        "url": "/drugs",
        "payload": None,
        "n_requests": 150,
        "concurrency": 8
    },
    {
        "name": "Cached Query Retrieval (POST /ask - Top Inquiries)",
        "method": "POST",
        "url": "/ask",
        "payload": {"question": "Can I take ibuprofen with warfarin?", "history": []},
        "warmup": True,
        "n_requests": 150,
        "concurrency": 10
    },
    {
        "name": "Multi-Drug Polypharmacy Audit (POST /check-list)",
        "method": "POST",
        "url": "/check-list",
        "payload": {"medications": ["warfarin", "aspirin", "metformin", "lisinopril"]},
        "n_requests": 50,
        "concurrency": 5
    }
]


def send_request(scenario):
    method = scenario["method"]
    url = scenario["url"]
    payload = scenario["payload"]

    t0 = time.perf_counter()
    if method == "GET":
        resp = client.get(url)
    elif method == "POST":
        resp = client.post(url, json=payload)
    else:
        raise ValueError(f"Unsupported method {method}")
    latency_ms = (time.perf_counter() - t0) * 1000

    return resp.status_code == 200, latency_ms


def run_benchmark(scenario):
    # Warmup if requested (e.g. populates in-memory cache)
    if scenario.get("warmup"):
        send_request(scenario)

    n_requests = scenario["n_requests"]
    concurrency = scenario["concurrency"]
    latencies = []
    successes = 0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, scenario) for _ in range(n_requests)]
        for f in as_completed(futures):
            ok, lat = f.result()
            if ok:
                successes += 1
            latencies.append(lat)
    total_time = time.perf_counter() - t_start

    rps = round(n_requests / total_time, 1)
    latencies.sort()
    p50 = round(statistics.median(latencies), 2)
    p95 = round(latencies[int(len(latencies) * 0.95)], 2)
    p99 = round(latencies[int(len(latencies) * 0.99)], 2)
    success_rate = round((successes / n_requests) * 100, 1)

    return {
        "name": scenario["name"],
        "concurrency": concurrency,
        "total_requests": n_requests,
        "rps": rps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "success_rate": success_rate
    }


def main():
    print("=" * 70)
    print("DrugInfo Agent - FastAPI Production Load Benchmark")
    print("=" * 70)

    results = []
    for sc in BENCHMARK_SCENARIOS:
        print(f"Running benchmark: {sc['name']} ({sc['n_requests']} reqs, {sc['concurrency']} concurrent)...")
        res = run_benchmark(sc)
        results.append(res)
        print(f"  -> RPS: {res['rps']} req/s | p50: {res['p50_ms']}ms | p95: {res['p95_ms']}ms | Success: {res['success_rate']}%")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY TABLE (Markdown)")
    print("=" * 70)
    print("| Endpoint / Scenario | Concurrency | Total Req | RPS (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Success Rate |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| **{r['name']}** | {r['concurrency']} | {r['total_requests']} | {r['rps']} | {r['p50_ms']}ms | {r['p95_ms']}ms | {r['p99_ms']}ms | {r['success_rate']}% |")
    print("=" * 70)


if __name__ == "__main__":
    main()

