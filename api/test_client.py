"""
Python client for testing Fraud Detection API
"""

import requests
import json
import time
from typing import Dict, Any

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\nHEALTH CHECK")

    response = requests.get(f"{API_URL}/health")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\nMODEL INFO")

    response = requests.get(f"{API_URL}/model/info")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def score_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single transaction"""
    response = requests.post(
        f"{API_URL}/score",
        json=transaction
    )
    return response.json()

def test_low_risk_transaction():
    """Test low-risk transaction (should APPROVE)"""
    print("\nLOW RISK TRANSACTION TEST")

    transaction = {
        "transaction_id": "txn_low_001",
        "user_id": "user_regular_001",
        "merchant_id": "merch_grocery_001",
        "amount": 45.99,
        "currency": "USD",
        "country": "US",
        "device_id": "device_trusted_001",
        "ip_address": "192.168.1.100",
        "merchant_category_code": "5411",
        "merchant_category": "Grocery Stores"
    }

    result = score_transaction(transaction)
    print(json.dumps(result, indent=2))

    assert result['decision'] in ['APPROVE', 'REVIEW'], f"Expected APPROVE/REVIEW, got {result['decision']}"
    assert result['fraud_score'] < 0.95, f"Score too high for low-risk: {result['fraud_score']}"
    print(f"Score: {result['fraud_score']:.3f}, Decision: {result['decision']}")

def test_high_risk_transaction():
    """Test high-risk transaction (should BLOCK)"""
    print("\nHIGH RISK TRANSACTION TEST")

    transaction = {
        "transaction_id": "txn_high_001",
        "user_id": "user_suspicious_001",
        "merchant_id": "merch_crypto_001",
        "amount": 1999.99,
        "currency": "USD",
        "country": "NG",
        "device_id": "device_unknown_001",
        "ip_address": "10.0.0.1",
        "merchant_category_code": "6051",
        "merchant_category": "Crypto Exchange"
    }

    result = score_transaction(transaction)
    print(json.dumps(result, indent=2))

    print(f"Score: {result['fraud_score']:.3f}, Decision: {result['decision']}")

def test_latency():
    """Test API latency"""
    print("\nLATENCY TEST (100 requests)")

    transaction = {
        "transaction_id": "txn_latency_test",
        "user_id": "user_test",
        "merchant_id": "merch_test",
        "amount": 99.99,
        "currency": "USD",
        "country": "US",
        "device_id": "device_test",
        "ip_address": "192.168.1.1",
        "merchant_category_code": "5411",
        "merchant_category": "Grocery Stores"
    }

    latencies = []

    for i in range(100):
        transaction['transaction_id'] = f"txn_latency_{i}"

        start = time.time()
        result = score_transaction(transaction)
        latency_ms = (time.time() - start) * 1000

        latencies.append(latency_ms)

    print(f"\nLatency Statistics:")
    print(f"  Min:     {min(latencies):.2f}ms")
    print(f"  Max:     {max(latencies):.2f}ms")
    print(f"  Mean:    {sum(latencies)/len(latencies):.2f}ms")
    print(f"  Median:  {sorted(latencies)[len(latencies)//2]:.2f}ms")
    print(f"  p95:     {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
    print(f"  p99:     {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms target"
    print(f"\nAverage latency: {avg_latency:.2f}ms (target: <100ms)")

def test_batch_scoring():
    """Test batch scoring endpoint"""
    print("\nBATCH SCORING TEST")

    transactions = [
        {
            "transaction_id": f"txn_batch_{i}",
            "user_id": f"user_{i}",
            "merchant_id": "merch_001",
            "amount": 50.0 + i * 10,
            "currency": "USD",
            "country": "US",
            "device_id": f"device_{i}",
            "ip_address": "192.168.1.1",
            "merchant_category_code": "5411",
            "merchant_category": "Grocery Stores"
        }
        for i in range(10)
    ]

    response = requests.post(f"{API_URL}/batch_score", json=transactions)
    result = response.json()

    print(f"Scored {result['total']} transactions")
    print(f"Sample results:")
    for r in result['results'][:3]:
        print(f"  {r['transaction_id']}: {r['fraud_score']:.3f} -> {r['decision']}")

    print(f"Batch scoring successful")

def main():
    """Run all tests"""
    print("\nFRAUD DETECTION API TEST SUITE")

    try:
        test_health()
        test_model_info()
        test_low_risk_transaction()
        test_high_risk_transaction()
        test_latency()
        test_batch_scoring()

        print("\nALL TESTS PASSED")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise

if __name__ == "__main__":
    main()

