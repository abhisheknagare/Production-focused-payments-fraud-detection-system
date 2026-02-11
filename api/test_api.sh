#!/bin/bash
#API Testing Script

API_URL="http://localhost:8000"

echo "FRAUD DETECTION API TESTING"

#Test 1: Health Check
echo -e "\n1. Health Check..."
curl -X GET "${API_URL}/health" | jq '.'

#Test 2: Model Info
echo -e "\n2. Model Info..."
curl -X GET "${API_URL}/model/info" | jq '.'

#Test 3: Low Risk Transaction (should APPROVE)
echo -e "\n3. Low Risk Transaction..."
curl -X POST "${API_URL}/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_low_risk_001",
    "user_id": "user_12345",
    "merchant_id": "merch_grocery_01",
    "amount": 45.99,
    "currency": "USD",
    "country": "US",
    "device_id": "device_iphone_abc",
    "ip_address": "192.168.1.100",
    "merchant_category_code": "5411",
    "merchant_category": "Grocery Stores"
  }' | jq '.'

#Test 4: High Risk Transaction (should BLOCK)
echo -e "\n4. High Risk Transaction..."
curl -X POST "${API_URL}/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_high_risk_001",
    "user_id": "user_99999",
    "merchant_id": "merch_crypto_01",
    "amount": 1999.99,
    "currency": "USD",
    "country": "NG",
    "device_id": "device_suspicious_xyz",
    "ip_address": "10.0.0.1",
    "merchant_category_code": "6051",
    "merchant_category": "Crypto Exchange"
  }' | jq '.'

#Test 5: Medium Risk Transaction (should REVIEW)
echo -e "\n5. Medium Risk Transaction..."
curl -X POST "${API_URL}/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_medium_risk_001",
    "user_id": "user_67890",
    "merchant_id": "merch_electronics_01",
    "amount": 599.99,
    "currency": "USD",
    "country": "GB",
    "device_id": "device_laptop_def",
    "ip_address": "172.16.0.50",
    "merchant_category_code": "5732",
    "merchant_category": "Electronics Stores"
  }' | jq '.'

echo "TESTS COMPLETE"