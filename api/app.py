"""
Real-Time Fraud Detection API
FastAPI service for production fraud scoring with <100ms latency
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
from functools import lru_cache

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time payment fraud detection service",
    version="1.0.0"
)

#REQUEST/RESPONSE MODELS

class Transaction(BaseModel):
    """Transaction request schema"""

    transaction_id: str = Field(..., description="Unique transaction ID")
    user_id: str = Field(..., description="User ID")
    merchant_id: str = Field(..., description="Merchant ID")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    currency: str = Field(default="USD", description="Currency code")
    country: str = Field(..., min_length=2, max_length=2, description="ISO country code")
    device_id: str = Field(..., description="Device identifier")
    ip_address: str = Field(..., description="IP address")
    merchant_category_code: str = Field(..., description="MCC code")
    merchant_category: str = Field(..., description="Merchant category")

    #Optional contextual fields
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    user_email_domain: Optional[str] = None
    is_first_transaction: Optional[bool] = False

    @validator('amount')
    def amount_must_be_reasonable(cls, v):
        if v > 10000:
            logger.warning(f"Large transaction amount: ${v}")
        return v

    @validator('country')
    def country_must_be_uppercase(cls, v):
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_123456",
                "user_id": "user_001",
                "merchant_id": "merch_001",
                "amount": 129.99,
                "currency": "USD",
                "country": "US",
                "device_id": "device_abc123",
                "ip_address": "192.168.1.1",
                "merchant_category_code": "5411",
                "merchant_category": "Grocery Stores"
            }
        }

class FraudScore(BaseModel):
    """Fraud score response schema"""

    transaction_id: str
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    decision: str = Field(..., description="APPROVE, REVIEW, or BLOCK")
    reason: str = Field(..., description="Primary reason for decision")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    processing_time_ms: float = Field(..., description="API processing time in milliseconds")
    model_version: str = Field(default="1.0", description="Model version")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_123456",
                "fraud_score": 0.87,
                "decision": "BLOCK",
                "reason": "High fraud score (87%) - multiple suspicious signals",
                "risk_level": "HIGH",
                "processing_time_ms": 45.2,
                "model_version": "1.0"
            }
        }

#MODEL LOADER (Singleton Pattern)

class ModelLoader:
    """Singleton class to load and cache model"""

    _instance = None
    _model = None
    _scaler = None
    _feature_names = None
    _threshold = 0.950  #Production threshold for 80% recall

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Load model, scaler, and feature info at startup"""
        try:
            logger.info("Loading production model...")

            #Load LightGBM model
            with open('../models/trained/lightgbm_production.pkl', 'rb') as f:
                self._model = pickle.load(f)

            #Load baseline model with scaler (for comparison)
            with open('../models/trained/logistic_regression_baseline.pkl', 'rb') as f:
                baseline = pickle.load(f)
                self._scaler = baseline.get('scaler')

            #Load feature names
            with open('../models/trained/feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
                self._feature_names = feature_info['feature_names']

            logger.info(f"Model loaded successfully")
            logger.info(f"   Features: {len(self._feature_names)}")
            logger.info(f"   Threshold: {self._threshold}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @property
    def model(self):
        return self._model

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def threshold(self):
        return self._threshold


#Initialize model loader at startup
model_loader = ModelLoader()

#FEATURE ENGINEERING (Real-time)

class FeatureComputer:
    """Compute features for a single transaction in real-time"""

    #In-memory cache for user/device/merchant history (production would use Redis)
    _history = {
        'user_transactions': {},
        'device_users': {},
        'merchant_transactions': {},
        'user_fraud_history': {},
        'device_fraud_history': {},
        'merchant_fraud_history': {}
    }

    def __init__(self, transaction: Transaction):
        self.txn = transaction
        self.timestamp = transaction.timestamp or datetime.utcnow()
        self.features = {}

    def compute_all_features(self) -> Dict[str, float]:
        """Compute all 35 features for the transaction"""

        #Simple features (always available)
        self._compute_temporal_features()
        self._compute_amount_features()
        self._compute_geo_features()

        #Historical features (use cache/database)
        self._compute_velocity_features()
        self._compute_device_risk_features()
        self._compute_historical_risk_features()

        return self.features

    def _compute_temporal_features(self):
        """Time-based features"""
        hour = self.timestamp.hour
        day_of_week = self.timestamp.weekday()

        self.features['feat_hour'] = hour
        self.features['feat_day_of_week'] = day_of_week
        self.features['feat_is_weekend'] = 1 if day_of_week >= 5 else 0
        self.features['feat_is_night'] = 1 if 0 <= hour < 6 else 0

        #Cyclical encoding
        self.features['feat_hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.features['feat_hour_cos'] = np.cos(2 * np.pi * hour / 24)
        self.features['feat_day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        self.features['feat_day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    def _compute_amount_features(self):
        """Amount-based features"""
        amount = self.txn.amount

        self.features['feat_is_small_amount'] = 1 if amount < 10 else 0
        self.features['feat_is_large_amount'] = 1 if amount > 500 else 0

        #User average (from cache/default)
        user_avg = self._get_user_avg_amount()
        user_std = self._get_user_std_amount()
        self.features['feat_amount_vs_user_avg'] = (amount - user_avg) / (user_std + 1)

        #Merchant average (from cache/default)
        merchant_avg = self._get_merchant_avg_amount()
        merchant_std = self._get_merchant_std_amount()
        self.features['feat_amount_vs_merchant_avg'] = (amount - merchant_avg) / (merchant_std + 1)

        #Percentile (simplified)
        self.features['feat_amount_percentile_user'] = min(amount / 1000, 1.0)

    def _compute_geo_features(self):
        """Geographic features"""
        HIGH_RISK_COUNTRIES = ['NG', 'PK', 'BD', 'VN', 'ID']

        self.features['feat_is_high_risk_country'] = 1 if self.txn.country in HIGH_RISK_COUNTRIES else 0
        self.features['feat_country_change'] = self._check_country_change()
        self.features['feat_unique_countries_user_7d'] = self._get_user_country_count()
        self.features['feat_user_country_entropy'] = 0.0  #Simplified for production

    def _compute_velocity_features(self):
        """Velocity features from recent history"""
        self.features['feat_tx_count_user_1h'] = self._get_user_tx_count_1h()
        self.features['feat_tx_count_user_24h'] = self._get_user_tx_count_24h()
        self.features['feat_amount_sum_user_24h'] = self._get_user_amount_sum_24h()
        self.features['feat_amount_avg_user_24h'] = self._get_user_amount_avg_24h()
        self.features['feat_time_since_last_tx_mins'] = self._get_time_since_last_tx()
        self.features['feat_tx_count_merchant_1h'] = self._get_merchant_tx_count_1h()

    def _compute_device_risk_features(self):
        """Device and IP risk features"""
        self.features['feat_unique_users_per_device_24h'] = self._get_device_user_count()
        self.features['feat_unique_countries_per_device_7d'] = self._get_device_country_count()
        self.features['feat_unique_users_per_ip_24h'] = self._get_ip_user_count()
        self.features['feat_device_age_days'] = self._get_device_age()
        self.features['feat_ip_age_days'] = self._get_ip_age()

    def _compute_historical_risk_features(self):
        """Historical fraud rates"""
        self.features['feat_user_fraud_rate_historical'] = self._get_user_fraud_rate()
        self.features['feat_merchant_fraud_rate_historical'] = self._get_merchant_fraud_rate()
        self.features['feat_device_fraud_rate_historical'] = self._get_device_fraud_rate()

    #Helper methods (these would query Redis/database in production)
    def _get_user_avg_amount(self): return 150.0
    def _get_user_std_amount(self): return 75.0
    def _get_merchant_avg_amount(self): return 100.0
    def _get_merchant_std_amount(self): return 50.0
    def _check_country_change(self): return 0
    def _get_user_country_count(self): return 1
    def _get_user_tx_count_1h(self): return 1
    def _get_user_tx_count_24h(self): return 3
    def _get_user_amount_sum_24h(self): return self.txn.amount * 3
    def _get_user_amount_avg_24h(self): return self.txn.amount
    def _get_time_since_last_tx(self): return 120.0
    def _get_merchant_tx_count_1h(self): return 10
    def _get_device_user_count(self): return 1
    def _get_device_country_count(self): return 1
    def _get_ip_user_count(self): return 1
    def _get_device_age(self): return 30.0
    def _get_ip_age(self): return 15.0
    def _get_user_fraud_rate(self): return 0.0
    def _get_merchant_fraud_rate(self): return 0.01
    def _get_device_fraud_rate(self): return 0.0

#SCORING ENGINE

def score_transaction(transaction: Transaction) -> FraudScore:
    """Score a single transaction"""

    start_time = time.time()

    try:
        #1. Compute features
        feature_computer = FeatureComputer(transaction)
        features_dict = feature_computer.compute_all_features()

        #2. Prepare feature vector in correct order
        feature_vector = [features_dict.get(name, 0.0) for name in model_loader.feature_names]
        X = np.array(feature_vector).reshape(1, -1)

        #3. Get prediction
        fraud_score = float(model_loader.model.predict(X)[0])

        #4. Make decision based on threshold
        threshold = model_loader.threshold

        if fraud_score >= threshold:
            decision = "BLOCK"
            risk_level = "HIGH"
            reason = f"High fraud score ({fraud_score*100:.1f}%) - multiple suspicious signals"
        elif fraud_score >= threshold * 0.7:
            decision = "REVIEW"
            risk_level = "MEDIUM"
            reason = f"Moderate fraud score ({fraud_score*100:.1f}%) - requires manual review"
        else:
            decision = "APPROVE"
            risk_level = "LOW"
            reason = f"Low fraud score ({fraud_score*100:.1f}%) - transaction approved"

        #5. Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        #Log performance
        logger.info(f"Transaction {transaction.transaction_id}: score={fraud_score:.3f}, decision={decision}, time={processing_time_ms:.1f}ms")

        return FraudScore(
            transaction_id=transaction.transaction_id,
            fraud_score=round(fraud_score, 4),
            decision=decision,
            reason=reason,
            risk_level=risk_level,
            processing_time_ms=round(processing_time_ms, 2),
            model_version="1.0"
        )

    except Exception as e:
        logger.error(f"Error scoring transaction {transaction.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

#API ENDPOINTS

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Fraud Detection API",
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": model_loader.model is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "features_count": len(model_loader.feature_names),
        "threshold": model_loader.threshold,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/score", response_model=FraudScore)
async def score_endpoint(transaction: Transaction):
    """
    Score a transaction for fraud

    Returns fraud score (0-1) and decision (APPROVE/REVIEW/BLOCK)
    """
    return score_transaction(transaction)


@app.post("/batch_score")
async def batch_score_endpoint(transactions: list[Transaction]):
    """
    Score multiple transactions in batch

    More efficient for bulk processing
    """
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 transactions")

    results = []
    for txn in transactions:
        try:
            score = score_transaction(txn)
            results.append(score)
        except Exception as e:
            logger.error(f"Failed to score {txn.transaction_id}: {e}")
            results.append({
                "transaction_id": txn.transaction_id,
                "error": str(e)
            })

    return {"results": results, "total": len(results)}


@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_version": "1.0",
        "model_type": "LightGBM",
        "threshold": model_loader.threshold,
        "features_count": len(model_loader.feature_names),
        "target_latency_ms": 100,
        "recall_target": 0.80,
        "precision_target": 0.62
    }

#STARTUP/SHUTDOWN EVENTS

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Fraud Detection API...")
    logger.info(f"   Model threshold: {model_loader.threshold}")
    logger.info(f"   Features: {len(model_loader.feature_names)}")
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Fraud Detection API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
