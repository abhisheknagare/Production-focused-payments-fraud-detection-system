# System Architecture

## High-Level Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATION                          │
│               (Web/Mobile App during checkout)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTPS POST /score
                            │ {transaction JSON}
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   API GATEWAY (AWS ALB)                         │
│              • Load balancing across pods                       │
│              • SSL termination                                  │
│              • Rate limiting                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │FastAPI   │     │FastAPI   │     │FastAPI   │
    │Pod 1     │     │Pod 2     │     │Pod 3     │
    │          │     │          │     │          │
    │ Validate │     │ Validate │     │ Validate │
    │    ↓     │     │    ↓     │     │    ↓     │
    │ Features │←───→│ Features │←───→│ Features │
    │    ↓     │     │    ↓     │     │    ↓     │
    │ Model    │     │ Model    │     │ Model    │
    │    ↓     │     │    ↓     │     │    ↓     │
    │ Decide   │     │ Decide   │     │ Decide   │
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                │
         └────────────────┴────────────────┘
                          │
                          │ Response JSON
                          ▼
                   ┌─────────────┐
                   │  Response   │
                   │  to Client  │
                   └─────────────┘


┌─────────────────────────────────────────────────────────────────┐
│              SUPPORTING INFRASTRUCTURE                          │
└─────────────────────────────────────────────────────────────────┘

  ┌─────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐
  │   Redis     │   │   Kinesis    │   │    S3    │   │CloudWatch│
  │  Cluster    │   │   Stream     │   │  Bucket  │   │          │
  │             │   │              │   │          │   │          │
  │• User hist  │   │• Predictions │   │• Training│   │• Metrics │
  │• Device     │   │• Decisions   │   │  data    │   │• Logs    │
  │• Merchant   │   │• Features    │   │• Models  │   │• Alerts  │
  │             │   │              │   │• Reports │   │          │
  │ 1-5ms       │   │ Real-time    │   │Data lake │   │Monitor   │
  └──────┬──────┘   └──────┬───────┘   └──────────┘   └──────────┘
         │                 │
         │                 ▼
         │      ┌────────────────────┐
         │      │ Flink/Spark        │
         │      │ Streaming Pipeline │
         │      │                    │
         └─────→│• Update features   │
                │• Detect drift      │
                │• Aggregate metrics │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Daily Drift Job   │
                │                    │
                │• Calculate PSI     │
                │• Monitor metrics   │
                │• Trigger retrain   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Monthly Retrain    │
                │                    │
                │• Fetch data (90d)  │
                │• Backfill labels   │
                │• Train challenger  │
                │• A/B test          │
                └────────────────────┘
```

## Simplified Flow Diagram
```
┌──────────────┐
│ Transaction  │
│    (JSON)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   FastAPI    │
│  Validator   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│  Feature Engineering         │
│                              │
│  Stateless     Stateful      │
│  (instant)     (Redis 1-5ms) │
│                              │
│  • hour        • user_hist   │
│  • amount      • device_hist │
│  • country     • merchant    │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────┐
│  LightGBM    │
│   Model      │
│ (Preloaded)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Decision    │
│   Engine     │
│              │
│ >0.95: BLOCK │
│ 0.67: REVIEW │
│ <0.67: OK    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Response    │
│  • score     │
│  • decision  │
│  • reason    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Logging    │
│  (Kinesis)   │
└──────────────┘
```

## Component Details

### API Layer (FastAPI)
- **Purpose**: Receive transactions, return decisions
- **Latency**: <2ms average
- **Scale**: 10K requests/second per pod
- **Features**: 
  - Pydantic validation
  - Auto-generated docs
  - Health checks
  - Error handling

### Feature Store (Redis)
- **Purpose**: Store user/device/merchant history
- **Latency**: 1-5ms lookup
- **Update**: Real-time via streaming pipeline
- **TTL**: 7 days for optimization
- **Fallback**: Default values if unavailable

### Model (LightGBM)
- **Loading**: Once at startup (singleton pattern)
- **Inference**: <0.1ms per prediction
- **Memory**: ~100MB per model
- **Versioning**: Champion/Challenger framework

### Monitoring
- **Real-time**: CloudWatch (latency, errors)
- **Daily**: Drift detection (PSI)
- **Weekly**: Performance metrics
- **Monthly**: Retrain decision

## Data Flow

**Request Path** (2ms total):
```
1. Transaction arrives       → 0.1ms
2. Validation (Pydantic)     → 0.2ms
3. Feature computation       → 1.0ms
   - Stateless: 0.1ms
   - Redis lookup: 0.9ms
4. Model prediction          → 0.1ms
5. Decision logic            → 0.1ms
6. JSON response             → 0.1ms
7. Async logging             → 0ms (non-blocking)
```

**Monitoring Path** (async):
```
1. Log to Kinesis            → Async
2. Spark aggregation         → Every 1 hour
3. Write to S3               → Batch
4. Daily drift job           → 2 AM daily
5. Alerts if needed          → Slack/Email
```

**Retraining Path** (monthly):
```
1. Detect drift              → Daily check
2. Fetch 90 days data        → S3 query
3. Backfill labels           → Label correction
4. Train challenger          → 2-3 hours
5. Evaluate                  → Compare to champion
6. A/B test (10% traffic)    → 7 days
7. Promote or rollback       → Based on metrics
```

## Scalability

### Current Capacity
- **Requests**: 10K/sec (3 pods × 4 workers)
- **Latency**: 1.27ms average, 11.6ms p99
- **Uptime**: 99.99% (multi-AZ)
- **Cost**: ~$650/month

### Scale to 50K requests/second
- **Horizontal**: Add 12 more pods (15 total)
- **Vertical**: Upgrade to c5.2xlarge instances
- **Redis**: Increase cluster size
- **Cost**: ~$2,000/month
- **Timeline**: Auto-scaling in <5 minutes

## Failure Handling

### API Failure
- **Detection**: Health check fails
- **Response**: Remove from load balancer
- **Healing**: Auto-restart pod
- **Fallback**: Route to healthy pods

### Redis Failure
- **Detection**: Connection timeout
- **Response**: Use default feature values
- **Impact**: Slightly lower accuracy, no downtime
- **Alert**: PagerDuty to ops team

### Model Failure
- **Detection**: Prediction error rate >1%
- **Response**: Automatic rollback to previous version
- **Timeline**: <30 seconds
- **Fallback**: Rule-based system (basic rules)

## Security

- **API**: HTTPS only, API key authentication
- **Data**: Encrypted at rest (S3) and in transit (TLS)
- **PII**: No storage of card numbers or personal data
- **Compliance**: GDPR-compliant logging (7-day retention)
- **Access**: IAM roles, least privilege principle

## Deployment

### Development
```bash
uvicorn api.app:app --reload
```

### Staging
```bash
docker build -t fraud-api:staging .
kubectl apply -f k8s/staging/
```

### Production
```bash
# Blue/Green deployment
kubectl apply -f k8s/production/green/
# Monitor for 1 hour
# Switch traffic
kubectl apply -f k8s/production/switch-to-green.yaml
```

## Monitoring Dashboard
```
┌────────────────────────────────────────────────┐
│        Fraud Detection Dashboard               │
├────────────────────────────────────────────────┤
│                                                │
│  Approval Rate:  [████████████] 99.5%          │
│  Target: 99%                                   │
│                                                │
│  Latency p99:    [██] 11.6ms                   │
│  Target: <100ms                                │
│                                                │
│  Score Drift:    [█] PSI=0.08                  │
│  Status: Stable                                │
│                                                │
│  Alerts (24h):   42 transactions               │
│  Average: 40                                   │
│                                                │
│  Error Rate:     [░] 0.1%                      │
│  Target: <1%                                   │
│                                                │
└────────────────────────────────────────────────┘
```

## Cost Breakdown

| Component | Monthly Cost |
|-----------|-------------|
| ECS Fargate (3 tasks) | $500 |
| Redis ElastiCache | $100 |
| Application Load Balancer | $50 |
| S3 Storage | $20 |
| CloudWatch | $30 |
| Kinesis | $100 |
| **Total** | **$800/month** |

**ROI**: Save $65K/quarter in fraud vs $2.4K infrastructure cost = **27× ROI**

---

## Decision Tree

**When does a transaction get blocked?**
```
Transaction arrives
       │
       ├─→ Compute 31 features
       │
       ├─→ LightGBM predicts score (0-1)
       │
       ├─→ Score >= 0.950?
       │        │
       │        ├─YES→ BLOCK (High fraud probability)
       │        │      Send to fraud team
       │        │
       │        └─NO→ Score >= 0.665?
       │                │
       │                ├─YES→ REVIEW (Manual check)
       │                │      Queue for analyst
       │                │
       │                └─NO→ APPROVE
       │                      Process payment
       │
       └─→ Log to monitoring
```

---

## Future Enhancements

### Phase 1 (Q1 2026)
- Real-time feature updates
- Network graph features
- SHAP explanations

### Phase 2 (Q2 2026)
- Multi-region deployment
- Ensemble models
- Advanced drift detection

### Phase 3 (Q3 2026)
- Deep learning experiments
- Automated retraining pipeline
- Real-time collaboration network