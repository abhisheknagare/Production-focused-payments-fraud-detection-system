# Model Retraining Strategy

## Overview

Fraud models decay over time as fraud patterns evolve. This document outlines our retraining strategy.

---

## Monitoring Schedule

### Daily Checks
- Approval rate (target: ≥95%)
- Alert volume (target: ~42/month)
- API latency (target: <100ms)
- Error rate (target: <1%)

### Weekly Analysis
- Score distribution drift (PSI)
- Top 10 feature drift
- Fraud detection rate (if labels available)
- Precision trends

### Monthly Deep Dive
- Full feature drift analysis (all 31 features)
- New fraud pattern detection
- Model performance by segment (country, MCC, amount)
- Champion vs Challenger comparison

---

## Retrain Triggers

### Automatic Triggers (Immediate Retrain)

1. **High Feature Drift**
   - PSI ≥ 0.25 in 3+ features
   - Action: Retrain within 48 hours

2. **Score Distribution Shift**
   - Score PSI ≥ 0.25
   - Score mean shifts >20%
   - Action: Investigate, likely retrain

3. **Business Metrics Degradation**
   - Approval rate < 95% for 3+ days
   - Fraud detection rate < 75% (when labels available)
   - Action: Retrain immediately

### Manual Triggers (Judgment Call)

1. **New Fraud Pattern**
   - Security team identifies new fraud technique
   - Action: Add features, retrain

2. **Business Changes**
   - New markets launched
   - New payment methods
   - Action: Retrain with new data

3. **Scheduled Retrain**
   - Monthly retrain (even if no drift)
   - Incorporate latest fraud labels
   - Action: Always retrain

---

## Retraining Process

### 1. Data Preparation
```python
# Collect data from last 90 days
train_start = today - 90 days
train_end = today - 7 days  # Exclude last week (label delay)

# Include delayed labels
# Fraud from day 1 might be discovered on day 30
# Only use labels discovered before train_end
```

### 2. Label Backfilling
```python
# Challenge: Fraud labels arrive with delay
# Solution: Backfill labels discovered after initial training

def backfill_labels(transactions_df, new_labels_df):
    """
    Update transaction labels with newly discovered fraud
    
    transactions_df: Original data
    new_labels_df: Fraud discovered since last train
    """
    #Update fraud labels
    for idx, row in new_labels_df.iterrows():
        txn_id = row['transaction_id']
        discovered_date = row['chargeback_date']
        
        #Only update if discovered before training cutoff
        if discovered_date <= train_end:
            transactions_df.loc[
                transactions_df['transaction_id'] == txn_id, 
                'is_fraud'
            ] = 1
    
    return transactions_df
```

### 3. Champion/Challenger Framework
```python
#Train new model (Challenger)
challenger_model = train_lightgbm(
    X_train_recent,
    y_train_recent,
    params=production_params
)

#Load current production model (Champion)
champion_model = load_model('lightgbm_production_v1.0.pkl')

#Compare on holdout set
champion_metrics = evaluate(champion_model, X_test, y_test)
challenger_metrics = evaluate(challenger_model, X_test, y_test)

#Decision rules
if challenger_metrics['dollar_recall'] > champion_metrics['dollar_recall']:
    if challenger_metrics['approval_rate'] >= 95:
        promote_to_production(challenger_model)
    else:
        log_warning("Challenger has better recall but approval rate too low")
else:
    keep_champion()
```

### 4. A/B Testing
```python
#Deploy challenger to 10% of traffic
if transaction_id % 10 == 0:
    score = challenger_model.predict(features)
    model_version = "v2.0_challenger"
else:
    score = champion_model.predict(features)
    model_version = "v1.0_champion"

#Log both for comparison
log({
    'transaction_id': transaction_id,
    'model_version': model_version,
    'score': score,
    'decision': decision
})
```

### 5. Promotion Criteria

Promote Challenger to Champion if:
- Dollar recall ≥ Champion
- Approval rate ≥ 95%
- Precision ≥ 50%
- No increase in customer complaints
- Stable performance for 7 days in A/B test

---

## Retraining Frequency

### Production Schedule

| Trigger | Frequency | Data Window | Notes |
|---------|-----------|-------------|-------|
| **Scheduled** | Monthly | Last 90 days | Regular refresh |
| **High Drift** | As needed | Last 60 days | Faster adaptation |
| **New Pattern** | Immediate | Last 30 days + pattern data | Targeted fix |
| **Quarterly** | Every 3 months | Last 180 days | Major update |

### Data Windows Explained

**Training Window**: 60-90 days
- Too short: Misses seasonal patterns
- Too long: Includes outdated patterns
- Sweet spot: 90 days

**Validation Window**: Last 7-14 days
- Recent enough to catch drift
- Long enough for statistical significance

**Label Lag**: Exclude last 7 days
- Fraud labels arrive with delay (avg 7 days)
- Training on incomplete labels biases model
- Wait for labels to arrive

---

## Example Timeline

### Month 1: Deploy v1.0
- Week 1-4: Monitor daily, collect data
- Result: Model performs well (80% recall, 99.5% approval)

### Month 2: Routine Monitoring
- Week 1-3: Stable performance
- Week 4: **Score PSI = 0.28** (drift detected!)
- Action: Trigger retrain

### Month 2: Retrain Process
- Day 1: Collect last 90 days data
- Day 2: Backfill new fraud labels
- Day 3: Train challenger model
- Day 4-5: Evaluate challenger vs champion
- Day 6: Deploy challenger to 10% traffic (A/B test)
- Day 7-13: Monitor A/B test results
- Day 14: **Promote challenger to v2.0** (better performance)

### Month 3: New Champion
- Week 1-4: Monitor v2.0 performance
- Result: Improved recall (82%), stable approval (99.4%)

---

## Rollback Plan

If new model underperforms:
```bash
# Immediate rollback (< 5 minutes)
kubectl set env deployment/fraud-api MODEL_VERSION=v1.0

# Investigate issue
# - Check feature drift
# - Validate training data
# - Review model metrics

# Fix and retry
# - Retrain with corrections
# - Test more thoroughly
# - Deploy to 5% first (not 10%)
```

---

## Monitoring Dashboard Metrics

### Real-Time (Updated Every Minute)
- Current approval rate
- Alert volume (last hour)
- API latency p99
- Error rate

### Daily (Updated at Midnight)
- Score distribution histogram
- Feature drift PSI (top 10)
- Fraud detection rate (when available)
- Precision / Recall

### Weekly (Updated Monday Morning)
- Full feature drift report
- Champion vs Challenger comparison
- Business impact ($$ saved)
- Retraining recommendation

---

## Success Criteria

Model retraining is successful if:

1. **Performance Maintained**
   - Dollar recall ≥ 80%
   - Approval rate ≥ 99%
   - Precision ≥ 60%

2. **Drift Reduced**
   - Score PSI < 0.1 after retrain
   - Feature PSI < 0.25 for all features

3. **Business Impact**
   - No increase in customer complaints
   - No decrease in revenue
   - Operational burden manageable

---

## Team Responsibilities

### Data Science Team
- Monitor drift weekly
- Retrain models monthly
- Evaluate champion vs challenger
- Update feature engineering

### ML Engineering Team
- Maintain monitoring infrastructure
- Automate retraining pipeline
- Deploy models to production
- Ensure rollback capability

### Fraud Ops Team
- Review false positives
- Identify new fraud patterns
- Provide feedback on model performance
- Approve threshold changes

### Product Team
- Monitor business impact
- Approve major model changes
- Balance fraud prevention vs. customer experience

---

## Key Takeaways

1. **Models decay** - Fraud patterns evolve, retraining is mandatory
2. **Monitor continuously** - Daily checks catch issues early
3. **Automate where possible** - Retraining should be one-click
4. **Test before promoting** - A/B test prevents disasters
5. **Plan for rollback** - Things will go wrong, be ready

---

**Last Updated**: 2026-02-08  
**Owner**: Data Science Team  
**Review Schedule**: Quarterly