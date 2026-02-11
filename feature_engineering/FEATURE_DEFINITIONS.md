# Feature Engineering Documentation

## 🎯 Overview

This document describes all engineered features for the payment fraud detection system. **All features are point-in-time correct** to prevent label leakage.

---

## 📊 Feature Categories

### 1. Velocity Features (6 features)

Velocity features detect rapid-fire transaction patterns common in fraud.

| Feature | Description | Window | Fraud Signal |
|---------|-------------|--------|--------------|
| `feat_tx_count_user_1h` | Number of transactions by user in last 1 hour | 1h | Fraudsters test cards rapidly |
| `feat_tx_count_user_24h` | Number of transactions by user in last 24 hours | 24h | Unusual activity spike |
| `feat_amount_sum_user_24h` | Total amount spent by user in last 24 hours | 24h | Large spending spree |
| `feat_amount_avg_user_24h` | Average transaction amount in last 24 hours | 24h | Spending pattern change |
| `feat_time_since_last_tx_mins` | Minutes since user's last transaction | N/A | Very fast = suspicious |
| `feat_tx_count_merchant_1h` | Transactions at merchant in last 1 hour | 1h | Merchant under attack |

**Why it matters:** Fraudsters often do velocity attacks - they test stolen cards with small amounts, then quickly make large purchases before the card is blocked.

---

### 2. Device & IP Risk Features (5 features)

These features detect device/IP reuse across multiple accounts - a strong fraud signal.

| Feature | Description | Window | Fraud Signal |
|---------|-------------|--------|--------------|
| `feat_unique_users_per_device_24h` | Number of different users on same device in 24h | 24h | Device sharing = fraud ring |
| `feat_unique_countries_per_device_7d` | Countries accessed from same device in 7 days | 7d | Impossible travel |
| `feat_unique_users_per_ip_24h` | Different users from same IP in 24h | 24h | IP sharing suspicious |
| `feat_device_age_days` | Days since device first seen | All time | New devices riskier |
| `feat_ip_age_days` | Days since IP first seen | All time | New IPs riskier |

**Why it matters:** Fraudsters often reuse devices/IPs across compromised accounts. Legitimate users typically have 1 user per device.

---

### 3. Geolocation Features (4 features)

Detect impossible travel and geographic anomalies.

| Feature | Description | Window | Fraud Signal |
|---------|-------------|--------|--------------|
| `feat_country_change` | Did user change countries since last transaction? | Last tx | Impossible travel |
| `feat_unique_countries_user_7d` | Number of countries user transacted from in 7 days | 7d | Country hopping |
| `feat_is_high_risk_country` | Transaction from high-risk country? | N/A | Geographic risk |
| `feat_user_country_entropy` | Diversity of countries for this user historically | All time | Erratic behavior |

**High-risk countries:** Nigeria (NG), Pakistan (PK), Bangladesh (BD), Vietnam (VN), Indonesia (ID)

**Why it matters:** Legitimate users typically transact from 1-2 countries. Fraudsters access accounts from many locations.

---

### 4. Historical Risk Features (3 features)

**⚠️ CRITICAL: These features prevent label leakage by using only past fraud labels.**

| Feature | Description | Calculation | Fraud Signal |
|---------|-------------|-------------|--------------|
| `feat_user_fraud_rate_historical` | User's historical fraud rate | Cumulative past frauds / cumulative past transactions | Repeat offenders |
| `feat_merchant_fraud_rate_historical` | Merchant's historical fraud rate | Same as above | Risky merchants |
| `feat_device_fraud_rate_historical` | Device's historical fraud rate | Same as above | Compromised devices |

**Leakage prevention:**
- Uses `.shift(1)` to exclude current transaction
- Only counts frauds from transactions that occurred BEFORE current one
- In production, would also need to account for label delay (fraud discovered later)

**Why it matters:** Past behavior predicts future fraud. Users/merchants/devices with fraud history are high risk.

---

### 5. Amount Features (5 features)

Detect unusual spending patterns.

| Feature | Description | Calculation | Fraud Signal |
|---------|-------------|-------------|--------------|
| `feat_amount_vs_user_avg` | Z-score of amount vs user's average | (amount - user_avg) / user_std | Unusual spending |
| `feat_amount_vs_merchant_avg` | Z-score of amount vs merchant's average | (amount - merchant_avg) / merchant_std | Unusual for merchant |
| `feat_is_small_amount` | Is amount < $10? | Binary | Testing stolen cards |
| `feat_is_large_amount` | Is amount > $500? | Binary | Large fraud attempt |
| `feat_amount_percentile_user` | Where does this amount rank in user's history? | Percentile rank | Top = unusual |

**Why it matters:** 
- Fraudsters test with small amounts ($1-$10)
- Then execute large transactions ($200-$2000)
- Amounts that deviate from user's normal pattern are suspicious

---

### 6. Temporal Features (8 features)

Time-based patterns in fraud.

| Feature | Description | Values | Fraud Signal |
|---------|-------------|--------|--------------|
| `feat_hour` | Hour of day (0-23) | 0-23 | Night = higher fraud |
| `feat_day_of_week` | Day of week (0=Mon, 6=Sun) | 0-6 | Weekend patterns |
| `feat_is_weekend` | Is weekend? | 0/1 | Weekend fraud spike |
| `feat_is_night` | Is 12 AM - 6 AM? | 0/1 | Night fraud spike |
| `feat_hour_sin` | Sine encoding of hour | -1 to 1 | Cyclical time |
| `feat_hour_cos` | Cosine encoding of hour | -1 to 1 | Cyclical time |
| `feat_day_sin` | Sine encoding of day | -1 to 1 | Cyclical day |
| `feat_day_cos` | Cosine encoding of day | -1 to 1 | Cyclical day |

**Why cyclical encoding?** 
- Hour 23 and hour 0 are close in time
- Linear encoding (0, 1, 2, ... 23) treats them as far apart
- Sin/cos encoding preserves circular relationship

**Why it matters:** Fraud is more common at night (12 AM - 6 AM) and on weekends when monitoring is reduced.

---

## 🔒 Label Leakage Prevention

### What is Label Leakage?

Using information from the future to predict the past. Example:
```python
# ❌ WRONG - Uses future fraud labels
df['fraud_rate'] = df.groupby('user_id')['is_fraud'].transform('mean')

# ✅ CORRECT - Uses only past fraud labels
df['fraud_rate'] = df.groupby('user_id')['is_fraud'].shift(1).expanding().mean()
```

### Our Safeguards

1. **Temporal Sorting**: All data sorted by timestamp before feature calculation
2. **Rolling Windows**: Only look backward in time (1h, 24h, 7d, 30d)
3. **Shift Operation**: For historical risk, we use `.shift(1)` to exclude current transaction
4. **Expanding Windows**: Features grow as we see more data, never shrink
5. **No Future Info**: Label delay means we can't use fraud labels until days later (production consideration)

### Production Considerations

In production, you'd also need to:
- Track when fraud labels arrive (chargeback_date)
- Only use fraud labels that arrived BEFORE current transaction time
- Build features that respect this delay

Example:
```python
# Only use fraud labels discovered before current transaction
mask = df['chargeback_date'] < df['timestamp']
df['fraud_rate'] = df[mask].groupby('user_id')['is_fraud'].expanding().mean()
```

---

## 📈 Feature Importance (Expected)

Based on fraud detection research, expected top features:

1. **feat_unique_users_per_device_24h** - Strong fraud signal
2. **feat_tx_count_user_1h** - Velocity attacks
3. **feat_device_fraud_rate_historical** - Past predicts future
4. **feat_amount_vs_user_avg** - Unusual spending
5. **feat_is_high_risk_country** - Geographic risk
6. **feat_is_night** - Temporal pattern
7. **feat_time_since_last_tx_mins** - Rapid transactions
8. **feat_user_fraud_rate_historical** - Repeat offenders

---

## 🎯 Interview Talking Points

### On Leakage Prevention
*"I explicitly prevent label leakage by using rolling windows and excluding the current transaction from historical calculations. In production, I'd also need to respect label delay - fraud labels arrive days after the transaction, so we can only use labels that were discovered before the current transaction time."*

### On Feature Engineering
*"Feature engineering is more important than model choice in fraud detection. I focused on velocity features to catch rapid-fire testing, device risk to detect account sharing, and historical risk to identify repeat patterns. All features are point-in-time correct."*

### On Domain Knowledge
*"I incorporated payment industry knowledge: fraudsters test with small amounts then go big, they reuse devices across accounts, and fraud spikes at night. These patterns are encoded in features like is_small_amount, unique_users_per_device, and is_night."*

---

## 🔄 Future Enhancements

1. **Network Features**: Build user-merchant-device graph features
2. **Email/Phone Risk**: If available, check email domain age, phone carrier
3. **Behavioral Biometrics**: Typing speed, mouse movements (mobile)
4. **Session Features**: Time spent on site before purchase
5. **Card BIN Features**: First 6 digits of card (issuing bank)
6. **3D Secure Status**: Whether 3DS authentication was used

---

## 📚 References

- [Stripe Radar ML](https://stripe.com/radar)
- [PayPal Risk ML](https://www.paypal.com/us/brc/article/machine-learning-fraud-detection)
- [Feature Engineering for Fraud Detection](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
