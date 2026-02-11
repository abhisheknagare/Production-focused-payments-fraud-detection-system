import sys
!{sys.executable} -m pip install faker

!pip install faker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

#Setting seeds for reproducibility

np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

#Configuration
#0.4% fraud rate (realistic)

NUM_TRANSACTIONS = 100000
FRAUD_RATE = 0.004
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

#High-risk countries and merchants

HIGH_RISK_COUNTRIES = ['NG', 'PK', 'BD', 'VN', 'ID']  #Nigeria, Pakistan, Bangladesh, Vietnam, Indonesia
LOW_RISK_COUNTRIES = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP', 'SG']

#Merchant Category Codes (MCC)

HIGH_RISK_MCC = {
    '5816': 'Digital Goods - Games',
    '5967': 'Direct Marketing',
    '5999': 'Miscellaneous',
    '6051': 'Crypto/Forex',
    '7995': 'Gambling'
}

LOW_RISK_MCC = {
    '5411': 'Grocery Stores',
    '5812': 'Restaurants',
    '5541': 'Gas Stations',
    '5311': 'Department Stores',
    '5912': 'Pharmacies'
}

def generate_ip_address(is_fraud=False):
    """Generate realistic IP addresses"""
    if is_fraud and random.random() < 0.3:
        return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"   #Fraudulent transactions often use VPN/Proxy IPs
    return fake.ipv4_public()


def generate_device_id(is_fraud=False):
    """Generate device IDs with fraud patterns"""
    if is_fraud and random.random() < 0.4:
        return f"FRAUD_DEVICE_{random.randint(1, 50)}"    #Fraudsters often reuse devices
    return f"DEVICE_{fake.uuid4()[:8]}"


def generate_amount(is_fraud=False):
    """Generate transaction amounts with realistic distributions"""
    if is_fraud:
        if random.random() < 0.3:      #Fraudsters often test with small amounts, then do large transactions
            return round(random.uniform(1, 10), 2)  #Small test transactions
        else:
            return round(random.uniform(200, 2000), 2)  #Large fraud amounts
    else:
        return round(np.random.lognormal(3.5, 1.2), 2)  #Normal transactions follow log-normal distribution


def generate_timestamp():
    """Generate random timestamp"""
    delta = END_DATE - START_DATE
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return START_DATE + timedelta(seconds=random_seconds)


def add_label_delay(timestamp, is_fraud):
    """Simulate delay in fraud detection (realistic!)"""
    if not is_fraud:                  #Fraud is typically discovered 1-30 days later
        return None, None
    delay_days = np.random.exponential(7)  #Average 7 days delay
    delay_days = min(delay_days, 90)  #Cap at 90 days
    delay_days = int(round(delay_days))

    chargeback_date = timestamp + timedelta(days=delay_days)

    return chargeback_date, delay_days

def generate_transactions():
    """Generate complete transaction dataset"""

    transactions = []
    num_fraud = int(NUM_TRANSACTIONS * FRAUD_RATE)
    num_legitimate = NUM_TRANSACTIONS - num_fraud

    print(f"Generating {NUM_TRANSACTIONS:,} transactions...")
    print(f"  - Legitimate: {num_legitimate:,}")
    print(f"  - Fraud: {num_fraud:,}")
    print(f"  - Fraud Rate: {FRAUD_RATE*100:.2f}%\n")

    #Generate user and merchant pools
    num_users = 20000
    num_merchants = 5000

    user_ids = [f"USER_{i:06d}" for i in range(num_users)]
    merchant_ids = [f"MERCH_{i:05d}" for i in range(num_merchants)]

    #Generate legitimate transactions
    for i in range(num_legitimate):
        timestamp = generate_timestamp()
        country = random.choice(LOW_RISK_COUNTRIES + LOW_RISK_COUNTRIES)  #Bias toward low-risk
        mcc_code = random.choice(list(LOW_RISK_MCC.keys()))

        transaction = {
            'transaction_id': f"TXN_{i:08d}",
            'user_id': random.choice(user_ids),
            'merchant_id': random.choice(merchant_ids),
            'merchant_category_code': mcc_code,
            'merchant_category': LOW_RISK_MCC[mcc_code],
            'amount': generate_amount(is_fraud=False),
            'currency': 'USD',
            'country': country,
            'device_id': generate_device_id(is_fraud=False),
            'ip_address': generate_ip_address(is_fraud=False),
            'timestamp': timestamp,
            'is_fraud': 0,
            'chargeback_date': None,
            'delay_days': None,
            'transaction_hour': timestamp.hour,
            'transaction_day_of_week': timestamp.weekday()
        }
        transactions.append(transaction)

    #Generate fraudulent transactions
    for i in range(num_fraud):
        timestamp = generate_timestamp()

        #Fraud patterns
        country = random.choice(HIGH_RISK_COUNTRIES) if random.random() < 0.6 else random.choice(LOW_RISK_COUNTRIES)
        mcc_code = random.choice(list(HIGH_RISK_MCC.keys())) if random.random() < 0.7 else random.choice(list(LOW_RISK_MCC.keys()))
        mcc_category = HIGH_RISK_MCC.get(mcc_code, LOW_RISK_MCC.get(mcc_code))

        #Fraudulent transactions often happen at odd hours
        if random.random() < 0.4:
            timestamp = timestamp.replace(hour=random.randint(0, 5))

        chargeback_date, delay_days = add_label_delay(timestamp, True)

        transaction = {
            'transaction_id': f"TXN_F_{i:08d}",
            'user_id': random.choice(user_ids),
            'merchant_id': random.choice(merchant_ids),
            'merchant_category_code': mcc_code,
            'merchant_category': mcc_category,
            'amount': generate_amount(is_fraud=True),
            'currency': 'USD',
            'country': country,
            'device_id': generate_device_id(is_fraud=True),
            'ip_address': generate_ip_address(is_fraud=True),
            'timestamp': timestamp,
            'is_fraud': 1,
            'chargeback_date': chargeback_date,
            'delay_days': delay_days,
            'transaction_hour': timestamp.hour,
            'transaction_day_of_week': timestamp.weekday()
        }
        transactions.append(transaction)

    #Convert to DataFrame and shuffle
    df = pd.DataFrame(transactions)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    #Sort by timestamp (realistic)
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['delay_days'] = df['delay_days'].astype('Int64')

    return df



def add_realistic_features(df):
    """Add additional realistic features"""

    #Velocity features (transaction count in last hour per user)
    df['user_transaction_count_1h'] = df.groupby('user_id').cumcount() + 1

    #Time since last transaction
    df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60
    df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(0)

    #Weekend flag
    df['is_weekend'] = df['transaction_day_of_week'].isin([5, 6]).astype(int)

    #Night transaction flag (12 AM - 6 AM)
    df['is_night'] = df['transaction_hour'].isin(range(0, 6)).astype(int)

    return df

def main():
    """Main execution"""
    print("=" * 60)
    print("SYNTHETIC PAYMENT FRAUD DATA GENERATOR")
    print("=" * 60)
    print()

    #Generate data
    df = generate_transactions()

    #Add realistic features
    df = add_realistic_features(df)

    #Save to CSV
    output_path = '../data/raw/transactions.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✅ Data generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total Transactions: {len(df):,}")
    print(f"Fraud Transactions: {df['is_fraud'].sum():,}")
    print(f"Fraud Rate: {df['is_fraud'].mean()*100:.3f}%")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Amount Range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Merchants: {df['merchant_id'].nunique():,}")
    print(f"\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    main()

