"""
Batch Score Full Test Dataset Through API
Validates that API predictions match offline model on complete generated dataset

This script:
1. Loads the saved test.csv (same data used in evaluation)
2. Scores all transactions through the API
3. Compares API predictions vs offline model predictions
4. Calculates fraud detection metrics
5. Measures latency statistics
"""

import pandas as pd
import requests
import numpy as np
import time
import json
import pickle
from tqdm import tqdm
from pathlib import Path
import sys

API_URL = "http://localhost:8000/score"
TEST_DATA_PATH = "../data/processed/test.csv"
MODEL_PATH = "../models/trained/lightgbm_production.pkl"
OUTPUT_DIR = "../evaluation/reports/api_validation/"

#Decision thresholds (match your production model)
BLOCK_THRESHOLD = 0.95
REVIEW_THRESHOLD = 0.665


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("API is running")
            return True
    except:
        pass
    
    print("API is not running!")
    print("\nPlease start the API in another terminal:")
    print("   cd your-project")
    print("   uvicorn api.app:app --reload")
    return False


def load_test_data():
    """Load test dataset"""
    print("\nLOADING TEST DATA")
    
    if not Path(TEST_DATA_PATH).exists():
        print(f"Test data not found at: {TEST_DATA_PATH}")
        print("\nPlease run training first to generate test split:")
        print("   python models/train_model.py")
        print("\nThis will create train.csv and test.csv in data/processed/")
        sys.exit(1)
    
    df = pd.read_csv(TEST_DATA_PATH)
    print(f"Loaded {len(df):,} test transactions")
    
    if 'is_fraud' in df.columns:
        print(f"   Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
        print(f"   Fraud cases: {df['is_fraud'].sum():,}")
        print(f"   Legitimate: {(~df['is_fraud']).sum():,}")
    else:
        print("Warning: 'is_fraud' column not found!")
        sys.exit(1)
    
    return df


def load_offline_model():
    """Load the same model the API uses for comparison"""
    print("LOADING OFFLINE MODEL")
    
    if not Path(MODEL_PATH).exists():
        print(f"Model not found at: {MODEL_PATH}")
        print("\nPlease train your model first:")
        print("   python models/train_model.py")
        sys.exit(1)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {MODEL_PATH}")
    return model


def prepare_transaction_for_api(row):
    """Convert dataframe row to API request format"""
    exclude_cols = ['is_fraud', 'timestamp']
    transaction = {k: v for k, v in row.items() 
                   if k not in exclude_cols and pd.notna(v)}
    
    #Convert numpy types to Python types
    transaction = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                   for k, v in transaction.items()}
    
    #Fix merchant_category_code - API expects string
    if 'merchant_category_code' in transaction:
        transaction['merchant_category_code'] = str(int(transaction['merchant_category_code']))
    
    return transaction


def score_through_api(test_df, sample_size=None, show_progress=True):
    """
    Score transactions through API with rate limiting and error handling
    """
    print("\nSCORING THROUGH API")
    
    #Sample if requested
    if sample_size:
        test_df = test_df.sample(min(sample_size, len(test_df)), random_state=42)
        print(f"Sampling {len(test_df):,} transactions for testing")
    else:
        print(f"Scoring all {len(test_df):,} transactions")
    
    results = []
    latencies = []
    errors = 0
    error_messages = []
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    print("\nProcessing transactions...")
    
    #Create session for connection pooling
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=3
    )
    session.mount('http://', adapter)
    
    iterator = tqdm(test_df.iterrows(), total=len(test_df), disable=not show_progress)
    
    for idx, row in iterator:
        transaction = prepare_transaction_for_api(row)
        
        #Measure latency
        start_time = time.time()
        
        try:
            response = session.post(
                API_URL,
                json=transaction,
                timeout=10
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                prediction = response.json()
                
                results.append({
                    'true_label': int(row['is_fraud']),
                    'fraud_score': prediction.get('fraud_score', prediction.get('fraud_probability', 0)),
                    'decision': prediction.get('decision', 'UNKNOWN'),
                    'latency_ms': latency_ms
                })
                
                latencies.append(latency_ms)
                consecutive_errors = 0  #Reset on success
                
            else:
                errors += 1
                consecutive_errors += 1
                error_msg = f"Status {response.status_code}: {response.text[:100]}"
                if errors <= 3:  
                    print(f"\n{error_msg}")
                if errors == 1:
                    error_messages.append(error_msg)
                
        except requests.exceptions.ConnectionError as e:
            errors += 1
            consecutive_errors += 1
            error_msg = f"Connection failed: {str(e)[:100]}"
            if errors <= 3:
                print(f"\n{error_msg}")
            if errors == 1:
                error_messages.append(error_msg)
            
            #Stop if API server has crashed
            if consecutive_errors >= max_consecutive_errors:
                print(f"\nToo many consecutive errors ({consecutive_errors})!")
                print("API server may have crashed. Stopping early.")
                print(f"Successfully processed: {len(results):,} transactions")
                break
                
        except Exception as e:
            errors += 1
            consecutive_errors += 1
            error_msg = f"Request failed: {str(e)[:100]}"
            if errors <= 3:
                print(f"\n{error_msg}")
            if errors == 1:
                error_messages.append(error_msg)
        
        #Rate limiting: pause every 100 requests
        if (len(results) + errors) % 100 == 0:
            time.sleep(0.05)  #50ms pause
        
        #Longer pause every 1000 requests
        if (len(results) + errors) % 1000 == 0 and (len(results) + errors) < len(test_df):
            time.sleep(0.5)  #500ms pause
    
    #Close session
    session.close()
    
    results_df = pd.DataFrame(results)
    
    print("\nLATENCY STATISTICS")
    print(f"Total requests:   {len(results_df):,}")
    print(f"Failed requests:  {errors:,}")
    
    if len(results_df) > 0:
        print(f"Success rate:     {len(results_df)/(len(results_df)+errors)*100:.1f}%")
    else:
        print(f"Success rate:     0.0%")
    
    if len(latencies) > 0:
        print(f"\nLatency Distribution:")
        print(f"  Mean:     {np.mean(latencies):>8.2f} ms")
        print(f"  Median:   {np.median(latencies):>8.2f} ms")
        print(f"  Std Dev:  {np.std(latencies):>8.2f} ms")
        print(f"  Min:      {np.min(latencies):>8.2f} ms")
        print(f"  Max:      {np.max(latencies):>8.2f} ms")
        print(f"  P50:      {np.percentile(latencies, 50):>8.2f} ms")
        print(f"  P95:      {np.percentile(latencies, 95):>8.2f} ms")
        print(f"  P99:      {np.percentile(latencies, 99):>8.2f} ms")
        
        #Throughput
        total_time = sum(latencies) / 1000  
        throughput = len(latencies) / total_time if total_time > 0 else 0
        print(f"\nThroughput:       {throughput:.1f} requests/second")
    
    if errors > 0 and error_messages:
        print(f"\nFirst error: {error_messages[0]}")
    
    return results_df, latencies, errors


def compare_with_offline_model(results_df, test_df, model):
    """
    Compare API predictions with offline model predictions
    """
    
    print("\nCOMPARING API vs OFFLINE MODEL PREDICTIONS")
    
    #Get ONLY feature columns (those starting with 'feat_')
    feature_columns = [col for col in test_df.columns if col.startswith('feat_')]
    
    X_test = test_df[feature_columns].head(len(results_df))
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    #Get offline predictions
    offline_scores = model.predict(X_test)
    
    api_scores = results_df['fraud_score'].values
    
    differences = np.abs(api_scores - offline_scores)
    
    print(f"Comparison Statistics:")
    print(f"  Mean absolute difference:  {np.mean(differences):.8f}")
    print(f"  Median difference:         {np.median(differences):.8f}")
    print(f"  Max difference:            {np.max(differences):.8f}")
    print(f"  Std dev of difference:     {np.std(differences):.8f}")
    
    #Count how many are within tolerance
    within_0001 = (differences < 0.0001).sum()
    within_001 = (differences < 0.001).sum()
    within_01 = (differences < 0.01).sum()
    
    total = len(differences)
    print(f"\nPrediction Alignment:")
    print(f"  Within 0.0001: {within_0001:>6,} ({within_0001/total*100:>5.1f}%)")
    print(f"  Within 0.001:  {within_001:>6,} ({within_001/total*100:>5.1f}%)")
    print(f"  Within 0.01:   {within_01:>6,} ({within_01/total*100:>5.1f}%)")
    
    if np.max(differences) < 0.001:
        print("\nEXCELLENT: API predictions match offline model perfectly!")
    elif np.max(differences) < 0.01:
        print("\nGOOD: API predictions match offline model closely")
    else:
        print("\nWARNING: API predictions differ from offline model")
        print("    This may indicate:")
        print("    - Different preprocessing in API vs training")
        print("    - Different model versions")
        print("    - Feature engineering discrepancies")
    
    return differences


def calculate_fraud_metrics(results_df):
    """
    Calculate fraud detection metrics from API predictions
    """
    
    print("\nFRAUD DETECTION METRICS (Multiple Thresholds)")
    
    from sklearn.metrics import (
        confusion_matrix, 
        roc_auc_score,
        average_precision_score
    )
    
    y_true = results_df['true_label'].values
    y_score = results_df['fraud_score'].values
    
    #Overall metrics
    print(f"\nOverall Statistics:")
    print(f"  Total transactions:    {len(results_df):,}")
    print(f"  Actual fraud cases:    {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    print(f"  Legitimate cases:      {(~y_true.astype(bool)).sum():,}")
    
    try:
        auc_roc = roc_auc_score(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        print(f"  AUC-ROC:               {auc_roc:.4f}")
        print(f"  Average Precision:     {avg_precision:.4f}")
    except:
        print("Could not calculate AUC/AP (need both classes)")
    
    #Test multiple thresholds
    thresholds_to_test = [0.5, 0.665, 0.8, 0.9, 0.95, 0.99]
    
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Approval %':<12}")
    
    metrics_by_threshold = {}
    
    for threshold in thresholds_to_test:
        y_pred = (y_score >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        approval_rate = (tn + fn) / (tn + fp + fn + tp)
        
        print(f"{threshold:<12.2f} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {approval_rate*100:<12.2f}")
        
        metrics_by_threshold[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'approval_rate': approval_rate,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    #Production threshold (0.95)
    print(f"PRODUCTION THRESHOLD ({BLOCK_THRESHOLD})")
    
    y_pred_prod = (y_score >= BLOCK_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_prod).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    approval_rate = (tn + fn) / (tn + fp + fn + tp)
    
    print(f"Confusion Matrix:")
    print(f"  True Positives (Caught Fraud):     {tp:>6,}")
    print(f"  False Positives (Blocked Legit):   {fp:>6,}")
    print(f"  True Negatives (Approved Legit):   {tn:>6,}")
    print(f"  False Negatives (Missed Fraud):    {fn:>6,}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Precision (When we block, % correct):    {precision*100:>6.2f}%")
    print(f"  Recall (% of fraud we catch):             {recall*100:>6.2f}%")
    print(f"  F1 Score:                                 {f1:>6.3f}")
    print(f"  Approval Rate (% customers not blocked):  {approval_rate*100:>6.2f}%")
    
    #Decision distribution
    print("\nDECISION DISTRIBUTION")
    
    decision_counts = results_df['decision'].value_counts()
    print(f"API Decisions:")
    for decision, count in decision_counts.items():
        print(f"  {decision:<10} {count:>8,} ({count/len(results_df)*100:>5.1f}%)")
    
    return metrics_by_threshold


def save_results(results_df, latencies, metrics, differences, errors):
    """Save all results to files"""
    
    print("SAVING RESULTS")
    
    #Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    #Save predictions
    results_file = f"{OUTPUT_DIR}/api_predictions.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved predictions to: {results_file}")
    
    summary = {
        'test_info': {
            'total_transactions': len(results_df),
            'fraud_cases': int(results_df['true_label'].sum()),
            'legitimate_cases': int((~results_df['true_label'].astype(bool)).sum()),
            'failed_requests': errors,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'latency_stats': {
            'mean_ms': float(np.mean(latencies)) if latencies else None,
            'median_ms': float(np.median(latencies)) if latencies else None,
            'std_ms': float(np.std(latencies)) if latencies else None,
            'min_ms': float(np.min(latencies)) if latencies else None,
            'max_ms': float(np.max(latencies)) if latencies else None,
            'p50_ms': float(np.percentile(latencies, 50)) if latencies else None,
            'p95_ms': float(np.percentile(latencies, 95)) if latencies else None,
            'p99_ms': float(np.percentile(latencies, 99)) if latencies else None,
            'throughput_per_sec': float(len(latencies) / (sum(latencies) / 1000)) if latencies and sum(latencies) > 0 else None
        },
        'model_comparison': {
            'mean_difference': float(np.mean(differences)),
            'max_difference': float(np.max(differences)),
            'predictions_match': bool(np.max(differences) < 0.001)
        },
        'metrics': {
            str(k): {
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1']),
                'approval_rate': float(v['approval_rate'])
            }
            for k, v in metrics.items()
        }
    }
    
    summary_file = f"{OUTPUT_DIR}/api_test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_file}")
    
    # Save latency distribution
    if latencies:
        latency_df = pd.DataFrame({'latency_ms': latencies})
        latency_file = f"{OUTPUT_DIR}/latency_distribution.csv"
        latency_df.to_csv(latency_file, index=False)
        print(f"Saved latencies to: {latency_file}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")


def main():
    """Main execution"""
    
    print("\nAPI VALIDATION ON FULL TEST DATASET")
    
    #Check API is running
    if not check_api_health():
        sys.exit(1)
    
    #Load data and model
    test_df = load_test_data()
    model = load_offline_model()
    
    print("\nTEST SIZE SELECTION")
    print("Options:")
    print("  1. Quick test (1,000 transactions) - ~2-3 seconds")
    print("  2. Medium test (5,000 transactions) - ~10-15 seconds")
    print("  3. Full test (all transactions) - may take 1-2 minutes")
    print()
    
    choice = input("Enter choice (1/2/3) or press Enter for full test: ").strip()
    
    if choice == '1':
        sample_size = 1000
    elif choice == '2':
        sample_size = 5000
    else:
        sample_size = None  #Full dataset
    
    #Score through API
    results_df, latencies, errors = score_through_api(test_df, sample_size=sample_size)
    
    if len(results_df) == 0:
        print("\nNo successful predictions! Please check API logs.")
        print("\nCommon issues:")
        print("  - API not handling feature names correctly")
        print("  - Missing required fields in API request")
        print("  - API expecting different data format")
        sys.exit(1)
    
    #Compare with offline model
    differences = compare_with_offline_model(results_df, test_df, model)
    
    #Calculate metrics
    metrics = calculate_fraud_metrics(results_df)
    
    #Save results
    save_results(results_df, latencies, metrics, differences, errors)
    
    #Final summary
    print("\nAPI VALIDATION COMPLETE")
    print(f"Scored {len(results_df):,} transactions")
    if latencies:
        print(f"Average latency: {np.mean(latencies):.2f}ms")
        print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"API predictions {'match' if np.max(differences) < 0.001 else 'differ from'} offline model")
    
    fraud_rate = results_df['true_label'].mean() * 100
    print(f"Test fraud rate: {fraud_rate:.2f}%")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()








