"""
Production Fraud Detection Model Evaluation
Evaluates models using business metrics that matter
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudModelEvaluator:
    """
    Production-grade fraud model evaluator
    Focuses on business metrics: dollars saved, approval rates, precision@recall
    """

    def __init__(self, test_df, feature_cols):

        self.test_df = test_df.copy()
        self.feature_cols = feature_cols

        #Prepare test data
        self.X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        self.y_test = test_df['is_fraud'].values
        self.amounts = test_df['amount'].values

        print(f"Initialized evaluator with {len(test_df):,} test transactions")
        print(f"Test fraud rate: {self.y_test.mean()*100:.3f}%")

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def get_predictions(self, model, model_type='sklearn'):

        if model_type == 'lightgbm':
            y_pred_proba = model.predict(self.X_test)
        else:
            #Handle sklearn model with optional scaler
            if isinstance(model, dict):
                X_test_scaled = model['scaler'].transform(self.X_test)
                y_pred_proba = model['model'].predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        return y_pred_proba

    def calculate_business_metrics(self, y_true, y_pred_proba, amounts, threshold=0.5):

        #Apply threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        #Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        #Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        #Dollar metrics
        fraud_caught_dollars = amounts[(y_true == 1) & (y_pred == 1)].sum()
        fraud_missed_dollars = amounts[(y_true == 1) & (y_pred == 0)].sum()
        total_fraud_dollars = amounts[y_true == 1].sum()

        legit_blocked_dollars = amounts[(y_true == 0) & (y_pred == 1)].sum()

        #Approval rate impact
        total_transactions = len(y_true)
        blocked_transactions = y_pred.sum()
        approval_rate = (total_transactions - blocked_transactions) / total_transactions * 100

        #Calculate dollar loss prevented %
        dollar_recall = fraud_caught_dollars / total_fraud_dollars * 100 if total_fraud_dollars > 0 else 0

        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'fraud_caught_dollars': fraud_caught_dollars,
            'fraud_missed_dollars': fraud_missed_dollars,
            'total_fraud_dollars': total_fraud_dollars,
            'dollar_recall': dollar_recall,
            'legit_blocked_dollars': legit_blocked_dollars,
            'approval_rate': approval_rate,
            'blocked_transactions': blocked_transactions
        }

    def find_optimal_threshold(self, y_pred_proba, target_metric='f1', target_value=None):

        thresholds = np.linspace(0.01, 0.99, 100)
        best_threshold = 0.5
        best_score = 0

        if target_metric == 'recall' and target_value:
            #Find threshold that achieves target recall with best precision
            for threshold in thresholds:
                metrics = self.calculate_business_metrics(self.y_test, y_pred_proba, self.amounts, threshold)
                if metrics['recall'] >= target_value and metrics['precision'] > best_score:
                    best_score = metrics['precision']
                    best_threshold = threshold

        elif target_metric == 'fpr' and target_value:
            #Find threshold that keeps FPR below target with best recall
            for threshold in thresholds:
                metrics = self.calculate_business_metrics(self.y_test, y_pred_proba, self.amounts, threshold)
                if metrics['fpr'] <= target_value and metrics['recall'] > best_score:
                    best_score = metrics['recall']
                    best_threshold = threshold

        else:
            #Optimize F1 score
            for threshold in thresholds:
                metrics = self.calculate_business_metrics(self.y_test, y_pred_proba, self.amounts, threshold)
                f1 = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
                if f1 > best_score:
                    best_score = f1
                    best_threshold = threshold

        return best_threshold

    def plot_precision_recall_curve(self, models_dict, save_path='../evaluation/reports/precision_recall_curve.png'):

        plt.figure(figsize=(10, 6))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

        for i, (model_name, y_pred_proba) in enumerate(models_dict.items()):
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            avg_precision = average_precision_score(self.y_test, y_pred_proba)

            plt.plot(recall, precision, color=colors[i % len(colors)],
                    linewidth=2, label=f'{model_name} (AP={avg_precision:.3f})')

        #Add baseline (random classifier)
        fraud_rate = self.y_test.mean()
        plt.axhline(y=fraud_rate, color='gray', linestyle='--', linewidth=1,
                   label=f'Random (AP={fraud_rate:.3f})')

        plt.xlabel('Recall (Fraud Detected)', fontsize=12, fontweight='bold')
        plt.ylabel('Precision (% Correct Alerts)', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve: Production Fraud Detection', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPrecision-Recall curve saved to: {save_path}")
        plt.close()

    def plot_threshold_analysis(self, y_pred_proba, model_name, save_path='../evaluation/reports/threshold_analysis.png'):

        thresholds = np.linspace(0.01, 0.99, 50)
        precisions = []
        recalls = []
        fprs = []
        dollar_recalls = []
        approval_rates = []

        for threshold in thresholds:
            metrics = self.calculate_business_metrics(self.y_test, y_pred_proba, self.amounts, threshold)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            fprs.append(metrics['fpr'])
            dollar_recalls.append(metrics['dollar_recall'])
            approval_rates.append(metrics['approval_rate'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        #Precision & Recall
        axes[0, 0].plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
        axes[0, 0].plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        axes[0, 0].set_xlabel('Threshold', fontweight='bold')
        axes[0, 0].set_ylabel('Score', fontweight='bold')
        axes[0, 0].set_title('Precision vs Recall', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        #FPR
        axes[0, 1].plot(thresholds, fprs, 'orange', linewidth=2)
        axes[0, 1].axhline(y=0.01, color='red', linestyle='--', label='1% FPR target')
        axes[0, 1].set_xlabel('Threshold', fontweight='bold')
        axes[0, 1].set_ylabel('False Positive Rate', fontweight='bold')
        axes[0, 1].set_title('False Positive Rate vs Threshold', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        #Dollar Recall
        axes[1, 0].plot(thresholds, dollar_recalls, 'green', linewidth=2)
        axes[1, 0].set_xlabel('Threshold', fontweight='bold')
        axes[1, 0].set_ylabel('% Fraud Dollars Prevented', fontweight='bold')
        axes[1, 0].set_title('Dollar Loss Prevention', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        #Approval Rate
        axes[1, 1].plot(thresholds, approval_rates, 'purple', linewidth=2)
        axes[1, 1].axhline(y=95, color='red', linestyle='--', label='95% target')
        axes[1, 1].set_xlabel('Threshold', fontweight='bold')
        axes[1, 1].set_ylabel('Approval Rate (%)', fontweight='bold')
        axes[1, 1].set_title('Customer Approval Rate', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{model_name}: Threshold Optimization Analysis', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nThreshold analysis saved to: {save_path}")
        plt.close()

    def create_model_comparison_table(self, models_results, save_path='../evaluation/reports/model_comparison.csv'):

        comparison_df = pd.DataFrame(models_results).T

        #Reorder columns for readability
        col_order = [
            'threshold', 'precision', 'recall', 'fpr',
            'dollar_recall', 'approval_rate',
            'fraud_caught_dollars', 'fraud_missed_dollars',
            'legit_blocked_dollars',
            'true_positives', 'false_positives', 'false_negatives', 'true_negatives'
        ]

        comparison_df = comparison_df[col_order]

        #Round for display
        comparison_df = comparison_df.round(3)

        #Save
        comparison_df.to_csv(save_path)
        print(f"\nModel comparison saved to: {save_path}")

        return comparison_df

def main():

    print("\nFRAUD DETECTION MODEL EVALUATION")

    #Load test data
    print("\nLoading test data...")

    df = pd.read_csv('../data/processed/transactions_with_features.csv')

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    #Load feature info
    with open('../models/trained/feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)

    feature_cols = feature_info['feature_names']

    #Temporal split (same as training)
    df = df.sort_values('timestamp')
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    total_days = (max_date - min_date).days
    train_days = int(total_days * 0.75)
    split_date = min_date + timedelta(days=train_days)

    test_df = df[df['timestamp'] >= split_date].copy()

    print(f"Test period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
    print(f"Test size: {len(test_df):,} transactions")

    #Initialize evaluator
    evaluator = FraudModelEvaluator(test_df, feature_cols)

    #Load models
    print("\nLOADING MODELS")

    baseline_model = evaluator.load_model('../models/trained/logistic_regression_baseline.pkl')
    lgb_model = evaluator.load_model('../models/trained/lightgbm_production.pkl')

    print("\nModels loaded successfully")

    #Get predictions
    print("\nGENERATING PREDICTIONS")

    baseline_proba = evaluator.get_predictions(baseline_model, 'sklearn')
    lgb_proba = evaluator.get_predictions(lgb_model, 'lightgbm')

    print(f"Predictions generated")

    #Find optimal thresholds
    print("\nFINDING OPTIMAL THRESHOLDS")

    #Strategy 1: Maximize F1
    baseline_threshold_f1 = evaluator.find_optimal_threshold(baseline_proba, 'f1')
    lgb_threshold_f1 = evaluator.find_optimal_threshold(lgb_proba, 'f1')

    #Strategy 2: 80% recall
    baseline_threshold_recall = evaluator.find_optimal_threshold(baseline_proba, 'recall', 0.80)
    lgb_threshold_recall = evaluator.find_optimal_threshold(lgb_proba, 'recall', 0.80)

    #Strategy 3: Max 1% FPR
    baseline_threshold_fpr = evaluator.find_optimal_threshold(baseline_proba, 'fpr', 0.01)
    lgb_threshold_fpr = evaluator.find_optimal_threshold(lgb_proba, 'fpr', 0.01)

    print(f"\nBaseline optimal thresholds:")
    print(f"  F1 optimized: {baseline_threshold_f1:.3f}")
    print(f"  80% recall: {baseline_threshold_recall:.3f}")
    print(f"  1% FPR: {baseline_threshold_fpr:.3f}")

    print(f"\nLightGBM optimal thresholds:")
    print(f"  F1 optimized: {lgb_threshold_f1:.3f}")
    print(f"  80% recall: {lgb_threshold_recall:.3f}")
    print(f"  1% FPR: {lgb_threshold_fpr:.3f}")

    #Calculate metrics at optimal thresholds
    print("BUSINESS METRICS EVALUATION")

    results = {
        'Baseline (F1)': evaluator.calculate_business_metrics(
            evaluator.y_test, baseline_proba, evaluator.amounts, baseline_threshold_f1
        ),
        'Baseline (80% Recall)': evaluator.calculate_business_metrics(
            evaluator.y_test, baseline_proba, evaluator.amounts, baseline_threshold_recall
        ),
        'Baseline (1% FPR)': evaluator.calculate_business_metrics(
            evaluator.y_test, baseline_proba, evaluator.amounts, baseline_threshold_fpr
        ),
        'LightGBM (F1)': evaluator.calculate_business_metrics(
            evaluator.y_test, lgb_proba, evaluator.amounts, lgb_threshold_f1
        ),
        'LightGBM (80% Recall)': evaluator.calculate_business_metrics(
            evaluator.y_test, lgb_proba, evaluator.amounts, lgb_threshold_recall
        ),
        'LightGBM (1% FPR)': evaluator.calculate_business_metrics(
            evaluator.y_test, lgb_proba, evaluator.amounts, lgb_threshold_fpr
        ),
    }

    #Create comparison table
    comparison_df = evaluator.create_model_comparison_table(results)

    print("\nMODEL COMPARISON")
    print(comparison_df[['precision', 'recall', 'fpr', 'dollar_recall', 'approval_rate']].to_string())

    #Generate visualizations
    print("GENERATING VISUALIZATIONS")

    evaluator.plot_precision_recall_curve({
        'Baseline (Logistic Regression)': baseline_proba,
        'LightGBM (Production)': lgb_proba
    })

    evaluator.plot_threshold_analysis(lgb_proba, 'LightGBM Production Model')

    print("\nEVALUATION COMPLETE!")
    print("\nResults saved:")
    print("  - evaluation/reports/model_comparison.csv")
    print("  - evaluation/reports/precision_recall_curve.png")
    print("  - evaluation/reports/threshold_analysis.png")

    #Print key recommendations
    print("\nKEY RECOMMENDATIONS")

    best_model = 'LightGBM (80% Recall)'
    best_metrics = results[best_model]

    print(f"\nRecommended Model: {best_model}")
    print(f"  Threshold: {best_metrics['threshold']:.3f}")
    print(f"  Precision: {best_metrics['precision']*100:.1f}%")
    print(f"  Recall: {best_metrics['recall']*100:.1f}%")
    print(f"  Fraud Dollars Prevented: ${best_metrics['fraud_caught_dollars']:,.2f} ({best_metrics['dollar_recall']:.1f}%)")
    print(f"  Approval Rate: {best_metrics['approval_rate']:.1f}%")
    print(f"  Legitimate Blocked: ${best_metrics['legit_blocked_dollars']:,.2f}")


if __name__ == "__main__":
    main()

