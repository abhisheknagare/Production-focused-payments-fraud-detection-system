"""
Production Fraud Detection Model Training
Trains baseline and advanced models with proper temporal validation
"""

!pip install lightgbm

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudModelTrainer:
    """
    Production-grade fraud model trainer with temporal validation
    """

    def __init__(self, df):
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        #Get feature columns
        self.feature_cols = [col for col in df.columns if col.startswith('feat_')]

        print(f"Initialized with {len(self.df):,} transactions")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Fraud rate: {self.df['is_fraud'].mean()*100:.3f}%")

    def temporal_train_test_split(self, train_months=9, test_months=3):
        """
        Split data temporally to prevent leakage

        In production, always train on past, test on future
        """
        print("\nTEMPORAL TRAIN/TEST SPLIT")

        #Get date range
        min_date = self.df['timestamp'].min()
        max_date = self.df['timestamp'].max()

        #Calculate split point
        total_days = (max_date - min_date).days
        train_days = int(total_days * train_months / (train_months + test_months))
        split_date = min_date + timedelta(days=train_days)

        #Split
        train_df = self.df[self.df['timestamp'] < split_date].copy()
        test_df = self.df[self.df['timestamp'] >= split_date].copy()

        print(f"\nTrain period: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")
        print(f"Test period:  {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
        print(f"\nTrain size: {len(train_df):,} ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"Test size:  {len(test_df):,} ({len(test_df)/len(self.df)*100:.1f}%)")
        print(f"\nTrain fraud rate: {train_df['is_fraud'].mean()*100:.3f}%")
        print(f"Test fraud rate:  {test_df['is_fraud'].mean()*100:.3f}%")

        #Extract features and labels
        X_train = train_df[self.feature_cols]
        y_train = train_df['is_fraud']
        X_test = test_df[self.feature_cols]
        y_test = test_df['is_fraud']

        #Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        #Handle inf values
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)

        return X_train, X_test, y_train, y_test, test_df

    def train_baseline_model(self, X_train, y_train):
        """
        Train baseline logistic regression
        """

        print("\nTRAINING BASELINE MODEL: Logistic Regression")

        #Scale features for better convergence
        from sklearn.preprocessing import StandardScaler

        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        #Use class weights to handle imbalance
        fraud_weight = len(y_train) / (2 * y_train.sum())
        legit_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()))

        print(f"\nClass weights:")
        print(f"  Fraud (1): {fraud_weight:.2f}")
        print(f"  Legitimate (0): {legit_weight:.2f}")

        model = LogisticRegression(
            max_iter=1000,
            class_weight={0: legit_weight, 1: fraud_weight},
            random_state=42,
            n_jobs=-1,
            solver='lbfgs'
        )

        model.fit(X_train_scaled, y_train)

        print("\nBaseline model trained!")

        return {'model': model, 'scaler': scaler}

    def train_lightgbm_model(self, X_train, y_train, X_test, y_test):
        """
        Train LightGBM with optimal parameters for fraud detection
        """

        print("\nTRAINING ADVANCED MODEL: LightGBM")

        #Calculate scale_pos_weight for imbalance
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        print(f"\nScale pos weight: {scale_pos_weight:.2f}")

        #LightGBM parameters optimized for fraud
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_child_samples': 20,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbose': -1
        }

        #Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        #Train with early stopping
        print("\nTraining with early stopping...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        print(f"\nLightGBM trained! Best iteration: {model.best_iteration}")

        return model

    def save_model(self, model, model_name):
        """Save model to disk"""

        output_path = f'../models/trained/{model_name}.pkl'

        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"\nModel saved to: {output_path}")

        return output_path

def main():
    """Main training pipeline"""

    print("\nFRAUD DETECTION MODEL TRAINING PIPELINE")

    #Load feature-engineered data
    print("\nLoading data with features...")

    try:
        df = pd.read_csv('../data/processed/transactions_with_features.csv')
        print("\nLoaded DuckDB-processed features")
    except FileNotFoundError:
        print("❌ Error: No processed features found!")
        print(" - Run feature engineering first:")
        print(" - python feature_engineering/build_features.py")
        return

    print(f"Loaded {len(df):,} transactions with {len([c for c in df.columns if c.startswith('feat_')])} features")

    #Initialize trainer
    trainer = FraudModelTrainer(df)

    #Temporal train/test split
    X_train, X_test, y_train, y_test, test_df = trainer.temporal_train_test_split()

    #Train baseline model
    baseline_model = trainer.train_baseline_model(X_train, y_train)
    trainer.save_model(baseline_model, 'logistic_regression_baseline')

    #Train LightGBM model
    lgb_model = trainer.train_lightgbm_model(X_train, y_train, X_test, y_test)
    trainer.save_model(lgb_model, 'lightgbm_production')

    #Save feature names
    feature_info = {
        'feature_names': trainer.feature_cols,
        'n_features': len(trainer.feature_cols)
    }

    with open('../models/trained/feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)

    print("\nTRAINING COMPLETE!")
    print("\nModels saved:")
    print(" - models/trained/logistic_regression_baseline.pkl")
    print(" - models/trained/lightgbm_production.pkl")
    print(" - models/trained/feature_info.pkl")
    print("\nNext step: Run evaluation")
    print(" - python evaluation/evaluate_model.py")


if __name__ == "__main__":
    main()
