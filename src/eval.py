import os
import re
import glob
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
import torch

from train import (
    AdvancedHallucinationClassifier, 
    TorchModelWrapper,
    PyTorchMLP, 
    AttentionModel,
    DEVICE
)
from datapreprocess import DataPreprocessor

TorchMLPWrapper = TorchModelWrapper

def parse_model_path(path):
    try:
        path = path.replace('\\', '/')
        pattern = r".*?(within_subject|cross_subject)[/\\]([A-Z_a-z]+)[/\\](?:subject_(\d+)_)?fold_(\d+)_.*\.pkl"
        match = re.search(pattern, path)
        if match:
            strategy, model_type, subject_id, fold_id = match.groups()
            return {
                "strategy": strategy,
                "model_type": model_type,
                "subject_id": int(subject_id) if subject_id else None,
                "fold_id": int(fold_id) - 1,
                "path": path
            }
    except Exception as e:
        print(f"Error: parsing model path {path}: {e}")
    return None

def check_and_log_nan(data, model_info, data_name="X_test"):
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        total_elements = data.size
        nan_percentage = (nan_count / total_elements) * 100
        nan_columns = np.where(np.isnan(data).any(axis=0))[0]
        # print(" Found NaN values ")
        # print(f"  Model: {model_info['path']}")
        # print(f"  Data: {data_name}")
        # print(f"  Data shape: {data.shape}")
        # print(f"  NaN count: {nan_count} / {total_elements} ({nan_percentage:.4f}%)")
        # print(f"  NaN columns: {nan_columns}")
        return True
    return False

def calculate_comprehensive_metrics(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    return metrics

def evaluate_within_subject(parsed_models, features, labels, subject_ids, model_type_to_eval, model_dir):
    all_fold_results = []
    for model_info in parsed_models:
        try:
            subject_id = model_info['subject_id']
            subject_mask = (subject_ids == subject_id)
            subject_features, subject_labels = features[subject_mask], labels[subject_mask]

            n_samples = len(subject_features)
            if n_samples == 0: continue

            n_splits = 10

            n_positive = np.sum(subject_labels)
            min_samples_per_class = min(n_positive, n_samples - n_positive)
            if min_samples_per_class < n_splits:
                print(f"Warning: subject {subject_id} has only {min_samples_per_class} samples in the minority class.")
                continue
            
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            fold_id = model_info['fold_id']
            all_splits = list(cv.split(subject_features, subject_labels))
            if fold_id >= len(all_splits):
                print(f"Warning: subject {subject_id} does not have fold {fold_id + 1}.")
                continue

            _, test_idx = list(cv.split(subject_features, subject_labels))[model_info['fold_id']]
            X_test, y_test = subject_features[test_idx], subject_labels[test_idx]

            model = joblib.load(model_info['path'])
            
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'model'):
                 model.classifier.model.to(DEVICE)

            check_and_log_nan(X_test, model_info)

            y_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_proba, axis=1)
            
            metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba[:, 1])
            metrics['subject_id'] = subject_id
            all_fold_results.append(metrics)
            
            print(f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error: evaluating model {model_info['path']}: {e}")

    if not all_fold_results:
        print("No successful evaluation results.")
        return

    results_df = pd.DataFrame(all_fold_results)
    per_subject_avg = results_df.groupby('subject_id').mean().reset_index()
    overall_avg = per_subject_avg.mean(numeric_only=True)
    overall_avg['subject_id'] = 'Overall_Average'
    csv_df = pd.concat([per_subject_avg, pd.DataFrame([overall_avg])], ignore_index=True)
    csv_df = csv_df.set_index('subject_id')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = '../data/results'
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"eval_within_subject_{model_type_to_eval}_{timestamp}.csv")
    csv_df.to_csv(csv_filename)

    print(f"Results saved to: {csv_filename}")
    print("Average metrics per subject:")
    print(per_subject_avg.to_string(index=False))
    print("Overall Average Metrics")
    print(overall_avg.drop('subject_id').to_string())


def evaluate_cross_subject(parsed_models, features, labels, subject_ids, model_type_to_eval, model_dir):
    gkf = GroupKFold(n_splits=27)
    splits = list(gkf.split(features, labels, groups=subject_ids))
    
    all_fold_results = []
    for model_info in parsed_models:
        try:
            fold_id = model_info['fold_id']
            _, test_idx = splits[fold_id]
            X_test, y_test = features[test_idx], labels[test_idx]

            model = joblib.load(model_info['path'])

            if hasattr(model, 'classifier') and hasattr(model.classifier, 'model'):
                 model.classifier.model.to(DEVICE)

            check_and_log_nan(X_test, model_info)

            y_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_proba, axis=1)
            
            metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba[:, 1])
            metrics['fold_id'] = fold_id + 1
            all_fold_results.append(metrics)
            
            print(f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error: evaluating model {model_info['path']}: {e}")

    if not all_fold_results:
        print("No successful evaluation results.")
        return

    results_df = pd.DataFrame(all_fold_results)
    overall_avg = results_df.mean(numeric_only=True)
    overall_avg['fold_id'] = 'Overall_Average'
    csv_df = pd.concat([results_df, pd.DataFrame([overall_avg])], ignore_index=True)
    csv_df = csv_df.set_index('fold_id')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = '../data/results'
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"eval_cross_subject_{model_type_to_eval}_{timestamp}.csv")
    csv_df.to_csv(csv_filename)

    print("Results saved to: {csv_filename}")
    print("Average metrics per fold:")
    print(results_df.to_string(index=False))
    print("Overall Average Metrics")
    print(overall_avg.drop('fold_id').to_string())


def run_evaluation(args):
    print(f"Evaluating strategy: {args.strategy}, model: {args.model}")
    print(f"Device: {DEVICE}")

    print("loading data...")
    data_loader = DataPreprocessor()
    features, labels, subject_ids = data_loader.load_multi_subject_data()
    
    if not check_and_log_nan(features, {'path': 'All'}, data_name="features"):
        print("No NaN values found in features.")

    model_paths = glob.glob(os.path.join(args.model_dir, '**', '*.pkl'), recursive=True)
    parsed_models = [parse_model_path(p) for p in model_paths]
    
    filtered_models = [
        m for m in parsed_models 
        if m is not None 
        and m['strategy'] == args.strategy 
        and m['model_type'] == args.model
    ]
    
    if not filtered_models:
        print(f"Error: no models found for strategy '{args.strategy}' and model '{args.model}' in directory '{args.model_dir}'.")
        return

    if args.strategy == 'within_subject':
        evaluate_within_subject(filtered_models, features, labels, subject_ids, args.model, args.model_dir)
    elif args.strategy == 'cross_subject':
        evaluate_cross_subject(filtered_models, features, labels, subject_ids, args.model, args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pre-trained models on hallucination EEG data")
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Root directory path containing saved models")
    parser.add_argument('--model', type=str, required=True, choices=['SVM', 'RF', 'GBDT', 'MLP', 'attention_based'],
                        help="Select the model type to evaluate")
    parser.add_argument('--strategy', type=str, required=True, choices=['within_subject', 'cross_subject'],
                        help="Select the training strategy to evaluate.")
    
    args = parser.parse_args()
    run_evaluation(args)