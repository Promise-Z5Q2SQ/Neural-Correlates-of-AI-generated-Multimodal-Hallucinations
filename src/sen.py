import numpy as np
import joblib
import torch
import warnings
import os
import glob
import argparse
from datetime import datetime
import pandas as pd
import mne
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm

from datapreprocess import DataPreprocessor
import train
from train import TorchModelWrapper, PyTorchMLP, AttentionModel

warnings.filterwarnings('ignore')

DEVICE = "cpu"

def smart_model_loader(model_path):
    print(f"Device: {DEVICE}")
    try:
        model = torch.load(model_path, map_location=DEVICE)
        return model
    except Exception:
        model = joblib.load(model_path)
        return model

def predict_sentence_score(word_features_in_sentence, model):
    if len(word_features_in_sentence) == 0:
        return 0.0
    word_probabilities = model.predict_proba(word_features_in_sentence)[:, 1]
    
    if np.any(np.isnan(word_probabilities)):
        return np.nan

    max_p = np.max(word_probabilities)
    mean_p = np.mean(word_probabilities)
    median_p = np.median(word_probabilities)
    return (0.5 * max_p + 0.3 * mean_p + 0.2 * median_p)

def create_sentence_dataset_from_features(preprocessor, force_recompute=False):
    cache_path = os.path.join(preprocessor.feature_cache_dir, 'sentence_dataset_by_subject.pkl')
    if os.path.exists(cache_path) and not force_recompute:
        return joblib.load(cache_path)

    print("Creating sentence-level dataset")
    word_features, word_labels, subject_ids = preprocessor.load_multi_subject_data()
    
    unique_word_labels, counts = np.unique(word_labels, return_counts=True)

    sentence_dataset_by_subject = {subj_id: [] for subj_id in preprocessor.find_available_subjects()}
    
    global_word_offset = 0
    for subj_idx in tqdm(subjects := preprocessor.find_available_subjects(), desc="Processing subjects"):
        try:
            epochs = preprocessor.load_subject_data(subj_idx)
            events = epochs.events
            event_id_map = epochs.event_id

            sentence_start_id = event_id_map.get('+')
            sentence_end_id = event_id_map.get('judge')
            
            if sentence_start_id is None or sentence_end_id is None:
                global_word_offset += np.sum(subject_ids == subj_idx)
                continue

            start_indices = np.where(events[:, 2] == sentence_start_id)[0]
            end_indices = np.where(events[:, 2] == sentence_end_id)[0]
            
            word_event_ids = [v for k, v in event_id_map.items() if k not in ['+', 'judge']]
            word_event_indices_in_events = np.where(np.isin(events[:, 2], word_event_ids))[0]
            event_to_word_map = {event_idx: word_idx for word_idx, event_idx in enumerate(word_event_indices_in_events)}

            for start_event_idx in start_indices:
                following_end_indices = end_indices[end_indices > start_event_idx]
                if len(following_end_indices) == 0: continue
                end_event_idx = following_end_indices[0]
                
                words_in_sentence_event_indices = word_event_indices_in_events[
                    (word_event_indices_in_events > start_event_idx) & 
                    (word_event_indices_in_events < end_event_idx)
                ]
                if len(words_in_sentence_event_indices) == 0: continue

                first_word_event_idx = words_in_sentence_event_indices[0]
                last_word_event_idx = words_in_sentence_event_indices[-1]
                
                sentence_word_start_idx = event_to_word_map[first_word_event_idx]
                sentence_word_end_idx = event_to_word_map[last_word_event_idx] + 1

                global_start = global_word_offset + sentence_word_start_idx
                global_end = global_word_offset + sentence_word_end_idx
                
                sentence_word_features = word_features[global_start:global_end]
                sentence_word_labels = word_labels[global_start:global_end]
                
                if len(sentence_word_features) == 0: continue
                
                POSITIVE_LABEL = 1 
                sentence_true_label = 1 if np.any(sentence_word_labels == POSITIVE_LABEL) else 0
                sentence_dataset_by_subject[subj_idx].append((sentence_word_features, sentence_true_label))

            global_word_offset += np.sum(subject_ids == subj_idx)
        except Exception as e:
            print(f"Error: processing subject {subj_idx} sentence structure: {e}")
            global_word_offset += np.sum(subject_ids == subj_idx)
            continue

    print(f"Dataset cached to: {cache_path}")
    joblib.dump(sentence_dataset_by_subject, cache_path)
    
    return sentence_dataset_by_subject

def evaluate_directory_sentence_level(model_dir, threshold=0.5, recompute_sentence_data=False):
    norm_path = os.path.normpath(model_dir)
    model_type_name = os.path.basename(norm_path)
    strategy_name = os.path.basename(os.path.dirname(norm_path))
    print(f"Strategy: {strategy_name}, model: {model_type_name}")

    preprocessor = DataPreprocessor()
    sentence_dataset_by_subject = create_sentence_dataset_from_features(preprocessor, force_recompute=recompute_sentence_data)
    
    all_metrics = []

    if strategy_name == 'within_subject':
        subjects = preprocessor.find_available_subjects()
        for subj_idx in tqdm(subjects, desc="Evaluating subjects"):
            model_path_pattern = os.path.join(model_dir, f'subject_{subj_idx}_fold_*.pkl')
            model_paths = glob.glob(model_path_pattern)
            
            if not model_paths:
                continue
            
            model_path = model_paths[0]
            if len(model_paths) > 1:
                print(f"Warning: Multiple models found for subject {subj_idx}, using {model_path}")
            
            model = smart_model_loader(model_path)

            subject_sentence_data = sentence_dataset_by_subject.get(subj_idx, [])

            if not subject_sentence_data:
                print(f"Warning: No sentence data found for subject {subj_idx}.")
                continue

            sentence_scores, true_labels = [], []
            for word_features, true_label in subject_sentence_data:
                score = predict_sentence_score(word_features, model)
                sentence_scores.append(score)
                true_labels.append(true_label)
            
            y_true = np.array(true_labels)
            y_proba = np.array(sentence_scores)

            nan_mask = np.isnan(y_proba)
            if np.all(nan_mask):
                print(f"Warning: All predictions for subject {subj_idx} are NaN.")
                auc_score = np.nan
                acc_score = np.nan
                recall = np.nan
                f1 = np.nan
            else:
                y_true_valid = y_true[~nan_mask]
                y_proba_valid = y_proba[~nan_mask]
                y_pred_valid = (y_proba_valid >= threshold).astype(int)
                
                if len(np.unique(y_true_valid)) < 2:
                    print(f"Warning: Subject {subj_idx} has only one class in valid data, cannot compute AUC.")
                    auc_score = np.nan
                else:
                    auc_score = roc_auc_score(y_true_valid, y_proba_valid)
                
                acc_score = accuracy_score(y_true_valid, y_pred_valid)
                recall = recall_score(y_true_valid, y_pred_valid, zero_division=0)
                f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)

            all_metrics.append({
                'model': os.path.basename(model_path),
                'accuracy': acc_score,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            })

    elif strategy_name == 'cross_subject':
        full_sentence_dataset = [item for sublist in sentence_dataset_by_subject.values() for item in sublist]
        
        model_paths = glob.glob(os.path.join(model_dir, '*.pkl'))
        for model_path in tqdm(model_paths, desc="Evaluating models"):
            model = smart_model_loader(model_path)
            
            sentence_scores, true_labels = [], []
            for word_features, true_label in full_sentence_dataset:
                score = predict_sentence_score(word_features, model)
                sentence_scores.append(score)
                true_labels.append(true_label)

            y_true = np.array(true_labels)
            y_proba = np.array(sentence_scores)

            nan_mask = np.isnan(y_proba)
            if np.all(nan_mask):
                print(f"Warning: All predictions for model {os.path.basename(model_path)} are NaN.")
                auc_score = np.nan
                acc_score = np.nan
                recall = np.nan
                f1 = np.nan
            else:
                y_true_valid = y_true[~nan_mask]
                y_proba_valid = y_proba[~nan_mask]
                y_pred_valid = (y_proba_valid >= threshold).astype(int)

                if len(np.unique(y_true_valid)) < 2:
                    print(f"Warning: Model {os.path.basename(model_path)} has only one class in valid data, cannot compute AUC.")
                    auc_score = np.nan
                else:
                    auc_score = roc_auc_score(y_true_valid, y_proba_valid)

                acc_score = accuracy_score(y_true_valid, y_pred_valid)
                recall = recall_score(y_true_valid, y_pred_valid, zero_division=0)
                f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)

            all_metrics.append({
                'model': os.path.basename(model_path),
                'accuracy': acc_score,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            })
    else:
        print(f"Error: Unknown strategy {strategy_name}.")
        return

    if not all_metrics:
        print("No metrics were computed.")
        return

    results_df = pd.DataFrame(all_metrics)
    mean_metrics = results_df.drop(columns='model').mean()
    std_metrics = results_df.drop(columns='model').std()
    
    output_dir = "../data/results/"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, f"sentence_level_eval_{strategy_name}_{model_type_name}.csv")

    results_df.to_csv(output_filename, index=False, float_format='%.4f')
    
    print(f"saved results to {output_filename}")
    print("average metrics:")
    print(mean_metrics)
    print("std of metrics:")
    print(std_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate sentence-level models.")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing trained models (.pkl).")
    parser.add_argument('--gpu', type=int, default=0, help="GPU id.")
    parser.add_argument('--recompute', action='store_true', help="Recompute sentence-level dataset.")
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        try:
            DEVICE = torch.device(f"cuda:{args.gpu}")
            torch.empty(1, device=DEVICE)
        except (RuntimeError, AssertionError):
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cpu")

    train.DEVICE = DEVICE
    
    print(f"Device: {DEVICE}")
    
    evaluate_directory_sentence_level(args.model_dir, recompute_sentence_data=args.recompute)