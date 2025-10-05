import os
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import mne
import warnings
from scipy.interpolate import interp1d
from datetime import datetime

import torch
from train import AttentionModel, TorchModelWrapper, PyTorchMLP
PYTORCH_AVAILABLE = True

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from datapreprocess import DataPreprocessor

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

BRAIN_REGIONS = {
    'l_temporal': ['FT7', 'FC5', 'T7', 'C5', 'TP7', 'CP5', 'P7', 'P5'],
    'r_temporal': ['FT8', 'FC6', 'T8', 'C6', 'TP8', 'CP6', 'P8', 'P6'],
    'central': ['CZ', 'FCZ', 'C1', 'C2', 'C3', 'C4', 'FC1', 'FC2', 'FC3', 'FC4'],
    'occipital': ['POZ', 'PO3', 'PO5', 'PO7', 'PO4', 'PO6', 'PO8', 'O1', 'O2', 'OZ', 'CB1', 'CB2']
}
FREQ_BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
TIME_WINDOW = (0.05, 0.75)
TIME_SEGMENTS = {'mid_early': (0.12, 0.28), 'mid_late': (0.28, 0.55), 'late': (0.55, 0.75)}
SELECTED_TIME_SEGMENTS = ['mid_early', 'mid_late', 'late']

def _isolated_compute_de(data, sfreq):
    n_epochs, n_channels, _ = data.shape
    info = mne.create_info(ch_names=[f'EEG{i}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
    epochs_obj = mne.EpochsArray(data=data, info=info, verbose=False)
    de_feat_list = []
    for _, (f_min, f_max) in FREQ_BANDS.items():
        psd = epochs_obj.compute_psd(method='welch', fmin=f_min, fmax=f_max, verbose=False).get_data()
        psd += 1e-10
        band_de_feat = np.sum(np.log(psd), axis=-1)
        de_feat_list.append(band_de_feat)
    return np.concatenate(de_feat_list, axis=1)

def _isolated_extract_time_domain_features(data, sampling_rate=5):
    n_channels, n_timepoints = data.shape
    if n_timepoints > sampling_rate:
        indices = np.linspace(0, n_timepoints - 1, sampling_rate, dtype=int)
        sampled_data = data[:, indices]
    else:
        x_old = np.arange(n_timepoints)
        x_new = np.linspace(0, n_timepoints - 1, sampling_rate)
        sampled_data = np.array([interp1d(x_old, ch_data, kind='linear', fill_value="extrapolate")(x_new) for ch_data in data])
    return sampled_data.flatten()

def _isolated_extract_features(epochs):
    base_epochs = epochs.copy()
    sfreq = base_epochs.info['sfreq']
    de_time_start = TIME_SEGMENTS[SELECTED_TIME_SEGMENTS[0]][0]
    de_time_end = TIME_SEGMENTS[SELECTED_TIME_SEGMENTS[-1]][1]
    epochs_for_de = base_epochs.copy().crop(tmin=de_time_start, tmax=de_time_end)
    all_de_features = []
    for region_name, channels in BRAIN_REGIONS.items():
        available_channels = [ch for ch in channels if ch in epochs_for_de.ch_names]
        if not available_channels: continue
        region_data = epochs_for_de.copy().pick(available_channels).get_data()
        de_features = _isolated_compute_de(region_data, sfreq)
        all_de_features.append(de_features)
    final_de_features = np.concatenate(all_de_features, axis=1)
    final_de_features_norm = MinMaxScaler().fit_transform(final_de_features)
    all_time_features = []
    epochs_for_time = base_epochs.copy().crop(tmin=TIME_WINDOW[0], tmax=TIME_WINDOW[1])
    for time_seg_name in SELECTED_TIME_SEGMENTS:
        time_start, time_end = TIME_SEGMENTS[time_seg_name]
        for region_name, channels in BRAIN_REGIONS.items():
            available_channels = [ch for ch in channels if ch in epochs_for_time.ch_names]
            if not available_channels: continue
            segment_data = epochs_for_time.copy().pick(available_channels).crop(tmin=time_start, tmax=time_end).get_data()
            time_features = np.array([_isolated_extract_time_domain_features(segment_data[i], sampling_rate=5) for i in range(segment_data.shape[0])])
            all_time_features.append(time_features)
    final_time_features = np.concatenate(all_time_features, axis=1)
    final_time_features_norm = MinMaxScaler().fit_transform(final_time_features)
    return np.concatenate([final_time_features_norm, final_de_features_norm], axis=1)

def get_word_level_data(preprocessor, subj_idx, model_type):
    cache_dir = f'../data/feat_word_level_{model_type}/'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"subj_{subj_idx}.pkl")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    
    try:
        raw_epochs = preprocessor.load_subject_data(subj_idx)
        events_to_keep = ['no_hallu', 'hallu', 'hallu_wrong']
        valid_events = [e for e in events_to_keep if e in raw_epochs.event_id]
        if not valid_events: return None, None
        epochs_subset = raw_epochs[valid_events].copy()
        if len(epochs_subset) == 0: return None, None
        positive_event_ids = [eid for name, eid in epochs_subset.event_id.items() if name in ['hallu', 'hallu_wrong']]
        labels = np.isin(epochs_subset.events[:, 2], positive_event_ids).astype(int)
        features = _isolated_extract_features(epochs_subset)
        if features.shape[1] != 760:
            raise ValueError(f"features shape mismatch: expected 760, got {features.shape[1]}")
        joblib.dump((features, labels), cache_path)
        return features, labels
    except Exception as e:
        print(f"Error processing subject {subj_idx}: {e}")
        return None, None

def create_sentence_dataset(preprocessor, model_type, force_recompute=False):
    cache_dir = f'../data/feat_sentence_level_{model_type}/'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'sentence_dataset_by_subject.pkl')
    if os.path.exists(cache_path) and not force_recompute:
        return joblib.load(cache_path)

    sentence_dataset_by_subject = {subj_id: [] for subj_id in preprocessor.find_available_subjects()}
    for subj_idx in tqdm(preprocessor.find_available_subjects(), desc="Creating sentence-level dataset"):
        try:
            epochs = preprocessor.load_subject_data(subj_idx)
            events, event_id_map = epochs.events, epochs.event_id
            sentence_start_id, sentence_end_id = event_id_map.get('+'), event_id_map.get('judge')
            if sentence_start_id is None or sentence_end_id is None: continue

            epochs_subset = epochs[['no_hallu', 'hallu', 'hallu_wrong']].copy()
            if len(epochs_subset) == 0: continue

            word_features = _isolated_extract_features(epochs_subset)
            positive_event_ids = [eid for name, eid in epochs_subset.event_id.items() if name in ['hallu', 'hallu_wrong']]
            word_labels = np.isin(epochs_subset.events[:, 2], positive_event_ids).astype(int)

            word_event_indices_in_epochs = np.where(np.isin(epochs.events[:, 2], epochs_subset.events[:, 2]))[0]
            event_to_word_map = {event_idx: word_idx for word_idx, event_idx in enumerate(word_event_indices_in_epochs)}

            start_indices = np.where(events[:, 2] == sentence_start_id)[0]
            end_indices = np.where(events[:, 2] == sentence_end_id)[0]

            for start_event_idx in start_indices:
                following_end_indices = end_indices[end_indices > start_event_idx]
                if len(following_end_indices) == 0: continue
                end_event_idx = following_end_indices[0]
                
                words_in_sentence_event_indices = [idx for idx in word_event_indices_in_epochs if start_event_idx < idx < end_event_idx]
                if not words_in_sentence_event_indices: continue

                sentence_word_indices = [event_to_word_map[idx] for idx in words_in_sentence_event_indices]
                sentence_word_features = word_features[sentence_word_indices]
                sentence_word_labels = word_labels[sentence_word_indices]
                
                if len(sentence_word_features) == 0: continue
                
                sentence_true_label = 1 if np.any(sentence_word_labels == 1) else 0
                sentence_dataset_by_subject[subj_idx].append((sentence_word_features, sentence_true_label))
        except Exception as e:
            print(f"Error processing subject {subj_idx}: {e}")
    
    joblib.dump(sentence_dataset_by_subject, cache_path)
    return sentence_dataset_by_subject

def predict_sentence_score(word_features_in_sentence, model):
    if len(word_features_in_sentence) == 0: return 0.0
    word_probabilities = model.predict_proba(word_features_in_sentence)[:, 1]
    if np.any(np.isnan(word_probabilities)): return np.nan
    max_p, mean_p, median_p = np.max(word_probabilities), np.mean(word_probabilities), np.median(word_probabilities)
    return (0.5 * max_p + 0.3 * mean_p + 0.2 * median_p)

def run_evaluation(model_type, eval_level, strategy, device, recompute, save_csv):
    preprocessor = DataPreprocessor()
    subjects = preprocessor.find_available_subjects()
    all_metrics = []
    
    model_dir_name = 'attention_based' if model_type == 'attention' else 'SVM'
    
    if eval_level == 'word':
        if strategy == 'cross_subject':
            all_features, all_labels = [], []
            for subj_idx in subjects:
                features, labels = get_word_level_data(preprocessor, subj_idx, model_type)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(labels)
            X_test, y_test = np.concatenate(all_features), np.concatenate(all_labels)
            
            model_paths = glob.glob(f'../data/model_params/cross_subject/{model_dir_name}/*.pkl')
            for model_path in tqdm(model_paths, desc="Evaluating models"):
                model = joblib.load(model_path)
                if model_type == 'attention' and PYTORCH_AVAILABLE:
                    model.named_steps.get('classifier').model.to(device)
                
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
                all_metrics.append({'model': os.path.basename(model_path), 'accuracy': accuracy_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred, zero_division=0), 'f1': f1_score(y_test, y_pred, zero_division=0), 'auc': auc})

        elif strategy == 'within_subject':
            for subj_idx in tqdm(subjects, desc="Evaluating subjects"):
                X_test, y_test = get_word_level_data(preprocessor, subj_idx, model_type)
                if X_test is None: continue
                
                model_path_pattern = f'../data/model_params/cross_subject/{model_dir_name}/fold_{subj_idx}_*.pkl'
                model_paths = glob.glob(model_path_pattern)
                if not model_paths: continue
                
                model = joblib.load(model_paths[0])
                if model_type == 'attention' and PYTORCH_AVAILABLE:
                    model.named_steps.get('classifier').model.to(device)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
                all_metrics.append({'model': f'subject_{subj_idx}', 'accuracy': accuracy_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred, zero_division=0), 'f1': f1_score(y_test, y_pred, zero_division=0), 'auc': auc})

    elif eval_level == 'sentence':
        sentence_dataset = create_sentence_dataset(preprocessor, model_type, force_recompute=recompute)
        if strategy == 'cross_subject':
            full_sentence_data = [item for sublist in sentence_dataset.values() for item in sublist]
            model_paths = glob.glob(f'../data/model_params/cross_subject/{model_dir_name}/*.pkl')
            for model_path in tqdm(model_paths, desc="Evaluating models"):
                model = joblib.load(model_path)
                if model_type == 'attention' and PYTORCH_AVAILABLE:
                    model.named_steps.get('classifier').model.to(device)
                
                scores = [predict_sentence_score(feats, model) for feats, _ in full_sentence_data]
                y_true = [label for _, label in full_sentence_data]
                y_pred = (np.array(scores) >= 0.5).astype(int)
                auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
                all_metrics.append({'model': os.path.basename(model_path), 'accuracy': accuracy_score(y_true, y_pred), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0), 'auc': auc})

        elif strategy == 'within_subject':
            for subj_idx in tqdm(subjects, desc="Evaluating subjects"):
                subject_sentence_data = sentence_dataset.get(subj_idx, [])
                if not subject_sentence_data: continue

                model_path_pattern = f'../data/model_params/cross_subject/{model_dir_name}/fold_{subj_idx}_*.pkl'
                model_paths = glob.glob(model_path_pattern)
                if not model_paths: continue

                model = joblib.load(model_paths[0])
                if model_type == 'attention' and PYTORCH_AVAILABLE:
                    model.named_steps.get('classifier').model.to(device)

                scores = [predict_sentence_score(feats, model) for feats, _ in subject_sentence_data]
                y_true = [label for _, label in subject_sentence_data]
                y_pred = (np.array(scores) >= 0.5).astype(int)
                auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
                all_metrics.append({'model': f'subject_{subj_idx}', 'accuracy': accuracy_score(y_true, y_pred), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0), 'auc': auc})

    if not all_metrics:
        print("No evaluation results to display.")
        return

    results_df = pd.DataFrame(all_metrics)
    mean_metrics = results_df.drop(columns='model').mean()
    
    print(f"Model Type: {model_type}, Evaluation Level: {eval_level}, Strategy: {strategy}")
    print("Metrics:")
    print(results_df.to_string(index=False))
    print("Average Metrics:")
    print(mean_metrics)

    if save_csv:
        output_dir = "../data/results/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{model_type}_{eval_level}_{strategy}_{timestamp}.csv"
        output_path = os.path.join(output_dir, filename)
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pre-trained models at word or sentence level.")
    parser.add_argument('--model_type', type=str, required=True, choices=['svm', 'attention'], help="Model type to evaluate.")
    parser.add_argument('--eval_level', type=str, required=True, choices=['word', 'sentence'], help="Evaluation level: 'word' or 'sentence'.")
    parser.add_argument('--strategy', type=str, required=True, choices=['cross_subject', 'within_subject'], help="Evaluation strategy: 'cross_subject' or 'within_subject'.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU id to use.")
    parser.add_argument('--recompute', action='store_true', help="Recompute sentence-level dataset even if cached data exists.")
    parser.add_argument('--save_csv', action='store_true', help="Save detailed results as a CSV file.")

    args = parser.parse_args()

    if args.model_type == 'attention' and not PYTORCH_AVAILABLE:
        raise ImportError("Pytorch is not available.")

    DEVICE = torch.device("cpu")
    if args.model_type == 'attention' and torch.cuda.is_available() and args.gpu >= 0:
        try:
            DEVICE = torch.device(f"cuda:{args.gpu}")
            torch.empty(1, device=DEVICE)
        except (RuntimeError, AssertionError):
            DEVICE = torch.device("cpu")
    
    print(f"Device: {DEVICE}")
    
    run_evaluation(
        model_type=args.model_type,
        eval_level=args.eval_level,
        strategy=args.strategy,
        device=DEVICE,
        recompute=args.recompute,
        save_csv=args.save_csv
    )