import os
import glob
import numpy as np
import mne
from sklearn.preprocessing import MinMaxScaler
from settings import *
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, time_window=(0.05, 0.75)):
        self.time_window = time_window
        self.time_segments = {'mid_early': (0.12, 0.28), 'mid_late': (0.28, 0.55), 'late': (0.55, 0.75)}
        self.selected_time_segments = ['mid_early', 'mid_late', 'late']
        self.brain_regions = {'central': central, 'l_temporal': l_temporal, 'r_temporal': r_temporal, 'occipital': occipital}
        self.feature_cache_dir = '../data/feat'
        os.makedirs(self.feature_cache_dir, exist_ok=True)

    def find_available_subjects(self, data_type='sen'):
        subjects = []
        paths_to_try = [EEG_PROCESSED_DIR, "../data/processed_data"]
        
        for base_path in paths_to_try:
            if os.path.exists(base_path):
                pattern = os.path.join(base_path, f"*_{data_type}_all-epo.fif")
                files = glob.glob(pattern)
                for file in files:
                    filename = os.path.basename(file)
                    subject_id = filename.split('_')[0]
                    try:
                        subject_num = int(subject_id)
                        subjects.append(subject_num)
                    except ValueError:
                        continue
                break
        
        subjects = sorted(list(set(subjects)))
        return subjects
    
    def load_subject_data(self, subj_idx, data_type='sen'):
        fpath = os.path.join(EEG_PROCESSED_DIR, f"{subj_idx}_{data_type}_all-epo.fif")
        if not os.path.exists(fpath):
            fpath = f"../data/processed_data/{subj_idx}_{data_type}_all-epo.fif"
        
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"file not found: {fpath}")
        
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        return epochs
    
    def extract_time_domain_features(self, data, sampling_rate=5):
        n_channels, n_timepoints = data.shape
        if n_timepoints > sampling_rate:
            indices = np.linspace(0, n_timepoints-1, sampling_rate, dtype=int)
            sampled_data = data[:, indices]
        else:
            from scipy.interpolate import interp1d
            x_old = np.arange(n_timepoints)
            x_new = np.linspace(0, n_timepoints-1, sampling_rate)
            sampled_data = np.array([interp1d(x_old, ch_data, kind='linear')(x_new) for ch_data in data])
        return sampled_data.flatten()
    
    def compute_de_features(self, data, sfreq, subject_id):
        if data.ndim != 3: raise ValueError(f"Error: data should be 3D (n_epochs, n_channels, n_times), got {data.shape}")
        n_epochs, n_channels, _ = data.shape
        de_cache_path = f'../data/de_feat/{subject_id}_de_modified.npy'
        if os.path.exists(de_cache_path): return np.load(de_cache_path)
        
        info = mne.create_info(ch_names=[f'EEG{i}' for i in range(1, n_channels + 1)], sfreq=sfreq, ch_types='eeg')
        epochs_obj = mne.EpochsArray(data=data, info=info, verbose=False)
        
        de_feat_list = []
        for band_name, (f_min, f_max) in FREQ_BANDS.items():
            try:
                psd = epochs_obj.compute_psd(method='welch', fmin=f_min, fmax=f_max, verbose=False).get_data()
                if psd.shape[-1] == 0: raise ValueError(f"Error: no frequency points found between fmin={f_min} and fmax={f_max}.")
                psd += 1e-10 
                band_de_feat = np.sum(np.log(psd), axis=-1)
                de_feat_list.append(band_de_feat)
            except Exception as e:
                print(e)
                de_feat_list.append(np.zeros((n_epochs, n_channels)))

        de_features = np.concatenate(de_feat_list, axis=1)
        os.makedirs('../data/de_feat', exist_ok=True)
        np.save(de_cache_path, de_features)
        return de_features

    def extract_segment_features(self, epochs, subject_id):
        full_time_window_start = self.time_segments[self.selected_time_segments[0]][0]
        full_time_window_end = self.time_segments[self.selected_time_segments[-1]][1]
        
        epochs_for_de = epochs.copy().crop(tmin=full_time_window_start, tmax=full_time_window_end)
        sfreq = epochs_for_de.info['sfreq']
        
        all_de_features = []
        for region_name, channels in self.brain_regions.items():
            available_channels = [ch for ch in channels if ch in epochs_for_de.ch_names]
            if not available_channels: continue
            region_data = epochs_for_de.copy().pick(available_channels).get_data()
            de_features = self.compute_de_features(region_data, sfreq, f"{subject_id}_full_{region_name}")
            all_de_features.append(de_features)
        
        final_de_features = np.concatenate(all_de_features, axis=1)
        final_de_features_norm = MinMaxScaler().fit_transform(final_de_features)

        all_time_features = []
        epochs_for_time = epochs.copy().crop(tmin=self.time_window[0], tmax=self.time_window[1])
        for time_seg_name in self.selected_time_segments:
            time_start, time_end = self.time_segments[time_seg_name]
            for region_name, channels in self.brain_regions.items():
                available_channels = [ch for ch in channels if ch in epochs_for_time.ch_names]
                if not available_channels: continue
                segment_data = epochs_for_time.copy().pick(available_channels).crop(tmin=time_start, tmax=time_end).get_data()
                time_features = np.array([self.extract_time_domain_features(segment_data[i], sampling_rate=5) for i in range(segment_data.shape[0])])
                all_time_features.append(time_features)
        
        final_time_features = np.concatenate(all_time_features, axis=1)
        final_time_features_norm = MinMaxScaler().fit_transform(final_time_features)

        return np.concatenate([final_time_features_norm, final_de_features_norm], axis=1)

    def prepare_labels(self, epochs, hallucination_events=[1]):
        return np.array([1 if epochs.events[i, 2] in hallucination_events else 0 for i in range(len(epochs))])

    def load_multi_subject_data(self, subjects=None, data_type='sen', force_recompute=False):
        if subjects is None: subjects = self.find_available_subjects(data_type)
        all_features, all_labels, all_subject_ids = [], [], []
        
        for subj_idx in subjects:
            feat_path = os.path.join(self.feature_cache_dir, f"subj_{subj_idx}_features.npy")
            labels_path = os.path.join(self.feature_cache_dir, f"subj_{subj_idx}_labels.npy")
            
            if os.path.exists(feat_path) and os.path.exists(labels_path) and not force_recompute:
                features = np.load(feat_path)
                labels = np.load(labels_path)
            else:
                try:
                    epochs = self.load_subject_data(subj_idx, data_type)
                    features = self.extract_segment_features(epochs, subj_idx)
                    labels = self.prepare_labels(epochs, hallucination_events=[1])
                    
                    np.save(feat_path, features)
                    np.save(labels_path, labels)
                except Exception as e:
                    print(f"Error processing subject {subj_idx}: {e}")
                    continue
            
            all_features.append(features)
            all_labels.append(labels)
            all_subject_ids.extend([subj_idx] * len(features))

        if not all_features: raise ValueError("No valid subject data found.")
        return np.vstack(all_features), np.concatenate(all_labels), np.array(all_subject_ids)

def main():
    preprocessor = DataPreprocessor()
    try:
        features, labels, subject_ids = preprocessor.load_multi_subject_data()
        print(f"Loaded data shape: features {features.shape}, labels {labels.shape}, subject_ids {subject_ids.shape}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()