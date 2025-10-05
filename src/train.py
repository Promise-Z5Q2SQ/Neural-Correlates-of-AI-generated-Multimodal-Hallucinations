import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import joblib
import warnings
import argparse
from datetime import datetime

from datapreprocess import DataPreprocessor

TORCH_AVAILABLE = False
DEVICE = "cpu"

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True

    class PyTorchMLP(nn.Module):
        def __init__(self, input_dim, hidden_layer_sizes=(100,), output_dim=2):
            super(PyTorchMLP, self).__init__()
            layers = []
            last_dim = input_dim
            for size in hidden_layer_sizes:
                layers.append(nn.Linear(last_dim, size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))
                last_dim = size
            layers.append(nn.Linear(last_dim, output_dim))
            self.layers = nn.Sequential(*layers)
        def forward(self, x): return self.layers(x)

    class AttentionModel(nn.Module):
        def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=2, output_dim=2):
            super(AttentionModel, self).__init__()
            self.input_proj = nn.Linear(input_dim, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.5)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(embed_dim, output_dim)
        def forward(self, x):
            x = self.input_proj(x).unsqueeze(1)
            x = self.transformer_encoder(x).squeeze(1)
            return self.output_proj(x)

    class TorchModelWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model_class, model_params, max_iter=300, learning_rate=0.001, batch_size=32):
            self.model_class = model_class
            self.model_params = model_params
            self.max_iter = max_iter
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.model = None
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            y_tensor = torch.LongTensor(y).to(DEVICE)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model = self.model_class(input_dim=X.shape[1], **self.model_params).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model.train()
            for epoch in range(self.max_iter):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            return self

        def predict_proba(self, X):
            self.model.to(DEVICE).eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(DEVICE)
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            return probabilities

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

except ImportError:
    pass

warnings.filterwarnings('ignore')

class AdvancedHallucinationClassifier:
    def get_model(self, model_name):
        common_steps = [('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]
        if model_name == 'SVM':
            return Pipeline(common_steps + [('classifier', SVC(C=1, kernel='rbf', random_state=42, probability=True))])
        elif model_name == 'RF':
            return Pipeline(common_steps + [('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        elif model_name == 'GBDT':
            return Pipeline(common_steps + [('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))])
        elif model_name == 'MLP':
            if TORCH_AVAILABLE:
                return Pipeline(common_steps + [('classifier', TorchModelWrapper(model_class=PyTorchMLP, model_params={'hidden_layer_sizes': (100,)}))])
            else:
                print("Warning: PyTorch unavailable, falling back to sklearn MLPClassifier.")
                return Pipeline(common_steps + [('classifier', SklearnMLPClassifier(hidden_layer_sizes=(100,), alpha=0.01, random_state=42, max_iter=300))])
        elif model_name == 'attention_based':
            if TORCH_AVAILABLE:
                return Pipeline(common_steps + [('classifier', TorchModelWrapper(model_class=AttentionModel, model_params={'embed_dim': 128, 'num_heads': 8}))])
            else:
                raise ImportError("Attention model requires PyTorch, which is not available.")
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _save_model(self, model, save_path):
        if TORCH_AVAILABLE:
            classifier_step = model.named_steps.get('classifier')
            if isinstance(classifier_step, TorchModelWrapper) and classifier_step.model is not None:
                classifier_step.model.to('cpu')
        
        joblib.dump(model, save_path)
        print(f"model saved to {save_path}")

    def cross_subject_validation(self, features, labels, subject_ids, model_name):
        if len(np.unique(subject_ids)) < 10: return
        gkf = GroupKFold(n_splits=27)
        for fold, (train_idx, _) in enumerate(gkf.split(features, labels, groups=subject_ids)):
            print(f"Training Cross-Subject: {model_name}, Fold {fold+1}/27")
            save_dir = f'../data/model_params/cross_subject/{model_name}/'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'fold_{fold+1}_{datetime.now().strftime("%Y%m%d")}.pkl')
            model = self.get_model(model_name)
            model.fit(features[train_idx], labels[train_idx])
            self._save_model(model, save_path)

    def within_subject_validation(self, features, labels, subject_ids, model_name):
        for subject in np.unique(subject_ids):
            subject_mask = subject_ids == subject
            subject_features, subject_labels = features[subject_mask], labels[subject_mask]
            n_samples, n_positive = len(subject_features), np.sum(subject_labels)
            if n_samples < 10 or n_positive < 2 or (n_samples - n_positive) < 2: continue
            
            print(f"Training Within-Subject: {model_name}, Subject {subject}")
            n_splits = min(10, n_positive, n_samples - n_positive)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fold, (train_idx, _) in enumerate(cv.split(subject_features, subject_labels)):
                save_dir = f'../data/model_params/within_subject/{model_name}/'
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'subject_{subject}_fold_{fold+1}_{datetime.now().strftime("%Y%m%d")}.pkl')
                model = self.get_model(model_name)
                model.fit(subject_features[train_idx], subject_labels[train_idx])
                self._save_model(model, save_path)

def run_experiment(args):
    preprocessor = DataPreprocessor()
    features, labels, subject_ids = preprocessor.load_multi_subject_data(force_recompute=args.recompute)
    classifier = AdvancedHallucinationClassifier()
    models_to_run = ['SVM', 'RF', 'GBDT', 'MLP', 'attention_based'] if args.model == 'all' else [args.model]
    for model_name in models_to_run:
        if args.strategy in ['cross', 'all']:
            classifier.cross_subject_validation(features, labels, subject_ids, model_name)
        if args.strategy in ['within', 'all']:
            classifier.within_subject_validation(features, labels, subject_ids, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Hallucination Detection Training Script")
    parser.add_argument('--model', type=str, default='all', choices=['SVM', 'RF', 'GBDT', 'MLP', 'attention_based', 'all'])
    parser.add_argument('--strategy', type=str, default='all', choices=['cross', 'within', 'all'])
    parser.add_argument('--recompute', action='store_true', help="Recompute features even if cached versions exist.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index to use.")
    args = parser.parse_args()

    if TORCH_AVAILABLE and torch.cuda.is_available() and args.gpu >= 0:
        try:
            DEVICE = torch.device(f"cuda:{args.gpu}")
            torch.empty(1, device=DEVICE)
        except (RuntimeError, AssertionError) as e:
            print(f"Warning: Unable to use specified GPU cuda:{args.gpu} ({e}). Switching to CPU.")
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cpu")

    print(f"Device: {DEVICE}")
    
    run_experiment(args)