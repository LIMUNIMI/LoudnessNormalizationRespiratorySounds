import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm

import essentia
import essentia.standard as es

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix


# =====================================
# ==== Configuration and Utilities ====
# =====================================

# ==== Configuration Dataclass ====
@dataclass
class Config:
    # Audio Loading Macros
    sample_rate : int = 44100

    # Audio Processing Macros
    target_duration : int = 10
    bandpass_frequency : float = 1150.0
    bandpass_band : float = 1700.0
    window_size : float = 0.025
    hop : float = 0.010

    # Feature Extraction Macros
    n_mfcc : int = 13
    kfolds : int = 5

    # Clustering Macros
    n_clusters : int = 5
    cluster_features: list[str] = field(
        default_factory=lambda: ["rms", "zcr", "centroid", "flux", "rolloff", "flatness"]
    )
    random_state : int = 42

# ==== Notes ====
#
# ==== End of Notes ====

# ==== Load Wrappers ====
def load_mono(path: str, sample_rate: int) -> np.ndarray:
    audio = es.MonoLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)

def load_eqloud(path: str, sample_rate: int) -> np.ndarray:
    audio = es.EqloudLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)

# ==== Noise Reduction Filters ====
def highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_frequency: float = 80.0) -> np.ndarray:
    audio_dc = es.DCRemoval()(audio)
    hp = es.HighPass(cutoffFrequency=cutoff_frequency, sampleRate=sample_rate)
    return hp(audio_dc)

def bandpass_filter(audio: np.ndarray, sample_rate: int, cutoff_frequency: float, bandwidth: float) -> np.ndarray:
    audio_dc = es.DCRemoval()(audio)
    bp = es.BandPass(cutoffFrequency=cutoff_frequency, sampleRate=sample_rate, bandwidth=bandwidth)
    return bp(audio)

def apply_filters(audio: np.ndarray, cfg: Config, use_hp: bool=False, use_bp=True) -> np.ndarray:
    y = audio
    if use_hp:
        y = highpass_filter(y, cfg.sample_rate)
    
    if use_bp:
        y = bandpass_filter(y, sample_rate=cfg.sample_rate, cutoff_frequency=cfg.bandpass_frequency, bandwidth=cfg.bandpass_band)
    return y


# ==== Duration Normalization Utilities ====
def normalize_duration(audio: np.ndarray, sample_rate: int, target_length: int) -> np.ndarray:
    target_samples = target_length * sample_rate
    if audio.shape[0] == target_samples:
        return audio
    elif audio.shape[0] < target_samples:
        return np.pad(audio, (0, target_samples - audio.shape[0]), mode='constant')
    else:
        return audio[:target_samples]

# ==== Loudness Normalization Utilities ====
def normalize_by_rms(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = es.RMS()(audio)
    return audio / (rms + eps)

def normalize_by_median(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    median = np.median(np.abs(audio))
    return audio / (median + eps)

def normalize_by_scalar(audio: np.ndarray, scalar: float, eps: float = 1e-8) -> np.ndarray:
    return audio / (scalar + eps)

# ==== Feature Extraction ====
def extract_features(audio: np.ndarray, cfg: Config) -> np.ndarray:
    # Framing
    n_mfcc = cfg.n_mfcc
    sample_rate = cfg.sample_rate
    wsize = int(cfg.window_size * sample_rate)
    hop = int(cfg.hop * sample_rate)

    # Spectrum
    window = es.Windowing(type='hann', size=wsize)
    spectrum = es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=n_mfcc)

    # Frame Iteration
    frames = es.FrameGenerator(audio, frameSize=wsize, hopSize=hop, startFromZero=True)
    mfccs = []
    for frame in frames:
        spec = spectrum(window(frame))
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)

    # MFCCs Aggregation
    mfccs = np.array(mfccs)
    if (mfccs.size == 0):
        mfccs = np.zeros((1, n_mfcc), dtype=np.float32)
    feat_mean = np.mean(mfccs, axis=0)
    feat_std = np.std(mfccs, axis=0)
    # Energy Features
    rms = es.RMS()(audio)
    zcr = es.ZeroCrossingRate()(audio)
    centroid = es.Centroid()(np.abs(es.Spectrum()(es.Windowing(type='hann')(audio[:min(len(audio), 2048)]))))

    return np.concatenate([feat_mean, feat_std, [rms, zcr, centroid]]).astype(np.float32)

# ==== Classification ====
def evaluate(X: np.ndarray, y: np.ndarray, cfg: Config) -> dict[str, float]:
    results = {}

    cv = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.random_state)

    # kNN Classifier
    knn_clf = SKPipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    acc_knn = cross_val_score(knn_clf, X, y, cv=cv, scoring='accuracy').mean()
    results['knn_accuracy'] = acc_knn

    # SVM RBF Classifier
    svm_clf = SKPipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])
    acc_svm = cross_val_score(svm_clf, X, y, cv=cv, scoring='accuracy').mean()
    results['svm_accuracy'] = acc_svm

    y = np.array(y)
    for name, clf in [('knn', knn_clf), ('svm', svm_clf)]:
        accs, sens, specs = [], [], []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))

            sens.append(recall_score(y_test, y_pred, pos_label=1, average='weighted'))

            specs.append(recall_score(y_test, y_pred, pos_label=0, average='weighted'))

        results[f'{name}_accuracy'] = np.mean(accs)
        results[f'{name}_sensitivity'] = np.mean(sens)
        results[f'{name}_specificity'] = np.mean(specs)



    return results

# ==== Dataset Utilities ====
def load_dataset(audio_dir: str, label_dir: str) -> Tuple[List[str], np.ndarray]:
    label_df = pd.read_csv(label_dir)

    ann_map = {
        row['file_name']: (row['n_wheeze'], row['n_crackle'])
        for _, row in label_df.iterrows()
    }

    files = [] 
    labels = []

    for f in os.listdir(audio_dir):
        if f.endswith(".wav"):
            path = os.path.join(audio_dir, f)
            files.append(path)

            f_base = os.path.splitext(f)[0]

            if f_base in ann_map:

                n_wheeze, n_crackle = ann_map[f_base]

                if int(n_wheeze) > 0 and int(n_crackle) > 0:
                    label = 'both'
                elif int(n_wheeze) > 0:
                    label = 'wheeze'
                elif int(n_crackle) > 0:
                    label = 'crackle'
                else:
                    label = 'none'
            else:
                label = 'none'
        
            labels.append(label)
        
    return files, labels

# ==== Clustering ====
def compute_cluster_features(paths: List[str], loader_fn: Callable[[str, int], np.ndarray], cfg: Config, apply_filtering: bool = False, features: list[str] = None) -> np.ndarray:
    vals = []
    for p in tqdm(paths, desc="Intensity Features"):
        y = loader_fn(p, cfg.sample_rate)
        if apply_filtering:
            y = apply_filters(y, cfg=cfg, use_bp=True)
        
        # Generating spectrum
        frame = y[:min(len(y), 2048)]
        spec = es.Spectrum()(es.Windowing(type='hann')(frame))

        # Feature map
        feature_map = {
            "rms": es.RMS()(y),
            "zcr": es.ZeroCrossingRate()(y),
            "centroid": es.Centroid()(spec),
            "flux": es.Flux()(spec),
            "rolloff": es.RollOff()(spec),
            "flatness": es.Flatness()(spec)
        }

        # Select only requested features
        selected = [feature_map[f] for f in features if f in feature_map]
        vals.append(selected)

    return np.array(vals, dtype=np.float64)

def fit_kmeans(features: np.ndarray, cfg: Config) -> KMeans:
    km = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state, n_init=10)
    km.fit(features)
    return km

  
# ===================
# ==== Pipelines ====
# ===================

# ==== Mono Loading (statistical) ====
def run_mono(paths: List[str], labels: np.ndarray, cfg: Config, use_filtering: bool = True, use_global_scalars: bool = False) -> Dict[str, Dict[str, float]]:
    results = {}

    # Sample Loading
    audio_list = []
    for p in tqdm(paths, desc="Mono Loading"):
        y = load_mono(p, cfg.sample_rate)
        if use_filtering:
            y = apply_filters(y, cfg=cfg, use_bp=True)

        y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=10)
        
        audio_list.append(y)
    
    # ---- Pipe A: Mean-based Normalization ----
    feats = []
    for y in audio_list:
        y_norm = normalize_by_rms(y)
        feats.append(extract_features(y_norm, cfg=cfg))
    
    X = np.vstack(feats)
    results['mono_rms'] = evaluate(X, labels, cfg)

    # ---- Pipe B: Median-based Normalization ----
    feats = []
    for y in audio_list:
        y_norm = normalize_by_median(y)
        feats.append(extract_features(y_norm, cfg=cfg))
    
    X = np.vstack(feats)
    results['mono_median'] = evaluate(X, labels, cfg)

    # ---- Pipe C: K-Means Clustering Normalization ----
    intensity = compute_cluster_features(paths, load_mono, cfg, apply_filtering=use_filtering, features=cfg.cluster_features)
    km = fit_kmeans(intensity, cfg)

    scaler = StandardScaler()
    intensity_scaled = scaler.fit_transform(intensity)

    feats = []
    for y, feat in zip(audio_list, intensity_scaled):
        # Ensure feat is np.float64 for sklearn
        feat = np.asarray(feat, dtype=np.float64)
        cluster_id = km.predict([feat])[0]
        # Ensure cluster_members is also np.float64
        cluster_members = intensity_scaled[km.labels_ == cluster_id]
        cluster_scalar = float(np.mean(cluster_members))
        y_norm = normalize_by_scalar(y, cluster_scalar)
        feats.append(extract_features(y_norm, cfg=cfg))
    
    X = np.vstack(feats)
    results['mono_cluster'] = evaluate(X, labels, cfg)

    return results

# ==== EqLoud Loading (perceptive) ====
def run_eqloud(paths: List[str], labels: np.ndarray, cfg: Config, use_filtering: bool = True, use_global_scalars: bool = False) -> Dict[str, Dict[str, float]]:
    results = {}
    filtered_dir = "filtered_data/"

    # Sample Loading
    audio_list = []
    for p in tqdm(paths, desc="EqLoud Loading"):
        if use_filtering:
            print("[DEBUG]: Using filtering for EqLoud")
            y = load_mono(p, cfg.sample_rate)
            y = apply_filters(y, cfg=cfg, use_bp=True)
            filtered_path = os.path.join(filtered_dir, os.path.basename(p))
            es.MonoWriter(filename=filtered_path, sampleRate=cfg.sample_rate)(y)
            y = load_eqloud(filtered_path, cfg.sample_rate)
        else:
            y = load_eqloud(p, cfg.sample_rate)
        
        y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=10)

        audio_list.append(y)
    
    # ---- Pipe A: Mean-based Normalization ----
    feats = []
    for y in audio_list:
        y_norm = normalize_by_rms(y)
        feats.append(extract_features(y_norm, cfg=cfg))
    
    X = np.vstack(feats)
    results['eqloud_rms'] = evaluate(X, labels, cfg)

    # ---- Pipe B: Median-based Normalization ----
    feats = []
    for y in audio_list:
        y_norm = normalize_by_median(y)
        feats.append(extract_features(y_norm, cfg=cfg))
    
    X = np.vstack(feats)
    results['eqloud_median'] = evaluate(X, labels, cfg)

    # ---- Pipe C: K-Means Clustering Normalization ----
    intensity = compute_cluster_features(paths, load_mono, cfg, apply_filtering=use_filtering, features=cfg.cluster_features)
    km = fit_kmeans(intensity, cfg)

    scaler = StandardScaler()
    intensity_scaled = scaler.fit_transform(intensity)

    feats = []
    for y, feat in zip(audio_list, intensity_scaled):
        # Ensure feat is np.float64 for sklearn
        feat = np.asarray(feat, dtype=np.float64)
        cluster_id = km.predict([feat])[0]
        # Ensure cluster_members is also np.float64
        cluster_members = intensity_scaled[km.labels_ == cluster_id]
        cluster_scalar = float(np.mean(cluster_members))
        y_norm = normalize_by_scalar(y, cluster_scalar)
        feats.append(extract_features(y_norm, cfg=cfg))
    
    X = np.vstack(feats)
    results['eqloud_cluster'] = evaluate(X, labels, cfg)

    return results


# ======================
# ==== Main Process ====
# ======================

# ==== Main Definition ====
def main():
    print("[DEBUG]: Configuring files...\n")
    cfg = Config(sample_rate=44100, n_mfcc=13, n_clusters=3, kfolds=5, random_state=42)

    dataset_directory = "resampled_data/"
    metadata_directory = "metadata/icbhi_summary.csv"
    print("[DEBUG]: Loading dataset...\n")
    paths, labels = load_dataset(dataset_directory, metadata_directory)
    from collections import Counter

    assert len(paths) == len(labels), "ERROR: Paths and Labels not in same number."
    assert len(paths) > 0, "ERROR: Can't find any path."

    # Running Mono Pipelines
    print("[DEBUG]: Running Mono Pipelines...\n")
    res_mono = run_mono(paths, labels, cfg, use_filtering=True, use_global_scalars=False)

    # Running EqLoud Pipelines
    print("[DEBUG]: Running EqLoud Pipelines...\n")
    res_eqloud = run_eqloud(paths, labels, cfg, use_filtering=True, use_global_scalars=False)

    # Printing Results
    def print_results(title: str, res: Dict[str, Dict[str, float]]):
        print(f"\n=== {title} ===")
        for k, v in res.items():
            print(f"{k:16s} | kNN acc: {v['knn_accuracy']:.4f} | SVM acc: {v['svm_accuracy']:.4f}")
            #print(f"{k:16s} | kNN sen: {v['knn_sensitivity']:.4f} | SVM sen: {v['svm_sensitivity']:.4f}")
            #print(f"{k:16s} | kNN spe: {v['knn_specificity']:.4f} | SVM spe: {v['svm_specificity']:.4f}")


    print_results("MonoLoader Normalization (statistical)", res_mono)
    print_results("EqLoudLoader Normalization (perceptive)", res_eqloud)

# ==== Creating Process ====
if __name__ == "__main__":
    main()