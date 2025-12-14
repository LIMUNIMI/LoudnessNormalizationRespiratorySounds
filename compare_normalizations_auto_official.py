import os
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

import essentia.standard as es


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline

from config import *
from filters import *
from normalization_utils import *
from features import *
from clustering import *
from classification import *
from dataset_utils import *

# ==== Load Wrappers ====
def load_mono(path: str, sample_rate: int) -> np.ndarray:
    audio = es.MonoLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)

def load_eqloud(path: str, sample_rate: int) -> np.ndarray:
    audio = es.EqloudLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)

  
# ===================
# ==== Processes ====
# ===================

def run_mono_official(train_paths: List[str], test_paths: List[str],
                      train_labels: np.ndarray, test_labels: np.ndarray,
                      cfg: Config, use_filtering: bool = True,
                      use_global_scalars: bool = False) -> Dict[str, Dict[str, float]]:
    results = {}

    # === Sample Loading (train) ===
    train_audio = []
    for p in tqdm(train_paths, desc="Mono Loading - Train"):
        y = load_mono(p, cfg.sample_rate)
        if use_filtering:
            y = apply_filters(y, cfg=cfg, use_bp=True)
        y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=10)
        train_audio.append(y)

    # === Sample Loading (test) ===
    test_audio = []
    for p in tqdm(test_paths, desc="Mono Loading - Test"):
        y = load_mono(p, cfg.sample_rate)
        if use_filtering:
            y = apply_filters(y, cfg=cfg, use_bp=True)
        y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=10)
        test_audio.append(y)

    # ---- Pipe A: Mean-based Normalization ----
    X_train = np.vstack([extract_features(normalize_by_rms(y), cfg=cfg) for y in train_audio])
    X_test  = np.vstack([extract_features(normalize_by_rms(y), cfg=cfg) for y in test_audio])
    results['mono_rms'] = evaluate_auto_official(X_train, train_labels, X_test, test_labels, cfg)

    # ---- Pipe B: Median-based Normalization ----
    X_train = np.vstack([extract_features(normalize_by_median(y), cfg=cfg) for y in train_audio])
    X_test  = np.vstack([extract_features(normalize_by_median(y), cfg=cfg) for y in test_audio])
    results['mono_median'] = evaluate_auto_official(X_train, train_labels, X_test, test_labels, cfg)

    # ---- Pipe C: K-Means Clustering Normalization ----
    intensity_train = compute_cluster_features(train_paths, load_mono, cfg,
                                               apply_filtering=use_filtering,
                                               features=cfg.cluster_features)
    intensity_test  = compute_cluster_features(test_paths, load_mono, cfg,
                                               apply_filtering=use_filtering,
                                               features=cfg.cluster_features)

    km = fit_kmeans(intensity_train, cfg)
    scaler = StandardScaler()
    intensity_train_scaled = scaler.fit_transform(intensity_train)
    intensity_test_scaled  = scaler.transform(intensity_test)

    feats_train, feats_test = [], []
    for y, feat in zip(train_audio, intensity_train_scaled):
        cluster_id = km.predict([feat])[0]
        cluster_members = intensity_train_scaled[km.labels_ == cluster_id]
        cluster_scalar = float(np.mean(cluster_members))
        y_norm = normalize_by_scalar(y, cluster_scalar)
        feats_train.append(extract_features(y_norm, cfg=cfg))

    for y, feat in zip(test_audio, intensity_test_scaled):
        cluster_id = km.predict([feat])[0]
        cluster_members = intensity_train_scaled[km.labels_ == cluster_id]
        cluster_scalar = float(np.mean(cluster_members))
        y_norm = normalize_by_scalar(y, cluster_scalar)
        feats_test.append(extract_features(y_norm, cfg=cfg))

    X_train = np.vstack(feats_train)
    X_test  = np.vstack(feats_test)
    results['mono_cluster'] = evaluate_auto_official(X_train, train_labels, X_test, test_labels, cfg)

    return results

def run_eqloud_official(train_paths: List[str], test_paths: List[str],
                        train_labels: np.ndarray, test_labels: np.ndarray,
                        cfg: Config, use_filtering: bool = True,
                        use_global_scalars: bool = False) -> Dict[str, Dict[str, float]]:
    results = {}
    filtered_dir = "filtered_data/"
    os.makedirs(filtered_dir, exist_ok=True)

    # === Sample Loading (train) ===
    train_audio = []
    for p in tqdm(train_paths, desc="EqLoud Loading - Train"):
        if use_filtering:
            y = load_mono(p, cfg.sample_rate)
            y = apply_filters(y, cfg=cfg, use_bp=True)
            filtered_path = os.path.join(filtered_dir, os.path.basename(p))
            es.MonoWriter(filename=filtered_path, sampleRate=cfg.sample_rate)(y)
            y = load_eqloud(filtered_path, cfg.sample_rate)
        else:
            y = load_eqloud(p, cfg.sample_rate)
        y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=10)
        train_audio.append(y)

    # === Sample Loading (test) ===
    test_audio = []
    for p in tqdm(test_paths, desc="EqLoud Loading - Test"):
        if use_filtering:
            y = load_mono(p, cfg.sample_rate)
            y = apply_filters(y, cfg=cfg, use_bp=True)
            filtered_path = os.path.join(filtered_dir, os.path.basename(p))
            es.MonoWriter(filename=filtered_path, sampleRate=cfg.sample_rate)(y)
            y = load_eqloud(filtered_path, cfg.sample_rate)
        else:
            y = load_eqloud(p, cfg.sample_rate)
        y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=10)
        test_audio.append(y)

    # ---- Pipe A: Mean-based Normalization ----
    X_train = np.vstack([extract_features(normalize_by_rms(y), cfg=cfg) for y in train_audio])
    X_test  = np.vstack([extract_features(normalize_by_rms(y), cfg=cfg) for y in test_audio])
    results['eqloud_rms'] = evaluate_auto_official(X_train, train_labels, X_test, test_labels, cfg)

    # ---- Pipe B: Median-based Normalization ----
    X_train = np.vstack([extract_features(normalize_by_median(y), cfg=cfg) for y in train_audio])
    X_test  = np.vstack([extract_features(normalize_by_median(y), cfg=cfg) for y in test_audio])
    results['eqloud_median'] = evaluate_auto_official(X_train, train_labels, X_test, test_labels, cfg)

    # ---- Pipe C: K-Means Clustering Normalization ----
    intensity_train = compute_cluster_features(train_paths, load_mono, cfg,
                                               apply_filtering=use_filtering,
                                               features=cfg.cluster_features)
    intensity_test  = compute_cluster_features(test_paths, load_mono, cfg,
                                               apply_filtering=use_filtering,
                                               features=cfg.cluster_features)

    km = fit_kmeans(intensity_train, cfg)
    scaler = StandardScaler()
    intensity_train_scaled = scaler.fit_transform(intensity_train)
    intensity_test_scaled  = scaler.transform(intensity_test)

    feats_train, feats_test = [], []
    for y, feat in zip(train_audio, intensity_train_scaled):
        cluster_id = km.predict([feat])[0]
        cluster_members = intensity_train_scaled[km.labels_ == cluster_id]
        cluster_scalar = float(np.mean(cluster_members))
        y_norm = normalize_by_scalar(y, cluster_scalar)
        feats_train.append(extract_features(y_norm, cfg=cfg))

    for y, feat in zip(test_audio, intensity_test_scaled):
        cluster_id = km.predict([feat])[0]
        cluster_members = intensity_train_scaled[km.labels_ == cluster_id]
        cluster_scalar = float(np.mean(cluster_members))
        y_norm = normalize_by_scalar(y, cluster_scalar)
        feats_test.append(extract_features(y_norm, cfg=cfg))

    X_train = np.vstack(feats_train)
    X_test  = np.vstack(feats_test)
    results['eqloud_cluster'] = evaluate_auto_official(X_train, train_labels, X_test, test_labels, cfg)

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
    official_split_directory = "metadata/ICBHI_challenge_train_test.txt"
    print("[DEBUG]: Loading dataset...\n")
    #paths, labels = load_dataset(dataset_directory, metadata_directory)
    train_labels, test_labels = load_labels_split(dataset_directory, metadata_directory, official_split_directory)
    from collections import Counter

    # Getting official split infos
    train_paths, test_paths = load_train_test_split(official_split_directory, dataset_directory)

    assert len(train_paths) == len(train_labels), "ERROR: Train paths and Labels not in same number."
    assert len(train_paths) > 0, "ERROR: Can't find any train path."

    assert len(test_paths) == len(test_labels), "ERROR: Test paths and Labels not in same number."
    assert len(test_paths) > 0, "ERROR: Can't find any test path."

    # Running Mono Pipelines
    print("[DEBUG]: Running Mono Pipelines with official split...\n")
    #res_mono = run_mono(paths, labels, cfg, use_filtering=True, use_global_scalars=False)
    res_mono = run_mono_official(train_paths=train_paths, test_paths=test_paths, train_labels=train_labels, test_labels=test_labels, cfg=cfg, use_filtering=True, use_global_scalars=False)

    # Running EqLoud Pipelines
    print("[DEBUG]: Running EqLoud Pipelines with official split...\n")
    #res_eqloud = run_eqloud(paths, labels, cfg, use_filtering=True, use_global_scalars=False)
    res_eqloud = run_eqloud_official(train_paths=train_paths, test_paths=test_paths, train_labels=train_labels, test_labels=test_labels, cfg=cfg, use_filtering=True, use_global_scalars=False)

    # Printing Results
    def print_results(title: str, res: Dict[str, Dict[str, float]]):
        print(f"\n=== {title} ===")
        for k, v in res.items():
            print(f"{k:16s} | Auto acc: {v['auto_accuracy_official_split']:.4f}")
            #print(f"{k:16s} | kNN sen: {v['knn_sensitivity']:.4f} | SVM sen: {v['svm_sensitivity']:.4f}")
            #print(f"{k:16s} | kNN spe: {v['knn_specificity']:.4f} | SVM spe: {v['svm_specificity']:.4f}")


    print_results("MonoLoader Normalization (statistical)", res_mono)
    print_results("EqLoudLoader Normalization (perceptive)", res_eqloud)

# ==== Creating Process ====
if __name__ == "__main__":
    main()