import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
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

# ==== Load Wrappers ====
def load_mono(path: str, sample_rate: int) -> np.ndarray:
    audio = es.MonoLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)

def load_eqloud(path: str, sample_rate: int) -> np.ndarray:
    audio = es.EqloudLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)

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
    diagnosis_directory = "ICBHI_Challenge_diagnosis.txt"
    print("[DEBUG]: Loading dataset...\n")
    paths, labels = load_dataset_macro(dataset_directory, diagnosis_directory)
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