import numpy as np
import essentia
import essentia.standard as es
from config import Config


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

def cluster_norm(y_feat, km, intensity_train_scaled, cfg: Config):
    y, feat = y_feat
    cluster_id = km.predict([feat])[0]
    cluster_members = intensity_train_scaled[km.labels_ == cluster_id]
    cluster_scalar = float(np.mean(cluster_members))
    y_norm = normalize_by_scalar(y, cluster_scalar)
    return extract_features(y_norm, cfg=cfg)