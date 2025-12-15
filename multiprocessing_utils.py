import os
import numpy as np
import essentia.standard as es
from filters import apply_filters

from config import *
from normalization_utils import normalize_duration

# ==== Load Wrappers ====
def load_mono(path: str, sample_rate: int) -> np.ndarray:
    audio = es.MonoLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)

def load_eqloud(path: str, sample_rate: int) -> np.ndarray:
    audio = es.EqloudLoader(filename=path, sampleRate=sample_rate)()
    return audio.astype(np.float32)


# === Multiprocessing Utils ===
def preprocess_file(path: str, cfg: Config, use_filtering: bool, use_duration_norm: bool) -> np.ndarray:
    y = load_mono(path, cfg.sample_rate)
    if use_filtering:
        y = apply_filters(y, cfg=cfg, use_bp=True)
    y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=cfg.target_duration)
    return y

def preprocess_eqloud(path: str, cfg: Config, use_filtering: bool, use_duration_norm: bool,filtered_dir: str):
    if use_filtering:
        y = load_mono(path, cfg.sample_rate)
        y = apply_filters(y, cfg=cfg, use_bp=True)
        filtered_path = os.path.join(filtered_dir, os.path.basename(path))
        es.MonoWriter(filename=filtered_path, sampleRate=cfg.sample_rate)(y)
        y = load_eqloud(filtered_path, cfg.sample_rate)
    else:
        y = load_eqloud(path, cfg.sample_rate)
    if use_duration_norm:
        y = normalize_duration(y, sample_rate=cfg.sample_rate, target_length=cfg.target_duration)
    return y