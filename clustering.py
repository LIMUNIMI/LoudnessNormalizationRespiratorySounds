import numpy as np
from typing import List, Callable
from tqdm import tqdm
from config import Config

from filters import *

from sklearn.cluster import KMeans

# ==== Clustering ====
def compute_cluster_features(paths: List[str], loader_fn: Callable[[str, int], np.ndarray], cfg: Config, apply_filtering: bool = False, features: list[str] = None) -> np.ndarray:
    vals = []
    for p in tqdm(paths, desc="Clustering Features"):
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