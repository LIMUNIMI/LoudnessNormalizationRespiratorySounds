import numpy as np
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import essentia.standard as es
from sklearn.cluster import KMeans

from filters import *

# ==== Clustering ====
def process_file_for_cluster(p: str, sample_rate: int, loader_fn: str,
                             use_hp: bool, use_bp: bool,
                             hp_cutoff: float, bp_cutoff: float, bp_bandwidth: float,
                             features: list[str]) -> list[float]:
    # Loader
    if loader_fn == 'mono':
        y = es.MonoLoader(filename=p, sampleRate=sample_rate)()
    else:
        y = es.EqloudLoader(filename=p, sampleRate=sample_rate)()

    # Filtri
    if use_bp or use_hp:
        y = apply_filters(y, sample_rate=sample_rate,
                          use_hp=use_hp, use_bp=use_bp,
                          hp_cutoff=hp_cutoff, bp_cutoff=bp_cutoff,
                          bp_bandwidth=bp_bandwidth)

    # Spectrum
    frame = y[:min(len(y), 2048)]
    spec = es.Spectrum()(es.Windowing(type='hann')(frame))

    # Feature map
    # Valuta aggiunta di LogMel per clustering
    feature_map = {
        "rms": es.RMS()(y),
        "zcr": es.ZeroCrossingRate()(y),
        "centroid": es.Centroid()(spec),
        "flux": es.Flux()(spec),
        "rolloff": es.RollOff()(spec),
        "flatness": es.Flatness()(spec)
    }

    # Seleziona solo le feature richieste
    return [feature_map[f] for f in features if f in feature_map]


def compute_cluster_features(directory: str, sample_rate: int,
                             loader_fn: str = 'mono',
                             use_hp: bool = False, use_bp: bool = False,
                             hp_cutoff: float = 80.0, bp_cutoff: float = 300.0,
                             bp_bandwidth: float = 100.0,
                             features: list[str] = None) -> np.ndarray:
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

    worker = partial(process_file_for_cluster,
                     sample_rate=sample_rate,
                     loader_fn=loader_fn,
                     use_hp=use_hp, use_bp=use_bp,
                     hp_cutoff=hp_cutoff, bp_cutoff=bp_cutoff,
                     bp_bandwidth=bp_bandwidth,
                     features=features)

    # Parallelizzazione
    with ProcessPoolExecutor() as executor:
        vals = list(executor.map(worker, file_paths))

    return np.array(vals, dtype=np.float64)


def fit_kmeans(features: np.ndarray, n_clusters: int = 2, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(features)
    return km