from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable

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

    # Automatic Classification
    autosklearn_time : int = 10
    autosklearn_per_run : int = 10