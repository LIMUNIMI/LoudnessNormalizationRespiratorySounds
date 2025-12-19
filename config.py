from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable



# ==== Configuration Dataclass ====
@dataclass
class Config:
    # Audio Loading Macros
    sample_rate : int = 44100

    # Audio Processing Macros
    target_duration : float = 60.0
    bandpass_frequency : float = 1150.0
    bandpass_bandwidth : float = 1700.0
    highpass_frequency : float = 300
    window_size : float = 0.5
    hop : float = 0.010

    # Feature Extraction Macros
    n_mfcc : int = 13
    kfolds : int = 5
    n_mel : int = 64

    # Clustering Macros
    n_clusters : int = 5
    cluster_features: list[str] = field(
        default_factory=lambda: ["rms", "zcr", "centroid", "flux", "rolloff", "flatness"]
    )
    random_state : int = 42

    # Automatic Classification
    autosklearn_time : int = 10
    autosklearn_per_run : int = 10
    autosklearn_memory : int = 4096

    # Step-by-step Processing
    duration_norm_toggle: bool = True
    highpass_toggle: bool = False
    bandpass_toggle: bool = False
    #wavelet_denoise_toggle: bool = False
    amplitude_norm_toggle: bool = False
    run_method: str = "all"  # Options: all, classification
    result_filename: str = "experiment_results.csv"


