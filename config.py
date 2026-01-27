from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable



# ==== Configuration Dataclass ====
@dataclass
class Config:
    # Audio Loading Macros
    sample_rate : int = 44100

    # Audio Processing Macros
    target_duration : float = 2.0
    bandpass_frequency : float = 965.0
    bandpass_bandwidth : float = 1930.0
    highpass_frequency : float = 300
    lowpass_frequency : float = 1800.0
    window_size : float = 0.5 # This to 0.5
    hop : float = 0.010

    # Feature Extraction Macros
    n_mfcc : int = 13
    kfolds : int = 5
    n_mel : int = 64

    # Clustering Macros
    n_clusters : int = 3 # This to 3
    cluster_features: list[str] = field(
        default_factory=lambda: ["rms", "zcr", "centroid", "flux", "rolloff", "flatness"]
    )
    random_state : int = 42

    # Classification
    tree_depth: int = 6 # This to 6

    # Automatic Classification
    autosklearn_time : int = 10
    autosklearn_per_run : int = 10
    autosklearn_memory : int = 4096

    # Step-by-step Processing
    duration_norm_toggle: bool = True # This to True
    highpass_toggle: bool = False
    bandpass_toggle: bool = True# This to True
    lowpass_toggle: bool = False
    fourth_filter_toggle: bool = True #This to True
    #wavelet_denoise_toggle: bool = False
    amplitude_norm_toggle: bool = True # This to True
    run_method: str = "all"  # Options: all, classification
    result_filename: str = "baseline_test.csv"


