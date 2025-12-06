import numpy as np
import essentia.standard as es
from config import Config

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