import numpy as np
import essentia.standard as es
from config import Config

# ==== Feature Extraction ====
def extract_features(audio: np.ndarray, cfg: Config) -> np.ndarray:
    # Framing
    n_mfcc = cfg.n_mfcc
    sample_rate = cfg.sample_rate
    wsize = int(cfg.window_size * sample_rate)
    hop = int(cfg.hop * sample_rate)

    # Spectrum
    window = es.Windowing(type='hann', size=wsize)
    spectrum = es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=n_mfcc)

    # Frame Iteration
    frames = es.FrameGenerator(audio, frameSize=wsize, hopSize=hop, startFromZero=True)
    mfccs = []
    for frame in frames:
        spec = spectrum(window(frame))
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)

    # MFCCs Aggregation
    mfccs = np.array(mfccs)
    if (mfccs.size == 0):
        mfccs = np.zeros((1, n_mfcc), dtype=np.float32)
    feat_mean = np.mean(mfccs, axis=0)
    feat_std = np.std(mfccs, axis=0)
    # Energy Features
    rms = es.RMS()(audio)
    zcr = es.ZeroCrossingRate()(audio)
    centroid = es.Centroid()(np.abs(es.Spectrum()(es.Windowing(type='hann')(audio[:min(len(audio), 2048)]))))

    return np.concatenate([feat_mean, feat_std, [rms, zcr, centroid]]).astype(np.float32)

def rms_features(y: np.ndarray, cfg: Config) -> np.ndarray:
    return extract_features(normalize_by_rms(y), cfg=cfg)

def median_features(y: np.ndarray, cfg: Config) -> np.ndarray:
    return extract_features(normalize_by_median(y), cfg=cfg)