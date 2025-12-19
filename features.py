import numpy as np
import essentia.standard as es
from config import Config
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from normalization_utils import *


# ==== Feature Extraction ====
def median_features(y: np.ndarray, cfg: Config) -> np.ndarray:
    return extract_features(normalize_by_median(y), cfg=cfg)


def rms_features(y: np.ndarray, cfg: Config) -> np.ndarray:
    return extract_features(normalize_by_rms(y), cfg=cfg)


def extract_features(audio: np.ndarray, cfg) -> np.ndarray:
    # Parametri
    n_mfcc = cfg.n_mfcc
    sample_rate = cfg.sample_rate
    wsize = int(cfg.window_size * sample_rate)
    hop = int(cfg.hop * sample_rate)

    # Moduli Essentia
    window = es.Windowing(type='hann', size=wsize)
    spectrum = es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=n_mfcc)
    melbands = es.MelBands(
        sampleRate=sample_rate,
        numberBands=cfg.n_mel,
        lowFrequencyBound=0,
        highFrequencyBound=sample_rate/2,
        normalize='unit_max',
        log=True
    )

    # Iterazione sui frame
    frames = es.FrameGenerator(audio, frameSize=wsize, hopSize=hop, startFromZero=True)
    mfccs, logmels = [], []
    for frame in frames:
        spec = spectrum(window(frame))
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)
        mel = melbands(spec)
        logmels.append(mel)

    # Aggregazione MFCC
    mfccs = np.array(mfccs)
    if mfccs.size == 0:
        mfccs = np.zeros((1, n_mfcc), dtype=np.float32)
    feat_mean = np.mean(mfccs, axis=0)
    feat_std = np.std(mfccs, axis=0)

    # Aggregazione Log-Mel
    logmels = np.array(logmels)
    if logmels.size == 0:
        logmels = np.zeros((1, cfg.n_mel), dtype=np.float32)
    logmel_mean = np.mean(logmels, axis=0)
    logmel_std = np.std(logmels, axis=0)

    # Energy Features
    rms = es.RMS()(audio)
    zcr = es.ZeroCrossingRate()(audio)
    centroid = es.Centroid()(np.abs(es.Spectrum()(es.Windowing(type='hann')(audio[:min(len(audio), 2048)]))))

    # Concatenazione finale
    return np.concatenate([
        feat_mean, feat_std,
        logmel_mean, logmel_std,
        [rms, zcr, centroid]
    ]).astype(np.float32)


def process_file_for_features(filename: str, source_dir: str, cfg: Config) -> np.ndarray:
    filepath = os.path.join(source_dir, filename)
    y = es.MonoLoader(filename=filepath, sampleRate=cfg.sample_rate)()
    return extract_features(y, cfg=cfg)


def extract_all_features(source_dir: str, cfg: Config) -> np.ndarray:
    # Lista dei file .wav
    file_list = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

    # Worker parzializzato
    worker = partial(process_file_for_features, source_dir=source_dir, cfg=cfg)

    # Parallelizzazione
    with ProcessPoolExecutor() as executor:
        feats = list(executor.map(worker, file_list))

    return np.array(feats)

# Think it's useless
def cluster_norm(y_feat, km, intensity_train_scaled, cfg: Config):
    y, feat = y_feat
    cluster_id = km.predict([feat])[0]
    cluster_members = intensity_train_scaled[km.labels_ == cluster_id]
    cluster_scalar = float(np.mean(cluster_members))
    y_norm = normalize_by_scalar(y, cluster_scalar)
    return extract_features(y_norm, cfg=cfg)
