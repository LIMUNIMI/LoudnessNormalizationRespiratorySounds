import os
import numpy as np
import essentia.standard as es
import concurrent.futures


# ==== Duration Normalization Utilities ====
def normalize_duration(audio: np.ndarray, sample_rate: int, target_length: float) -> np.ndarray:
    target_samples = int(target_length * sample_rate)
    if audio.shape[0] == target_samples:
        return audio
    elif audio.shape[0] < target_samples:
        return np.pad(audio, (0, target_samples - audio.shape[0]), mode='constant')
    else:
        return audio[:target_samples]
    

def normalize_segment_duration(file_path: str, file_dest: str, sample_rate: int, target_length: int):
    target_samples = target_length * sample_rate
    loader = es.MonoLoader(filename=file_path, sampleRate=sample_rate)
    y = loader()
    y_norm = normalize_duration(y, sample_rate, target_length)
    writer = es.MonoWriter(filename=file_dest, sampleRate=sample_rate)
    writer(y_norm)


def normalize_all_segments_duration(source_dir: str, output_dir: str, sample_rate: int, target_length: int):
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(source_dir, filename)
            file_dest = os.path.join(output_dir, filename)
            normalize_segment_duration(filepath, file_dest, sample_rate, target_length)



# ==== Loudness Normalization Utilities ====
def normalize_by_rms(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = es.RMS()(audio)
    return audio / (rms + eps)

def normalize_by_median(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    median = np.median(np.abs(audio))
    return audio / (median + eps)

def normalize_by_scalar(audio: np.ndarray, scalar: float, eps: float = 1e-8) -> np.ndarray:
    return audio / (scalar + eps)

def normalize_by_cluster(audio: np.ndarray, cluster_scalar: float, eps: float = 1e-8) -> np.ndarray:
    return audio / (cluster_scalar + eps)



# ==== File-level Normalization Utilities ====
def normalize_amplitude(file_path: str, file_dest: str, sample_rate: int, method: str = 'rms', loader: str = 'mono', scalar: float = 1.0):
    if loader == 'mono':
        loader = es.MonoLoader(filename=file_path, sampleRate=sample_rate)
    elif loader == 'eqloud':
        loader = es.EqloudLoader(filename=file_path, sampleRate=sample_rate)
    else:
        raise ValueError(f"Undefined loader type: {loader}")

    y = loader()
    if method == 'rms':
        y_norm = normalize_by_rms(y)
    elif method == 'median':
        y_norm = normalize_by_median(y)
    else:
        raise ValueError(f"Undefined normalization method: {method}")
    writer = es.MonoWriter(filename=file_dest, sampleRate=sample_rate)
    writer(y_norm)


def normalize_all_files(source_dir: str, output_dir: str, sample_rate:int, method: str = 'rms', loader: str = 'mono', scalar: float = 1.0):
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(source_dir, filename)
            file_dest = os.path.join(output_dir, filename)
            normalize_amplitude(filepath, file_dest, sample_rate, method, loader, scalar)


def normalize_all_files_cluster(
    source_dir: str,
    output_dir: str,
    sample_rate: int,
    km,
    scaled_features,
    loader: str = 'mono'
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.wav')]
    for p, feat in zip(file_paths, scaled_features):
        if loader == 'mono':
            y = es.MonoLoader(filename=p, sampleRate=sample_rate)()
        elif loader == 'eqloud':
            y = es.EqloudLoader(filename=p, sampleRate=sample_rate)()
        else:
            raise ValueError(f"Loader non supportato: {loader}")
        feat = np.asarray(feat, dtype=np.float64)
        cluster_id = km.predict([feat])[0]
        cluster_members = scaled_features[km.labels_ == cluster_id]
        cluster_scalar = float(np.mean(cluster_members))
        y_norm = normalize_by_scalar(y, cluster_scalar)
        out_path = os.path.join(output_dir, os.path.basename(p))
        es.MonoWriter(filename=out_path, sampleRate=sample_rate)(y_norm)



# ==== Parallel Processing Utilities ====
def process_amplitude(args):
    return normalize_amplitude(*args)


def parallel_normalize_all_files(*, source_dir, output_dir, sample_rate, method, loader):
    file_paths = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    args = [(os.path.join(source_dir, f), os.path.join(output_dir, f), sample_rate, method, loader) for f in file_paths]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_amplitude, args))


def process_duration(args):
    from normalization_utils import normalize_segment_duration
    return normalize_segment_duration(*args)


def parallel_normalize_all_segments_duration(source_dir, output_dir, sample_rate, target_length):
    file_paths = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    args = [(os.path.join(source_dir, f), os.path.join(output_dir, f), sample_rate, target_length) for f in file_paths]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_duration, args))


def process_cluster_normalization(args):
    p, feat, sample_rate, km, scaled_features, loader, output_dir = args
    if loader == 'mono':
        y = es.MonoLoader(filename=p, sampleRate=sample_rate)()
    elif loader == 'eqloud':
        y = es.EqloudLoader(filename=p, sampleRate=sample_rate)()
    else:
        raise ValueError(f"Loader non supportato: {loader}")
    feat = np.asarray(feat, dtype=np.float64)
    cluster_id = km.predict([feat])[0]
    cluster_members = scaled_features[km.labels_ == cluster_id]
    cluster_scalar = float(np.mean(cluster_members))
    y_norm = normalize_by_scalar(y, cluster_scalar)
    out_path = os.path.join(output_dir, os.path.basename(p))
    es.MonoWriter(filename=out_path, sampleRate=sample_rate)(y_norm)

def parallel_normalize_all_files_cluster(*, source_dir, output_dir, sample_rate, km, scaled_features, loader):
    file_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.wav')]
    args = [(p, feat, sample_rate, km, scaled_features, loader, output_dir) for p, feat in zip(file_paths, scaled_features)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_cluster_normalization, args))