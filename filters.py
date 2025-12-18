import os
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

def apply_filters(audio: np.ndarray, sample_rate: int, use_hp: bool=False, use_bp=False, hp_cutoff: float=80.0, bp_cutoff: float=80.0, bp_bandwidth: float=40.0) -> np.ndarray:
    y = audio
    if use_hp:
        y = highpass_filter(y, sample_rate, cutoff_frequency=hp_cutoff)

    if use_bp:
        y = bandpass_filter(y, sample_rate, cutoff_frequency=bp_cutoff, bandwidth=bp_bandwidth)
    return y

def filter_file(file_path: str, file_dest: str, sample_rate: int, use_hp: bool=False, hp_cutoff: float=80.0, use_bp: bool=False, bp_cutoff: float=80.0, bp_bandwidth: float=40.0):
    loader = es.MonoLoader(filename=file_path, sampleRate=sample_rate)
    y = loader()
    y_filtered = apply_filters(
        audio=y,
        sample_rate=sample_rate,
        use_hp=use_hp, use_bp=use_bp,
        hp_cutoff=hp_cutoff,
        bp_cutoff=bp_cutoff,
        bp_bandwidth=bp_bandwidth)
    writer = es.MonoWriter(filename=file_dest, sampleRate=sample_rate)
    writer(y_filtered)

def filter_all_files(source_dir: str, output_dir: str, sample_rate: int, use_hp: bool=False, hp_cutoff: float=80.0, use_bp: bool=False, bp_cutoff: float=80.0, bp_bandwidth: float=40.0):
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(source_dir, filename)
            file_dest = os.path.join(output_dir, filename)
            filter_file(
                file_path=filepath,
                file_dest=file_dest,
                sample_rate=sample_rate,
                use_hp=use_hp,
                hp_cutoff=hp_cutoff,
                use_bp=use_bp,
                bp_cutoff=bp_cutoff,
                bp_bandwidth=bp_bandwidth)
            

# ==== Parallel Processing Utilities ====
def process_filter(args):
    import essentia.standard as es
    in_path, out_path, sample_rate, use_hp, hp_cutoff, use_bp, bp_cutoff, bp_bandwidth = args
    y = es.MonoLoader(filename=in_path, sampleRate=sample_rate)()
    cfg = Config()
    y_filt = apply_filters(y, sample_rate=cfg.sample_rate, use_hp=cfg.highpass_toggle, hp_cutoff=cfg.highpass_frequency, use_bp=cfg.bandpass_toggle, bp_cutoff=cfg.bandpass_frequency, bp_bandwidth=cfg.bandpass_bandwidth)
    es.MonoWriter(filename=out_path, sampleRate=sample_rate)(y_filt)

def parallel_filter_all_files(source_dir, output_dir, sample_rate, use_hp, hp_cutoff, use_bp, bp_cutoff, bp_bandwidth):
    file_paths = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    args = [(
        os.path.join(source_dir, f),
        os.path.join(output_dir, f),
        sample_rate,
        use_hp,
        hp_cutoff,
        use_bp,
        bp_cutoff,
        bp_bandwidth
    ) for f in file_paths]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_filter, args))