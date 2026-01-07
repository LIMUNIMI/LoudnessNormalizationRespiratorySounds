import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import welch



def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def compute_basic_stats(y, sr):
    duration = len(y) / sr
    rms = np.sqrt(np.mean(y**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return {
        "duration_sec": duration,
        "rms": rms,
        "zcr": zcr
    }


def compute_spectrum(y, sr, n_fft=2048):
    freqs, psd = welch(y, sr, nperseg=n_fft)
    return freqs, psd


def compute_spectrogram(y, sr, n_fft=1024, hop=256):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db


def plot_spectrogram(S_db, sr, hop, title="Spectrogram", save_path=None, show=True):
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                             x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_dataset(directory, sr=16000):
    stats = []
    durations = []
    rms_vals = []
    zcr_vals = []

    for fname in os.listdir(directory):
        if not fname.endswith(".wav"):
            continue

        path = os.path.join(directory, fname)
        y, sr = load_audio(path, sr)

        s = compute_basic_stats(y, sr)
        stats.append(s)

        durations.append(s["duration_sec"])
        rms_vals.append(s["rms"])
        zcr_vals.append(s["zcr"])

    dataset_stats = {
        "num_files": len(stats),
        "mean_duration": np.mean(durations),
        "std_duration": np.std(durations),
        "min_duration": np.min(durations),
        "max_duration": np.max(durations),
        "mean_rms": np.mean(rms_vals),
        "mean_zcr": np.mean(zcr_vals)
    }

    return dataset_stats, stats


def main():
    dataset_dir = "resampled_data/"
    segment_dir = f'{dataset_dir}segments/'

    print("Analisi del dataset ICBHI 2017...")
    dataset_stats, file_stats = analyze_dataset(dataset_dir)

    print("\n=== GLOBAL STATS ===")
    for k, v in dataset_stats.items():
        print(f"{k}: {v}")

    example_file_crackle = os.path.join(segment_dir, "106_2b1_Pl_mc_LittC2SE.wav_segment0.wav")
    y, sr = load_audio(example_file_crackle)
    S_db_ck = compute_spectrogram(y, sr)
    plot_spectrogram(S_db_ck, sr, hop=64, title=f"Crackle Spectrogram: {example_file_crackle}", save_path="plots/crackle_spectrogram.png", show=False)

    example_file_wheeze = os.path.join(segment_dir, "107_3p2_Ar_mc_AKGC417L.wav_segment0.wav")
    y, sr = load_audio(example_file_wheeze)
    S_db_wh = compute_spectrogram(y, sr)
    plot_spectrogram(S_db_wh, sr, hop=64, title=f"Wheeze Spectrogram: {example_file_wheeze}", save_path="plots/wheeze_spectrogram.png", show=False)

    example_file_normal = os.path.join(segment_dir, "107_2b5_Pl_mc_AKGC417L.wav_segment0.wav")
    y, sr = load_audio(example_file_normal)
    S_db_nm = compute_spectrogram(y, sr)
    plot_spectrogram(S_db_nm, sr, hop=64, title=f"Normal Spectrogram: {example_file_normal}", save_path="plots/normal_spectrogram.png", show=False)


if __name__ == "__main__":
    main()