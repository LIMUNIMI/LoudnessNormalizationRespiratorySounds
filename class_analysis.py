import pandas as pd
import numpy as np
import essentia.standard as es
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def extract_features(audio: np.ndarray, cfg, extractors) -> np.ndarray:
    """
    Usa estrattori pre-inizializzati per massimizzare la velocità.
    """
    # Decomponiamo gli estrattori passati come dizionario
    window = extractors['window']
    spectrum = extractors['spectrum']
    mfcc = extractors['mfcc']
    melbands = extractors['melbands']
    
    # Frame generation (Essentia FrameGenerator è veloce)
    wsize = int(cfg.window_size * cfg.sample_rate)
    hop = int(cfg.hop * cfg.sample_rate)
    frames = es.FrameGenerator(audio, frameSize=wsize, hopSize=hop, startFromZero=True)
    
    mfccs, logmels = [], []
    for frame in frames:
        spec = spectrum(window(frame))
        _, mfcc_coeffs = mfcc(spec)
        mfccs.append(mfcc_coeffs)
        logmels.append(melbands(spec))

    mfccs = np.array(mfccs) if mfccs else np.zeros((1, cfg.n_mfcc))
    logmels = np.array(logmels) if logmels else np.zeros((1, cfg.n_mel))

    # Calcolo statistiche aggregando in vettori
    return np.concatenate([
        np.mean(mfccs, axis=0), np.std(mfccs, axis=0),
        np.mean(logmels, axis=0), np.std(logmels, axis=0),
        [es.RMS()(audio), es.ZeroCrossingRate()(audio), 0.0] # Centroid rimosso per velocità o sostituito
    ]).astype(np.float32)

def process_single_file(filename: str, source_dir: str, cfg) -> dict:
    """
    Worker: Estrae metadati dal nome e feature dall'audio.
    """
    # Inizializziamo gli algoritmi UNA VOLTA per processo (Lazy initialization)
    if not hasattr(process_single_file, "extractors"):
        process_single_file.extractors = {
            'window': es.Windowing(type='hann', size=int(cfg.window_size * cfg.sample_rate)),
            'spectrum': es.Spectrum(),
            'mfcc': es.MFCC(numberCoefficients=cfg.n_mfcc),
            'melbands': es.MelBands(sampleRate=cfg.sample_rate, numberBands=cfg.n_mel, 
                                    lowFrequencyBound=0, highFrequencyBound=cfg.sample_rate/2,
                                    normalize='unit_max', log=True)
        }

    try:
        # Parsing metadati ICBHI
        parts = filename.replace(".wav", "").split("_")
        # 101_1b1_Al_sc_Medusa -> [ID, Rec, Location, Mode, Equipment]
        meta = {
            'filename': filename,
            'chest_location': parts[2] if len(parts) > 2 else "NA",
            'rec_equipment': parts[4] if len(parts) > 4 else "NA"
        }

        filepath = os.path.join(source_dir, filename)
        loader = es.MonoLoader(filename=filepath, sampleRate=cfg.sample_rate)
        audio = loader()
        
        # Estrazione
        feat_vector = extract_features(audio, cfg, process_single_file.extractors)
        
        # Aggiungiamo le feature al dizionario con nomi dinamici
        for i, val in enumerate(feat_vector):
            meta[f'f_{i}'] = val
            
        return meta
    except Exception as e:
        print(f"Errore su {filename}: {e}")
        return None

def extract_all_features_to_df(source_dir: str, cfg) -> pd.DataFrame:
    file_list = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
    
    worker = partial(process_single_file, source_dir=source_dir, cfg=cfg)

    # Il multiprocessing è ottimale per carichi CPU-bound come FFT/MFCC
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker, file_list))

    # Filtriamo eventuali None dovuti a file corrotti e creiamo il DF
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)


def filter_by_chest_location(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Ritorna una copia del dataframe contenente solo la posizione specificata.
    """
    # Usiamo .copy() per evitare il SettingWithCopyWarning nelle operazioni successive
    filtered_df = df[df['chest_location'] == location].copy()
    
    if filtered_df.empty:
        print(f"Attenzione: Nessuna tupla trovata per la location '{location}'")
    
    return filtered_df


def filter_by_equipment(df: pd.DataFrame, equipment: str) -> pd.DataFrame:
    """
    Ritorna una copia del dataframe contenente solo l'equipment specificato.
    """
    filtered_df = df[df['rec_equipment'] == equipment].copy()
    
    if filtered_df.empty:
        print(f"Attenzione: Nessuna tupla trovata per l'equipment '{equipment}'")
    
    return filtered_df



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_feature_distributions(df: pd.DataFrame, class_name: str, save_path: str = None):
    """
    Dashboard completa per l'analisi delle feature in un'unica immagine.
    """
    # 1. Preparazione Dati e Metriche
    features_only = df.filter(regex='f_')
    rms_col = features_only.columns[-3]  # RMS
    zcr_col = features_only.columns[-2]  # Zero Crossing Rate
    
    centroid = features_only.mean()
    dist_from_centroid = np.linalg.norm(features_only - centroid, axis=1).mean()
    total_var = features_only.var().mean()

    # 2. Configurazione Plot (Griglia 3x2 per includere più info)
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle(f'FEATURE ANALYSIS DASHBOARD: {class_name}\n(N. campioni: {len(df)})', 
                 fontsize=22, fontweight='bold', y=0.98)

    # --- PLOT 1: RMS (Energy) ---
    sns.histplot(df[rms_col], kde=True, ax=axes[0, 0], color='teal', line_kws={'linewidth': 3})
    axes[0, 0].set_title('Energy Distribution (RMS)', fontsize=14, fontweight='bold')

    # --- PLOT 2: MFCC Density (First 5) ---
    mfcc_cols = [f'f_{i}' for i in range(5)]
    for col in mfcc_cols:
        sns.kdeplot(df[col], ax=axes[0, 1], label=f'MFCC {col.split("_")[1]}', fill=True, alpha=0.1)
    axes[0, 1].set_title('Spectral Shape: MFCCs Mean (0-4)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='upper right', fontsize='small')

    # --- PLOT 3: Log-Mel Bands ---
    mel_start = 26 # Indice ipotetico basato sul tuo estrattore
    mel_cols = [f'f_{i}' for i in range(mel_start, mel_start + 10)]
    for i, col in enumerate(mel_cols):
        sns.kdeplot(df[col], ax=axes[1, 0], alpha=0.7, lw=1)
    axes[1, 0].set_title('Log-Mel Energy Bands (First 10)', fontsize=14, fontweight='bold')

    # --- PLOT 4: Boxplot Variabilità & Outliers ---
    # Normalizziamo i dati al volo solo per il boxplot per renderli comparabili visivamente
    subset_feat = df[[rms_col, 'f_0', 'f_26', zcr_col]]
    normalized_sub = (subset_feat - subset_feat.mean()) / subset_feat.std()
    sns.boxplot(data=normalized_sub, ax=axes[1, 1], orient='h')
    axes[1, 1].set_title('Feature Spread (Standardized Z-Score)', fontsize=14, fontweight='bold')
    axes[1, 1].set_yticklabels(['RMS', 'MFCC_0', 'Mel_0', 'ZCR'])

    # --- PLOT 5: Zero Crossing Rate ---
    sns.violinplot(x=df[zcr_col], ax=axes[2, 0], color='orange', inner="quart")
    axes[2, 0].set_title('Zero Crossing Rate (Temporal Complexity)', fontsize=14, fontweight='bold')

    # --- AREA 6: METRICHE TESTUALI (Summary Card) ---
    axes[2, 1].axis('off')  # Nascondiamo gli assi per scrivere testo
    stats_text = (
        f"--- CLASS STATISTICS ---\n\n"
        f"• Avg Distance to Centroid: {dist_from_centroid:.4f}\n"
        f"• Mean Total Variance: {total_var:.4f}\n"
        f"• RMS Mean: {df[rms_col].mean():.4f} (±{df[rms_col].std():.4f})\n"
        f"• ZCR Mean: {df[zcr_col].mean():.4f}\n"
        f"• MFCC_0 Mean: {df['f_0'].mean():.4f}\n\n"
        f"Interpretazione: Una distanza dal centroide bassa\n"
        f"indica una classe molto coesa e compatta."
    )
    axes[2, 1].text(0.1, 0.5, stats_text, fontsize=15, 
                    bbox=dict(facecolor='wheat', alpha=0.3, boxstyle='round,pad=1'),
                    verticalalignment='center')

    # Ottimizzazione layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grafico salvato in: {save_path}")
    
    plt.show()




def compare_chestLoc_distribution(dict_dfs: dict, feature_idx: int, save_name: str = "comparison_7_classes.png"):
    """
    Confronta una feature tra 7 classi (es. 'Al', 'Ar', 'Pl', 'Pr', 'Tc', 'Ll', 'Lr').
    dict_dfs: {'NomeClasse': dataframe}
    """
    col = f'f_{feature_idx}'
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="white")
    
    plt.xlim(-1300, -700)
    plt.ylim(0, 0.01)

    # Palette qualitativa per 7 classi
    palette = sns.color_palette("husl", 7)
    
    for (label, df), color in zip(dict_dfs.items(), palette):
        if not df.empty and col in df.columns:
            sns.kdeplot(df[col], label=label, fill=True, color=color, alpha=0.2, bw_adjust=0.8)

    plt.title(f'Distribution Comparison for Chest Location', fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(title="Chest Locations")
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    print(f"Grafico 7 classi salvato come: {save_name}")
    plt.show()




def compare_recEquipment_distribution(dict_dfs: dict, feature_idx: int, save_name: str = "comparison_4_classes.png"):
    """
    Confronta una feature tra 4 classi (es. 'AKGC417L', 'Meditron', 'Littmann', 'Medusa').
    dict_dfs: {'NomeClasse': dataframe}
    """
    col = f'f_{feature_idx}'
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="ticks")
    
    plt.xlim(-1300, -700)
    plt.ylim(0, 0.025)

    # Palette decisa per 4 classi
    palette = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
    
    for (label, df), color in zip(dict_dfs.items(), palette):
        if not df.empty and col in df.columns:
            sns.kdeplot(df[col], label=label, fill=True, color=color, alpha=0.3, linewidth=2.5)

    plt.title(f'Distribution Comparison for Recording Equipment', fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(title="Recording Equipment")
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    print(f"Grafico 4 classi salvato come: {save_name}")
    plt.show()