import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm

import essentia.standard as es


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline

from config import *
from filters import *
from normalization_utils import *
from features import *
from clustering import *
from classification import *

# ==== Dataset Utilities ====
def load_dataset(audio_dir: str, label_dir: str) -> Tuple[List[str], np.ndarray]:
    label_df = pd.read_csv(label_dir)

    ann_map = {
        row['file_name']: (row['n_wheeze'], row['n_crackle'])
        for _, row in label_df.iterrows()
    }

    files = [] 
    labels = []

    for f in os.listdir(audio_dir):
        if f.endswith(".wav"):
            path = os.path.join(audio_dir, f)
            files.append(path)

            f_base = os.path.splitext(f)[0]

            if f_base in ann_map:

                n_wheeze, n_crackle = ann_map[f_base]

                if int(n_wheeze) > 0 and int(n_crackle) > 0:
                    label = 'both'
                elif int(n_wheeze) > 0:
                    label = 'wheeze'
                elif int(n_crackle) > 0:
                    label = 'crackle'
                else:
                    label = 'none'
            else:
                label = 'none'
        
            labels.append(label)
        
    return files, labels

def load_train_test_split(split_file: str) -> tuple[list[str], list[str]]:
    train_paths, test_paths = [], []
    with open(split_file, "r") as f:
        for line in f
        fname, set_type = line.strip().split()
        if set_type.lower() == "train":
            train_paths.append(fname)
        elif set_type.lower() == "test":
            test_paths.append(fname)
    return train_paths, test_paths

def load_labels_split(audio_dir: str, label_file: str, split_file: str):
    """
    Restituisce solo train_labels e test_labels in base allo split ufficiale ICBHI2017.
    
    Args:
        audio_dir (str): directory con i file audio .wav
        label_file (str): csv con annotazioni (file_name, n_wheeze, n_crackle)
        split_file (str): txt con split ufficiale (file_name train/test)
    
    Returns:
        train_labels (np.ndarray), test_labels (np.ndarray)
    """
    # Carica annotazioni
    label_df = pd.read_csv(label_file)
    ann_map = {
        row['file_name']: (row['n_wheeze'], row['n_crackle'])
        for _, row in label_df.iterrows()
    }

    # Carica split ufficiale
    train_files, test_files = [], []
    with open(split_file, "r") as f:
        for line in f:
            fname, set_type = line.strip().split()
            if set_type.lower() == "train":
                train_files.append(fname)
            elif set_type.lower() == "test":
                test_files.append(fname)

    def get_label(f_base: str):
        if f_base in ann_map:
            n_wheeze, n_crackle = ann_map[f_base]
            if int(n_wheeze) > 0 and int(n_crackle) > 0:
                return "both"
            elif int(n_wheeze) > 0:
                return "wheeze"
            elif int(n_crackle) > 0:
                return "crackle"
            else:
                return "none"
        else:
            return "none"

    # Costruisci train_labels e test_labels
    train_labels = [get_label(os.path.splitext(f)[0]) for f in train_files]
    test_labels  = [get_label(os.path.splitext(f)[0]) for f in test_files]

    return np.array(train_labels), np.array(test_labels)