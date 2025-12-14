import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm

import essentia.standard as es

from sklearn.pipeline import Pipeline as SKPipeline

from config import *
from filters import *
from normalization_utils import *
from features import *
from clustering import *
from classification import *

# ==== Dataset Utilities ====
def load_dataset_micro(audio_dir: str, label_dir: str) -> Tuple[List[str], np.ndarray]:
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

import os
import re
import pandas as pd
import numpy as np
from typing import List, Tuple

def load_dataset_macro(audio_dir: str, diagnosis_file: str) -> Tuple[List[str], np.ndarray]:
    # Carica diagnosi
    diag_df = pd.read_csv(diagnosis_file, sep='\t', header=None, names=['patient_id', 'diagnosis'])
    diag_df['patient_id'] = diag_df['patient_id'].astype(str)
    diag_map = dict(zip(diag_df['patient_id'], diag_df['diagnosis']))

    files = []
    labels = []

    def extract_patient_id(file_name: str) -> str:
        match = re.match(r'(\d{3})', file_name)
        return match.group(1) if match else None

    # Scansiona cartella audio
    for f in os.listdir(audio_dir):
        if f.lower().endswith(".wav"):
            path = os.path.join(audio_dir, f)
            files.append(path)

            f_base = os.path.splitext(f)[0]
            patient_id = extract_patient_id(f_base)

            if patient_id and patient_id in diag_map:
                if diag_map[patient_id] == "Healthy":
                    label = Healty
                else:
                    label = "Unhealty"
            else:
                label = "unknown"

            labels.append(label)

    return files, np.array(labels)

def load_train_test_split(split_file: str, dir: str) -> tuple[list[str], list[str]]:
    train_paths, test_paths = [], []
    with open(split_file, "r") as f:
        for line in f:
            fname, set_type = line.strip().split()
            # Normalizza estensione
            if not fname.lower().endswith(".wav"):
                fname = fname + ".wav"
            else:
                # Se è .WAV maiuscolo, convertilo
                fname = fname[:-4] + ".wav"
            path = os.path.join(dir, fname)

            if set_type.lower() == "train":
                train_paths.append(path)
            elif set_type.lower() == "test":
                test_paths.append(path)
    return train_paths, test_paths

def load_labels_split_micro(audio_dir: str, label_file: str, split_file: str):
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

import os
import re
import pandas as pd
import numpy as np

def load_labels_split_macro(audio_dir: str, diagnosis_file: str, split_file: str):
    # Carica diagnosi
    diag_df = pd.read_csv(diagnosis_file, sep='\t', header=None, names=['patient_id', 'diagnosis'])
    diag_df['patient_id'] = diag_df['patient_id'].astype(str)
    diag_map = dict(zip(diag_df['patient_id'], diag_df['diagnosis']))

    # Carica split ufficiale
    train_files, test_files = [], []
    with open(split_file, "r") as f:
        for line in f:
            fname, set_type = line.strip().split()
            if set_type.lower() == "train":
                train_files.append(fname)
            elif set_type.lower() == "test":
                test_files.append(fname)
    #print(train_files)
    #print(test_files)

    # Funzione per estrarre patient_id dai nomi file
    def extract_patient_id(file_name: str) -> str:
        match = re.match(r'(\d{3})', file_name)
        return match.group(1) if match else None

    def get_label(f_base: str):
        patient_id = extract_patient_id(f_base)
        if patient_id and patient_id in diag_map:
            if diag_map[patient_id] == "Healthy":
                label = "Healty"
            else:
                label = "Unhealty"
        else:
            return "unknown"
        return label

    # Costruisci train_labels e test_labels
    train_labels = [get_label(os.path.splitext(f)[0]) for f in train_files]
    test_labels  = [get_label(os.path.splitext(f)[0]) for f in test_files]
    return np.array(train_labels), np.array(test_labels)