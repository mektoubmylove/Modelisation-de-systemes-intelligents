import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

from google.colab import drive
drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/datasetSujet3.zip -d /content/datasetSujet3


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

base_dir = '/content/datasetSujet3/content/dataset'

def extract_label_from_name(filename):
    matches = re.findall(r'\(([^)]*)\)', filename)
    if len(matches) >= 4:
        if matches[3] == '1':
            return 'homme'
        elif matches[3] == '2':
            return 'femme'
    return None

def load_split(split_name):
    split_dir = os.path.join(base_dir, split_name)
    data = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.endswith('.txt'):
                label = extract_label_from_name(f)
                if label:
                    data.append({
                        'path': os.path.join(root, f),
                        'label': label
                    })
    return pd.DataFrame(data)

def read_text(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

train_df = load_split('train')
val_df = load_split('val')
test_df = load_split('test')

for df in [train_df, val_df, test_df]:
    df['text'] = df['path'].apply(read_text)


print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("\nDistribution des classes:")
print("Train:\n", train_df['label'].value_counts())
print("Val:\n", val_df['label'].value_counts())
print("Test:\n", test_df['label'].value_counts())

"""
Train: 852 | Val: 284 | Test: 285

Distribution des classes:
Train:
 label
femme    492
homme    360
Name: count, dtype: int64
Val:
 label
femme    164
homme    120
Name: count, dtype: int64
Test:
 label
femme    165
homme    120
Name: count, dtype: int64

"""