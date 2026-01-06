import pandas as pd
import glob, os
from sklearn.model_selection import train_test_split
import shutil


from google.colab import drive
drive.mount('/content/drive')

# Décompresser le zip dans /content/textes
!unzip -q "/content/drive/MyDrive/textes.zip" -d "/content"

# Vérifier la structure obtenue
!ls -R /content/textes



# Dossier racine
extract_dir = '/content/textes'

# Charger les fichiers
data = []
for label in ['femme', 'homme']:
    files = glob.glob(os.path.join(extract_dir, label, '*.txt'))
    for f in files:
        data.append({'path': f, 'label': label})

df = pd.DataFrame(data)
print("Total:", len(df))
print(df['label'].value_counts())

"""Total: 1422
label
femme    822
homme    600
Name: count, dtype: int64
"""

# Train (60%) / temp (40%)
train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df['label'], random_state=42
)

# Validation (20%) / Test (20%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
print("\nProportions dans chaque split :")
print("Train\n", train_df['label'].value_counts(normalize=True))
print("Val\n", val_df['label'].value_counts(normalize=True))
print("Test\n", test_df['label'].value_counts(normalize=True))


"""
Train: 853 Val: 284 Test: 285

Proportions dans chaque split :
Train
 label
femme    0.57796
homme    0.42204
Name: proportion, dtype: float64
Val
 label
femme    0.577465
homme    0.422535
Name: proportion, dtype: float64
Test
 label
femme    0.578947
homme    0.421053
Name: proportion, dtype: float64
"""



base_dir = '/content/dataset'

# Création des dossiers train, val, test
for split_name, split_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
    split_path = os.path.join(base_dir, split_name)
    os.makedirs(split_path, exist_ok=True)

    # Copier chaque fichier tel quel
    for src in split_df['path']:
        dst = os.path.join(split_path, os.path.basename(src))
        shutil.copy(src, dst)

    print(f"{split_name} -> {len(split_df)} fichiers copiés dans {split_path}")

"""
train -> 853 fichiers copiés dans /content/dataset/train
val -> 284 fichiers copiés dans /content/dataset/val
test -> 285 fichiers copiés dans /content/dataset/test
"""

!zip -r /content/dataset.zip /content/dataset
from google.colab import files
files.download('/content/dataset.zip')
