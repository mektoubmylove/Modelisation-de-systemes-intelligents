from collections import Counter
import re

def tokenize(text):
    
    return re.findall(r"\b\w+\b", text.lower())

global_counter = Counter()

for df in [train_df, val_df, test_df]:
    for text in df['text']:
        global_counter.update(tokenize(text))

# mots apparaissant exactement une seule fois dans tout le dataset
hapax = {word for word, count in global_counter.items() if count == 1}

print("Nombre total de mots uniques (hapax) :", len(hapax))

"""
Nombre total de mots uniques (hapax) : 107643
"""

def count_hapax_in_df(df, hapax_set):
    counter = Counter()
    for text in df['text']:
        tokens = set(tokenize(text)) 
        for token in tokens:
            if token in hapax_set:
                counter[token] += 1
    return len(counter)

hapax_train = count_hapax_in_df(train_df, hapax)
hapax_val   = count_hapax_in_df(val_df, hapax)
hapax_test  = count_hapax_in_df(test_df, hapax)

print("Répartition des mots apparaissant une seule fois :")
print(f"Train : {hapax_train}")
print(f"Val   : {hapax_val}")
print(f"Test  : {hapax_test}")

""""
Répartition des mots apparaissant une seule fois :
Train : 67323
Val   : 18293
Test  : 22027
"""


