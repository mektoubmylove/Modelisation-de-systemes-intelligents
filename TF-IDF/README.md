# Fonctionnement de TF-IDF

La méthode TF-IDF (Term Frequency – Inverse Document Frequency) permet de transformer un texte en un vecteur numérique interprétable où chaque dimension correspond à un mot du vocabulaire.

## Term Frequency (TF)

La fréquence de terme mesure l’importance d’un mot à l’intérieur d’un document.
Un mot apparaissant fréquemment dans un texte est considéré comme représentatif de son contenu.

## Inverse Document Frequency (IDF)

L’IDF pénalise les mots apparaissant dans un grand nombre de documents du corpus.
Les mots très fréquents dans l’ensemble des textes ont un faible pouvoir discriminant et reçoivent donc un poids réduit.

## Pondération TF-IDF

Le poids final d’un mot est le produit de TF et IDF :

élevé si le mot est fréquent dans un document,

faible s’il est commun à la majorité des textes.

# Paramètres TF-IDF utilisés

max_features = 40 000: taille maximale du vocabulaire.

min_df = 2 : élimine les hapax.

max_df = 0.85 : supprime les mots trop fréquents (peu discriminants).

ngram_range = (1,1) : on ne prend en compte que les unigrammes.