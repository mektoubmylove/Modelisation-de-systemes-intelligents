# Classification de textes par sexe – TF-IDF + Réseau de neurones

## Objectif du projet

Ce projet a pour objectif de classifier des textes selon le sexe de leur auteur (homme ou femme) en combinant une représentation TF-IDF des textes et un réseau de neurones.
On étudie, en fonction des couches, les termes rétropropagés les plus discriminants.

- Classifier des textes selon le genre de leur auteur avec une précision élevée
- Identifier les termes rétropropagés correspondant aux poids les plus discriminants
- Analyser l'influence des caractéristiques à travers les différentes couches du réseau

## Architecture

### Modèle de Classification

TF-IDF (40 000 features) → Couche 1 (64 neurones) → Couche 2 (32 neurones) → Sortie (2 classes)


**Composants :**
- **Représentation TF-IDF** : Extraction de 40 000 features avec `min_df=2` et `max_df=0.85`
- **Couche 1** : 64 neurones + BatchNorm + ReLU + Dropout (0.6)
- **Couche 2** : 32 neurones + BatchNorm + ReLU + Dropout (0.6)
- **Couche de sortie** : 2 neurones (logits pour homme/femme)

**Entraînement :**
- Optimiseur : Adam (lr=0.0003, weight_decay=5e-4)
- Loss : CrossEntropyLoss
- Early stopping avec patience de 3 époques
- Validation sur ensemble de validation

---

## Méthodes d'Explicabilité

On test quatre méthodes complémentaires pour analyser les décisions du modèle :

### 1. Gradient × Input

**Principe :** Calcule la contribution de chaque terme en multipliant sa valeur TF-IDF par le gradient de la sortie par rapport à ce terme.

**Formule :** `C_i = x_i × (∂z_c/∂x_i)`

### 2. Poids × Activation

Principe : Propage l'importance depuis la sortie vers l'entrée en utilisant uniquement les poids et activations, sans calcul de gradient.

Formule : importance_terme[i] = Σⱼ importance_L1[j] × fraction(i → j)


### 3. Layer-wise Relevance Propagation (LRP)

Principe : Décompose exactement le logit de sortie en contributions additives des termes d'entrée en propageant la relevance de la sortie vers l'entrée.

Propriété : Σᵢ R₀[i] = logit_c (conservation exacte)


### 4. Integrated Gradients

Principe : Intègre les gradients le long du chemin reliant un texte de référence (vide) au texte actuel.

Propriété : Σᵢ IG_i = logit(texte) - logit(baseline) (complétude)