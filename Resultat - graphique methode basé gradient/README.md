
new_visualizer.ipynb contient les visualisations par chunk avec uniquement les mots surlignés en vert qui poussent vers la classe prédite. 

old_visualisation.ipynb integre les mots qui poussent vers la classe non prédite.

Pour le dossier text_analysis_results_and_balance , dans /csv_data on a pour chaque texte de l'ensemble de test :
Un fichier CSV listant l’ensemble des mots présents, leurs valeurs TF-IDF, leurs contributions par couche (entrée, L1, L2, sortie),leurs contributions séparées pour les classes homme et femme,
Dans /plots il y a les graphiques montrant, pour les mots les plus discriminants, l’évolution des contributions par couche selon la balance homme/femme.


Le dossier explanations_test contient pour chaque texte de l’ensemble de test, les sorties détaillées (contributions par couche, etc.) sous forme de fichiers texte.