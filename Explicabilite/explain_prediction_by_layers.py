def explain_prediction_by_layers(model, tfidf, le, text, true_label, device, top_k=15, visualize=True):
    """
    analyse l'influence des termes sur la prédiction à travers chaque couche du réseau.
    rétropropagation pour identifier les poids les plus discriminants.
    """

    model.eval()

    # vectorisation du texte avec le vectoriseur TF-IDF entraîné
    X = tfidf.transform([text]).toarray()
    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor.requires_grad = True

    feature_names = tfidf.get_feature_names_out()  # récupère la liste des mots du vocabulaire TF-IDF


    # passe avant, récupère les sorties ET les activations des couches cachées
    outputs, activations = model(X_tensor, return_activations=True)


    probs = torch.softmax(outputs.detach(), dim=1)  # convertit les logits en probabilités avec softmax
    pred_class = outputs.argmax(1).item()  # trouve l'indice de la classe avec la plus haute probabilité
    pred_label = le.inverse_transform([pred_class])[0]  # convertit l'indice numérique en label textuel (ex: 0 -> 'homme')
    confidence = probs[0, pred_class].item()  # récupère la valeur de confiance (probabilité) pour la classe prédite



    print(f"texte analysé: {text[:200]}...")
    print(f"prédiction: {pred_label.upper()} (confiance: {confidence:.1%})")
    print(f"vérité terrain: {true_label.upper()}")
    if pred_label != true_label:
        print("  erreur de classification")


    if visualize:
        # calcule les gradients de la sortie par rapport à l'entrée et aux activations
        gradients_tuple = torch.autograd.grad(
            outputs=outputs[0, pred_class],  # on prend le gradient pour la classe prédite
            inputs=[X_tensor, activations['a1'], activations['a2']],  # par rapport à l'entrée et aux couches cachées
            retain_graph=True  # conserve le graphe de calcul pour les prochaines étapes
        )

        grad_input = gradients_tuple[0].cpu().numpy().flatten()  # gradient de la sortie par rapport à l'entrée TF-IDF
        contributions = grad_input * X[0]  # contribution de chaque terme = gradient × valeur TF-IDF

        # identification des mots  présents dans le texte (valeur TF-IDF > 0)
        present_indices = np.where(X[0] > 0)[0]
        # création d'un dictionnaire {mot: contribution}
        word_scores = {feature_names[i]: contributions[i] for i in present_indices}

        # appelle la fonction pour la visualisation du chink avec les mots colores
        visualize_text_explanations(
            text=text,
            word_scores=word_scores,
            pred_label=pred_label,
            confidence=confidence,
            true_label=true_label
        )


    # on boucle sur les deux classes (homme et femme)
    for target_class_idx, target_class_name in enumerate(le.classes_):
        marker = ""  # marqueur pour indiquer si c'est la classe prédite ou la vraie classe
        if target_class_name == pred_label:
            marker = "  prédite"
        elif target_class_name == true_label:
            marker = "  vraie classe"


        print(f"# analyse pour la classe: {target_class_name.upper()}{marker}")
        print(f"# logit: {outputs[0, target_class_idx].item():.4f}")  # valeur du logit (avant softmax) pour cette classe


        # calcul des gradients pour la classe courante
        gradients_tuple = torch.autograd.grad(
            outputs=outputs[0, target_class_idx],  # gradient pour la classe cible
            inputs=[X_tensor, activations['a1'], activations['a2']],
            retain_graph=True
        )

        # extraction des gradients pour chaque couche
        grad_input = gradients_tuple[0].cpu().numpy().flatten()  # gradient par rapport à l'entrée
        grad_a1 = gradients_tuple[1].cpu().numpy()[0]  # gradient par rapport aux activations de la couche 1
        grad_a2 = gradients_tuple[2].cpu().numpy()[0]  # gradient par rapport aux activations de la couche 2

        # extraction des activations (valeurs après application de la fonction d'activation)
        a1 = activations['a1'][0].detach().cpu().numpy()  # activations de la couche 1 (64 neurones)
        a2 = activations['a2'][0].detach().cpu().numpy()  # activations de la couche 2 (32 neurones)


        print("chain rule : ∂sortie/∂terme = Σ ∂sortie/∂L2 × ∂L2/∂L1 × ∂L1/∂terme")
        print("ordre de la rétropropagation : sortie → L2 → L1 → terme")


        # contrib finale via tt le reseau
        print(f" couche de sortie → termes via tout le réseau")
        print(f"calcul : ∂sortie/∂terme = Σ [∂sortie/∂L2 × ∂L2/∂L1 × ∂L1/∂terme]")


        # extraction des matrices de poids du modèle
        W1 = model.fc1.weight.data.cpu().numpy()  # poids de la couche 1: [64 neurones × vocab_size]
        W2 = model.fc2.weight.data.cpu().numpy()  # poids de la couche 2: [32 neurones × 64 neurones]
        W3 = model.fc3.weight.data.cpu().numpy()  # poids de la couche de sortie: [2 classes × 32 neurones]
        class_weights = W3[target_class_idx]  # poids spécifiques à la classe cible (connexions L2 → sortie)

        # calcul de la contribution de chaque terme via toute l'architecture: terme → L1 → L2 → sortie
        term_contrib_output = np.zeros(len(feature_names))  # initialisation du vecteur de contributions
        for l2_idx in range(len(a2)):  # pour chaque neurone de la couche L2 (32 neurones)
            l2_weight_to_output = class_weights[l2_idx]  # poids de connexion L2[l2_idx] → sortie[classe]
            l2_weights_from_l1 = W2[l2_idx]  # poids des connexions L1 → L2[l2_idx]
            for l1_idx in range(len(a1)):  # pour chaque neurone de la couche L1 (64 neurones)
                l1_weights = W1[l1_idx]  # poids des connexions termes → L1[l1_idx]
                # contribution totale via ce chemin: terme → L1[l1_idx] → L2[l2_idx] → sortie
                term_contrib_output += l1_weights * X[0] * grad_a1[l1_idx] * l2_weights_from_l1[l1_idx] * l2_weight_to_output

        # séparation des termes en deux groupes: ceux qui poussent vers la classe et ceux qui l'éloignent
        terms_pushing_output = [(feature_names[i], term_contrib_output[i], X[0, i])
                                for i in range(len(term_contrib_output))
                                if term_contrib_output[i] > 0 and X[0, i] > 0]  # contributions positives ET terme présent
        terms_pulling_output = [(feature_names[i], term_contrib_output[i], X[0, i])
                                for i in range(len(term_contrib_output))
                                if term_contrib_output[i] < 0 and X[0, i] > 0]  # contributions négatives ET terme présent

        # tri des termes par contribution absolue (du plus influent au moins influent)
        terms_pushing_output.sort(key=lambda x: x[1], reverse=True)  # termes positifs triés par valeur décroissante
        terms_pulling_output.sort(key=lambda x: x[1])  # termes négatifs triés par valeur croissante (plus négatif d'abord)

        print(f"\n termes poussant vers '{target_class_name.upper()}' (contribution finale):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(terms_pushing_output[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")

        print(f"\n termes éloignant de '{target_class_name.upper()}' (contribution finale):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(terms_pulling_output[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")

        # contrib via la couche l2
        print(f"couche 2 (32 neurones) → influence via L1 → L2")
        print(f"calcul : ∂sortie/∂terme via L2 = Σ [∂sortie/∂L2 × ∂L2/∂L1 × ∂L1/∂terme]")


        # calcul de la contribution de chaque terme via les couches 1 et 2 seulement (sans la sortie)
        term_contrib_l2 = np.zeros(len(feature_names))
        for l2_idx in range(len(a2)):
            l2_weights = W2[l2_idx]  # poids L1 → L2[l2_idx]
            l2_grad = grad_a2[l2_idx]  # gradient de la sortie par rapport à L2[l2_idx]
            for l1_idx in range(len(a1)):
                l1_weights = W1[l1_idx]  # poids termes → L1[l1_idx]
                # contribution via ce chemin: terme → L1[l1_idx] → L2[l2_idx]
                term_contrib_l2 += l1_weights * X[0] * grad_a1[l1_idx] * l2_weights[l1_idx] * l2_grad

        # séparation et tri des termes pour l'étape 2
        terms_pushing_l2 = [(feature_names[i], term_contrib_l2[i], X[0, i])
                            for i in range(len(term_contrib_l2))
                            if term_contrib_l2[i] > 0 and X[0, i] > 0]
        terms_pulling_l2 = [(feature_names[i], term_contrib_l2[i], X[0, i])
                            for i in range(len(term_contrib_l2))
                            if term_contrib_l2[i] < 0 and X[0, i] > 0]

        terms_pushing_l2.sort(key=lambda x: x[1], reverse=True)
        terms_pulling_l2.sort(key=lambda x: x[1])

        print(f"\n termes poussant vers '{target_class_name.upper()}' (via L1 → L2):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(terms_pushing_l2[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")

        print(f"\n termes éloignant de '{target_class_name.upper()}' (via L1 → L2):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(terms_pulling_l2[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")


        print(f"couche 1 (64 neurones) → influence via L1 seulement")
        print(f"calcul : ∂sortie/∂terme via L1 = Σ [∂sortie/∂L1 × ∂L1/∂terme]")


        # calcul de la contribution de chaque terme via la couche 1 seulement
        term_contrib_l1 = np.zeros(len(feature_names))
        for neuron_idx in range(len(a1)):
            neuron_weights = W1[neuron_idx]  # poids termes → L1[neuron_idx]
            neuron_grad = grad_a1[neuron_idx]  # gradient de la sortie par rapport à L1[neuron_idx]
            term_contrib_l1 += neuron_weights * X[0] * neuron_grad

        # séparation et tri des termes pour l'étape 3
        terms_pushing_l1 = [(feature_names[i], term_contrib_l1[i], X[0, i])
                            for i in range(len(term_contrib_l1))
                            if term_contrib_l1[i] > 0 and X[0, i] > 0]
        terms_pulling_l1 = [(feature_names[i], term_contrib_l1[i], X[0, i])
                            for i in range(len(term_contrib_l1))
                            if term_contrib_l1[i] < 0 and X[0, i] > 0]

        terms_pushing_l1.sort(key=lambda x: x[1], reverse=True)
        terms_pulling_l1.sort(key=lambda x: x[1])

        print(f"\n termes poussant vers '{target_class_name.upper()}' (via L1):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(terms_pushing_l1[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")

        print(f"\n termes éloignant de '{target_class_name.upper()}' (via L1):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(terms_pulling_l1[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")


        print(f" couche d'entrée (tf-idf) → gradient direct")
        print(f"calcul : ∂sortie/∂terme = gradient_input × valeur_tf-idf")


        # calcul de la contribution directe: gradient × valeur TF-IDF
        input_contribution = grad_input * X[0]

        # séparation des termes pour l'étape 4
        positive_contrib = [(i, feature_names[i], input_contribution[i], X[0, i])
                           for i in range(len(input_contribution))
                           if input_contribution[i] > 0 and X[0, i] > 0]
        negative_contrib = [(i, feature_names[i], input_contribution[i], X[0, i])
                           for i in range(len(input_contribution))
                           if input_contribution[i] < 0 and X[0, i] > 0]

        positive_contrib.sort(key=lambda x: x[2], reverse=True)
        negative_contrib.sort(key=lambda x: x[2])

        print(f"\n termes poussant vers '{target_class_name.upper()}' (gradient direct):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (idx, term, contrib, tfidf_val) in enumerate(positive_contrib[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")

        print(f"\n termes éloignant de '{target_class_name.upper()}' (gradient direct):")
        print(f"   {'rang':<6} {'terme':<30} {'contribution':<15} {'tf-idf':<10}")
        print(f"   {'-'*65}")
        for rank, (idx, term, contrib, tfidf_val) in enumerate(negative_contrib[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")


        print(f"synthèse : chemin complet de la rétropropagation")
        print(f"pour chaque terme : ∂sortie/∂terme = Σ [∂sortie/∂L2 × ∂L2/∂L1 × ∂L1/∂terme]")


        # affichage de l'évolution de la contribution des termes à travers les couches
        print(f"\nexemples d'évolution de contribution (top 5 termes):")
        print(f"{'terme':<25} {'via L1':<15} {'via L1→L2':<15} {'final':<15} {'tf-idf':<10}")
        print(f"{'-'*80}")

        # prend les 20 termes les plus influents selon l'analyse finale
        top_terms_final = [term for term, _, _ in terms_pushing_output[:20]]
        for term in top_terms_final:
            # recherche des contributions de ce terme dans chaque couche
            contrib_l1 = next((c for t, c, _ in terms_pushing_l1 if t == term), 0.0)
            contrib_l2 = next((c for t, c, _ in terms_pushing_l2 if t == term), 0.0)
            contrib_final = next((c for t, c, _ in terms_pushing_output if t == term), 0.0)
            tfidf_val = next((v for t, _, v in terms_pushing_output if t == term), 0.0)

            print(f"{term:<25} {contrib_l1:+.6f}     {contrib_l2:+.6f}     {contrib_final:+.6f}     {tfidf_val:.4f}")


        print(f"logit final pour '{target_class_name}': {outputs[0, target_class_idx].item():.4f}")



    return pred_label, confidence