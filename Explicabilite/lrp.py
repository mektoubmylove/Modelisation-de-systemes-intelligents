def explain_lrp(model, tfidf, le, text, true_label, device, top_k=15, epsilon=1e-9):
    """
    Layer-wise Relevance Propagation (LRP)
    Propriété fondamentale : sum(R_input) = logit (conservation de la relevance)
    Propage la "relevance" depuis la sortie vers l'entrée
    """
    model.eval()

    X = tfidf.transform([text]).toarray()
    X_tensor = torch.FloatTensor(X).to(device)

    feature_names = tfidf.get_feature_names_out()

    # Forward pass SANS gradients (on va tout calculer manuellement)
    with torch.no_grad():
        outputs, activations = model(X_tensor, return_activations=True)

    probs = torch.softmax(outputs, dim=1)
    pred_class = outputs.argmax(1).item()
    pred_label = le.inverse_transform([pred_class])[0]
    confidence = probs[0, pred_class].item()

    print(f"\n{'='*80}")
    print(f"MÉTHODE : LAYER-WISE RELEVANCE PROPAGATION (LRP)")
    print(f"{'='*80}")
    print(f"Texte : {text[:150]}...")
    print(f"Prédiction : {pred_label.upper()} (confiance : {confidence:.1%})")
    print(f"Vérité terrain : {true_label.upper()}")
    if pred_label != true_label:
        print("  ERREUR DE CLASSIFICATION")

    # Extraction des poids et activations
    W1 = model.fc1.weight.data.cpu().numpy()
    b1 = model.fc1.bias.data.cpu().numpy()
    W2 = model.fc2.weight.data.cpu().numpy()
    b2 = model.fc2.bias.data.cpu().numpy()
    W3 = model.fc3.weight.data.cpu().numpy()
    b3 = model.fc3.bias.data.cpu().numpy()

    a0 = X[0]
    a1 = activations['a1'][0].cpu().numpy()
    a2 = activations['a2'][0].cpu().numpy()

    # Recalcul des pré-activations
    z1 = W1 @ a0 + b1
    z2 = W2 @ a1 + b2
    z3 = W3 @ a2 + b3

    # Analyse pour chaque classe
    for target_class_idx, target_class_name in enumerate(le.classes_):
        marker = ""
        if target_class_name == pred_label:
            marker = " ← PRÉDITE"
        elif target_class_name == true_label:
            marker = " ← VRAIE CLASSE"

        print(f"\n{'─'*80}")
        print(f"CLASSE : {target_class_name.upper()}{marker}")
        print(f"Logit : {z3[target_class_idx]:.4f}")
        print(f"{'─'*80}")

        #  LRP : Propagation de la relevance 

        # Initialisation : R_output = logit de la classe cible
        R_output = z3[target_class_idx]

        print(f"\n PROPAGATION LRP (de la sortie vers l'entrée)")
        print(f"Relevance initiale (logit) : {R_output:.4f}")

        #  Couche 2 → Sortie 
        print(f"\n COUCHE 2 → SORTIE")
        R2 = np.zeros_like(a2)

        for j in range(len(a2)):
            # Contribution de ce neurone au logit
            contribution = W3[target_class_idx, j] * a2[j]
            # Normalisation par le logit total
            R2[j] = R_output * (contribution / (z3[target_class_idx] + epsilon))

        print(f"Somme des relevances L2 : {R2.sum():.4f} (doit être ≈ {R_output:.4f})")

        # Top neurones L2
        neurons_l2_sorted = sorted(enumerate(R2), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n Neurones L2 les plus pertinents (top 10) :")
        print(f"   {'Neurone':<10} {'Relevance':<15} {'Activation':<15}")
        print(f"   {'-'*40}")
        for i, rel in neurons_l2_sorted[:10]:
            print(f"   L2[{i:<3}]   {rel:+.6f}       {a2[i]:.4f}")

        #  Couche 1 → Couche 2 
        print(f"\n COUCHE 1 → COUCHE 2")
        R1 = np.zeros_like(a1)

        for i in range(len(a1)):
            for j in range(len(a2)):
                contribution = W2[j, i] * a1[i]
                R1[i] += R2[j] * (contribution / (z2[j] + epsilon))

        print(f"Somme des relevances L1 : {R1.sum():.4f} (doit être ≈ {R_output:.4f})")

        # Top neurones L1
        neurons_l1_sorted = sorted(enumerate(R1), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n Neurones L1 les plus pertinents (top 10) :")
        print(f"   {'Neurone':<10} {'Relevance':<15} {'Activation':<15}")
        print(f"   {'-'*40}")
        for i, rel in neurons_l1_sorted[:10]:
            print(f"   L1[{i:<3}]   {rel:+.6f}       {a1[i]:.4f}")

        #  Entrée → Couche 1 
        print(f"\n ENTRÉE (TF-IDF) → COUCHE 1")
        R0 = np.zeros_like(a0)

        for idx in range(len(a0)):
            for j in range(len(a1)):
                contribution = W1[j, idx] * a0[idx]
                R0[idx] += R1[j] * (contribution / (z1[j] + epsilon))

        print(f"Somme des relevances INPUT : {R0.sum():.4f} (doit être ≈ {R_output:.4f})")
        print(f" VÉRIFICATION : conservation de la relevance = {abs(R0.sum() - R_output) < 0.01}")

        #  Termes les plus pertinents 
        present_indices = np.where(X[0] > 0)[0]

        positive = [(feature_names[i], R0[i], X[0, i])
                   for i in present_indices if R0[i] > 0]
        negative = [(feature_names[i], R0[i], X[0, i])
                   for i in present_indices if R0[i] < 0]

        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        print(f"\n Termes POUSSANT vers '{target_class_name.upper()}' (LRP) :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Relevance':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, rel, tfidf_val) in enumerate(positive[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {rel:+.6f}       {tfidf_val:.4f}")

        print(f"\n Termes ÉLOIGNANT de '{target_class_name.upper()}' (LRP) :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Relevance':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, rel, tfidf_val) in enumerate(negative[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {rel:+.6f}       {tfidf_val:.4f}")

    print(f"\n{'='*80}\n")
    return pred_label, confidence