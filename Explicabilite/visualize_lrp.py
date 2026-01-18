def explain_lrp(model, tfidf, le, text, true_label, device, top_k=15, epsilon=1e-9, visualize=True):
    """
    Layer-wise Relevance Propagation (LRP)
    Propriété fondamentale : sum(R_input) = logit (conservation de la relevance)
    Propage la "relevance" depuis la sortie vers l'entrée
    """
    model.eval()

    X = tfidf.transform([text]).toarray()
    X_tensor = torch.FloatTensor(X).to(device)

    feature_names = tfidf.get_feature_names_out()

    # Forward pass SANS gradients
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

    # Poids et activations
    W1 = model.fc1.weight.data.cpu().numpy()
    b1 = model.fc1.bias.data.cpu().numpy()
    W2 = model.fc2.weight.data.cpu().numpy()
    b2 = model.fc2.bias.data.cpu().numpy()
    W3 = model.fc3.weight.data.cpu().numpy()
    b3 = model.fc3.bias.data.cpu().numpy()

    a0 = X[0]
    a1 = activations['a1'][0].cpu().numpy()
    a2 = activations['a2'][0].cpu().numpy()

    # Pré-activations
    z1 = W1 @ a0 + b1
    z2 = W2 @ a1 + b2
    z3 = W3 @ a2 + b3

    # Analyse par classe
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

        R_output = z3[target_class_idx]

        # Sortie → L2
        R2 = np.zeros_like(a2)
        for j in range(len(a2)):
            contribution = W3[target_class_idx, j] * a2[j]
            R2[j] = R_output * (contribution / (z3[target_class_idx] + epsilon))

        # L2 → L1
        R1 = np.zeros_like(a1)
        for i in range(len(a1)):
            for j in range(len(a2)):
                contribution = W2[j, i] * a1[i]
                R1[i] += R2[j] * (contribution / (z2[j] + epsilon))

        # L1 → Entrée
        R0 = np.zeros_like(a0)
        for idx in range(len(a0)):
            for j in range(len(a1)):
                contribution = W1[j, idx] * a0[idx]
                R0[idx] += R1[j] * (contribution / (z1[j] + epsilon))

        print(f"Somme des relevances INPUT : {R0.sum():.4f} (doit être ≈ {R_output:.4f})")
        print(f" VÉRIFICATION : conservation = {abs(R0.sum() - R_output) < 0.01}")

        if visualize and target_class_name == pred_label:
            present_indices = np.where(X[0] > 0)[0]
            word_scores = {
                feature_names[i]: R0[i]
                for i in present_indices
            }

            visualize_text_explanations(
                text=text,
                word_scores=word_scores,
                pred_label=pred_label,
                confidence=confidence,
                true_label=true_label
            )

        # Impression des termes
        present_indices = np.where(X[0] > 0)[0]
        positive = [(feature_names[i], R0[i], X[0, i]) for i in present_indices if R0[i] > 0]
        negative = [(feature_names[i], R0[i], X[0, i]) for i in present_indices if R0[i] < 0]

        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        print(f"\n Termes POUSSANT vers '{target_class_name.upper()}' (LRP) :")
        for rank, (term, rel, tfidf_val) in enumerate(positive[:top_k], 1):
            print(f"   {rank:<3} {term:<30} {rel:+.6f}  {tfidf_val:.4f}")

        print(f"\n Termes ÉLOIGNANT de '{target_class_name.upper()}' (LRP) :")
        for rank, (term, rel, tfidf_val) in enumerate(negative[:top_k], 1):
            print(f"   {rank:<3} {term:<30} {rel:+.6f}  {tfidf_val:.4f}")

    return pred_label, confidence
