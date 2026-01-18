def explain_integrated_gradients(
    model, tfidf, le, text, true_label, device,
    top_k=15, steps=50, visualize=True
):
    """
    Integrated Gradients : moyenne des gradients le long du chemin baseline → input
    Propriété : sum(attributions) ≈ logit(input) - logit(baseline)
    """
    model.eval()

    X = tfidf.transform([text]).toarray()
    baseline = np.zeros_like(X)

    feature_names = tfidf.get_feature_names_out()

    # Prédiction sur l'input original
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        outputs_original = model(X_tensor)

    probs = torch.softmax(outputs_original, dim=1)
    pred_class = outputs_original.argmax(1).item()
    pred_label = le.inverse_transform([pred_class])[0]
    confidence = probs[0, pred_class].item()

    print(f"\n{'='*80}")
    print(f"MÉTHODE : INTEGRATED GRADIENTS ({steps} étapes)")
    print(f"{'='*80}")
    print(f"Texte : {text[:150]}...")
    print(f"Prédiction : {pred_label.upper()} (confiance : {confidence:.1%})")
    print(f"Vérité terrain : {true_label.upper()}")
    if pred_label != true_label:
        print("  ERREUR DE CLASSIFICATION")

    # Analyse pour chaque classe
    for target_class_idx, target_class_name in enumerate(le.classes_):
        marker = ""
        if target_class_name == pred_label:
            marker = " ← PRÉDITE"
        elif target_class_name == true_label:
            marker = " ← VRAIE CLASSE"

        print(f"\n{'─'*80}")
        print(f"CLASSE : {target_class_name.upper()}{marker}")

        # Logits input / baseline
        with torch.no_grad():
            logit_input = outputs_original[0, target_class_idx].item()
            baseline_tensor = torch.FloatTensor(baseline).to(device)
            outputs_baseline = model(baseline_tensor)
            logit_baseline = outputs_baseline[0, target_class_idx].item()

        print(f"Logit (input) : {logit_input:.4f}")
        print(f"Logit (baseline) : {logit_baseline:.4f}")
        print(f"Différence : {logit_input - logit_baseline:.4f}")
        print(f"{'─'*80}")

        #  Integrated Gradients 
        integrated_grads = np.zeros_like(X[0])

        for step in range(steps):
            alpha = (step + 1) / steps
            X_interpolated = baseline + alpha * (X - baseline)

            X_interp_tensor = torch.FloatTensor(X_interpolated).to(device)
            X_interp_tensor.requires_grad = True

            outputs = model(X_interp_tensor)

            grad = torch.autograd.grad(
                outputs=outputs[0, target_class_idx],
                inputs=X_interp_tensor,
                create_graph=False
            )[0]

            integrated_grads += grad.cpu().numpy().flatten()

        integrated_grads /= steps
        attributions = integrated_grads * (X[0] - baseline[0])

        print(f"Somme des attributions : {attributions.sum():.4f}")
        print(f"Différence logit : {logit_input - logit_baseline:.4f}")
        print(f" VÉRIFICATION : écart = {abs(attributions.sum() - (logit_input - logit_baseline)):.4f}")

        if visualize and target_class_name == pred_label:
            present_indices = np.where(X[0] > 0)[0]
            word_scores = {
                feature_names[i]: attributions[i]
                for i in present_indices
            }

            visualize_text_explanations(
                text=text,
                word_scores=word_scores,
                pred_label=pred_label,
                confidence=confidence,
                true_label=true_label
            )

        #  Termes les plus pertinents 
        present_indices = np.where(X[0] > 0)[0]

        positive = [(feature_names[i], attributions[i], X[0, i])
                   for i in present_indices if attributions[i] > 0]
        negative = [(feature_names[i], attributions[i], X[0, i])
                   for i in present_indices if attributions[i] < 0]

        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        print(f"\n Termes POUSSANT vers '{target_class_name.upper()}' (IG) :")
        for rank, (term, attr, tfidf_val) in enumerate(positive[:top_k], 1):
            print(f"   {rank:<3} {term:<30} {attr:+.6f}  {tfidf_val:.4f}")

        print(f"\n Termes ÉLOIGNANT de '{target_class_name.upper()}' (IG) :")
        for rank, (term, attr, tfidf_val) in enumerate(negative[:top_k], 1):
            print(f"   {rank:<3} {term:<30} {attr:+.6f}  {tfidf_val:.4f}")

    print(f"\n{'='*80}\n")
    return pred_label, confidence
