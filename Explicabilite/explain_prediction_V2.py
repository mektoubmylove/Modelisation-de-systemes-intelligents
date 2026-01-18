def explain_gradient_input(model, tfidf, le, text, true_label, device, top_k=15):
    """
    Méthode basée uniquement sur les gradients : contribution = gradient × input
    Cette méthode est simple, rapide et théoriquement fondée.
    ATTENTION : sum(contributions) ≠ logit
    """
    model.eval()

    X = tfidf.transform([text]).toarray()
    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor.requires_grad = True

    feature_names = tfidf.get_feature_names_out()

    # Forward pass avec activations
    outputs, activations = model(X_tensor, return_activations=True)

    probs = torch.softmax(outputs.detach(), dim=1)
    pred_class = outputs.argmax(1).item()
    pred_label = le.inverse_transform([pred_class])[0]
    confidence = probs[0, pred_class].item()

    print(f"\n{'='*80}")
    print(f"MÉTHODE : GRADIENT × INPUT")
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
        print(f"Logit : {outputs[0, target_class_idx].item():.4f}")
        print(f"{'─'*80}")

        # Calcul des gradients pour chaque couche
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()

        gradients_tuple = torch.autograd.grad(
            outputs=outputs[0, target_class_idx],
            inputs=[X_tensor, activations['a1'], activations['a2']],
            retain_graph=True,
            create_graph=False
        )

        grad_input = gradients_tuple[0].cpu().numpy().flatten()
        grad_a1 = gradients_tuple[1].cpu().numpy()[0]
        grad_a2 = gradients_tuple[2].cpu().numpy()[0]

        a1 = activations['a1'][0].detach().cpu().numpy()
        a2 = activations['a2'][0].detach().cpu().numpy()

        #  GRADIENT D'ENTRÉE (couche TF-IDF) 
        print(f"\n GRADIENT D'ENTRÉE (TF-IDF)")
        print(f"Formule : contribution = ∂logit/∂terme × valeur_tfidf")

        input_contrib = grad_input * X[0]
        present_indices = np.where(X[0] > 0)[0]

        positive = [(feature_names[i], input_contrib[i], X[0, i])
                   for i in present_indices if input_contrib[i] > 0]
        negative = [(feature_names[i], input_contrib[i], X[0, i])
                   for i in present_indices if input_contrib[i] < 0]

        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        print(f"\n Termes POUSSANT vers '{target_class_name.upper()}' :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Contribution':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(positive[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")

        print(f"\n Termes ÉLOIGNANT de '{target_class_name.upper()}' :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Contribution':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, contrib, tfidf_val) in enumerate(negative[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {contrib:+.6f}       {tfidf_val:.4f}")

        # GRADIENT COUCHE 1 (64 neurones)
        print(f"\nGRADIENT COUCHE 1 (64 neurones)")
        print(f"Formule : ∂logit/∂a1[i] représente l'influence du neurone i")

        neurons_l1 = [(i, grad_a1[i], a1[i]) for i in range(len(a1))]
        neurons_l1_pos = [(i, g, a) for i, g, a in neurons_l1 if g > 0]
        neurons_l1_neg = [(i, g, a) for i, g, a in neurons_l1 if g < 0]

        neurons_l1_pos.sort(key=lambda x: x[1], reverse=True)
        neurons_l1_neg.sort(key=lambda x: x[1])

        print(f"\n Neurones POUSSANT vers '{target_class_name.upper()}' (top {min(10, len(neurons_l1_pos))}):")
        print(f"   {'Neurone':<10} {'Gradient':<15} {'Activation':<15}")
        print(f"   {'-'*40}")
        for i, grad, activ in neurons_l1_pos[:10]:
            print(f"   L1[{i:<3}]   {grad:+.6f}       {activ:.4f}")

        print(f"\n Neurones ÉLOIGNANT de '{target_class_name.upper()}' (top {min(10, len(neurons_l1_neg))}):")
        print(f"   {'Neurone':<10} {'Gradient':<15} {'Activation':<15}")
        print(f"   {'-'*40}")
        for i, grad, activ in neurons_l1_neg[:10]:
            print(f"   L1[{i:<3}]   {grad:+.6f}       {activ:.4f}")

        # Termes les plus influents via L1
        print(f"\n Termes influents via COUCHE 1 (via poids × gradient):")
        W1 = model.fc1.weight.data.cpu().numpy()

        term_influence_l1 = np.zeros(len(feature_names))
        for neuron_idx in range(len(a1)):
            neuron_weights = W1[neuron_idx]
            neuron_grad = grad_a1[neuron_idx]
            term_influence_l1 += neuron_weights * X[0] * neuron_grad

        terms_l1_pos = [(feature_names[i], term_influence_l1[i], X[0, i])
                       for i in present_indices if term_influence_l1[i] > 0]
        terms_l1_neg = [(feature_names[i], term_influence_l1[i], X[0, i])
                       for i in present_indices if term_influence_l1[i] < 0]

        terms_l1_pos.sort(key=lambda x: x[1], reverse=True)
        terms_l1_neg.sort(key=lambda x: x[1])

        print(f"\n Top termes (influence via L1) :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Influence':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, infl, tfidf_val) in enumerate(terms_l1_pos[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {infl:+.6f}       {tfidf_val:.4f}")

        #  GRADIENT COUCHE 2 (32 neurones) 
        print(f"\n GRADIENT COUCHE 2 (32 neurones)")

        neurons_l2 = [(i, grad_a2[i], a2[i]) for i in range(len(a2))]
        neurons_l2_pos = [(i, g, a) for i, g, a in neurons_l2 if g > 0]
        neurons_l2_neg = [(i, g, a) for i, g, a in neurons_l2 if g < 0]

        neurons_l2_pos.sort(key=lambda x: x[1], reverse=True)
        neurons_l2_neg.sort(key=lambda x: x[1])

        print(f"\n Neurones POUSSANT vers '{target_class_name.upper()}' (top {min(10, len(neurons_l2_pos))}):")
        print(f"   {'Neurone':<10} {'Gradient':<15} {'Activation':<15}")
        print(f"   {'-'*40}")
        for i, grad, activ in neurons_l2_pos[:10]:
            print(f"   L2[{i:<3}]   {grad:+.6f}       {activ:.4f}")

        # Termes via L1→L2
        print(f"\n Termes influents via COUCHES 1→2 :")
        W2 = model.fc2.weight.data.cpu().numpy()

        term_influence_l2 = np.zeros(len(feature_names))
        for l2_idx in range(len(a2)):
            l2_weights = W2[l2_idx]
            l2_grad = grad_a2[l2_idx]
            for l1_idx in range(len(a1)):
                l1_weights = W1[l1_idx]
                term_influence_l2 += l1_weights * X[0] * grad_a1[l1_idx] * l2_weights[l1_idx] * l2_grad

        terms_l2_pos = [(feature_names[i], term_influence_l2[i], X[0, i])
                       for i in present_indices if term_influence_l2[i] > 0]

        terms_l2_pos.sort(key=lambda x: x[1], reverse=True)

        print(f"\n Top termes (influence via L1→L2) :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Influence':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, infl, tfidf_val) in enumerate(terms_l2_pos[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {infl:+.6f}       {tfidf_val:.4f}")

    print(f"\n{'='*80}\n")
    return pred_label, confidence
