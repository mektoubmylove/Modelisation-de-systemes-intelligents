import numpy as np
import torch
import re
from IPython.display import display, HTML


def render_token(token, word_scores, max_abs_score, pred_label):
    lower_token = token.lower()
    if lower_token in word_scores:
        score = word_scores[lower_token]
        alpha = 0.2 + (abs(score) / max_abs_score) * 0.7

        if score > 0:
            color = f"rgba(0, 128, 0, {alpha:.2f})"
        else:
            color = f"rgba(178, 34, 34, {alpha:.2f})"

        return (
            f"<span style='background-color:{color}; color:white; padding:1px 3px; "
            f"border-radius:4px; margin:0 1px;' "
            f"title='Contribution vers \"{pred_label}\": {score:+.4e}'>"
            f"{token}</span>"
        )

    return f"<span style='color:#999;'>{token}</span>"

def generate_explanation_html(text, word_scores, pred_label, confidence, true_label):
    max_abs_score = max(abs(v) for v in word_scores.values())
    tokens = re.findall(r"\w+|\W+", text)
    html_tokens = "".join(
        render_token(t, word_scores, max_abs_score, pred_label) for t in tokens
    )

    correct = pred_label == true_label
    result_color = "#2ecc71" if correct else "#e74c3c"
    result_text = "CORRECT" if correct else "ERREUR"

    return f"""
    <div style="font-family:Segoe UI; padding:25px; background:#2c3e50;
                border-radius:12px; color:white; margin-bottom:25px;">
        <h3>Visualisation – Poids × Activation</h3>

        <p><b>Classe prédite :</b> {pred_label.upper()} |
           <b>Confiance :</b> {confidence:.1%}</p>
        <p><b>Vérité terrain :</b> {true_label.upper()} |
           <b>Résultat :</b>
           <span style="color:{result_color}; font-weight:bold;">
           {result_text}</span></p>

        <hr>

        <div style="background:#1a252f; padding:20px; border-radius:8px;
                    line-height:2; max-height:400px; overflow-y:auto;">
            {html_tokens}
        </div>
    </div>
    """

def visualize_text_explanations(text, word_scores, pred_label, confidence, true_label):
    html = generate_explanation_html(
        text, word_scores, pred_label, confidence, true_label
    )
    display(HTML(html))



def explain_weight_activation(model, tfidf, le, text, true_label, device, top_k=15, epsilon=1e-9):
    """
    Méthode d'explicabilité basée uniquement sur les poids et activations.
    Ne nécessite AUCUN calcul de gradient.

    Principe : On propage l'importance depuis la sortie vers l'entrée en utilisant
    uniquement les poids W et les activations a.
    """
    model.eval()

    X = tfidf.transform([text]).toarray()
    X_tensor = torch.FloatTensor(X).to(device)

    feature_names = tfidf.get_feature_names_out()

    # Forward pass pour obtenir les activations (SANS gradients)
    with torch.no_grad():
        outputs, activations = model(X_tensor, return_activations=True)

    probs = torch.softmax(outputs, dim=1)
    pred_class = outputs.argmax(1).item()
    pred_label = le.inverse_transform([pred_class])[0]
    confidence = probs[0, pred_class].item()

    print(f"MÉTHODE : POIDS × ACTIVATION")
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

    x = X[0]
    a1 = activations['a1'][0].cpu().numpy()
    a2 = activations['a2'][0].cpu().numpy()

    present_indices = np.where(x > 0)[0]

    # Boucle sur chaque classe
    for target_class_idx, target_class_name in enumerate(le.classes_):
        marker = ""
        if target_class_name == pred_label:
            marker = " ← PRÉDITE"
        elif target_class_name == true_label:
            marker = " ← VRAIE CLASSE"

        print(f"\n{'─'*80}")
        print(f"CLASSE : {target_class_name.upper()}{marker}")
        logit = outputs[0, target_class_idx].item()
        print(f"Logit : {logit:.4f}")
        print(f"{'─'*80}")

        
        print(f"\n PROPAGATION DE L'IMPORTANCE (Sortie → Entrée)")
        print(f"Méthode : importance basée sur poids × activations uniquement")

        # Couche 2 → sortie
        importance_L2 = W3[target_class_idx] * a2
        sum_L2 = np.sum(importance_L2)
        print(f"\n COUCHE 2 → SORTIE")
        print(f"Somme des contributions L2 : {sum_L2:.4f}")
        print(f"Logit (avec biais) : {logit:.4f}")
        print(f"Biais de sortie : {b3[target_class_idx]:.4f}")
        print(f"Vérification : {sum_L2:.4f} + {b3[target_class_idx]:.4f} = {sum_L2 + b3[target_class_idx]:.4f} ≈ {logit:.4f}")

        # Top neurones L2
        neurons_L2_sorted = sorted(enumerate(importance_L2), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n Neurones L2 les plus importants (top 10) :")
        print(f"   {'Neurone':<10} {'Importance':<15} {'Poids':<12} {'Activation':<12}")
        print(f"   {'-'*55}")
        for i, imp in neurons_L2_sorted[:10]:
            print(f"   L2[{i:<3}]   {imp:+.6f}       {W3[target_class_idx,i]:+.4f}    {a2[i]:.4f}")

        # Couche 1 → Couche 2
        importance_L1 = np.zeros(len(a1))
        for j in range(len(a1)):
            if a1[j] == 0:
                continue
            for k in range(len(a2)):
                contrib_j_to_k = W2[k, j] * a1[j]
                total_input_to_k = np.sum(W2[k] * a1) + b2[k]
                fraction = contrib_j_to_k / total_input_to_k if abs(total_input_to_k) > epsilon else 0
                importance_L1[j] += importance_L2[k] * fraction

        sum_L1 = np.sum(importance_L1)
        print(f"\n COUCHE 1 → COUCHE 2")
        print(f"Somme des importances L1 : {sum_L1:.4f}")
        print(f"Conservation : {sum_L1:.4f} ≈ {sum_L2:.4f} (écart : {abs(sum_L1 - sum_L2):.4f})")

        neurons_L1_sorted = sorted(enumerate(importance_L1), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n Neurones L1 les plus importants (top 10) :")
        print(f"   {'Neurone':<10} {'Importance':<15} {'Activation':<12}")
        print(f"   {'-'*40}")
        for i, imp in neurons_L1_sorted[:10]:
            print(f"   L1[{i:<3}]   {imp:+.6f}       {a1[i]:.4f}")

        # Entrée → Couche 1
        importance_terms = np.zeros(len(x))
        for i in present_indices:
            for j in range(len(a1)):
                if a1[j] == 0:
                    continue
                contrib_i_to_j = W1[j, i] * x[i]
                total_input_to_j = np.sum(W1[j] * x) + b1[j]
                fraction = contrib_i_to_j / total_input_to_j if abs(total_input_to_j) > epsilon else 0
                importance_terms[i] += importance_L1[j] * fraction

        sum_terms = np.sum(importance_terms)
        print(f"\n ENTRÉE (TF-IDF) → COUCHE 1")
        print(f"Somme des importances des termes : {sum_terms:.4f}")
        print(f"Conservation : {sum_terms:.4f} ≈ {sum_L1:.4f} (écart : {abs(sum_terms - sum_L1):.4f})")

        # Termes importants
        positive = [(feature_names[i], importance_terms[i], x[i]) for i in present_indices if importance_terms[i] > 0]
        negative = [(feature_names[i], importance_terms[i], x[i]) for i in present_indices if importance_terms[i] < 0]
        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        print(f"\n Termes POUSSANT vers '{target_class_name.upper()}' (Poids×Activation) :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Importance':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, imp, tfidf_val) in enumerate(positive[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {imp:+.6f}       {tfidf_val:.4f}")

        print(f"\n Termes ÉLOIGNANT de '{target_class_name.upper()}' (Poids×Activation) :")
        print(f"   {'Rang':<6} {'Terme':<30} {'Importance':<15} {'TF-IDF':<10}")
        print(f"   {'-'*65}")
        for rank, (term, imp, tfidf_val) in enumerate(negative[:top_k], 1):
            print(f"   {rank:<6} {term:<30} {imp:+.6f}       {tfidf_val:.4f}")

        
        if target_class_name == pred_label:
            word_scores = {feature_names[i]: importance_terms[i] for i in present_indices if abs(importance_terms[i]) > 0}
            visualize_text_explanations(
                text=text,
                word_scores=word_scores,
                pred_label=pred_label,
                confidence=confidence,
                true_label=true_label
            )

    print(f"\n{'='*80}\n")
    return pred_label, confidence
