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
        <h3>Visualisation – Gradient × Input</h3>

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


def explain_gradient_input(model, tfidf, le, text, true_label, device, top_k=15):
    """
    Méthode basée uniquement sur les gradients : contribution = gradient × input
    ATTENTION : sum(contributions) ≠ logit
    """

    model.eval()

    X = tfidf.transform([text]).toarray()
    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor.requires_grad = True

    feature_names = tfidf.get_feature_names_out()

    outputs, activations = model(X_tensor, return_activations=True)

    probs = torch.softmax(outputs.detach(), dim=1)
    pred_class = outputs.argmax(1).item()
    pred_label = le.inverse_transform([pred_class])[0]
    confidence = probs[0, pred_class].item()

    print("=" * 80)
    print("MÉTHODE : GRADIENT × INPUT")
    print("=" * 80)
    print(f"Texte : {text[:150]}...")
    print(f"Prédiction : {pred_label.upper()} (confiance : {confidence:.1%})")
    print(f"Vérité terrain : {true_label.upper()}")
    if pred_label != true_label:
        print("ERREUR DE CLASSIFICATION")

    gradients = torch.autograd.grad(
        outputs=outputs[0, pred_class],
        inputs=X_tensor,
        retain_graph=True
    )[0].cpu().numpy().flatten()

    input_contrib = gradients * X[0]
    present_indices = np.where(X[0] > 0)[0]

    word_scores = {
        feature_names[i]: input_contrib[i]
        for i in present_indices
    }

    visualize_text_explanations(
        text=text,
        word_scores=word_scores,
        pred_label=pred_label,
        confidence=confidence,
        true_label=true_label
    )

    for target_class_idx, target_class_name in enumerate(le.classes_):

        marker = ""
        if target_class_name == pred_label:
            marker = " ← PRÉDITE"
        elif target_class_name == true_label:
            marker = " ← VRAIE CLASSE"

        print("\n" + "─" * 80)
        print(f"CLASSE : {target_class_name.upper()}{marker}")
        print(f"Logit : {outputs[0, target_class_idx].item():.4f}")
        print("─" * 80)

        grads = torch.autograd.grad(
            outputs=outputs[0, target_class_idx],
            inputs=X_tensor,
            retain_graph=True
        )[0].cpu().numpy().flatten()

        contrib = grads * X[0]

        pos = [(feature_names[i], contrib[i], X[0, i])
               for i in present_indices if contrib[i] > 0]
        neg = [(feature_names[i], contrib[i], X[0, i])
               for i in present_indices if contrib[i] < 0]

        pos.sort(key=lambda x: x[1], reverse=True)
        neg.sort(key=lambda x: x[1])

        print("\nTermes POUSSANT :")
        for r, (t, c, v) in enumerate(pos[:top_k], 1):
            print(f"{r:>2}. {t:<25} {c:+.6f} (tf-idf={v:.4f})")

        print("\nTermes ÉLOIGNANT :")
        for r, (t, c, v) in enumerate(neg[:top_k], 1):
            print(f"{r:>2}. {t:<25} {c:+.6f} (tf-idf={v:.4f})")

    print("=" * 80)
    return pred_label, confidence
