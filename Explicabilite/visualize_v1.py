from IPython.display import display, HTML
import re


def render_token(token, word_scores, max_abs_score, pred_label):
    """
    Colore un mot en fonction de sa contribution à la prédiction.
    """
    lower_token = token.lower()
    if lower_token in word_scores:
        score = word_scores[lower_token]
        alpha = 0.2 + (abs(score) / max_abs_score) * 0.7

        # Couleurs selon la contribution
        if score > 0:
            color = f"rgba(0, 128, 0, {alpha:.2f})"  # Vert pour contributions positives
        else:
            color = f"rgba(178, 34, 34, {alpha:.2f})" # Rouge pour contributions négatives

        return (
            f"<span style='background-color:{color}; color: white; padding:1px 3px; "
            f"border-radius:4px; margin:0 1px; cursor:help;' "
            f"title='Contribution vers \"{pred_label}\": {score:+.4e}'>{token}</span>"
        )

    # Texte normal sans couleur de fond
    return f"<span style='color: #999;'>{token}</span>"

def generate_explanation_html(text, word_scores, pred_label, confidence, true_label):
    """
    Génère le code HTML complet pour la visualisation des explications.
    """
    if not word_scores:
        return "<div style='color: red; padding: 20px;'>Aucun mot du dictionnaire trouvé dans le texte.</div>"

    max_abs_score = max(abs(v) for v in word_scores.values())

    # Tokenisation du texte
    tokens = re.findall(r"\w+|\W+", text)
    html_output = "".join(render_token(t, word_scores, max_abs_score, pred_label) for t in tokens)

    # Déterminer si la prédiction est correcte
    is_correct = pred_label == true_label
    correctness_color = "#2ecc71" if is_correct else "#e74c3c"
    correctness_text = " CORRECT" if is_correct else " ERREUR"

    # Générer l'HTML complet
    html = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 25px; background-color: #2c3e50; border-radius: 12px;
                color: white; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">

        <h3 style="margin-top:0; color: #ecf0f1; border-bottom: 2px solid #3498db;
                   padding-bottom: 10px;">
            Visualisation
        </h3>

        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;
                    margin-bottom: 20px;">
            <div style="flex: 1; min-width: 250px; margin-right: 20px;">
                <p style="margin: 8px 0;">
                    <b style="color: #bdc3c7;">Classe Prédite :</b>
                    <span style="color: #3498db; font-size: 1.3em; font-weight: bold;">
                        {pred_label.upper()}
                    </span>
                </p>
                <p style="margin: 8px 0;">
                    <b style="color: #bdc3c7;">Confiance :</b>
                    <span style="font-size: 1.2em; font-weight: bold;">
                        {confidence:.1%}
                    </span>
                </p>
            </div>

            <div style="flex: 1; min-width: 250px;">
                <p style="margin: 8px 0;">
                    <b style="color: #bdc3c7;">Vérité Terrain :</b>
                    <span style="color: #e74c3c; font-size: 1.2em; font-weight: bold;">
                        {true_label.upper()}
                    </span>
                </p>
                <p style="margin: 8px 0;">
                    <b style="color: #bdc3c7;">Résultat :</b>
                    <span style="color: {correctness_color}; font-size: 1.2em; font-weight: bold;">
                        {correctness_text}
                    </span>
                </p>
            </div>
        </div>

        <div style="margin-bottom: 20px; padding: 12px; background-color: #34495e;
                    border-radius: 8px;">
            <h4 style="margin-top: 0; color: #ecf0f1;">Légende des couleurs :</h4>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: rgba(0, 128, 0, 0.8);
                                border-radius: 4px; margin-right: 8px;"></div>
                    <span>Contribution positive → {pred_label.upper()}</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: rgba(178, 34, 34, 0.8);
                                border-radius: 4px; margin-right: 8px;"></div>
                    <span>Contribution négative → {pred_label.upper()}</span>
                </div>

            </div>
            <p style="margin: 8px 0 0 0; font-size: 12px; color: #bdc3c7;">
                <i>Survolez les mots colorés pour voir leur contribution exacte.</i>
            </p>
        </div>

        <hr style="border: 0; border-top: 1px solid #7f8c8d; margin: 20px 0;">

        <div style="background-color: #1a252f; padding: 20px; border-radius: 8px;
                    max-height: 400px; overflow-y: auto; line-height: 2.0;
                    font-size: 15px; text-align: justify;">
            {html_output}
        </div>

        <hr style="border: 0; border-top: 1px solid #7f8c8d; margin: 20px 0;">

        <div style="font-size: 13px; color: #bdc3c7;">
            <p style="margin: 5px 0;">
                <b>Interprétation :</b> Plus la couleur est intense (verte ou rouge),
                plus le mot influence fortement la prédiction vers ou contre la classe "{pred_label}".
            </p>

        </div>
    </div>
    """

    return html

def visualize_text_explanations(text, word_scores, pred_label, confidence, true_label):
    """
    Affiche la visualisation  des explications.
    """

    print("Visualisation des contrib par mot ")

    html_content = generate_explanation_html(text, word_scores, pred_label, confidence, true_label)
    display(HTML(html_content))