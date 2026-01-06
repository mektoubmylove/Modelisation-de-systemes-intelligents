# eval
model.load_state_dict(torch.load('best_gender_classifier.pth'))
model.eval()

y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        y_pred.extend(outputs.argmax(1).cpu().numpy())
        y_true.extend(y_batch.numpy())

test_acc = np.mean(np.array(y_pred) == np.array(y_true))
print(f"Test Accuracy: {test_acc:.4f}")
print("\nRapport de classification:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_,
            yticklabels=le.classes_, ax=ax, cbar_kws={'label': 'Nombre de prédictions'})
ax.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
ax.set_ylabel('Vérité terrain', fontsize=12, fontweight='bold')
ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Sauvegarde
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("\n Modèles sauvegardés")

"""
Test Accuracy: 0.8877

Rapport de classification:
              precision    recall  f1-score   support

       femme       0.89      0.92      0.90       165
       homme       0.89      0.84      0.86       120

    accuracy                           0.89       285
   macro avg       0.89      0.88      0.88       285
weighted avg       0.89      0.89      0.89       285
"""