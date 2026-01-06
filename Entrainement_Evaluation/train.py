device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = GenderClassifier(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)

n_epochs = 15
patience = 3
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print(f"Architecture: {X_train.shape[1]} → 64 → 32 → 2")
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(n_epochs):
    model.train()
    train_loss, train_correct = 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == y_batch).sum().item()

    train_loss /= len(train_loader)
    train_acc = train_correct / len(train_dataset)

    model.eval()
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == y_batch).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_dataset)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_gender_classifier.pth')
        print(f" Epoch {epoch+1}/{n_epochs} - Train: {train_acc:.3f} | Val: {val_acc:.3f} [SAVED]")
    else:
        patience_counter += 1
        print(f"  Epoch {epoch+1}/{n_epochs} - Train: {train_acc:.3f} | Val: {val_acc:.3f} [patience: {patience_counter}/{patience}]")
        if patience_counter >= patience:
            print(f"\nEarly stopping à l'epoch {epoch+1}")
            break

# Visu
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history['train_loss']) + 1)
ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
ax2.plot(epochs_range, history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

"""Device: cpu
Architecture: 40000 → 64 → 32 → 2
Paramètres: 2,562,402
 Epoch 1/15 - Train: 0.546 | Val: 0.423 
 Epoch 2/15 - Train: 0.629 | Val: 0.842 
 Epoch 3/15 - Train: 0.718 | Val: 0.849 
 Epoch 4/15 - Train: 0.784 | Val: 0.842 
 Epoch 5/15 - Train: 0.804 | Val: 0.838 
 Epoch 6/15 - Train: 0.857 | Val: 0.852 
 Epoch 7/15 - Train: 0.876 | Val: 0.856 
  Epoch 8/15 - Train: 0.907 | Val: 0.845 [patience: 1/3]
 Epoch 9/15 - Train: 0.923 | Val: 0.845 
 Epoch 10/15 - Train: 0.933 | Val: 0.852 
 Epoch 11/15 - Train: 0.954 | Val: 0.859 
 Epoch 12/15 - Train: 0.965 | Val: 0.866 
 Epoch 13/15 - Train: 0.968 | Val: 0.859 
 Epoch 14/15 - Train: 0.961 | Val: 0.859 
 Epoch 15/15 - Train: 0.975 | Val: 0.859 
 """