class GenderClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32):
        super(GenderClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x, return_activations=False):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        a1 = self.relu1(h1)
        a1_drop = self.dropout1(a1)

        h2 = self.fc2(a1_drop)
        h2 = self.bn2(h2)
        a2 = self.relu2(h2)
        a2_drop = self.dropout2(a2)

        output = self.fc3(a2_drop)

        if return_activations:
            return output, {'h1': h1, 'a1': a1, 'h2': h2, 'a2': a2}
        return output
    
