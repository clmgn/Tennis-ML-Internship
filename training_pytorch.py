import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# === CONFIG ===
root_dir = './Data/Single Strokes'
label_map = {
    'Dretes': 0,
    'Reves': 1,
    'Serve': 2,
    'Smash': 3,
    'VD': 4,
    'VR': 5
}
label_names = {v: k for k, v in label_map.items()}

players = ["J1", "J2", "J3", "J4", "J5", "J6"]
single_player = "J6"  #For testing with a single player

# === DATASET ===
class StrokeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === FEATURE EXTRACTION ===
def load_stroke_data(root_dir, single_player, label_map):
    X, y = [], []

    for player in os.listdir(root_dir):
        if player != single_player:
            continue

        player_path = os.path.join(root_dir, player)
        if not os.path.isdir(player_path):
            continue

        for stroke_dir in os.listdir(player_path):
            if not stroke_dir.endswith('ACC'):
                continue

            label = next((label_map[k] for k in label_map if k in stroke_dir), None)
            if label is None:
                continue

            acc_dir = os.path.join(player_path, stroke_dir)
            gyr_dir = acc_dir.replace('ACC', 'GYR')

            for filename in os.listdir(acc_dir):
                if not filename.endswith('.csv'):
                    continue

                acc_path = os.path.join(acc_dir, filename)
                gyr_path = os.path.join(gyr_dir, filename)

                if not os.path.exists(gyr_path):
                    print(f"GYR manquant pour {acc_path}, ignoré.")
                    continue

                acc = pd.read_csv(acc_path)[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy()
                gyr = pd.read_csv(gyr_path)[["GYR_X", "GYR_Y", "GYR_Z"]].to_numpy()

                min_len = min(len(acc), len(gyr))
                acc = acc[:min_len]
                gyr = gyr[:min_len]

                combined = np.hstack([acc, gyr]).T  # shape: (6, time)

                if combined.shape[1] < 100:
                    pad = np.zeros((6, 100 - combined.shape[1]))
                    combined = np.hstack([combined, pad])
                elif combined.shape[1] > 100:
                    combined = combined[:, :100]

                X.append(combined)
                y.append(label)

    return np.array(X), np.array(y)

# === CNN MODEL ===
class CNN1D(nn.Module):
    def __init__(self, n_classes=6):
        super(CNN1D, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),      # cnn.0
            nn.BatchNorm1d(32),                              # cnn.1
            nn.ReLU(),                                       # cnn.2
            nn.MaxPool1d(2),                                 # cnn.3

            nn.Conv1d(32, 64, kernel_size=5, padding=2),     # cnn.4
            nn.BatchNorm1d(64),                              # cnn.5
            nn.ReLU(),                                       # cnn.6
            nn.MaxPool1d(2),                                 # cnn.7

            nn.Conv1d(64, 128, kernel_size=5, padding=2),    # cnn.8
            nn.BatchNorm1d(128),                             # cnn.9
            nn.ReLU(),                                       # cnn.10
            nn.AdaptiveMaxPool1d(1),                         # cnn.11
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                    # fc.0
            nn.Linear(128, 64),                              # fc.1
            nn.ReLU(),                                       # fc.2
            nn.Dropout(0.6),                                 # fc.3
            nn.Linear(64, n_classes)                         # fc.4
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


# === TRAINING ===
X, y = load_stroke_data(root_dir, single_player, label_map)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_dataset = StrokeDataset(X_train, y_train)
test_dataset = StrokeDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN1D().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Calcul de la perte sur le test
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    print(f"Epoch {epoch+1}/30, Train Loss: {total_loss:.4f}, Test Loss: {test_loss:.4f}")

# === EVALUATION ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred, normalize='true')
ConfusionMatrixDisplay(cm, display_labels=[label_names[i] for i in range(6)]).plot(cmap='Blues')
plt.title("Confusion Matrix (Normalized)")
plt.tight_layout()
plt.show()

# === SAVE MODEL ===
torch.save(model.state_dict(), 'cnn_model.pth')
