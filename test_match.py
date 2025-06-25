# import pandas as pd
# import numpy as np
# import joblib
# import os

# # === PARAMÈTRES ===
# files_path = "./Data/Mixed Dataset/"
# acc_file = os.path.join(files_path, "joined_ACC.csv")
# gyr_file = acc_file.replace("ACC", "GYR")
# hit_times_file = os.path.join(files_path, "metadata.csv")
# model_path = "random_forest_model.pkl"

# # Vérification de l'existence des fichiers
# if not all(os.path.exists(f) for f in [acc_file, gyr_file, hit_times_file, model_path]):
#     raise FileNotFoundError("One or several files are not found. Please check your path.")

# # === CHARGEMENT DES DONNÉES ===
# df_acc = pd.read_csv(acc_file)
# df_gyr = pd.read_csv(gyr_file)
# hit_times_df = pd.read_csv(hit_times_file)

# # Chargement du modèle Random Forest
# model = joblib.load(model_path)

# # === FONCTION D'EXTRACTION DE FEATURES ===
# def extract_features(acc_segment, gyr_segment):
#     acc = acc_segment[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy()
#     gyr = gyr_segment[["GYR_X", "GYR_Y", "GYR_Z"]].to_numpy()
#     min_len = min(len(acc), len(gyr))
#     acc = acc[:min_len]
#     gyr = gyr[:min_len]
#     combined = np.hstack([acc, gyr])
#     return np.concatenate([combined.mean(axis=0), combined.std(axis=0)])



# # === DICTIONNAIRE DES LABELS ===
# label_names = {
#     0: "Dretes",
#     1: "Reves",
#     2: "Serve",
#     3: "Smash",
#     4: "VD",
#     5: "VR"
# }

# # === RÉCUPÉRATION DES PAIRES DE LIGNES ===
# lines = list(hit_times_df["Line_Number"])
# hit_pairs = [(lines[i], lines[i + 1]) for i in range(len(lines) - 1)]

# # === PRÉDICTIONS ===
# predictions = []

# for start_line, end_line in hit_pairs:
#     mask = (df_acc["Line_Number"] >= start_line) & (df_acc["Line_Number"] < end_line)
#     acc_seg = df_acc[mask]
#     gyr_seg = df_gyr[mask]

#     if len(acc_seg) == 0 or len(gyr_seg) == 0:
#         predictions.append("Unknown")
#         print(f"[WARNING] Empty between lines {start_line}-{end_line}")
#         continue

#     feat = extract_features(acc_seg, gyr_seg).reshape(1, -1)
#     pred_label = model.predict(feat)[0]
#     predictions.append(label_names.get(pred_label, "Unknown"))

# # === EXTRACTION DES VRAIS LABELS ===
# true_labels = []
# for acc_file_path in hit_times_df["acc_file"][:-1]:  # On ignore la dernière ligne "END"
#     folder = acc_file_path.split("/")[-2]  # Ex: 'J1 Serve ACC'
#     label = folder.replace("ACC", "").strip().split()[-1]  # Ex: 'Serve'
#     true_labels.append(label)

# # === AFFICHAGE ET ÉVALUATION ===
# correct = 0
# total = len(true_labels)

# print("\n--- Prediction results ---\n")

# for i, (start_line, end_line), pred, true in zip(range(total), hit_pairs, predictions, true_labels):
#     result = "✅" if pred == true else "❌"
#     if pred == true:
#         correct += 1
#     print(f"Stroke {i+1:02d} | Lines: {start_line}-{end_line} | Prediction: {pred:<7} | True: {true:<7} {result}")

# accuracy = (correct / total) * 100 if total > 0 else 0
# print(f"\n🎯 Modele accuracy : {accuracy:.2f}% ({correct}/{total} accurate)")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score

# Parameters
files_path = "./Data/Mixed Dataset/Match Set J4/"
acc_file = os.path.join(files_path, "joined_ACC.csv")
gyr_file = acc_file.replace("ACC", "GYR")
hit_times_file = os.path.join(files_path, "metadata.csv")
model_path = "cnn_model.pth"

# Check files exist
if not all(os.path.exists(f) for f in [acc_file, gyr_file, hit_times_file, model_path]):
    raise FileNotFoundError("Some files are missing. Check paths.")

# Load data
df_acc = pd.read_csv(acc_file)
df_gyr = pd.read_csv(gyr_file)
hit_times_df = pd.read_csv(hit_times_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN model same as training
class CNN1D(nn.Module):
    def __init__(self, n_classes=6):
        super(CNN1D, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


# Load model
model = CNN1D()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

label_names = {
    0: "Dretes",
    1: "Reves",
    2: "Serve",
    3: "Smash",
    4: "VD",
    5: "VR"
}

def preprocess_segment(acc_segment, gyr_segment):
    acc = acc_segment[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy()
    gyr = gyr_segment[["GYR_X", "GYR_Y", "GYR_Z"]].to_numpy()
    min_len = min(len(acc), len(gyr), 100)
    acc = acc[:min_len]
    gyr = gyr[:min_len]
    combined = np.hstack([acc, gyr])
    if combined.shape[0] < 100:
        pad = np.zeros((100 - combined.shape[0], 6))
        combined = np.vstack([combined, pad])
    return combined.astype(np.float32).T  # shape (6, 100)

# Retrieve hit pairs from metadata.csv
lines = list(hit_times_df["Line_Number"])
hit_pairs = [(lines[i], lines[i + 1]) for i in range(len(lines) - 1)]

predictions = []

for start_line, end_line in hit_pairs:
    mask = (df_acc["Line_Number"] >= start_line) & (df_acc["Line_Number"] < end_line)
    acc_seg = df_acc[mask]
    gyr_seg = df_gyr[mask]

    if len(acc_seg) == 0 or len(gyr_seg) == 0:
        predictions.append("Unknown")
        print(f"[WARNING] Empty segment lines {start_line}-{end_line}")
        continue

    feat = preprocess_segment(acc_seg, gyr_seg)
    input_tensor = torch.tensor(feat).unsqueeze(0).to(device)  # shape (1, 6, 100)
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()
        predictions.append(label_names.get(pred_label, "Unknown"))

# Extract true labels from hit_times metadata
true_labels = []
for acc_file_path in hit_times_df["acc_file"][:-1]:  # ignore last "END"
    folder = acc_file_path.split("/")[-2]
    label = folder.replace("ACC", "").strip().split()[-1]
    true_labels.append(label)

# Evaluation
correct = 0
total = len(true_labels)

print("\n--- Prediction results ---\n")
for i, ((start_line, end_line), pred, true) in enumerate(zip(hit_pairs, predictions, true_labels)):
    result = "✅" if pred == true else "❌"
    if pred == true:
        correct += 1
    print(f"Stroke {i+1:02d} | Lines: {start_line}-{end_line} | Prediction: {pred:<7} | True: {true:<7} {result}")

accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n🎯 Model accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
