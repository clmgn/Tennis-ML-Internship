import numpy as np
import pandas as pd
import joblib

# === Paramètres ===
model_path = "random_forest_model.pkl"
player_id = "3"
stroke_type = "Dretes"
hit_number = 12
files_path = "Data/Single Strokes/J" + player_id + "/"
acc_path = files_path + f"J{player_id} {stroke_type} ACC/hit_"+f"{hit_number}"+".csv"
gyr_path = files_path + f"J{player_id} {stroke_type} GYR/hit_"+f"{hit_number}"+".csv"

label_map_inv = {
    0: 'Dretes',
    1: 'Reves',
    2: 'Serve',
    3: 'Smash',
    4: 'VD',
    5: 'VR'
}

# === Charger le modèle Random Forest ===
model = joblib.load(model_path)

# === Fonction de features identique à l'entraînement ===
def extract_features_from_hit(acc_path, gyr_path):
    acc_df = pd.read_csv(acc_path)
    gyr_df = pd.read_csv(gyr_path)

    acc = acc_df[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy()
    gyr = gyr_df[["GYR_X", "GYR_Y", "GYR_Z"]].to_numpy()

    min_len = min(len(acc), len(gyr))
    acc = acc[:min_len]
    gyr = gyr[:min_len]
    combined = np.hstack([acc, gyr])

    means = combined.mean(axis=0)
    stds = combined.std(axis=0)
    features = np.concatenate([means, stds])
    return features

# === Extraire les features ===
features = extract_features_from_hit(acc_path, gyr_path)
if features is None:
    raise ValueError("Erreur lors de l'extraction des features.")

# === Prédiction ===
features = features.reshape(1, -1)
pred = model.predict(features)[0]
proba = model.predict_proba(features)[0]

# === Affichage ===
print(f"\n🧠 Prediction : {label_map_inv[pred]}")
print("Probability per class :")
for i, p in enumerate(proba):
    print(f"  {label_map_inv[i]:<6} : {p:.4f}")
