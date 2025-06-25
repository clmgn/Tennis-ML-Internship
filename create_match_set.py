import os
import random
import pandas as pd

base_path = "./Data/Single Strokes"
player_id = "J6"
num_hits = 100
output_dir = "./Data/Mixed Dataset/Match Set "+ player_id

def join_random_hits_dual(base_path, player_id, num_hits, output_dir="output", random_seed=None):
    acc_files = []

    player_path = os.path.join(base_path, player_id)
    for root, dirs, files in os.walk(player_path):
        if "ACC" in root:
            for file in files:
                if file.startswith("hit_") and file.endswith(".csv"):
                    acc_path = os.path.join(root, file)
                    gyr_path = acc_path.replace("ACC", "GYR")  # Associe le fichier GYR
                    if os.path.exists(gyr_path):
                        acc_files.append((acc_path, gyr_path))

    if len(acc_files) < num_hits:
        raise ValueError(f"Seulement {len(acc_files)} coups disponibles, impossible d’en sélectionner {num_hits}.")

    if random_seed is not None:
        random.seed(random_seed)

    selected_files = random.sample(acc_files, num_hits)

    joined_acc = pd.DataFrame()
    joined_gyr = pd.DataFrame()
    metadata = []

    acc_line = 0

    for acc_path, gyr_path in selected_files:
        df_acc = pd.read_csv(acc_path)
        df_gyr = pd.read_csv(gyr_path)

        joined_acc = pd.concat([joined_acc, df_acc], ignore_index=True)
        joined_gyr = pd.concat([joined_gyr, df_gyr], ignore_index=True)

        metadata.append({
            "acc_file": os.path.relpath(acc_path, base_path),
            "gyr_file": os.path.relpath(gyr_path, base_path),
            "Line_Number": acc_line
        })

        acc_line += len(df_acc)

    # Ajouter une dernière ligne au fichier metadata indiquant la fin
    metadata.append({
    "acc_file": "END",
    "gyr_file": "END",
    "Line_Number": len(joined_acc)
    })


    # Ajouter Line_Number global après concaténation
    joined_acc['Line_Number'] = range(len(joined_acc))
    joined_gyr['Line_Number'] = range(len(joined_gyr))

    os.makedirs(output_dir, exist_ok=True)

    joined_acc.to_csv(os.path.join(output_dir, "joined_ACC.csv"), index=False)
    joined_gyr.to_csv(os.path.join(output_dir, "joined_GYR.csv"), index=False)
    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"Fichiers créés dans '{output_dir}' avec {num_hits} coups sélectionnés.")

# Lancement
join_random_hits_dual(base_path, player_id, num_hits, output_dir)
