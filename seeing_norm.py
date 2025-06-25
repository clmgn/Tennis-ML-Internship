import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

player_id = "1"
stroke_type = "Dretes"
hit_number = "4"
acc_path = "Data/Single Strokes/J" + player_id + "/J" + player_id + " " + stroke_type + " ACC/hit_"+hit_number+".csv"

acc_df = pd.read_csv(acc_path)
if acc_df.empty:
    raise ValueError(f"Le fichier {acc_path} n'existe pas.")

norm = acc_df["norm"].to_numpy()

plt.figure(figsize=(10, 5))
plt.plot(norm, label='Norme de l\'accélération', color='blue')
plt.title(f'Norme de l\'accélération pour le coup {stroke_type} de joueur {player_id} (hit {hit_number})')
plt.xlabel('Index')
plt.ylabel('Norme')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

