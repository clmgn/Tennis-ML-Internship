import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt

# Configuration
path = "./Data/Proves Stroke Count/Jugador 6/Enregistraments J6/"
window_size = 80
prominence = 10

def process_acc_file(input_file_acc):
    # Extract the base name of the file without extension
    file_base_acc = os.path.basename(input_file_acc).replace(".csv", "")
    output_folder_acc = os.path.join("data", file_base_acc)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_acc, exist_ok=True)

    # Load the ACC data
    df_acc = pd.read_csv(input_file_acc)

    # Calculate the norm for ACC data
    df_acc["norm"] = np.sqrt(df_acc["ACC_X"]**2 + df_acc["ACC_Y"]**2 + df_acc["ACC_Z"]**2)

    # Plot the total acceleration
    plt.figure(figsize=(12, 4))
    plt.plot(df_acc["norm"], label="Norm (acc)")
    plt.title(f"Total Signal Norm - {file_base_acc}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Extract hits from ACC data
    cut_df_acc = df_acc.copy()
    hit_segments = []

    while True:
        data = cut_df_acc["norm"].values
        peaks, _ = find_peaks(data, prominence=prominence, distance=window_size)
        if len(peaks) == 0:
            break

        first_peak = peaks[0]
        start = max(0, first_peak - window_size)
        end = min(len(cut_df_acc), first_peak + window_size)

        # Save the segment indices
        hit_segments.append((start, end))

        # Save the ACC hit
        hit_df_acc = cut_df_acc.iloc[start:end]
        hit_df_acc.to_csv(os.path.join(output_folder_acc, f"hit_{len(hit_segments)-1}.csv"), index=False)

        # Cut the detected segment
        cut_df_acc = pd.concat([cut_df_acc.iloc[:start], cut_df_acc.iloc[end:]]).reset_index(drop=True)

    return hit_segments

def apply_segments_to_gyr(input_file_gyr, hit_segments):
    # Extract the base name of the file without extension
    file_base_gyr = os.path.basename(input_file_gyr).replace(".csv", "")
    output_folder_gyr = os.path.join("data", file_base_gyr)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_gyr, exist_ok=True)

    # Load the GYR data
    df_gyr = pd.read_csv(input_file_gyr)

    # Apply the same segments to GYR data
    for i, (start, end) in enumerate(hit_segments):
        hit_df_gyr = df_gyr.iloc[start:end]
        hit_df_gyr.to_csv(os.path.join(output_folder_gyr, f"hit_{i}.csv"), index=False)

    print(f"✅ Total number of hits extracted : {len(hit_segments)}")

# Process the ACC file to get the segments, plot the acceleration, and save ACC hits
hit_segments = process_acc_file(path + "J6 Dretes ACC.csv")

# Apply the same segments to the GYR file and save GYR hits
apply_segments_to_gyr(path + "J6 Dretes GYR.csv", hit_segments)
