import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os


def detect_shots(y, sr, threshold_factor=4, min_interval=0.5, plot=False):
    # # 1. Charger l'audio
    # y, sr = librosa.load(audio_path, sr=None, mono=True)

   # 2. Calculer l'énergie RMS
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # 3. Détection initiale des pics (candidats)
    threshold = np.mean(rms) * threshold_factor
    peaks, properties = scipy.signal.find_peaks(rms, height=threshold)
    peak_times = times[peaks]
    peak_values = rms[peaks]

    # 4. Regroupement des pics trop proches
    merged_times = []
    last_time = -np.inf
    group_max_value = -np.inf
    group_max_time = None

    for t, val in zip(peak_times, peak_values):
        if t - last_time > min_interval:
            # Nouveau groupe
            if group_max_time is not None:
                merged_times.append(group_max_time)
            group_max_value = val
            group_max_time = t
        else:
            # Même groupe, on garde le pic le plus fort
            if val > group_max_value:
                group_max_value = val
                group_max_time = t
        last_time = t

    # Ajouter le dernier pic
    if group_max_time is not None:
        merged_times.append(group_max_time)

    merged_times = np.array(merged_times)

    # 5. Affichage
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(times, rms, label="RMS Energy")
        plt.plot(peak_times, peak_values, "rx", label="Peaks")
        plt.plot(merged_times, [threshold]*len(merged_times), "go", label="Possible Hits")
        plt.axhline(threshold, color='gray', linestyle='--', label=f"Seuil = {threshold:.5f}")
        plt.xlabel("Time (s)")
        plt.ylabel("RMS Energy")
        plt.legend()
        plt.title("Tennis hits detection")
        plt.grid()
        plt.show()

    return merged_times

def denoise_signal(y, sr, cutoff_freq=500):

    # Conception du filtre passe-haut Butterworth
    nyq = 0.5 * sr
    norm_cutoff = cutoff_freq / nyq
    b, a = butter(N=4, Wn=norm_cutoff, btype='high', analog=False)

    # Application du filtre
    y_filtered = filtfilt(b, a, y)

    # Normalisation du signal
    y_filtered = y_filtered / np.max(np.abs(y_filtered))

    return y_filtered

def compute_fft(y, sr, plot=False):

    n = len(y)
    yf = np.fft.rfft(y)
    xf = np.fft.rfftfreq(n, 1 / sr)
    magnitude = np.abs(yf)

    # if plot:
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(xf, magnitude)
    #     plt.title("Transformée de Fourier (spectre de fréquences)")
    #     plt.xlabel("Fréquence (Hz)")
    #     plt.ylabel("Amplitude")
    #     plt.grid(True)
    #     plt.xlim(0, sr/2)
    #     plt.show()

    return xf, magnitude

def plot_signal(y, sr, title="Audio Signal", plot_fft=False):
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(times, y)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # Exemple d'utilisation
    # audio_file = "tennis_short.mp3"  # Remplace par le chemin de ton fichier

    # signal, sr = librosa.load(audio_file, sr=None, mono=True)

    # plot_signal(signal, sr, title="Original Audio Signal")

    # denoised_signal = denoise_signal(signal, sr, cutoff_freq=500)

    # plot_signal(denoised_signal, sr, title="Denoised Audio Signal")

    # coup_times = detect_shots(denoised_signal, sr, plot=True)

    # print("Detected strokes (in seconds) :", coup_times)
    # print("Number of detected strokes:", len(coup_times))
    
    segment_dir = "segments"
    hits_count = 0
    for filename in sorted(os.listdir(segment_dir)):
        if filename.endswith(".wav"):
            file_path = os.path.join(segment_dir, filename)
            signal, sr = librosa.load(file_path, sr=None, mono=True)
            denoised_signal = denoise_signal(signal, sr, cutoff_freq=500)
            coup_times = detect_shots(denoised_signal, sr, plot=True)
            print(f"Detected strokes in {filename} (in seconds):", coup_times)
            print(f"Number of detected strokes in {filename}:", len(coup_times))
            hits_count += len(coup_times)
    print(f"Total number of detected strokes across all segments: {hits_count}")