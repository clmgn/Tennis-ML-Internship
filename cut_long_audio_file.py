import os
import librosa
import soundfile as sf
import numpy as np
from detecting_hits_audio import detect_shots

def split_audio_by_shots(audio_path, output_dir, max_silence=3.0, threshold_factor=2.5, min_interval=0.3):
    """
    Découpe un fichier audio en plusieurs fichiers si une pause > max_silence est détectée entre les coups.

    audio_path : chemin du fichier .wav
    output_dir : dossier où enregistrer les segments
    max_silence : durée maximale sans coup avant découpage (en secondes)
    threshold_factor : facteur pour le seuil de détection des coups
    min_interval : durée min entre 2 coups distincts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Chargement de l'audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Détection des coups (réutilise la fonction précédente)
    shot_times = detect_shots(y, sr, threshold_factor=threshold_factor, min_interval=min_interval, plot=False)

    if len(shot_times) == 0:
        print("Aucun coup détecté.")
        return []

    # Ajouter le début et la fin
    shot_times = np.insert(shot_times, 0, 0.0)
    shot_times = np.append(shot_times, librosa.get_duration(y=y, sr=sr))

    segments = []
    current_segment = []
    last_shot = 0.0

    for t in shot_times:
        if t - last_shot > max_silence and current_segment:
            # Nouvelle séquence : extraire audio entre le 1er et dernier coup du segment
            start_time = current_segment[0]
            end_time = current_segment[-1] + 1.0  # on ajoute 1 sec de marge à droite
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]

            filename = f"segment_{len(segments)}.wav"
            output_path = os.path.join(output_dir, filename)
            sf.write(output_path, segment_audio, sr)
            segments.append((start_time, end_time, output_path))

            current_segment = []

        current_segment.append(t)
        last_shot = t

    # Dernier segment
    if current_segment:
        start_time = current_segment[0]
        end_time = current_segment[-1] + 1.0
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = y[start_sample:end_sample]

        filename = f"segment_{len(segments)}.wav"
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, segment_audio, sr)
        segments.append((start_time, end_time, output_path))

    print(f"{len(segments)} segments sauvegardés dans '{output_dir}'.")
    return segments

split_audio_by_shots(
    audio_path="tennis_long.mp3",
    output_dir="segments",
    max_silence=4.0,           # couper après 3 sec sans coup
    threshold_factor=2.8,
    min_interval=0.25
)