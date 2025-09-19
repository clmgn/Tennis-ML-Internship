# Tennis-ML-Internship

This project was developed as part of an engineering internship, with the aim of analyzing muscle use during tennis practice using inertial sensor data (accelerometer and gyroscope).
The goal is to detect the type of stroke performed (forehand, backhand, serve, etc.) so that experts cant then study the muscle groups mostly impacted based on strokes frequency.

# Objectives :

- Collect motion data using a smartwatch (accelerometer + gyroscope)
- Apply Machine Learning techniques to classify different tennis strokes
- Identify the most frequent strokes for each player
- Investigate the potential muscular impact caused by repetitive motion

N.B. : This project could not be achieved completely, since the real match data was never sent.

# Description :

The main dataset that has been used for this project was composed of many files containing series of repeated strokes, from different players.
It was constructed like this :

<pre> Data/
├── Jugador 1/
│   ├── Enregistraments 1/
│   │   ├── J1 Dretes ACC.csv
│   │   ├── J1 Dretes GYR.csv
│   │   ├── J1 Reves ACC.csv
│   │   ├── J1 Reves GYR.csv
│   │   ├── J1 Smash ACC.csv
│   │   ├── J1 Smash GYR.csv
│   │   ├── J1 Serve ACC.csv
│   │   ├── J1 Serve GYR.csv
│   │   ├── J1 VD ACC.csv
│   │   ├── J1 VD GYR.csv
│   │   ├── J1 VR ACC.csv
│   │   └── J1 VR GYR.csv
│   └── INFO Jugador 1.docx
├── Jugador 2/
│   ├── Enregistraments 2/
│   │   ├── J2 Dretes ACC.csv
│   │   ├── J2 Dretes GYR.csv
│   │   ├── J2 Reves ACC.csv
│   │   ├── J2 Reves GYR.csv
│   │   ...
│   └── INFO Jugador 2.docx
...
└── Jugador 6/ </pre>

Other data have been used to do an audio detection of the hits, which can be .mp3 or .wav files, and downloaded from youtube.

**Code Description**

Using motion data :

- separating_hits.py : This file was used to cut the original csv files into several smaller files containing single strokes. It made training a model easier.
- training_pytorch.py : This is the final algorithm that I used to train a Machine Learning model detecting the strokes. It returns a trained model that can be used in other files, as well as a confusion matrix showing the test results.
- stroke_detection.py : This file was used to test the model trained in the previous file : for one specific stroke file, it returns what type of stroke it is and with what probability.
- create_match_set.py : This file allows to create a fake match-alike file, using the already separated strokes. The code can be modified to select from which player the files should be taken. It returns three different csv files : one containing the accelerometer data, an other containing the gyroscop data, and a final one, containing the lines where a stroke ends and a new one start, as well as the type of stroke for each one.
- test_match.py : This file allows to try using the model on what should be a recorded match. For each stroke, it prints the prediction and what it really is. It then calculates the purcentage of accuracy of the prediction.


Using audio files :

- cut_long_audio_file.py : This code will detect if the intensity of the sound does not exceed a specific treshold for too long, then the initial file will be cut in small files.
- detecting_hits_audio.py : This file shows the norm of the audio signal, points the strokes that have been detected and prints the number of strokes detected in one file.





