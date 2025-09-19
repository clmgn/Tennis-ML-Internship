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

The dataset that has been used for this project was composed of many files containing series of repeated strokes, from different players.
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



