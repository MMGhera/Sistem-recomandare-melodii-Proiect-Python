import os
import csv
import librosa
import numpy as np
import soundfile as sf

def extract_simple_features(signal, sr):
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
    tempo = float(tempo[0])  # <-- îl convertim în număr simplu

    rms = float(np.mean(librosa.feature.rms(y=signal)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)))
    mfcc_mean = float(np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)))

    return [tempo, rms, centroid, zcr, mfcc_mean]

output_path = 'features.csv'

# Scriem header o singură dată
write_header = not os.path.exists(output_path)

with open(output_path, 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    if write_header:
        writer.writerow(["tempo", "rms", "centroid", "zcr", "mfcc_mean"])

    for filename in os.listdir('test_wav'):
        if filename.endswith('.wav'):
            path = os.path.join('test_wav', filename)

            signal, sr = sf.read(path, always_2d=False)
            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)

            features = extract_simple_features(signal, sr)
            writer.writerow(features)
