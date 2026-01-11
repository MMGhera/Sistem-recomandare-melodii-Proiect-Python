import os
import csv
import librosa
import numpy as np
import soundfile as sf

CSV_FILE = "features.csv"
WAV_DIR = "wav"

def feats(sig, sr):
    tempo, _ = librosa.beat.beat_track(y=sig, sr=sr)
    rms = np.mean(librosa.feature.rms(y=sig))
    centroid = np.mean(librosa.feature.spectral_centroid(y=sig, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=sig))
    mfcc = np.mean(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13))
    return [float(tempo), float(rms), float(centroid), float(zcr), float(mfcc)]

def get_features_from_path(file_path):
    try:
        sig, sr = librosa.load(file_path, duration=30, mono=True)
        return np.array(feats(sig, sr), dtype=np.float32)
    except:
        return np.zeros(5, dtype=np.float32)

def all_feats():
    files = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav")]
    if not files:
        print("No WAV files found")
        return

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["title","tempo","rms","centroid","zcr","mfcc_mean"])
        for fn in files:
            path = os.path.join(WAV_DIR, fn)
            sig, sr = sf.read(path, always_2d=False)
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            w.writerow([fn[:-4]] + feats(sig, sr))

if __name__ == "__main__":
    all_feats()