import os
import requests
from pydub import AudioSegment
import extract
import soundfile as sf
import numpy as np
import csv

ARTIST = "-"
TOP = 10
MY_SONG = "test_song.wav"

MP3 = "mp3"
WAV = "wav"
CSV = "features.csv"

os.makedirs(MP3, exist_ok=True)
os.makedirs(WAV, exist_ok=True)

# remove old files
for folder in [MP3, WAV]:
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))

res = requests.get(f"https://api.deezer.com/search?q={ARTIST}")
artist_id = res.json()["data"][0]["artist"]["id"]
tracks = requests.get(f"https://api.deezer.com/artist/{artist_id}/top?limit={TOP}").json()["data"]

for t in tracks:
    title = "".join(c for c in t['title'] if c.isalnum() or c in " _-")
    mp3_file = os.path.join(MP3, f"{title}.mp3")
    wav_file = os.path.join(WAV, f"{title}.wav")
    with open(mp3_file, "wb") as f:
        f.write(requests.get(t["preview"]).content)
    AudioSegment.from_mp3(mp3_file).export(wav_file, format="wav")
    print(f"{title} downloaded")

extract.all_feats()
print("Features extracted")

def norm(v):
    return v / np.linalg.norm(v)

if os.path.exists(MY_SONG):
    sig, sr = sf.read(MY_SONG, always_2d=False)
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)

    user_f = norm(np.array(extract.feats(sig, sr)))
    best, best_score = None, -1

    with open(CSV, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            title = row[0]
            feats_arr = norm(np.array([float(x) for x in row[1:]]))
            score = np.dot(user_f, feats_arr)
            print(f"{title}: {score:.3f}")
            if score > best_score:
                best, best_score = title, score

    print(f"\nBest match: {best} ({best_score:.3f})")
else:
    print("No song found")
