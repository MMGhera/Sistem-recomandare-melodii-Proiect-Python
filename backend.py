# from fastapi import FastAPI, HTTPException, Query
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# from threading import Lock
# from typing import List
# import json
#
# app = FastAPI(title="Local Music Preferences Backend")
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Allow your Vite frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # ---- FILES ----; replace later????
# DATA_FILE = Path("users.json")
# SONGS_FILE = Path("songs.json")
# DATA_LOCK = Lock()  # thread-safe writes
#
# # ---- Ensure files exist ----
# if not DATA_FILE.exists():
#     DATA_FILE.write_text("{}")  # empty JSON
#
# if not SONGS_FILE.exists():
#     # Example song list; replace with db
#     SONGS_FILE.write_text(json.dumps([
#         "Metallica - Enter Sandman",
#         "Metallica - One",
#         "Iron Maiden - The Trooper",
#         "Nirvana - Smells Like Teen Spirit",
#         "AC/DC - Back In Black",
#         "Megadeth - Symphony of Destruction"
#     ], indent=4))
#
# # ---- Load songs once ----
# with SONGS_FILE.open("r", encoding="utf-8") as f:
#     ALL_SONGS = json.load(f)
#
# # ---- MODELS ----
# class LoginRequest(BaseModel):
#     username: str
#
# class PrefsRequest(BaseModel):
#     username: str
#     songs: list[str]
#
# # ---- HELPER FUNCTIONS ----
# def load_data():
#     with DATA_LOCK:
#         with DATA_FILE.open("r", encoding="utf-8") as f:
#             return json.load(f)
#
# def save_data(data: dict):
#     with DATA_LOCK:
#         with DATA_FILE.open("w", encoding="utf-8") as f:
#             json.dump(data, f, indent=4)
#
# # ---- ENDPOINTS ----
# @app.post("/login")
# def login(request: LoginRequest):
#     data = load_data()
#     if request.username not in data:
#         data[request.username] = {"songs": []}
#         save_data(data)
#     return {"status": "ok", "username": request.username}
#
# @app.get("/prefs/{username}")
# def get_prefs(username: str):
#     data = load_data()
#     if username not in data:
#         raise HTTPException(status_code=404, detail="User not found")
#     return {"username": username, "songs": data[username]["songs"]}
#
# @app.post("/prefs")
# def save_prefs(request: PrefsRequest):
#     data = load_data()
#     if request.username not in data:
#         raise HTTPException(status_code=404, detail="User not found")
#     data[request.username]["songs"] = request.songs
#     save_data(data)
#     return {"status": "ok", "username": request.username, "songs": request.songs}
#
# @app.get("/autocomplete", response_model=List[str])
# def autocomplete(q: str = Query(..., min_length=1)):
#     """
#     Return list of song names that contain the query string (case-insensitive).
#     """
#     query_lower = q.lower()
#     results = [song for song in ALL_SONGS if query_lower in song.lower()]
#     return results[:10]  # return only top 10 matches
#
# @app.get("/")
# def root():
#     return {"message": "Backend is running"}
#
# if __name__ == "__main__":
#     import uvicorn
#     # Run the server on localhost:8000
#     uvicorn.run(app, host="127.0.0.1", port=8000)

#V2V2V2V2V2V2V2V2V2V2V2V2V2V2V2V2
# from fastapi import FastAPI, HTTPException, Query
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# from threading import Lock
# from typing import List
# import json
# import uvicorn
#
# app = FastAPI(title="Local Music Preferences Backend")
#
# # --- CONFIGURARE CORS ---
# # Permitem frontend-ului de pe portul 5173 (Vite) să vorbească cu backend-ul
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # --- GESTIONARE FIȘIERE ---
# DATA_FILE = Path("users.json")
# SONGS_FILE = Path("songs.json")
# DATA_LOCK = Lock()  # Previne erorile dacă două cereri vin simultan
#
# # 1. Inițializare fișier utilizatori (dacă nu există)
# if not DATA_FILE.exists():
#     DATA_FILE.write_text("{}")
#
# # 2. Inițializare fișier melodii (dacă nu există, creăm o listă default)
# if not SONGS_FILE.exists():
#     default_songs = [
#         "Metallica - Enter Sandman",
#         "Metallica - One",
#         "Iron Maiden - The Trooper",
#         "Nirvana - Smells Like Teen Spirit",
#         "AC/DC - Back In Black",
#         "Megadeth - Symphony of Destruction",
#         "Pink Floyd - Comfortably Numb",
#         "Led Zeppelin - Stairway to Heaven",
#         "Queen - Bohemian Rhapsody",
#         "Black Sabbath - Paranoid",
#         "Guns N' Roses - Sweet Child O' Mine"
#     ]
#     SONGS_FILE.write_text(json.dumps(default_songs, indent=4))
#
# # Încărcăm melodiile în memorie la pornire (pentru viteză la autocomplete)
# with SONGS_FILE.open("r", encoding="utf-8") as f:
#     ALL_SONGS = json.load(f)
#
#
# # --- MODELE DE DATE (Pydantic) ---
# class LoginRequest(BaseModel):
#     username: str
#
#
# class PrefsRequest(BaseModel):
#     username: str
#     songs: List[str]
#
#
# # --- FUNCȚII AJUTĂTOARE ---
# def load_users_data():
#     """Citește baza de date cu utilizatori."""
#     with DATA_LOCK:
#         try:
#             with DATA_FILE.open("r", encoding="utf-8") as f:
#                 content = f.read().strip()
#                 return json.loads(content) if content else {}
#         except json.JSONDecodeError:
#             return {}  # Returnăm dict gol dacă fișierul e corupt
#
#
# def save_users_data(data: dict):
#     """Scrie baza de date cu utilizatori."""
#     with DATA_LOCK:
#         with DATA_FILE.open("w", encoding="utf-8") as f:
#             json.dump(data, f, indent=4)
#
#
# # --- ENDPOINTS (RUTELE) ---
#
# @app.get("/")
# def root():
#     return {"message": "Music Backend is running correctly"}
#
#
# # 1. Login (sau Register automat dacă userul nu există)
# @app.post("/login")
# def login(request: LoginRequest):
#     data = load_users_data()
#     # Dacă e utilizator nou, îl creăm cu lista goală
#     if request.username not in data:
#         data[request.username] = {"songs": []}
#         save_users_data(data)
#     return {"status": "ok", "username": request.username}
#
#
# # 2. Obține preferințele unui utilizator
# @app.get("/prefs/{username}")
# def get_prefs(username: str):
#     data = load_users_data()
#     if username not in data:
#         # Dacă userul nu există, nu dăm eroare, ci returnăm listă goală (mai sigur pt frontend)
#         return {"username": username, "songs": []}
#     return {"username": username, "songs": data[username]["songs"]}
#
#
# # 3. Salvează preferințele (Lista completă de melodii a userului)
# @app.post("/prefs")
# def save_prefs(request: PrefsRequest):
#     data = load_users_data()
#
#     # Asigurăm că userul există înainte să salvăm
#     if request.username not in data:
#         data[request.username] = {"songs": []}
#
#     data[request.username]["songs"] = request.songs
#     save_users_data(data)
#     return {"status": "ok", "username": request.username, "saved_songs_count": len(request.songs)}
#
#
# # 4. Autocomplete (Căutare melodii)
# @app.get("/autocomplete", response_model=List[str])
# def autocomplete(q: str = Query(..., min_length=1)):
#     """
#     Returnează primele 10 melodii care conțin textul căutat 'q'.
#     """
#     query_lower = q.lower()
#     # Filtrăm lista de melodii încărcată în memorie
#     results = [song for song in ALL_SONGS if query_lower in song.lower()]
#     return results[:10]
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)


#V3V3V3V3V3V3V3V3V3V3
import sys
import os
import json
import torch
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

# --- 1. SETUP IMPORTURI AI ---
# Adăugăm folderul 'ai' la path pentru a putea importa modulele tale
sys.path.append(os.path.join(os.path.dirname(__file__), "ai"))

# Importăm clasele din fișierele tale
from ai.model import MusiCNN, INSTRUMENT_MAP
from ai.dataset import generate_melspectrogram, MelConfig

# --- 2. CONFIGURARE GLOBALA ---
AI_MODEL_PATH = "ai/checkpoints/big_sample_rate/best.pt"  # Verifică calea exactă!
AUDIO_LIBRARY_PATH = "audio_library"  # Folderul unde pui mp3-urile
VECTORS_FILE = Path("song_vectors.json")
DATA_FILE = Path("users.json")

# Variabile globale pentru model
model_instance = None
ai_config = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 3. LIFESPAN (Rulează la pornirea serverului) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Aici încărcăm modelul o singură dată, la start
    global model_instance, ai_config

    if os.path.exists(AI_MODEL_PATH):
        print(f"🔄 Loading AI Model from {AI_MODEL_PATH} on {device}...")
        try:
            # Logica de încărcare preluată din test.py
            # Adăugăm weights_only=False pentru a permite încărcarea claselor custom (MelConfig)
            state = torch.load(AI_MODEL_PATH, map_location=device, weights_only=False)

            mel_conf = state["mel_config"]
            # Reconstruim modelul
            model = MusiCNN(num_classes=len(INSTRUMENT_MAP), num_mels=mel_conf.n_mels)
            model.load_state_dict(state["model"])
            model.to(device)
            model.eval()

            model_instance = model
            ai_config = {
                "mel_config": mel_conf,
                "frames_per_window": state["frames_per_window"],
                "instrument_list": state["instrument_list"]
            }
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Warning: Model file not found at {AI_MODEL_PATH}")

    yield  # Aici rulează aplicația

    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pentru dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 4. FUNCȚIA DE ANALIZĂ (PRELUATĂ DIN TEST.PY) ---
def analyze_audio_file(file_path):
    if model_instance is None:
        raise Exception("AI Model not loaded")

    # Load audio
    audio, sr = librosa.load(file_path, sr=None)

    # Configs
    mel_conf = ai_config["mel_config"]
    frames_win = ai_config["frames_per_window"]

    # Resample & Mel Spectrogram
    audio = librosa.resample(audio, orig_sr=sr, target_sr=mel_conf.sample_rate)
    mel = generate_melspectrogram(audio, mel_conf)

    # Chunking logic (exact ca in test.py)
    chunks = []
    total_frames = mel.shape[1]
    for offset in range(0, total_frames, frames_win):
        chunk = mel[:, offset:offset + frames_win]
        if chunk.shape[1] < frames_win:
            pad_width = frames_win - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
        chunks.append(chunk)

    if not chunks:
        return None

    # Inference
    batch_tensor = np.stack(chunks)
    batch_tensor = torch.from_numpy(batch_tensor).float().unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model_instance(batch_tensor)
        probs = logits.cpu().numpy()

    # Max Pooling over time (Song-level prediction)
    song_vector = np.max(probs, axis=0)
    return song_vector.tolist()  # Returnăm ca listă Python simplă


# --- 5. ENDPOINTS ---

class LoginRequest(BaseModel):
    username: str


class PrefsRequest(BaseModel):
    username: str
    songs: list[str]


def load_vectors():
    if VECTORS_FILE.exists():
        with open(VECTORS_FILE, 'r') as f:
            return json.load(f)
    return {}


def load_users():
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)


@app.post("/login")
def login(request: LoginRequest):
    data = load_users()
    if request.username not in data:
        data[request.username] = {"songs": []}
        save_users(data)
    return {"status": "ok", "username": request.username}


@app.get("/prefs/{username}")
def get_prefs(username: str):
    data = load_users()
    if username not in data: return {"songs": []}
    return {"songs": data[username]["songs"]}


@app.post("/prefs")
def save_prefs(request: PrefsRequest):
    data = load_users()
    if request.username not in data: data[request.username] = {"songs": []}
    data[request.username]["songs"] = request.songs
    save_users(data)
    return {"status": "ok"}


@app.get("/autocomplete")
def autocomplete(q: str):
    # Căutăm în fișierul de vectori (care reprezintă biblioteca analizată)
    vectors = load_vectors()
    all_songs = list(vectors.keys())
    results = [s for s in all_songs if q.lower() in s.lower()]
    return results[:10]


# --- RUTA MAGICĂ: SCANARE BIBLIOTECĂ ---
@app.post("/scan_library")
def scan_library():
    """
    Citește folderul 'audio_library', analizează fiecare melodie cu AI-ul
    și salvează vectorii în song_vectors.json.
    """
    if not os.path.exists(AUDIO_LIBRARY_PATH):
        return {"error": f"Folderul {AUDIO_LIBRARY_PATH} nu există. Creează-l și pune muzică."}

    vectors = load_vectors()
    processed_count = 0

    for file in os.listdir(AUDIO_LIBRARY_PATH):
        if file.endswith((".mp3", ".wav", ".m4a", ".flac")):
            song_name = os.path.splitext(file)[0]  # Numele fișierului fără extensie

            # Analizăm doar dacă nu există deja
            if song_name not in vectors:
                print(f"🎵 Analyzing: {song_name}...")
                try:
                    full_path = os.path.join(AUDIO_LIBRARY_PATH, file)
                    vec = analyze_audio_file(full_path)
                    if vec:
                        vectors[song_name] = vec
                        processed_count += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # Salvăm baza de date cu vectori
    with open(VECTORS_FILE, 'w') as f:
        json.dump(vectors, f)

    return {"status": "completed", "new_songs_analyzed": processed_count, "total_songs": len(vectors)}


# --- RUTA MAGICĂ: RECOMANDARE ---
@app.get("/recommend/{username}")
def recommend(username: str):
    users = load_users()
    vectors = load_vectors()

    if username not in users or not users[username]["songs"]:
        return {"recommendations": ["Alege câteva melodii mai întâi!"]}

    user_songs = users[username]["songs"]

    # 1. Calculăm Profilul Utilizatorului (Media vectorilor melodiilor preferate)
    user_vector_sum = None
    count = 0

    for song_name in user_songs:
        if song_name in vectors:
            vec = np.array(vectors[song_name])
            if user_vector_sum is None:
                user_vector_sum = vec
            else:
                user_vector_sum += vec
            count += 1

    if count == 0:
        return {"recommendations": ["Nu am date analizate pentru melodiile tale. Rulează /scan_library."]}

    user_profile = user_vector_sum / count  # Media

    # 2. Căutăm melodii similare (Cosine Similarity)
    scores = []
    for song_name, vec_list in vectors.items():
        if song_name in user_songs: continue  # Excludem ce ascultă deja

        song_vec = np.array(vec_list)

        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        similarity = np.dot(user_profile, song_vec) / (np.linalg.norm(user_profile) * np.linalg.norm(song_vec))

        scores.append((song_name, similarity))

    # 3. Sortăm descrescător după similaritate
    scores.sort(key=lambda x: x[1], reverse=True)

    # Returnăm top 5
    return {"recommendations": [s[0] for s in scores[:5]]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)