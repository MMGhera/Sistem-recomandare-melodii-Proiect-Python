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
# import sys
# import os
# import json
# import torch
# import numpy as np
# import librosa
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from pathlib import Path
# from contextlib import asynccontextmanager
# from typing import List
#
# # --- 1. SETUP IMPORTURI AI ---
# # Adăugăm folderul 'ai' la path pentru a putea importa modulele tale
# sys.path.append(os.path.join(os.path.dirname(__file__), "ai"))
#
# # Importăm clasele din fișierele tale
# from ai.model import MusiCNN, INSTRUMENT_MAP
# from ai.dataset import generate_melspectrogram, MelConfig
#
# # --- 2. CONFIGURARE GLOBALA ---
# AI_MODEL_PATH = "ai/checkpoints/big_sample_rate/best.pt"  # Verifică calea exactă!
# AUDIO_LIBRARY_PATH = "audio_library"  # Folderul unde pui mp3-urile
# VECTORS_FILE = Path("song_vectors.json")
# DATA_FILE = Path("users.json")
#
# # Variabile globale pentru model
# model_instance = None
# ai_config = {}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # --- 3. LIFESPAN (Rulează la pornirea serverului) ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Aici încărcăm modelul o singură dată, la start
#     global model_instance, ai_config
#
#     if os.path.exists(AI_MODEL_PATH):
#         print(f"🔄 Loading AI Model from {AI_MODEL_PATH} on {device}...")
#         try:
#             # Logica de încărcare preluată din test.py
#             # Adăugăm weights_only=False pentru a permite încărcarea claselor custom (MelConfig)
#             state = torch.load(AI_MODEL_PATH, map_location=device, weights_only=False)
#
#             mel_conf = state["mel_config"]
#             # Reconstruim modelul
#             model = MusiCNN(num_classes=len(INSTRUMENT_MAP), num_mels=mel_conf.n_mels)
#             model.load_state_dict(state["model"])
#             model.to(device)
#             model.eval()
#
#             model_instance = model
#             ai_config = {
#                 "mel_config": mel_conf,
#                 "frames_per_window": state["frames_per_window"],
#                 "instrument_list": state["instrument_list"]
#             }
#             print("✅ Model loaded successfully!")
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#     else:
#         print(f"⚠️ Warning: Model file not found at {AI_MODEL_PATH}")
#
#     yield  # Aici rulează aplicația
#
#     print("Shutting down...")
#
#
# app = FastAPI(lifespan=lifespan)
#
# # --- CORS ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Pentru dev
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # --- 4. FUNCȚIA DE ANALIZĂ (PRELUATĂ DIN TEST.PY) ---
# def analyze_audio_file(file_path):
#     if model_instance is None:
#         raise Exception("AI Model not loaded")
#
#     # Load audio
#     audio, sr = librosa.load(file_path, sr=None)
#
#     # Configs
#     mel_conf = ai_config["mel_config"]
#     frames_win = ai_config["frames_per_window"]
#
#     # Resample & Mel Spectrogram
#     audio = librosa.resample(audio, orig_sr=sr, target_sr=mel_conf.sample_rate)
#     mel = generate_melspectrogram(audio, mel_conf)
#
#     # Chunking logic (exact ca in test.py)
#     chunks = []
#     total_frames = mel.shape[1]
#     for offset in range(0, total_frames, frames_win):
#         chunk = mel[:, offset:offset + frames_win]
#         if chunk.shape[1] < frames_win:
#             pad_width = frames_win - chunk.shape[1]
#             chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
#         chunks.append(chunk)
#
#     if not chunks:
#         return None
#
#     # Inference
#     batch_tensor = np.stack(chunks)
#     batch_tensor = torch.from_numpy(batch_tensor).float().unsqueeze(1).to(device)
#
#     with torch.no_grad():
#         logits = model_instance(batch_tensor)
#         probs = logits.cpu().numpy()
#
#     # Max Pooling over time (Song-level prediction)
#     song_vector = np.max(probs, axis=0)
#     return song_vector.tolist()  # Returnăm ca listă Python simplă
#
#
# # --- 5. ENDPOINTS ---
#
# class LoginRequest(BaseModel):
#     username: str
#
#
# class PrefsRequest(BaseModel):
#     username: str
#     songs: list[str]
#
#
# def load_vectors():
#     if VECTORS_FILE.exists():
#         with open(VECTORS_FILE, 'r') as f:
#             return json.load(f)
#     return {}
#
#
# def load_users():
#     if DATA_FILE.exists():
#         with open(DATA_FILE, 'r') as f:
#             return json.load(f)
#     return {}
#
#
# def save_users(data):
#     with open(DATA_FILE, 'w') as f:
#         json.dump(data, f, indent=4)
#
#
# @app.post("/login")
# def login(request: LoginRequest):
#     data = load_users()
#     if request.username not in data:
#         data[request.username] = {"songs": []}
#         save_users(data)
#     return {"status": "ok", "username": request.username}
#
#
# @app.get("/prefs/{username}")
# def get_prefs(username: str):
#     data = load_users()
#     if username not in data: return {"songs": []}
#     return {"songs": data[username]["songs"]}
#
#
# @app.post("/prefs")
# def save_prefs(request: PrefsRequest):
#     data = load_users()
#     if request.username not in data: data[request.username] = {"songs": []}
#     data[request.username]["songs"] = request.songs
#     save_users(data)
#     return {"status": "ok"}
#
#
# @app.get("/autocomplete")
# def autocomplete(q: str):
#     # Căutăm în fișierul de vectori (care reprezintă biblioteca analizată)
#     vectors = load_vectors()
#     all_songs = list(vectors.keys())
#     results = [s for s in all_songs if q.lower() in s.lower()]
#     return results[:10]
#
#
# # --- RUTA MAGICĂ: SCANARE BIBLIOTECĂ ---
# @app.post("/scan_library")
# def scan_library():
#     """
#     Citește folderul 'audio_library', analizează fiecare melodie cu AI-ul
#     și salvează vectorii în song_vectors.json.
#     """
#     if not os.path.exists(AUDIO_LIBRARY_PATH):
#         return {"error": f"Folderul {AUDIO_LIBRARY_PATH} nu există. Creează-l și pune muzică."}
#
#     vectors = load_vectors()
#     processed_count = 0
#
#     for file in os.listdir(AUDIO_LIBRARY_PATH):
#         if file.endswith((".mp3", ".wav", ".m4a", ".flac")):
#             song_name = os.path.splitext(file)[0]  # Numele fișierului fără extensie
#
#             # Analizăm doar dacă nu există deja
#             if song_name not in vectors:
#                 print(f"🎵 Analyzing: {song_name}...")
#                 try:
#                     full_path = os.path.join(AUDIO_LIBRARY_PATH, file)
#                     vec = analyze_audio_file(full_path)
#                     if vec:
#                         vectors[song_name] = vec
#                         processed_count += 1
#                 except Exception as e:
#                     print(f"Error processing {file}: {e}")
#
#     # Salvăm baza de date cu vectori
#     with open(VECTORS_FILE, 'w') as f:
#         json.dump(vectors, f)
#
#     return {"status": "completed", "new_songs_analyzed": processed_count, "total_songs": len(vectors)}
#
#
# # --- RUTA MAGICĂ: RECOMANDARE ---
# @app.get("/recommend/{username}")
# def recommend(username: str):
#     users = load_users()
#     vectors = load_vectors()
#
#     if username not in users or not users[username]["songs"]:
#         return {"recommendations": ["Alege câteva melodii mai întâi!"]}
#
#     user_songs = users[username]["songs"]
#
#     # 1. Calculăm Profilul Utilizatorului (Media vectorilor melodiilor preferate)
#     user_vector_sum = None
#     count = 0
#
#     for song_name in user_songs:
#         if song_name in vectors:
#             vec = np.array(vectors[song_name])
#             if user_vector_sum is None:
#                 user_vector_sum = vec
#             else:
#                 user_vector_sum += vec
#             count += 1
#
#     if count == 0:
#         return {"recommendations": ["Nu am date analizate pentru melodiile tale. Rulează /scan_library."]}
#
#     user_profile = user_vector_sum / count  # Media
#
#     # 2. Căutăm melodii similare (Cosine Similarity)
#     scores = []
#     for song_name, vec_list in vectors.items():
#         if song_name in user_songs: continue  # Excludem ce ascultă deja
#
#         song_vec = np.array(vec_list)
#
#         # Cosine Similarity: (A . B) / (||A|| * ||B||)
#         similarity = np.dot(user_profile, song_vec) / (np.linalg.norm(user_profile) * np.linalg.norm(song_vec))
#
#         scores.append((song_name, similarity))
#
#     # 3. Sortăm descrescător după similaritate
#     scores.sort(key=lambda x: x[1], reverse=True)
#
#     # Returnăm top 5
#     return {"recommendations": [s[0] for s in scores[:5]]}
#
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="127.0.0.1", port=8000)

#V4V4
import sys
import os
import json
import torch
import numpy as np
import librosa
import ai.dataset
# Îi spunem lui Python: "Când cineva caută 'dataset', dă-i 'ai.dataset'"
sys.modules['dataset'] = ai.dataset

import requests
import uuid # Pentru nume unice la fișiere temporare
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
from sqlalchemy.orm import Session

# --- IMPORTURI DB ---
from models import create_tables, get_db, User, Song, UserPreference, SessionLocal

# --- SETUP IMPORTURI AI ---
sys.path.append(os.path.dirname(__file__))
try:
    from ai.model import MusiCNN, INSTRUMENT_MAP
    from ai.dataset import generate_melspectrogram
except ImportError:
    pass  # Ignorăm dacă nu merge importul local, doar pentru test

# --- CONFIGURARE AI ---
AI_MODEL_PATH = os.path.join("ai", "checkpoints", "big_sample_rate", "best.pt")
AUDIO_LIBRARY_PATH = "audio_library"
ai_context = {"model": None, "config": None, "device": None}


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Inițializăm Baza de Date SQL
    create_tables()
    print("💾 Baza de date SQL conectată.")

    # 2. Încărcăm AI-ul
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(AI_MODEL_PATH):
        try:
            state = torch.load(AI_MODEL_PATH, map_location=device, weights_only=False)
            mel_conf = state["mel_config"]
            instrument_list = state["instrument_list"]
            model = MusiCNN(num_classes=len(instrument_list), num_mels=mel_conf.n_mels)
            model.load_state_dict(state["model"])
            model.to(device)
            model.eval()

            ai_context.update({"model": model, "device": device, "config": {
                "mel_config": mel_conf, "frames_per_window": state["frames_per_window"]
            }})
            print("✅ Model AI încărcat.")
        except Exception as e:
            print(f"❌ Eroare AI: {e}")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# --- LOGICA AI (Aceeași ca înainte) ---

def analyze_audio_file(file_path):
    # Verificări preliminare
    model = ai_context["model"]
    cfg = ai_context["config"]
    device = ai_context["device"]

    if model is None:
        return None

    try:
        # 1. Încărcare Audio (cu librosa)
        # Folosind try-catch intern pentru a prinde erorile specifice de codec
        audio, sr = librosa.load(file_path, sr=None)

        # 2. Resample la frecvența modelului
        audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg["mel_config"].sample_rate)

        # 3. Generare Spectrogramă
        mel = generate_melspectrogram(audio, cfg["mel_config"])

        # 4. Împărțire în bucăți (Chunking)
        frames_win = cfg["frames_per_window"]
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

        # 5. Predicție AI
        batch_tensor = np.stack(chunks)
        batch_tensor = torch.from_numpy(batch_tensor).float().unsqueeze(1).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = logits.cpu().numpy()

        # 6. Agregare rezultate (Max Pooling)
        song_vector = np.max(probs, axis=0)
        return song_vector.tolist()

    except Exception as e:
        # Dacă apare o eroare, o afișăm discret în consolă și returnăm None
        # astfel încât scanarea să continue cu următoarea melodie
        print(f"⚠️ Nu s-a putut analiza {os.path.basename(file_path)}: {e}")
        return None


# --- MODELE PYDANTIC (Pentru API) ---
class LoginRequest(BaseModel):
    username: str


class PrefsRequest(BaseModel):
    username: str
    songs: list[str]


# --- ENDPOINTS NOI CU SQL ---

@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    # Căutăm userul în SQL
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        # Creăm user nou
        new_user = User(username=request.username)
        db.add(new_user)
        db.commit()
    return {"status": "ok", "username": request.username}


@app.get("/prefs/{username}")
def get_prefs(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"songs": []}

    # Extragem numele melodiilor din tabelul de preferințe
    song_titles = [pref.song.title for pref in user.preferences]
    return {"songs": song_titles}


@app.post("/prefs")
def save_prefs(request: PrefsRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Ștergem preferințele vechi (cea mai simplă metodă de update)
    db.query(UserPreference).filter(UserPreference.user_id == user.id).delete()

    # Adăugăm noile preferințe
    for song_name in request.songs:
        # Verificăm dacă melodia există în baza de date
        song = db.query(Song).filter(Song.title == song_name).first()
        if song:
            new_pref = UserPreference(user_id=user.id, song_id=song.id)
            db.add(new_pref)

    db.commit()
    return {"status": "ok"}


@app.get("/autocomplete")
def autocomplete(q: str, db: Session = Depends(get_db)):
    # Căutare SQL (LIKE %q%)
    songs = db.query(Song).filter(Song.title.contains(q)).limit(10).all()
    return [s.title for s in songs]


@app.post("/scan_library")
def scan_library(db: Session = Depends(get_db)):
    # Importăm logica de analiză din contextul global sau funcția definită
    # Nota: Trebuie să incluzi funcția analyze_audio_file completă în acest fișier
    from ai.dataset import generate_melspectrogram  # Re-import pt siguranță

    if not os.path.exists(AUDIO_LIBRARY_PATH):
        return {"error": "No audio folder"}

    processed = 0
    files = os.listdir(AUDIO_LIBRARY_PATH)

    for file in files:
        if file.endswith((".mp3", ".wav", ".m4a")):
            song_name = os.path.splitext(file)[0]

            # Verificăm dacă există deja în SQL
            exists = db.query(Song).filter(Song.title == song_name).first()
            if not exists:
                print(f"🎵 Analizez: {song_name}...")
                full_path = os.path.join(AUDIO_LIBRARY_PATH, file)

                # AICI apelăm AI-ul tău
                # vector = analyze_audio_file(full_path)
                # (Simulare vector pentru exemplul DB - tu decomentează analiza reală)
                #vector = [0.1, 0.2, 0.3]  # Placeholder dacă nu merge analiza pe moment
                try:
                    vector = analyze_audio_file(full_path)
                except Exception as e:
                    print(f"Eroare la analiza {song_name}: {e}")
                    vector = None

                if vector:
                    new_song = Song(
                        title=song_name,
                        vector_data=json.dumps(vector)  # Salvăm vectorul ca text JSON
                    )
                    db.add(new_song)
                    processed += 1
                    db.commit()  # Salvăm fiecare melodie pe rând

    return {"status": "completed", "new_songs": processed}


@app.get("/recommend/{username}")
def recommend(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.preferences:
        return {"recommendations": ["Adaugă preferințe!"]}

    # 1. Construim profilul userului
    user_vectors = []
    for pref in user.preferences:
        user_vectors.append(json.loads(pref.song.vector_data))

    if not user_vectors:
        return {"recommendations": []}

    # Media vectorilor
    user_profile = np.mean(np.array(user_vectors), axis=0)

    # 2. Luăm toate melodiile din DB
    all_songs = db.query(Song).all()
    scores = []

    my_song_ids = [p.song_id for p in user.preferences]

    for song in all_songs:
        if song.id in my_song_ids: continue  # Excludem ce ascultă deja

        song_vec = np.array(json.loads(song.vector_data))

        # Cosine Similarity
        similarity = np.dot(user_profile, song_vec) / (np.linalg.norm(user_profile) * np.linalg.norm(song_vec))

        if similarity > 0.35:  # Prag minim
            scores.append((song.title, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return {"recommendations": [s[0] for s in scores[:5]]}


# --- ENDPOINT NOU: ANALIZĂ EXTERNĂ LIVE ---
@app.get("/analyze_external")
def analyze_external(q: str, username: str, db: Session = Depends(get_db)):
    """
    Caută pe iTunes, analizează, SALVEAZĂ în DB și adaugă la PREFERINȚELE utilizatorului.
    """
    print(f"🌍 {username} caută și adaugă: {q}")

    # 1. Căutare pe iTunes
    itunes_url = f"https://itunes.apple.com/search?term={q}&media=music&entity=song&limit=1"
    try:
        resp = requests.get(itunes_url).json()
        if not resp["results"]:
            return {"error": "Melodia nu a fost găsită pe iTunes."}

        track = resp["results"][0]
        full_name = f"{track['artistName']} - {track['trackName']}"
        preview_url = track["previewUrl"]

        # 2. Gestionare Melodie în DB (Verificăm / Adăugăm)
        song_in_db = db.query(Song).filter(Song.title == full_name).first()

        target_vector = None

        if song_in_db:
            print("⚡ Melodia există deja. O refolosim.")
            target_vector = json.loads(song_in_db.vector_data)
        else:
            # Nu există -> O descărcăm și analizăm
            print("⬇️ Descarc și analizez melodia nouă...")
            temp_path = os.path.join(AUDIO_LIBRARY_PATH, f"temp_{uuid.uuid4()}.m4a")
            os.makedirs(AUDIO_LIBRARY_PATH, exist_ok=True)

            r = requests.get(preview_url)
            with open(temp_path, 'wb') as f:
                f.write(r.content)

            target_vector = analyze_audio_file(temp_path)

            if os.path.exists(temp_path):
                os.remove(temp_path)  # Ștergem fișierul audio, păstrăm doar matematica

            if target_vector:
                # SALVARE PERMANENTĂ ÎN BAZA DE DATE
                song_in_db = Song(title=full_name, vector_data=json.dumps(target_vector))
                db.add(song_in_db)
                db.commit()
                db.refresh(song_in_db)
            else:
                return {"error": "AI-ul nu a putut analiza fișierul."}

        # 3. Adăugare la Preferințele Utilizatorului (Link User <-> Song)
        user = db.query(User).filter(User.username == username).first()
        if user:
            # Verificăm dacă nu o are deja
            existing_pref = db.query(UserPreference).filter(
                UserPreference.user_id == user.id,
                UserPreference.song_id == song_in_db.id
            ).first()

            if not existing_pref:
                new_pref = UserPreference(user_id=user.id, song_id=song_in_db.id)
                db.add(new_pref)
                db.commit()
                print(f"✅ Adăugat '{full_name}' la preferințele lui {username}.")

    except Exception as e:
        return {"error": f"Eroare server: {str(e)}"}

    # 4. Recomandări (Bazat pe melodia tocmai adăugată)
    all_songs = db.query(Song).all()
    scores = []
    target_np = np.array(target_vector)

    for song in all_songs:
        if song.id == song_in_db.id: continue  # Nu ne recomandăm pe noi înșine

        song_vec = np.array(json.loads(song.vector_data))
        similarity = np.dot(target_np, song_vec) / (np.linalg.norm(target_np) * np.linalg.norm(song_vec))
        scores.append((song.title, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "source_song": full_name,
        "recommendations": [s[0] for s in scores[:5]],
        "added_to_library": True
    }


@app.get("/itunes_autocomplete")
def itunes_autocomplete(q: str):
    """
    Returnează o listă scurtă de sugestii de la iTunes (Titlu + Artist).
    """
    if not q or len(q) < 2:
        return []

    # Cerem doar 5 rezultate pentru viteză
    url = f"https://itunes.apple.com/search?term={q}&media=music&entity=song&limit=5"
    try:
        resp = requests.get(url).json()
        results = []
        for track in resp.get("results", []):
            # Formatăm frumos: "Artist - Piesă"
            display_name = f"{track['artistName']} - {track['trackName']}"
            results.append(display_name)
        # Eliminăm duplicatele (set) și returnăm lista
        return list(set(results))
    except:
        return []


# --- ENDPOINT ȘTERGERE PREFERINȚĂ ---
@app.delete("/pref")
def delete_pref(username: str, song: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"error": "User not found"}

    # Găsim melodia
    song_obj = db.query(Song).filter(Song.title == song).first()
    if song_obj:
        # Ștergem legătura dintre user și melodie
        db.query(UserPreference).filter(
            UserPreference.user_id == user.id,
            UserPreference.song_id == song_obj.id
        ).delete()
        db.commit()
        print(f"🗑️ {username} a șters: {song}")
        return {"status": "deleted"}

    return {"status": "song not found (ignored)"}

if __name__ == "__main__":
    import uvicorn
    # Asta ține programul deschis și ascultă cereri
    uvicorn.run(app, host="127.0.0.1", port=8000)