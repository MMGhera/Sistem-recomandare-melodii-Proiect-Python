#V4V4
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import torch
import numpy as np
import librosa
# import ai.dataset
# # Îi spunem lui Python: "Când cineva caută 'dataset', dă-i 'ai.dataset'"
# sys.modules['dataset'] = ai.dataset

# --- SETUP IMPORTURI AI ---
sys.path.append(os.path.dirname(__file__))

# 1. Importăm modulul dataset (cel care este folderul colegului)
import dataset

try:
    # Importăm componentele AI din locația lor reală
    from ai.model import MusiCNN, INSTRUMENT_MAP
    # Importăm ȘI MelConfig explicit
    from ai.dataset import generate_melspectrogram, MelConfig

    # Îi spunem Python-ului: "Dacă cineva caută MelConfig în dataset, dă-i-l pe ăsta!"
    dataset.MelConfig = MelConfig

except ImportError as e:
    print(f"⚠️ Warning importuri AI: {e}")

from dataset.extract import get_features_from_path

import requests
import uuid # Pentru nume unice la fișiere temporare
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
from sqlalchemy.orm import Session
from sklearn.preprocessing import normalize

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
    # Verificări preliminare AI
    model = ai_context["model"]
    cfg = ai_context["config"]
    device = ai_context["device"]

    if model is None:
        return None

    try:
        # --- PARTEA 1: AI (Deep Learning - MusiCNN) ---
        # 1. Încărcare Audio
        audio, sr = librosa.load(file_path, sr=None)  # Încărcăm tot fișierul pentru AI

        # 2. Resample
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=cfg["mel_config"].sample_rate)

        # 3. Spectrogramă
        mel = generate_melspectrogram(audio_resampled, cfg["mel_config"])

        # 4. Chunking
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
            ai_vector = np.zeros(128)  # Fallback dacă e gol (depinde de output-ul modelului tău)
        else:
            # 5. Predicție
            batch_tensor = np.stack(chunks)
            batch_tensor = torch.from_numpy(batch_tensor).float().unsqueeze(1).to(device)

            with torch.no_grad():
                logits = model(batch_tensor)
                probs = logits.cpu().numpy()

            # 6. Max Pooling
            ai_vector = np.max(probs, axis=0)

        # --- PARTEA 2: CLASIC (Metoda Colegului) ---
        # Aici apelăm funcția importată din dataset/extract.py
        classic_vector = get_features_from_path(file_path)

        # --- PARTEA 3: FUZIUNEA (Concatenare + Normalizare) ---

        # Reshape pentru sklearn (vrea matrice 2D)
        ai_vector_2d = ai_vector.reshape(1, -1)
        classic_vector_2d = classic_vector.reshape(1, -1)

        # Normalizare L2 (aduce vectorii la scară comună)
        ai_norm = normalize(ai_vector_2d, axis=1, norm='l2')[0]
        classic_norm = normalize(classic_vector_2d, axis=1, norm='l2')[0]

        # Concatenare
        final_vector = np.concatenate([ai_norm, classic_norm])

        print(f"✅ Analizat hibrid: {os.path.basename(file_path)} (Len: {len(final_vector)})")
        return final_vector.tolist()

    except Exception as e:
        print(f"⚠️ Eroare la analiză hibridă {file_path}: {e}")
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