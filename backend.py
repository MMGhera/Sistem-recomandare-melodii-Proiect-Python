from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from threading import Lock
from typing import List
import json

app = FastAPI(title="Local Music Preferences Backend")

# ---- FILES ----
DATA_FILE = Path("users.json")
SONGS_FILE = Path("songs.json")
DATA_LOCK = Lock()  # thread-safe writes

# ---- Ensure files exist ----
if not DATA_FILE.exists():
    DATA_FILE.write_text("{}")  # empty JSON

if not SONGS_FILE.exists():
    # Example song list, you can expand this
    SONGS_FILE.write_text(json.dumps([
        "Metallica - Enter Sandman",
        "Metallica - One",
        "Iron Maiden - The Trooper",
        "Nirvana - Smells Like Teen Spirit",
        "AC/DC - Back In Black",
        "Megadeth - Symphony of Destruction"
    ], indent=4))

# ---- Load songs once ----
with SONGS_FILE.open("r", encoding="utf-8") as f:
    ALL_SONGS = json.load(f)

# ---- MODELS ----
class LoginRequest(BaseModel):
    username: str

class PrefsRequest(BaseModel):
    username: str
    songs: list[str]

# ---- HELPER FUNCTIONS ----
def load_data():
    with DATA_LOCK:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

def save_data(data: dict):
    with DATA_LOCK:
        with DATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

# ---- ENDPOINTS ----
@app.post("/login")
def login(request: LoginRequest):
    data = load_data()
    if request.username not in data:
        data[request.username] = {"songs": []}
        save_data(data)
    return {"status": "ok", "username": request.username}

@app.get("/prefs/{username}")
def get_prefs(username: str):
    data = load_data()
    if username not in data:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": username, "songs": data[username]["songs"]}

@app.post("/prefs")
def save_prefs(request: PrefsRequest):
    data = load_data()
    if request.username not in data:
        raise HTTPException(status_code=404, detail="User not found")
    data[request.username]["songs"] = request.songs
    save_data(data)
    return {"status": "ok", "username": request.username, "songs": request.songs}

@app.get("/autocomplete", response_model=List[str])
def autocomplete(q: str = Query(..., min_length=1)):
    """
    Return list of song names that contain the query string (case-insensitive).
    """
    query_lower = q.lower()
    results = [song for song in ALL_SONGS if query_lower in song.lower()]
    return results[:10]  # return top 10 matches only

@app.get("/")
def root():
    return {"message": "Backend is running"}

