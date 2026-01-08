import requests
import os
import time

# --- CONFIGURARE ---
# Lista de artiști pe care vrei să îi cauți
ARTISTS = [
    "Metallica", "Eminem", "Mozart", "Dua Lipa", "Pink Floyd",
    "The Weeknd", "Hans Zimmer", "AC/DC", "Taylor Swift", "Skrillex",
    "Miles Davis", "Nirvana", "Queen", "Drake", "Rammstein"
]

AUDIO_LIBRARY_PATH = "audio_library"
SERVER_URL = "http://127.0.0.1:8000/scan_library"

# Asigură-te că folderul există
os.makedirs(AUDIO_LIBRARY_PATH, exist_ok=True)


def search_itunes(artist_name):
    """Caută melodii pe iTunes (gratuit, fără cheie API)."""
    url = f"https://itunes.apple.com/search?term={artist_name}&media=music&entity=song&limit=5"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["results"]
    except Exception as e:
        print(f"❌ Eroare conexiune iTunes: {e}")
    return []


def download_preview(preview_url, save_path):
    """Descarcă clipul de 30s."""
    try:
        r = requests.get(preview_url)
        with open(save_path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"❌ Eroare download: {e}")
        return False


def trigger_server_scan():
    """Îi spune serverului Python să analizeze ce e nou în folder."""
    try:
        print("⏳ Trimit cerere de analiză către Backend...")
        # Apelăm endpoint-ul POST pe care l-am creat deja
        r = requests.post(SERVER_URL)
        if r.status_code == 200:
            print("✅ Backend-ul a confirmat analiza!")
            return True
        else:
            print(f"⚠️ Backend-ul a răspuns cu eroare: {r.status_code}")
    except:
        print("❌ Nu mă pot conecta la Backend. Asigură-te că 'python backend.py' rulează!")


if __name__ == "__main__":
    print("--- 🎵 HARVESTER MUSIC AUTOMATION 🎵 ---")
    print("⚠️  IMPORTANT: Serverul backend trebuie să ruleze într-un alt terminal!")

    songs_downloaded = 0

    for artist in ARTISTS:
        print(f"\n🔍 Căutăm: {artist}...")
        results = search_itunes(artist)

        for song in results:
            # Curățăm numele fișierului de caractere ciudate (/ \ :)
            safe_name = f"{song['artistName']} - {song['trackName']}"
            safe_name = "".join([c for c in safe_name if c.isalnum() or c in " -_()"]).strip()
            filename = f"{safe_name}.m4a"

            file_path = os.path.join(AUDIO_LIBRARY_PATH, filename)

            # Verificăm dacă există deja fizic
            if os.path.exists(file_path):
                print(f"  ⏭️  Deja descărcat: {filename}")
                continue

            print(f"  ⬇️  Descarc: {filename}")
            if download_preview(song['previewUrl'], file_path):
                songs_downloaded += 1

            # Pauză mică să nu blocăm iTunes
            time.sleep(0.2)

    if songs_downloaded > 0:
        print(f"\n✨ Am descărcat {songs_downloaded} melodii noi.")
        # La final, declanșăm analiza AI o singură dată pentru toate
        trigger_server_scan()
    else:
        print("\n💤 Nu am găsit melodii noi de descărcat.")