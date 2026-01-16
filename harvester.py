import requests
import os
import time

SONGS_PER_ARTIST = 25

ARTISTS = [
    #  ROCK & METAL
    "Metallica", "AC/DC", "Iron Maiden", "Nirvana", "Guns N' Roses",
    "Linkin Park", "System of a Down", "Rammstein", "Queen", "Led Zeppelin",
    "Pink Floyd", "Bon Jovi", "Red Hot Chili Peppers", "Slipknot",

    #  POP & MODERN
    "Taylor Swift", "Dua Lipa", "The Weeknd", "Ariana Grande", "Bruno Mars",
    "Ed Sheeran", "Michael Jackson", "Madonna", "Lady Gaga", "Justin Bieber",
    "Katy Perry", "Rihanna", "Beyonce", "Coldplay",

    #  HIP-HOP & RAP
    "Eminem", "Drake", "Snoop Dogg", "Jay-Z", "Kendrick Lamar",
    "50 Cent", "Dr. Dre", "Travis Scott", "Kanye West", "Post Malone",
    "Tupac", "Notorious B.I.G.",

    #  ELECTRONIC & EDM
    "Skrillex", "Daft Punk", "Avicii", "Calvin Harris", "David Guetta",
    "The Prodigy", "Tiesto", "Martin Garrix", "Marshmello", "The Chainsmokers",

    #  CLASSICAL & SOUNDTRACK
    "Mozart", "Beethoven", "Bach", "Chopin", "Vivaldi",
    "Hans Zimmer", "John Williams", "Ludovico Einaudi", "Tchaikovsky",

    #  JAZZ, BLUES & SOUL
    "Miles Davis", "Louis Armstrong", "Frank Sinatra", "Ray Charles",
    "John Coltrane", "Aretha Franklin", "Stevie Wonder", "Amy Winehouse",

    #  REGGAE & LATIN
    "Bob Marley", "Shakira", "Enrique Iglesias", "Bad Bunny", "Luis Fonsi"
]

AUDIO_LIBRARY_PATH = "audio_library"
SERVER_URL = "http://127.0.0.1:8000/scan_library"

os.makedirs(AUDIO_LIBRARY_PATH, exist_ok=True)


def search_itunes(artist_name):
    """Caută melodii pe iTunes folosind limita configurată."""
    # AICI era modificarea: am înlocuit limit=5 cu limit={SONGS_PER_ARTIST}
    url = f"https://itunes.apple.com/search?term={artist_name}&media=music&entity=song&limit={SONGS_PER_ARTIST}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["results"]
    except Exception as e:
        print(f"❌ Eroare conexiune iTunes: {e}")
    return []


def download_preview(preview_url, save_path):
    try:
        r = requests.get(preview_url)
        with open(save_path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"❌ Eroare download: {e}")
        return False


def trigger_server_scan():
    try:
        print("⏳ Trimit cerere de analiză către Backend...")
        r = requests.post(SERVER_URL)
        if r.status_code == 200:
            print("✅ Backend-ul a confirmat analiza!")
            return True
        else:
            print(f"⚠️ Backend-ul a răspuns cu eroare: {r.status_code}")
    except:
        print("❌ Nu mă pot conecta la Backend. Asigură-te că 'python backend.py' rulează!")


if __name__ == "__main__":
    print(f" 🎵 HARVESTER: Descărcăm top {SONGS_PER_ARTIST} melodii/artist 🎵 ")

    songs_downloaded = 0

    for artist in ARTISTS:
        print(f"\n🔍 Căutăm: {artist}...")
        results = search_itunes(artist)

        # Un mic set pentru a evita duplicatele în cadrul aceluiași artist
        # (ex: Album Version vs Single Version care au același nume)
        seen_songs = set()

        for song in results:
            track_name = song.get('trackName', 'Unknown')

            # Filtrare simplă: Dacă am descărcat deja o melodie cu numele ăsta pentru acest artist, o sărim
            if track_name in seen_songs:
                continue
            seen_songs.add(track_name)

            # Curățăm numele fișierului
            safe_name = f"{song['artistName']} - {track_name}"
            safe_name = "".join([c for c in safe_name if c.isalnum() or c in " -_()"]).strip()
            filename = f"{safe_name}.m4a"

            file_path = os.path.join(AUDIO_LIBRARY_PATH, filename)

            if os.path.exists(file_path):
                # Nu afișăm mesaj pentru fiecare skip, ca să nu poluăm consola
                continue

            print(f"  ⬇️  Descarc: {filename}")
            # 1. Extragem URL-ul folosind .get() (returnează None dacă nu există, nu dă eroare)
            preview_url = song.get('previewUrl')

            # 2. Verificăm dacă am primit un URL valid înainte să descărcăm
            if preview_url and download_preview(preview_url, file_path):
                songs_downloaded += 1

            time.sleep(0.1)

    if songs_downloaded > 0:
        print(f"\n✨ Am descărcat {songs_downloaded} melodii noi.")
        trigger_server_scan()
    else:
        print("\n💤 Nu am găsit melodii noi (sau le ai deja pe toate).")