# Sistem-recomandare-melodii-Proiect-Python

Un sistem full-stack de recomandare muzicală care folosește **Deep Learning** pentru a analiza conținutul audio al melodiilor (nu doar metadatele). Aplicația "ascultă" piese, extrage caracteristici audio complexe folosind o rețea neuronală (CNN) și recomandă melodii similare pe baza distanței vectoriale.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![React](https://img.shields.io/badge/React-18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)

##  Funcționalități Principale

* ** Motor de Căutare Live & Analiză Instant:**
    * Utilizatorul caută o melodie (prin iTunes API).
    * Backend-ul descarcă un preview audio de 30s.
    * AI-ul generează o spectrogramă și extrage vectorul de caracteristici în timp real.
    * Melodia este adăugată automat în bibliotecă și legată de profilul utilizatorului.
* ** Recomandări Bazate pe Conținut (Content-Based Filtering):**
    * Folosește **Cosine Similarity** pentru a găsi melodii care "sună" la fel, nu doar care au același gen în tag-uri.
    * Analizează timbrul, ritmul și instrumentația.
* ** The Harvester (Colector Automat de Date):**
    * Un script automatizat care populează baza de date.
    * Scanează artiști, descarcă sample-uri, le trece prin AI și stochează vectorii, ștergând fișierele audio pentru a economisi spațiu.
* ** Interfață Modernă:**
    * Frontend React rapid cu Vite.
    * Autocomplete (Live Search) cu Debounce.
    * Management vizual al bibliotecii personale.

##  Tehnologii Folosite

### Backend
* **Python & FastAPI:** Pentru API-ul REST rapid.
* **PyTorch & Torchaudio:** Pentru încărcarea și rularea modelului AI (arhitectură MusiCNN).
* **Librosa:** Pentru procesarea semnalului audio (re-sampling, generare Mel-spectrograms).
* **SQLAlchemy & SQLite:** Stocarea structurată a utilizatorilor, melodiilor și vectorilor (serializați JSON).
* **FFmpeg:** Decodare audio universală (.m4a, .mp3).

### Frontend
* **React.js (Vite):** Framework UI.
* **CSS Modules:** Stilizare modernă și responsivă.

##  Instalare și Configurare

### Cerințe Preliminare (Prerequisites)
Înainte de a începe, asigură-te că ai instalate următoarele pe calculator:
* **Python 3.10+** (Bifează "Add to PATH" la instalare)
* **Node.js** (Pentru interfață)
* **FFmpeg** (Pentru procesarea audio)

---

###  Metoda Rapidă (Windows)

Am automatizat tot procesul pentru tine!

#### 1. Instalare Dependențe
Dă dublu-click pe fișierul:
`install_all.bat`

>  **Ce face acest script?**
> * Creează mediul virtual Python (`.venv`).
> * Instalează toate bibliotecile necesare (`PyTorch`, `Librosa`, `FastAPI`).
> * Intră în folderul de frontend și instalează pachetele `React` (`node_modules`).

#### 2. Pornire Aplicație
După ce instalarea e gata, dă dublu-click pe:
`run_app.bat`

>  **Ce face acest script?**
> * Pornește serverul Backend într-o fereastră.
> * Pornește serverul Frontend în altă fereastră.
> * Deschide automat browserul tău la adresa aplicației (`http://localhost:5173`).

---

###  Metoda Manuală (Mac / Linux / Debugging)

Dacă nu folosești Windows sau preferi terminalul:

#### 1. Backend Setup
```bash
# Activare mediu virtual
python -m venv .venv
source .venv/bin/activate  # Pe Mac/Linux

# Instalare pachete
pip install -r requirements.txt

# Pornire server
python backend.py
```

#### 2. Frontend Setup
```
cd muzica_UI
npm install
npm run dev
```

Aceasta aplicatie este proiectul realizat de studentii Baiaș Andrei Silviu, Gherasim Mihnea Matei, Dragomir Mihai Andrei si  Dicu Tudor Andrei la disciplina **Proiect Python**.
