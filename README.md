# Sistem-recomandare-melodii-Proiect-Python

Un sistem full-stack de recomandare muzicală care folosește **Deep Learning** și **Signal Processing** pentru a analiza conținutul audio al melodiilor (nu doar metadatele). Aplicația "ascultă" piese folosind o arhitectură hibridă și recomandă melodii similare pe baza distanței vectoriale combinate.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![React](https://img.shields.io/badge/React-18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)

## Arhitectura Hibridă (Hybrid Engine)

Inovația principală a acestui proiect constă în fuziunea a două metode distincte de analiză audio, eliminând punctele slabe ale fiecăreia:

1. **Deep Learning (Visual/Abstract):**
   - Folosește o rețea neuronală convoluțională (MusiCNN) antrenată cu PyTorch.
   - Analizează spectrogramele vizuale pentru a detecta structuri complexe (refren, atmosferă, progresie).

2. **Signal Processing (Statistical/Math):**
   - Folosește algoritmi matematici clasici (DSP) prin Librosa.
   - Extrage indicatori fizici preciși: Tempo (BPM), Energie (RMS), Timbrul (MFCC), Spectral Centroid și Zero Crossing Rate.

Cei doi vectori sunt normalizați și concatenați într-un singur "Super-Vector" salvat în baza de date pentru o precizie maximă a recomandărilor.

## Funcționalități Principale

* **Motor de Căutare Live & Analiză Hibridă:**
  - Utilizatorul caută o melodie (prin iTunes API).
  - Backend-ul descarcă un preview audio de 30s.
  - Sistemul rulează simultan modelele AI și DSP, generând vectorul combinat în timp real.
  - Melodia este adăugată automat în bibliotecă și legată de profilul utilizatorului.

* **Recomandări Bazate pe Conținut (Content-Based Filtering):**
  - Folosește **Cosine Similarity** pe vectorii hibrizi pentru a găsi melodii similare auditiv.
  - Analizează simultan stilul (prin AI) și caracteristicile fizice (Ritm/Timbru).

* **The Harvester (Colector Automat de Date):**
  - Un script automatizat care populează baza de date.
  - Scanează artiști, descarcă sample-uri, le trece prin pipeline-ul hibrid și stochează vectorii, ștergând fișierele audio pentru a economisi spațiu.

* **Interfață Modernă:**
  - Frontend React rapid cu Vite.
  - Autocomplete (Live Search) cu Debounce.
  - Management vizual al bibliotecii personale.

## Tehnologii Folosite

### Backend
* **Python & FastAPI:** API REST rapid pentru comunicarea dintre module.
* **PyTorch & Torchaudio:** Modulul de Deep Learning.
* **Scikit-Learn:** Pentru normalizarea și fuziunea vectorilor.
* **Librosa:** Pentru procesarea semnalului audio și extragerea statistică.
* **SQLAlchemy & SQLite:** Stocarea structurată a datelor și vectorilor JSON.

### Frontend
* **React.js (Vite):** Framework UI.
* **CSS Modules:** Stilizare modernă și responsivă.

## Instalare și Configurare

### Cerințe Preliminare
Înainte de a începe, asigură-te că ai instalate următoarele pe calculator:
* **Python 3.10+** (Bifează "Add to PATH" la instalare)
* **Node.js** (Pentru interfață)
* **FFmpeg** (Pentru procesarea audio)

---

### Metoda Rapidă (Windows)

Am automatizat tot procesul pentru tine!

#### 1. Instalare Dependențe
Dă dublu-click pe fișierul:
`install_all.bat`

> **Ce face acest script?**
> - Creează mediul virtual Python (`.venv`).
> - Instalează toate bibliotecile necesare (PyTorch, Librosa, Scikit-Learn, FastAPI).
> - Intră în folderul de frontend și instalează pachetele React (`node_modules`).

#### 2. Pornire Aplicație
După ce instalarea e gata, dă dublu-click pe:
`run_app.bat`

> **Ce face acest script?**
> - Pornește serverul Backend într-o fereastră.
> - Pornește serverul Frontend în altă fereastră.
> - Deschide automat browserul la adresa aplicației (`http://localhost:5173`).

---

### Metoda Manuală (Mac / Linux / Debugging)

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
```bash
cd muzica_UI
npm install
npm run dev
```

Această aplicație este proiectul realizat de studenții Baiaș Andrei Silviu, Gherasim Mihnea Matei, Dragomir Mihai Andrei și Dicu Tudor Andrei la disciplina Proiect Python.