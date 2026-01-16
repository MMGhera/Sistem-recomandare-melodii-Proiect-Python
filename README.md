# Sistem-recomandare-melodii-Proiect-Python

Un sistem full-stack de recomandare muzicală care folosește **Deep Learning** și **Signal Processing** pentru a analiza conținutul audio al melodiilor (nu doar metadatele). Aplicația "ascultă" piese folosind o arhitectură hibridă și recomandă melodii similare pe baza distanței vectoriale combinate.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![React](https://img.shields.io/badge/React-18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)

---

## Arhitectura Hibridă (Hybrid Engine)

Inovația principală a acestui proiect constă în fuziunea a două metode distincte de analiză audio, eliminând punctele slabe ale fiecăreia:

1. **Deep Learning (Visual / Abstract)**

   * Rețea neuronală convoluțională (MusiCNN) antrenată cu PyTorch.
   * Analizează spectrograme pentru a detecta structuri muzicale complexe (refren, atmosferă, progresie).

2. **Signal Processing (Statistical / Math)**

   * Algoritmi DSP clasici prin Librosa.
   * Extrage caracteristici fizice: Tempo (BPM), Energie (RMS), Timbru (MFCC), Spectral Centroid, Zero Crossing Rate.

Cei doi vectori sunt normalizați și concatenați într-un singur **vector hibrid**, stocat în baza de date și folosit pentru recomandări precise.

---

## Funcționalități Principale

* **Motor de Căutare Live & Analiză Hibridă**

  * Căutare melodii prin iTunes API.
  * Descărcare preview audio (30s).
  * Analiză simultană AI + DSP.
  * Adăugare automată în bibliotecă și asociere cu utilizatorul.

* **Recomandări Bazate pe Conținut (Content-Based Filtering)**

  * Similaritate Cosinus pe vectorii hibrizi.
  * Corelare între stil muzical și caracteristici audio fizice.

* **The Harvester (Colector Automat de Date)**

  * Script automat de populare a bazei de date.
  * Procesează artiști, generează vectori și șterge fișierele audio pentru economie de spațiu.

* **Interfață Modernă**

  * Frontend React (Vite).
  * Live Search cu debounce.
  * Management vizual al bibliotecii personale.

---

## Tehnologii Folosite

### Backend

* Python & FastAPI
* PyTorch & Torchaudio
* Librosa (DSP)
* Scikit-Learn (normalizare și similaritate)
* SQLAlchemy & SQLite

### Frontend

* React.js (Vite)
* CSS Modules

---

## Instalare și Configurare

### Cerințe Preliminare

Asigură-te că sunt instalate:

* **Python 3.10+** (cu *Add to PATH*)
* **Node.js**
* **FFmpeg**

> **Important:** Asigură-te că te afli în folderul **root** al proiectului (acolo unde se află `backend.py`) înainte de a continua.

---

## Metoda Rapidă (Windows)

**Am automatizat tot procesul pentru tine!**

### 1. Instalare Dependențe

Dă dublu-click pe fișierul:

```
install_all.bat
```

**Ce face acest script?**

* Creează mediul virtual Python (`.venv`).
* Instalează bibliotecile necesare (PyTorch, Librosa, Scikit-Learn, FastAPI).
* Instalează dependențele frontend (`node_modules`).

### 2. Pornire Aplicație

Dă dublu-click pe fișierul:

```
run_app.bat
```

**Ce face acest script?**

* Pornește backend-ul într-o fereastră separată.
* Pornește frontend-ul React într-o altă fereastră.
* Deschide automat aplicația în browser la:

```
http://localhost:5173
```

---

## Rezolvarea Problemelor (Setup Manual)

Dacă scripturile `.bat` nu pornesc, se închid imediat sau apar erori, urmează pașii manual de mai jos **în ordinea prezentată**.

### 1. Instalare Dependențe (Manual)

Deschide un terminal în folderul **root** al proiectului.

#### A. Configurare Backend (Python)

```bash
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
```

#### B. Configurare Frontend (React)

```bash
cd muzica_UI
npm install
cd ..
```

### 2. Pornire Aplicație (Manual)

Vei avea nevoie de **două terminale separate**.

#### Terminal 1 – Backend (API Python)

```bash
call .venv\Scripts\activate
python backend.py
```

Lasă terminalul deschis. Dacă apare mesajul *"Application startup complete"*, backend-ul funcționează.

#### Terminal 2 – Frontend (Interfață React)

```bash
cd muzica_UI
npm run dev
```

Deschide browserul la adresa afișată (de obicei `http://localhost:5173`).

---

## Autori

Această aplicație este realizată de studenții:

* **Baiaș Andrei Silviu**
* **Gherasim Mihnea Matei**
* **Dragomir Mihai Andrei**
* **Dicu Tudor Andrei**

în cadrul disciplinei **Proiect Python**.
