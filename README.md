# Sistem-recomandare-melodii-Proiect-Python

Un sistem full-stack de recomandare muzicală care folosește **Deep Learning** pentru a analiza conținutul audio al melodiilor (nu doar metadatele). Aplicația "ascultă" piese, extrage caracteristici audio complexe folosind o rețea neuronală (CNN) și recomandă melodii similare pe baza distanței vectoriale.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![React](https://img.shields.io/badge/React-18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)

## ✨ Funcționalități Principale

* **🔍 Motor de Căutare Live & Analiză Instant:**
    * Utilizatorul caută o melodie (prin iTunes API).
    * Backend-ul descarcă un preview audio de 30s.
    * AI-ul generează o spectrogramă și extrage vectorul de caracteristici în timp real.
    * Melodia este adăugată automat în bibliotecă și legată de profilul utilizatorului.
* **🧠 Recomandări Bazate pe Conținut (Content-Based Filtering):**
    * Folosește **Cosine Similarity** pentru a găsi melodii care "sună" la fel, nu doar care au același gen în tag-uri.
    * Analizează timbrul, ritmul și instrumentația.
* **🤖 The Harvester (Colector Automat de Date):**
    * Un script automatizat care populează baza de date.
    * Scanează artiști, descarcă sample-uri, le trece prin AI și stochează vectorii, ștergând fișierele audio pentru a economisi spațiu.
* **⚡ Interfață Modernă:**
    * Frontend React rapid cu Vite.
    * Autocomplete (Live Search) cu Debounce.
    * Management vizual al bibliotecii personale.

## 🛠️ Tehnologii Folosite

### Backend
* **Python & FastAPI:** Pentru API-ul REST rapid.
* **PyTorch & Torchaudio:** Pentru încărcarea și rularea modelului AI (arhitectură MusiCNN).
* **Librosa:** Pentru procesarea semnalului audio (re-sampling, generare Mel-spectrograms).
* **SQLAlchemy & SQLite:** Stocarea structurată a utilizatorilor, melodiilor și vectorilor (serializați JSON).
* **FFmpeg:** Decodare audio universală (.m4a, .mp3).

### Frontend
* **React.js (Vite):** Framework UI.
* **CSS Modules:** Stilizare modernă și responsivă.

###  
Aceasta aplicatie este proiectul realizat de studentii Baiaș Andrei Silviu, Gherasim Mihnea Matei, Dragomir Mihai Andrei si  Dicu Tudor Andrei la disciplina **Proiect Python**.
