// import { useState, useEffect } from 'react'
//
// export default function MusicPreferences({ username }) {
//   // Stare pentru melodiile utilizatorului
//   const [mySongs, setMySongs] = useState([])
//
//   // Stare pentru căutare
//   const [searchTerm, setSearchTerm] = useState('')
//   const [searchResults, setSearchResults] = useState([])
//
//   // Stare pentru feedback vizual
//   const [msg, setMsg] = useState('')
//
//   // 1. La încărcarea componentei, aducem preferințele salvate din Python
//   useEffect(() => {
//     fetch(`http://127.0.0.1:8000/prefs/${username}`)
//       .then(res => res.json())
//       .then(data => {
//         setMySongs(data.songs || [])
//       })
//       .catch(err => console.error("Eroare la încărcare preferințe:", err))
//   }, [username])
//
//   // 2. Funcția de Căutare (Autocomplete)
//   const handleSearch = async (text) => {
//     setSearchTerm(text)
//     if (text.length < 1) {
//       setSearchResults([])
//       return
//     }
//
//     try {
//       const res = await fetch(`http://127.0.0.1:8000/autocomplete?q=${text}`)
//       const data = await res.json()
//       setSearchResults(data)
//     } catch (err) {
//       console.error("Eroare la căutare:", err)
//     }
//   }
//
//   // 3. Adăugarea unei melodii în listă
//   const addSong = (song) => {
//     if (!mySongs.includes(song)) {
//       setMySongs([...mySongs, song])
//       setSearchTerm('') // Curățăm căutarea
//       setSearchResults([])
//     }
//   }
//
//   // 4. Ștergerea unei melodii
//   const removeSong = (songToDelete) => {
//     setMySongs(mySongs.filter(song => song !== songToDelete))
//   }
//
//   // 5. Salvarea listei în Backend
//   const savePreferences = async () => {
//     try {
//       setMsg("Se salvează...")
//       const res = await fetch('http://127.0.0.1:8000/prefs', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           username: username,
//           songs: mySongs
//         })
//       })
//
//       if (res.ok) {
//         setMsg("Listă salvată cu succes! ✅")
//         // Ascundem mesajul după 3 secunde
//         setTimeout(() => setMsg(''), 3000)
//       } else {
//         setMsg("Eroare la salvare ❌")
//       }
//     } catch (err) {
//       console.error(err)
//       setMsg("Eroare conexiune backend")
//     }
//   }
//
//   return (
//     <div style={{ textAlign: 'left', maxWidth: '500px', margin: '0 auto' }}>
//       <h3>Lista ta de melodii</h3>
//
//       {/* ZONA DE CĂUTARE */}
//       <div style={{ marginBottom: '20px', position: 'relative' }}>
//         <input
//           type="text"
//           placeholder="Caută o melodie..."
//           value={searchTerm}
//           onChange={(e) => handleSearch(e.target.value)}
//           style={{ width: '100%', padding: '10px', boxSizing: 'border-box' }}
//         />
//
//         {/* Lista de sugestii (Dropdown) */}
//         {searchResults.length > 0 && (
//           <ul style={{
//             listStyle: 'none',
//             padding: 0,
//             margin: 0,
//             border: '1px solid #ccc',
//             position: 'absolute',
//             width: '100%',
//             backgroundColor: '#242424',
//             zIndex: 10
//           }}>
//             {searchResults.map((song, idx) => (
//               <li
//                 key={idx}
//                 onClick={() => addSong(song)}
//                 style={{ padding: '10px', cursor: 'pointer', borderBottom: '1px solid #444' }}
//                 className="suggestion-item"
//               >
//                 + {song}
//               </li>
//             ))}
//           </ul>
//         )}
//       </div>
//
//       {/* LISTA DE MELODII ALESE */}
//       <ul style={{ listStyle: 'none', padding: 0 }}>
//         {mySongs.map((song, idx) => (
//           <li key={idx} style={{
//             display: 'flex',
//             justifyContent: 'space-between',
//             padding: '8px',
//             background: '#333',
//             marginBottom: '5px',
//             borderRadius: '4px'
//           }}>
//             <span>{song}</span>
//             <button
//               onClick={() => removeSong(song)}
//               style={{ background: 'red', border: 'none', color: 'white', cursor: 'pointer', padding: '2px 8px' }}
//             >
//               X
//             </button>
//           </li>
//         ))}
//       </ul>
//
//       {mySongs.length === 0 && <p style={{ color: '#888' }}>Nu ai selectat nicio melodie încă.</p>}
//
//       <div style={{ marginTop: '20px', borderTop: '1px solid #555', paddingTop: '10px' }}>
//         <button onClick={savePreferences} style={{ backgroundColor: '#646cff', width: '100%' }}>
//           Salvează Preferințele
//         </button>
//         {msg && <p style={{ textAlign: 'center', marginTop: '10px', fontWeight: 'bold' }}>{msg}</p>}
//       </div>
//     </div>
//   )
// }

//V2V2

import { useState, useEffect } from 'react'

export default function MusicPreferences({ username }) {
  // --- STARE (State) ---
  const [mySongs, setMySongs] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [msg, setMsg] = useState('')

  // Stare nouă pentru Recomandări AI
  const [recomandari, setRecomandari] = useState([])
  const [loadingAI, setLoadingAI] = useState(false)

  // 1. Încărcare preferințe la start
  useEffect(() => {
    fetch(`http://127.0.0.1:8000/prefs/${username}`)
      .then(res => res.json())
      .then(data => {
        setMySongs(data.songs || [])
      })
      .catch(err => console.error("Eroare la încărcare:", err))
  }, [username])

  // 2. Căutare (Autocomplete)
  const handleSearch = async (text) => {
    setSearchTerm(text)
    if (text.length < 1) {
      setSearchResults([])
      return
    }
    try {
      const res = await fetch(`http://127.0.0.1:8000/autocomplete?q=${text}`)
      const data = await res.json()
      setSearchResults(data)
    } catch (err) {
      console.error("Eroare search:", err)
    }
  }

  // 3. Adăugare melodie
  const addSong = (song) => {
    if (!mySongs.includes(song)) {
      setMySongs([...mySongs, song])
      setSearchTerm('')
      setSearchResults([])
    }
  }

  // 4. Ștergere melodie
  const removeSong = (songToDelete) => {
    setMySongs(mySongs.filter(song => song !== songToDelete))
  }

  // 5. Salvare în Backend
  const savePreferences = async () => {
    try {
      setMsg("Se salvează...")
      const res = await fetch('http://127.0.0.1:8000/prefs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, songs: mySongs })
      })
      if (res.ok) {
        setMsg("Listă salvată cu succes! ✅")
        setTimeout(() => setMsg(''), 3000)
      }
    } catch (err) {
      setMsg("Eroare conexiune ❌")
    }
  }

  // 6. FUNCȚIA MAGICĂ: Cere Recomandări AI
  const getAIRecommendations = async () => {
    setLoadingAI(true)
    setRecomandari([]) // Resetăm lista veche
    try {
        // Apelăm endpoint-ul nou creat în backend.py
        const res = await fetch(`http://127.0.0.1:8000/recommend/${username}`)
        const data = await res.json()

        if (data.recommendations) {
            setRecomandari(data.recommendations)
        }
    } catch (err) {
        console.error(err)
        alert("Nu am putut primi recomandări. Verifică dacă backend-ul rulează.")
    } finally {
        setLoadingAI(false)
    }
  }

  return (
    <div style={{ textAlign: 'left', maxWidth: '500px', margin: '0 auto' }}>

      {/* --- ZONA LISTA TA --- */}
      <h3>Lista ta de melodii</h3>

      <div style={{ marginBottom: '20px', position: 'relative' }}>
        <input
          type="text"
          placeholder="Caută o melodie..."
          value={searchTerm}
          onChange={(e) => handleSearch(e.target.value)}
          style={{ width: '100%', padding: '10px', boxSizing: 'border-box' }}
        />

        {/* Sugestii Dropdown */}
        {searchResults.length > 0 && (
          <ul style={{
            listStyle: 'none', padding: 0, margin: 0,
            border: '1px solid #444', position: 'absolute', width: '100%',
            backgroundColor: '#2a2a2a', zIndex: 10
          }}>
            {searchResults.map((song, idx) => (
              <li key={idx} onClick={() => addSong(song)}
                style={{ padding: '10px', cursor: 'pointer', borderBottom: '1px solid #444' }}
              >
                + {song}
              </li>
            ))}
          </ul>
        )}
      </div>

      <ul style={{ listStyle: 'none', padding: 0 }}>
        {mySongs.map((song, idx) => (
          <li key={idx} style={{
            display: 'flex', justifyContent: 'space-between',
            padding: '10px', background: '#333', marginBottom: '5px', borderRadius: '4px'
          }}>
            <span>{song}</span>
            <button onClick={() => removeSong(song)}
              style={{ background: 'red', border: 'none', color: 'white', cursor: 'pointer', padding: '5px 10px', borderRadius: '3px' }}
            >X</button>
          </li>
        ))}
      </ul>

      {mySongs.length === 0 && <p style={{ color: '#888' }}>Lista e goală.</p>}

      <div style={{ marginTop: '20px' }}>
        <button onClick={savePreferences} style={{ backgroundColor: '#646cff', width: '100%', padding: '10px', fontSize: '1rem' }}>
          Salvează Preferințele
        </button>
        {msg && <p style={{ textAlign: 'center', marginTop: '10px', color: '#4caf50' }}>{msg}</p>}
      </div>

      {/* --- ZONA NOUĂ: AI RECOMANDĂRI --- */}
      <div style={{ marginTop: '40px', paddingTop: '20px', borderTop: '2px dashed #444' }}>
          <h3 style={{ color: '#a0a0ff' }}>🎵 Descoperă Muzică Nouă</h3>

          <button
              onClick={getAIRecommendations}
              disabled={loadingAI}
              style={{
                  background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
                  color: 'white',
                  border: 0,
                  width: '100%',
                  fontWeight: 'bold',
                  padding: '12px',
                  fontSize: '1rem',
                  cursor: 'pointer',
                  opacity: loadingAI ? 0.7 : 1
              }}
          >
              {loadingAI ? 'AI-ul analizează... 🤖' : 'Cere Recomandări AI ✨'}
          </button>

          {/* Afișarea rezultatelor */}
          {recomandari.length > 0 && (
              <div style={{ marginTop: '20px', background: '#1a1a1a', padding: '15px', borderRadius: '8px', border: '1px solid #FF8E53' }}>
                  <h4 style={{marginTop: 0, color: '#FF8E53'}}>Recomandări pentru tine:</h4>
                  <ul style={{ paddingLeft: '20px', textAlign: 'left' }}>
                      {recomandari.map((rec, idx) => (
                          <li key={idx} style={{ marginBottom: '8px', fontSize: '1.1rem' }}>
                            {rec}
                          </li>
                      ))}
                  </ul>
              </div>
          )}
      </div>

    </div>
  )
}