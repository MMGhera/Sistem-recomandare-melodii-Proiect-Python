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

// import { useState, useEffect } from 'react'
//
// export default function MusicPreferences({ username }) {
//   // --- STARE (State) ---
//   const [mySongs, setMySongs] = useState([])
//   const [searchTerm, setSearchTerm] = useState('')
//   const [searchResults, setSearchResults] = useState([])
//   const [msg, setMsg] = useState('')
//
//   // Stare nouă pentru Recomandări AI
//   const [recomandari, setRecomandari] = useState([])
//   const [loadingAI, setLoadingAI] = useState(false)
//
//   //new state
//   const [magicSearch, setMagicSearch] = useState('')
//   const [magicResults, setMagicResults] = useState(null)
//   const [loadingMagic, setLoadingMagic] = useState(false)
//
//
//   // 1. Încărcare preferințe la start
//   useEffect(() => {
//     fetch(`http://127.0.0.1:8000/prefs/${username}`)
//       .then(res => res.json())
//       .then(data => {
//         setMySongs(data.songs || [])
//       })
//       .catch(err => console.error("Eroare la încărcare:", err))
//   }, [username])
//
//   // 2. Căutare (Autocomplete)
//   const handleSearch = async (text) => {
//     setSearchTerm(text)
//     if (text.length < 1) {
//       setSearchResults([])
//       return
//     }
//     try {
//       const res = await fetch(`http://127.0.0.1:8000/autocomplete?q=${text}`)
//       const data = await res.json()
//       setSearchResults(data)
//     } catch (err) {
//       console.error("Eroare search:", err)
//     }
//   }
//
//   // 3. Adăugare melodie
//   const addSong = (song) => {
//     if (!mySongs.includes(song)) {
//       setMySongs([...mySongs, song])
//       setSearchTerm('')
//       setSearchResults([])
//     }
//   }
//
//   // 4. Ștergere melodie
//   const removeSong = (songToDelete) => {
//     setMySongs(mySongs.filter(song => song !== songToDelete))
//   }
//
//   // 5. Salvare în Backend
//   const savePreferences = async () => {
//     try {
//       setMsg("Se salvează...")
//       const res = await fetch('http://127.0.0.1:8000/prefs', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ username, songs: mySongs })
//       })
//       if (res.ok) {
//         setMsg("Listă salvată cu succes! ✅")
//         setTimeout(() => setMsg(''), 3000)
//       }
//     } catch (err) {
//       setMsg("Eroare conexiune ❌")
//     }
//   }
//
//   // 6. FUNCȚIA MAGICĂ: Cere Recomandări AI
//   const getAIRecommendations = async () => {
//     setLoadingAI(true)
//     setRecomandari([]) // Resetăm lista veche
//     try {
//         // Apelăm endpoint-ul nou creat în backend.py
//         const res = await fetch(`http://127.0.0.1:8000/recommend/${username}`)
//         const data = await res.json()
//
//         if (data.recommendations) {
//             setRecomandari(data.recommendations)
//         }
//     } catch (err) {
//         console.error(err)
//         alert("Nu am putut primi recomandări. Verifică dacă backend-ul rulează.")
//     } finally {
//         setLoadingAI(false)
//     }
//   }
//
//   const handleMagicSearch = async () => {
//     if (!magicSearch) return
//     setLoadingMagic(true)
//     setMagicResults(null)
//
//     try {
//         // --- MODIFICARE: Adăugăm &username=${username} în URL ---
//         const res = await fetch(`http://127.0.0.1:8000/analyze_external?q=${magicSearch}&username=${username}`)
//         const data = await res.json()
//
//         if (data.error) {
//             alert(data.error)
//         } else {
//             setMagicResults(data)
//
//             // --- MODIFICARE: Dacă s-a adăugat cu succes, actualizăm lista vizuală ---
//             if (data.added_to_library) {
//                 // Verificăm să nu fie deja în listă ca să evităm dublurile vizuale
//                 if (!mySongs.includes(data.source_song)) {
//                     setMySongs(prevSongs => [...prevSongs, data.source_song])
//                     setMsg("Melodie adăugată și analizată! 💾")
//                     setTimeout(() => setMsg(''), 4000)
//                 }
//             }
//         }
//     } catch (err) {
//         console.error(err)
//         alert("Eroare la căutare.")
//     } finally {
//         setLoadingMagic(false)
//         setMagicSearch('') // Curățăm câmpul de input
//     }
//   }
//
//   return (
//     <div style={{ textAlign: 'left', maxWidth: '500px', margin: '0 auto' }}>
//
//         {/* ======================================================= */}
//
//       <div style={{
//           marginBottom: '30px', padding: '20px',
//           background: 'linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%)',
//           borderRadius: '12px', border: '1px solid #646cff',
//           boxShadow: '0 4px 15px rgba(0,0,0,0.3)'
//       }}>
//           <h3 style={{ marginTop: 0, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
//             🔎 Motor Căutare AI
//           </h3>
//           <p style={{ fontSize: '0.85rem', color: '#ccc', marginBottom: '15px' }}>
//               Scrie orice melodie (chiar dacă nu o ai). O descarc, o ascult și îți spun ce seamănă cu ea din baza ta!
//           </p>
//
//           <div style={{ display: 'flex', gap: '10px' }}>
//               <input
//                   type="text"
//                   placeholder="Ex: Rammstein - Du Hast"
//                   value={magicSearch}
//                   onChange={(e) => setMagicSearch(e.target.value)}
//                   onKeyDown={(e) => e.key === 'Enter' && handleMagicSearch()}
//                   style={{ flex: 1, padding: '10px', borderRadius: '5px', border: 'none', outline: 'none' }}
//               />
//               <button
//                   onClick={handleMagicSearch}
//                   disabled={loadingMagic}
//                   style={{
//                     background: '#646cff', color: 'white', fontWeight: 'bold',
//                     border: 'none', borderRadius: '5px', cursor: 'pointer', padding: '0 20px'
//                   }}
//               >
//                   {loadingMagic ? '⏳...' : 'Caută'}
//               </button>
//           </div>
//
//           {/* REZULTATELE CĂUTĂRII */}
//           {magicResults && (
//               <div style={{ marginTop: '20px', textAlign: 'left', background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px' }}>
//                   <div style={{ color: '#4caf50', marginBottom: '10px', fontWeight: 'bold', fontSize: '0.9rem' }}>
//                       ✅ Analizat: "{magicResults.source_song}"
//                   </div>
//
//                   <h4 style={{ color: '#aaa', marginBottom: '5px', fontSize: '0.9rem', marginTop: 0 }}>
//                     Melodii similare din baza ta:
//                   </h4>
//
//                   <ul style={{ paddingLeft: '20px', margin: 0 }}>
//                       {magicResults.recommendations.map((rec, idx) => (
//                           <li key={idx} style={{ marginBottom: '5px', color: '#fff' }}>
//                               {rec}
//                           </li>
//                       ))}
//                   </ul>
//
//                   {magicResults.recommendations.length === 0 && (
//                       <p style={{color: 'orange', fontSize: '0.9rem'}}>
//                         Nu am găsit nimic similar în baza de date locală. Încearcă să scanezi mai multă muzică!
//                       </p>
//                   )}
//               </div>
//           )}
//       </div>
//
//       {/* ======================================================= */}
//
//
//       {/* --- ZONA LISTA TA --- */}
//       <h3>Lista ta de melodii</h3>
//
//       <div style={{ marginBottom: '20px', position: 'relative' }}>
//         <input
//           type="text"
//           placeholder="Caută o melodie..."
//           value={searchTerm}
//           onChange={(e) => handleSearch(e.target.value)}
//           style={{ width: '100%', padding: '10px', boxSizing: 'border-box' }}
//         />
//
//         {/* Sugestii Dropdown */}
//         {searchResults.length > 0 && (
//           <ul style={{
//             listStyle: 'none', padding: 0, margin: 0,
//             border: '1px solid #444', position: 'absolute', width: '100%',
//             backgroundColor: '#2a2a2a', zIndex: 10
//           }}>
//             {searchResults.map((song, idx) => (
//               <li key={idx} onClick={() => addSong(song)}
//                 style={{ padding: '10px', cursor: 'pointer', borderBottom: '1px solid #444' }}
//               >
//                 + {song}
//               </li>
//             ))}
//           </ul>
//         )}
//       </div>
//
//       <ul style={{ listStyle: 'none', padding: 0 }}>
//         {mySongs.map((song, idx) => (
//           <li key={idx} style={{
//             display: 'flex', justifyContent: 'space-between',
//             padding: '10px', background: '#333', marginBottom: '5px', borderRadius: '4px'
//           }}>
//             <span>{song}</span>
//             <button onClick={() => removeSong(song)}
//               style={{ background: 'red', border: 'none', color: 'white', cursor: 'pointer', padding: '5px 10px', borderRadius: '3px' }}
//             >X</button>
//           </li>
//         ))}
//       </ul>
//
//       {mySongs.length === 0 && <p style={{ color: '#888' }}>Lista e goală.</p>}
//
//       <div style={{ marginTop: '20px' }}>
//         <button onClick={savePreferences} style={{ backgroundColor: '#646cff', width: '100%', padding: '10px', fontSize: '1rem' }}>
//           Salvează Preferințele
//         </button>
//         {msg && <p style={{ textAlign: 'center', marginTop: '10px', color: '#4caf50' }}>{msg}</p>}
//       </div>
//
//       {/* --- ZONA NOUĂ: AI RECOMANDĂRI --- */}
//       <div style={{ marginTop: '40px', paddingTop: '20px', borderTop: '2px dashed #444' }}>
//           <h3 style={{ color: '#a0a0ff' }}>🎵 Descoperă Muzică Nouă</h3>
//
//           <button
//               onClick={getAIRecommendations}
//               disabled={loadingAI}
//               style={{
//                   background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
//                   color: 'white',
//                   border: 0,
//                   width: '100%',
//                   fontWeight: 'bold',
//                   padding: '12px',
//                   fontSize: '1rem',
//                   cursor: 'pointer',
//                   opacity: loadingAI ? 0.7 : 1
//               }}
//           >
//               {loadingAI ? 'AI-ul analizează... 🤖' : 'Cere Recomandări AI ✨'}
//           </button>
//
//           {/* Afișarea rezultatelor */}
//           {recomandari.length > 0 && (
//               <div style={{ marginTop: '20px', background: '#1a1a1a', padding: '15px', borderRadius: '8px', border: '1px solid #FF8E53' }}>
//                   <h4 style={{marginTop: 0, color: '#FF8E53'}}>Recomandări pentru tine:</h4>
//                   <ul style={{ paddingLeft: '20px', textAlign: 'left' }}>
//                       {recomandari.map((rec, idx) => (
//                           <li key={idx} style={{ marginBottom: '8px', fontSize: '1.1rem' }}>
//                             {rec}
//                           </li>
//                       ))}
//                   </ul>
//               </div>
//           )}
//       </div>
//
//     </div>
//   )
// }

//V3V3
import { useState, useEffect } from 'react'

export default function MusicPreferences({ username }) {
  // --- STATE ---
  const [mySongs, setMySongs] = useState([])

  // Căutare & Autocomplete
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState([]) // <--- LISTA SUGESTII

  // Analiză & UI
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [recomandari, setRecomandari] = useState([])
  const [loadingAI, setLoadingAI] = useState(false)
  const [msg, setMsg] = useState('')

  // 1. Încărcare inițială
  useEffect(() => {
    fetch(`http://127.0.0.1:8000/prefs/${username}`)
      .then(res => res.json())
      .then(data => setMySongs(data.songs || []))
      .catch(err => console.error(err))
  }, [username])

  // ==============================================
  // 🟢 2. AUTOCOMPLETE LOGIC (DEBOUNCE) 🟢
  // ==============================================
  useEffect(() => {
    // Dacă am șters tot sau e text scurt, golim sugestiile
    if (searchQuery.length < 2) {
      setSuggestions([])
      return
    }

    // Așteptăm 300ms după ce userul se oprește din tastat
    const delayDebounceFn = setTimeout(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/itunes_autocomplete?q=${searchQuery}`)
        const data = await res.json()
        setSuggestions(data)
      } catch (err) {
        console.error(err)
      }
    }, 300)

    // Curățenie: Dacă userul scrie iar înainte de 300ms, anulăm căutarea anterioară
    return () => clearTimeout(delayDebounceFn)
  }, [searchQuery])

  // Când dai click pe o sugestie
  const selectSuggestion = (text) => {
    setSearchQuery(text)    // Punem textul în input
    setSuggestions([])      // Ascundem lista
    // Opțional: Putem declanșa direct adăugarea!
    // handleAddSong(text)
  }

  // ==============================================

  // 3. Adăugare Melodie (Modificat să accepte parametru opțional)
  const handleAddSong = async (manualQuery = null) => {
    const queryToUse = manualQuery || searchQuery
    if (!queryToUse) return

    setIsAnalyzing(true)
    setAnalysisResult(null)
    setSuggestions([]) // Ascundem sugestiile dacă au rămas

    try {
        const res = await fetch(`http://127.0.0.1:8000/analyze_external?q=${queryToUse}&username=${username}`)
        const data = await res.json()

        if (data.error) {
            alert(data.error)
        } else {
            setAnalysisResult(data)
            if (data.added_to_library) {
                if (!mySongs.includes(data.source_song)) {
                    setMySongs(prev => [...prev, data.source_song])
                    setMsg("Melodie adăugată! 💾")
                    setTimeout(() => setMsg(''), 3000)
                }
            }
        }
    } catch (err) {
        console.error(err)
    } finally {
        setIsAnalyzing(false)
        setSearchQuery('')
    }
  }

  // ... (Restul funcțiilor removeSong, savePreferences, getAIRecommendations rămân la fel) ...
  const removeSong = (s) => setMySongs(mySongs.filter(song => song !== s))
  const savePreferences = async () => { /* ... codul tău vechi ... */ }
  const getAIRecommendations = async () => {
    setLoadingAI(true); setRecomandari([])
    try {
        const res = await fetch(`http://127.0.0.1:8000/recommend/${username}`)
        const data = await res.json()
        if(data.recommendations) setRecomandari(data.recommendations)
    } catch(e) { console.error(e) }
    finally { setLoadingAI(false) }
  }


  return (
    <div style={{ textAlign: 'left', maxWidth: '600px', margin: '0 auto', fontFamily: 'Inter, sans-serif' }}>

      {/* ZONA ADĂUGARE CU AUTOCOMPLETE */}
      <div style={{
          marginBottom: '30px', padding: '25px', background: '#232323',
          borderRadius: '12px', border: '1px solid #444', boxShadow: '0 8px 20px rgba(0,0,0,0.4)',
          position: 'relative' // Important pentru poziționarea listei
      }}>
          <h3 style={{ marginTop: 0, color: '#fff' }}>➕ Adaugă o melodie</h3>

          <div style={{ display: 'flex', gap: '10px', position: 'relative' }}>
              <input
                  type="text"
                  placeholder="Scrie numele artistului..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  // Dacă apeși Enter, ia textul curent
                  onKeyDown={(e) => e.key === 'Enter' && handleAddSong()}
                  style={{
                      flex: 1, padding: '12px', borderRadius: '6px',
                      border: '1px solid #555', background: '#111', color: 'white', outline: 'none'
                  }}
              />
              <button
                  onClick={() => handleAddSong()}
                  disabled={isAnalyzing}
                  style={{
                    background: '#646cff', color: 'white', fontWeight: 'bold',
                    border: 'none', borderRadius: '6px', cursor: 'pointer', padding: '0 25px'
                  }}
              >
                  {isAnalyzing ? '...' : 'Adaugă'}
              </button>

              {/* --- LISTA DROP-DOWN SUGESTII --- */}
              {suggestions.length > 0 && (
                <ul style={{
                  position: 'absolute', top: '100%', left: 0, right: '100px', // sub input
                  background: '#2a2a2a', border: '1px solid #444',
                  borderRadius: '0 0 6px 6px', listStyle: 'none', padding: 0, margin: 0,
                  zIndex: 100, boxShadow: '0 4px 10px rgba(0,0,0,0.5)'
                }}>
                  {suggestions.map((sug, idx) => (
                    <li key={idx}
                      onClick={() => selectSuggestion(sug)}
                      style={{
                        padding: '10px 15px', cursor: 'pointer', borderBottom: '1px solid #333',
                        color: '#ddd', transition: 'background 0.2s'
                      }}
                      onMouseOver={(e) => e.target.style.background = '#333'}
                      onMouseOut={(e) => e.target.style.background = 'transparent'}
                    >
                      🎵 {sug}
                    </li>
                  ))}
                </ul>
              )}
          </div>

          {/* Rezultat Analiză */}
          {analysisResult && (
              <div style={{ marginTop: '15px', padding: '10px', background: 'rgba(76, 175, 80, 0.1)', borderRadius: '6px', borderLeft: '3px solid #4caf50' }}>
                  <div style={{ color: '#4caf50', fontWeight: 'bold' }}>✅ "{analysisResult.source_song}" adăugată!</div>
              </div>
          )}
      </div>

      {/* --- LISTA BIBLIOTECĂ --- */}
      <h3>Biblioteca ta ({mySongs.length})</h3>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {mySongs.map((song, idx) => (
          <li key={idx} style={{
            display: 'flex', justifyContent: 'space-between', padding: '12px',
            background: '#2a2a2a', marginBottom: '8px', borderRadius: '6px', borderLeft: '4px solid #646cff'
          }}>
            <span>{song}</span>
            <button onClick={() => removeSong(song)} style={{ background: 'transparent', border: 'none', color: '#ff4d4d', cursor: 'pointer' }}>X</button>
          </li>
        ))}
      </ul>

      {/* --- ZONA RECOMANDĂRI --- */}
      <div style={{ marginTop: '40px', paddingTop: '20px', borderTop: '1px dashed #444' }}>
          <button onClick={getAIRecommendations} style={{ width: '100%', padding: '15px', background: '#FF8E53', border: 'none', borderRadius: '8px', color: 'white', fontWeight: 'bold', cursor: 'pointer' }}>
             ✨ Generează Recomandări AI
          </button>

          {recomandari.length > 0 && (
             <div style={{ marginTop: '20px' }}>
                {recomandari.map((rec, idx) => (
                    <div key={idx} style={{ padding: '10px', background: '#1a1a1a', marginBottom: '5px', borderRadius: '5px', border: '1px solid #FF8E53' }}>
                        #{idx+1} {rec}
                    </div>
                ))}
             </div>
          )}
      </div>
    </div>
  )
}