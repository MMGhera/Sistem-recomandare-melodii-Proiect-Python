/*
//V3V3
import { useState, useEffect } from 'react'

export default function MusicPreferences({ username }) {
  // --- STATE ---
  const [mySongs, setMySongs] = useState([])

  // Căutare & Autocomplete
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])

  // Analiză & Feedback UI
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)

  // Recomandări AI
  const [recomandari, setRecomandari] = useState([])
  const [loadingAI, setLoadingAI] = useState(false)
  const [msg, setMsg] = useState('')

  // 1. Încărcare inițială a listei
  useEffect(() => {
    if (!username) return; // Protecție dacă nu e user
    fetch(`http://127.0.0.1:8000/prefs/${username}`)
      .then(res => res.json())
      .then(data => {
        // Ne asigurăm că e mereu array, chiar dacă vine null
        setMySongs(data.songs || [])
      })
      .catch(err => console.error("Eroare la încărcare:", err))
  }, [username])

  // 2. AUTOCOMPLETE LOGIC (Debounce)
  useEffect(() => {
    if (searchQuery.length < 2) {
      setSuggestions([])
      return
    }

    const delayDebounceFn = setTimeout(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/itunes_autocomplete?q=${searchQuery}`)
        const data = await res.json()
        if (Array.isArray(data)) {
            setSuggestions(data)
        }
      } catch (err) {
        console.error(err)
      }
    }, 300)

    return () => clearTimeout(delayDebounceFn)
  }, [searchQuery])

  // Helper: Când dai click pe o sugestie
  const selectSuggestion = (text) => {
    setSearchQuery(text)
    setSuggestions([])
    // Putem declanșa adăugarea automat dacă vrei:
    // handleAddSong(text)
  }

  // 3. ADĂUGARE MELODIE (Live Analysis)
  const handleAddSong = async (manualQuery = null) => {
    const queryToUse = manualQuery || searchQuery
    if (!queryToUse) return

    setIsAnalyzing(true)
    setAnalysisResult(null)
    setSuggestions([])

    try {
        const res = await fetch(`http://127.0.0.1:8000/analyze_external?q=${queryToUse}&username=${username}`)
        const data = await res.json()

        if (data.error) {
            alert(data.error)
        } else {
            setAnalysisResult(data)
            if (data.added_to_library) {
                // Verificăm dublurile
                if (!mySongs.includes(data.source_song)) {
                    setMySongs(prev => [...prev, data.source_song])
                    setMsg("Melodie adăugată! 💾")
                    setTimeout(() => setMsg(''), 3000)
                }
            }
        }
    } catch (err) {
        console.error(err)
        alert("Eroare conexiune server.")
    } finally {
        setIsAnalyzing(false)
        setSearchQuery('')
    }
  }

  // 4. ȘTERGERE MELODIE (Permanentă)
  const removeSong = async (songToDelete) => {
    // Backup vizual
    const previousSongs = [...mySongs]

    // Ștergem vizual instant
    setMySongs(mySongs.filter(song => song !== songToDelete))

    try {
        // Trimitem comanda DELETE la backend
        // encodeURIComponent e vital pentru nume cu spații sau &
        const res = await fetch(`http://127.0.0.1:8000/pref?username=${username}&song=${encodeURIComponent(songToDelete)}`, {
            method: 'DELETE'
        })

        if (!res.ok) {
            throw new Error("Eroare server")
        }
    } catch (err) {
        console.error("Nu s-a putut șterge:", err)
        alert("Eroare la ștergere! Verifică dacă backend-ul rulează.")
        // Restaurăm lista dacă a eșuat
        setMySongs(previousSongs)
    }
  }

  // 5. CERE RECOMANDĂRI AI
  const getAIRecommendations = async () => {
    setLoadingAI(true)
    setRecomandari([])
    try {
        const res = await fetch(`http://127.0.0.1:8000/recommend/${username}`)
        const data = await res.json()
        if(data.recommendations) setRecomandari(data.recommendations)
    } catch(e) {
        console.error(e)
        alert("Eroare la recomandări.")
    }
    finally { setLoadingAI(false) }
  }

  // --- INTERFAȚA (UI) ---
  return (
    <div style={{ textAlign: 'left', maxWidth: '600px', margin: '0 auto', fontFamily: 'Inter, sans-serif' }}>

      {/* ZONA ADĂUGARE */ /*}
      <div style={{
          marginBottom: '30px', padding: '25px', background: '#232323',
          borderRadius: '12px', border: '1px solid #444', boxShadow: '0 8px 20px rgba(0,0,0,0.4)',
          position: 'relative', zIndex: 1000
      }}>
          <h3 style={{ marginTop: 0, color: '#fff' }}>➕ Adaugă o melodie</h3>

          <div style={{ display: 'flex', gap: '10px', position: 'relative' }}>
              <input
                  type="text"
                  placeholder="Scrie numele artistului..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
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

              {/* LISTA SUGESTII (Dropdown) *//*}
              {suggestions.length > 0 && (
                <ul style={{
                  position: 'absolute', top: '100%', left: 0, right: '100px',
                  background: '#2a2a2a', border: '1px solid #444',
                  borderRadius: '0 0 6px 6px', listStyle: 'none', padding: 0, margin: 0,
                  zIndex: 2000, boxShadow: '0 4px 10px rgba(0,0,0,0.5)'
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

          {/* Feedback Vizual *//*}
          {analysisResult && (
              <div style={{ marginTop: '15px', padding: '10px', background: 'rgba(76, 175, 80, 0.1)', borderRadius: '6px', borderLeft: '3px solid #4caf50' }}>
                  <div style={{ color: '#4caf50', fontWeight: 'bold' }}>✅ "{analysisResult.source_song}" adăugată!</div>
              </div>
          )}
      </div>

      {/* LISTA BIBLIOTECĂ *//*}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3>Biblioteca ta ({mySongs.length})</h3>
          {msg && <span style={{ color: '#4caf50' }}>{msg}</span>}
      </div>

      <ul style={{ listStyle: 'none', padding: 0 }}>
        {mySongs.map((song, idx) => (
          <li key={idx} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '12px', background: '#2a2a2a', marginBottom: '8px', borderRadius: '6px',
            borderLeft: '4px solid #646cff'
          }}>
            <span>{song}</span>
            <button
                onClick={() => removeSong(song)}
                style={{
                    background: 'transparent', border: 'none', color: '#ff4d4d',
                    cursor: 'pointer', fontWeight: 'bold', fontSize: '1.2rem', padding: '0 10px'
                }}
            >
                &times;
            </button>
          </li>
        ))}
      </ul>

      {mySongs.length === 0 && <p style={{color: '#666'}}>Lista e goală.</p>}

      {/* BUTON RECOMANDĂRI *//*}
      <div style={{ marginTop: '40px', paddingTop: '20px', borderTop: '1px dashed #444' }}>
          <button
            onClick={getAIRecommendations}
            disabled={loadingAI}
            style={{
                width: '100%', padding: '15px', background: '#FF8E53',
                border: 'none', borderRadius: '8px', color: 'white', fontWeight: 'bold', cursor: 'pointer',
                opacity: loadingAI ? 0.7 : 1
            }}
          >
             {loadingAI ? 'Gândesc... 🧠' : '✨ Generează Recomandări AI'}
          </button>

          {recomandari.length > 0 && (
             <div style={{ marginTop: '20px' }}>
                {recomandari.map((rec, idx) => (
                    <div key={idx} style={{ padding: '12px', background: '#1a1a1a', marginBottom: '5px', borderRadius: '5px', border: '1px solid #FF8E53' }}>
                        <span style={{color: '#FF8E53', fontWeight: 'bold'}}>#{idx+1}</span> {rec}
                    </div>
                ))}
             </div>
          )}
      </div>
    </div>
  )
}
*/
import { useState, useEffect } from 'react'
import './MusicPreferences.css'

export default function MusicPreferences({ username }) {
  const [mySongs, setMySongs] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  
  // Date separate
  const [suggestions, setSuggestions] = useState([]) 
  const [recomandari, setRecomandari] = useState([]) 
  
  // Stări UI
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [loadingAI, setLoadingAI] = useState(false)
  const [showOverlay, setShowOverlay] = useState(false)

  // 1. Încărcare
  useEffect(() => {
    if (!username) return;
    fetch(`http://127.0.0.1:8000/prefs/${username}`)
      .then(res => res.json())
      .then(data => {
         setMySongs(data.songs || [])
         // Încărcăm recomandări "silently"
         if(data.songs && data.songs.length > 0) {
             getAIRecommendations(true) 
         }
      })
      .catch(err => console.error(err))
  }, [username])

  // 2. Autocomplete
  useEffect(() => {
    if (searchQuery.length < 2) {
      setSuggestions([])
      return
    }
    
    // --- FIX CRITIC: Dacă avem text, forțăm overlay-ul deschis ---
    setShowOverlay(true) 

    const delayDebounceFn = setTimeout(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/itunes_autocomplete?q=${searchQuery}`)
        const data = await res.json()
        if (Array.isArray(data)) setSuggestions(data)
      } catch (err) { console.error(err) }
    }, 300)
    return () => clearTimeout(delayDebounceFn)
  }, [searchQuery])

  // Selectare
  const selectSuggestion = (songData) => {
    const fullTitle = songData.full_text || `${songData.artist} - ${songData.title}`
    handleAddSong(fullTitle)
    setSearchQuery('')
    setSuggestions([])
  }

  // Selectare AI (JOS)
  const selectAIRecommendation = (songData) => {
    const fullTitle = songData.full_text || `${songData.artist} - ${songData.title}`
    handleAddSong(fullTitle)
    setRecomandari(prev => prev.filter(item => item.title !== songData.title))
  }

  // 3. Adăugare
  const handleAddSong = async (textToAdd) => {
    if (!textToAdd) return

    setIsAnalyzing(true)
    setAnalysisResult(null)
    
    try {
        const res = await fetch(`http://127.0.0.1:8000/analyze_external?q=${textToAdd}&username=${username}`)
        const data = await res.json()

        if (data.error) {
            alert(data.error)
        } else {
            setAnalysisResult(data)
            if (data.added_to_library) {
                if (!mySongs.includes(data.source_song)) {
                    setMySongs(prev => [...prev, data.source_song])
                    
                    setSearchQuery('') 
                    setSuggestions([]) 
                    getAIRecommendations()
                    setShowOverlay(true) 
                }
            }
        }
    } catch (err) {
        alert("Eroare server.")
    } finally {
        setIsAnalyzing(false)
    }
  }

  // 4. Ștergere
  const removeSong = async (songToDelete) => {
    const previousSongs = [...mySongs]
    setMySongs(mySongs.filter(song => song !== songToDelete))
    try {
        await fetch(`http://127.0.0.1:8000/pref?username=${username}&song=${encodeURIComponent(songToDelete)}`, { method: 'DELETE' })
    } catch (err) { setMySongs(previousSongs) }
  }

  // 5. Recomandări AI
  const getAIRecommendations = async (silent = false) => {
    if (!silent) setLoadingAI(true)
    try {
        const res = await fetch(`http://127.0.0.1:8000/recommend/${username}`)
        const data = await res.json()
        if(data.recommendations) {
            setRecomandari(data.recommendations)
        }
    } catch(e) { console.error(e) }
    finally { if (!silent) setLoadingAI(false) }
  }

  // Ce afișăm în gridul de sus?
  const itemsDisplay = searchQuery.length >= 2 ? suggestions : []
  const isLibraryEmpty = mySongs.length === 0;

  return (
    <div style={{ width: '100%' }}>
        
        {/* --- HERO SECTION --- */}
        {isLibraryEmpty ? (
            <div className="hero-container">
                <h1 className="hero-title">
                    Discover Your Next <br />
                    <span className="hero-highlight">Favorite</span> Song
                </h1>

                {!showOverlay && (
                    <p className="hero-subtitle">
                        Explorează milioane de piese. Playlist-ul tău perfect este la o căutare distanță.
                    </p>
                )}

                <div className="hero-search-wrapper">
                    
                    {/* GRIDUL DE SEARCH (HERO) */}
                    {showOverlay && itemsDisplay.length > 0 && (
                        <div className="results-area">
                            {itemsDisplay.map((item, idx) => (
                                <div key={idx} className="grid-card" onClick={() => selectSuggestion(item)}>
                                    <img 
                                        src={item.cover || 'https://via.placeholder.com/150?text=Music'} 
                                        alt="art" 
                                        className="grid-cover"
                                    />
                                    <div className="grid-info">
                                        <span className="grid-title">{item.title}</span>
                                        <span className="grid-artist">{item.artist}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    <input 
                        type="text" 
                        placeholder="Caută o piesă anume..." 
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleAddSong(searchQuery)}
                        className="hero-input"
                        onFocus={() => setShowOverlay(true)}
                        autoFocus
                    />
                    <button onClick={() => handleAddSong(searchQuery)} className="hero-search-btn">🔍</button>
                </div>
                {isAnalyzing && <p style={{marginTop: 20}}>Se analizează...</p>}
            </div>
        ) : (
            // --- DASHBOARD SECTION ---
            <div className="dashboard-container">
                 <div className="dashboard-header">
                    <p>Ai {mySongs.length} piese în bibliotecă.</p>
                </div>

                <div className="glass-card">
                     {/* GRIDUL DE SEARCH (DASHBOARD) */}
                     {showOverlay && itemsDisplay.length > 0 && (
                        <div className="results-area">
                            {itemsDisplay.map((item, idx) => (
                                <div key={idx} className="grid-card" onClick={() => selectSuggestion(item)}>
                                    <img src={item.cover || 'https://via.placeholder.com/150?text=Music'} alt="art" className="grid-cover"/>
                                    <div className="grid-info">
                                        <span className="grid-title">{item.title}</span>
                                        <span className="grid-artist">{item.artist}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="search-row">
                        <input
                            type="text"
                            placeholder="Caută și adaugă..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleAddSong(searchQuery)}
                            onFocus={() => setShowOverlay(true)}
                            className="dashboard-input"
                        />
                        <button onClick={() => handleAddSong(searchQuery)} disabled={isAnalyzing} className="add-btn">
                            {isAnalyzing ? '...' : 'Adaugă'}
                        </button>
                    </div>

                    {analysisResult && (
                        <div className="success-msg">✅ Adăugat: "{analysisResult.source_song}"</div>
                    )}
                </div>

                {/* Lista Bibliotecă */}
                <div className="library-grid">
                    {mySongs.map((song, idx) => (
                        <div key={idx} className="chip">
                            <span>{song}</span>
                            <button onClick={() => removeSong(song)} className="remove-btn">×</button>
                        </div>
                    ))}
                </div>
                
                {/* AI Section (JOS) */}
                <div className="ai-section">
                    <button 
                        onClick={() => getAIRecommendations()} 
                        disabled={loadingAI} 
                        className="generate-btn"
                    >
                        {loadingAI ? 'Se generează... 🧠' : '✨ Generează Recomandări AI'}
                    </button>

                    {recomandari.length > 0 && (
                        <div style={{marginTop: '20px'}}>
                            <div className="ai-section-title">
                                Recomandate pentru tine ({recomandari.length})
                                <span 
                                    style={{float:'right', cursor:'pointer', fontSize:'0.8rem', color:'#60a5fa'}}
                                    onClick={() => getAIRecommendations()}
                                >
                                    🔄 Update
                                </span>
                            </div>
                            
                            <div className="ai-grid">
                                {recomandari.map((rec, idx) => (
                                    <div key={idx} className="grid-card ai-card-wrapper" onClick={() => selectAIRecommendation(rec)}>
                                        <div className="plus-overlay-btn" title="Adaugă">+</div>
                                        <img 
                                            src={rec.cover || 'https://via.placeholder.com/150?text=AI'} 
                                            alt="art" 
                                            className="grid-cover"
                                        />
                                        <div className="grid-info">
                                            <span className="grid-title">{rec.title}</span>
                                            <span className="grid-artist">{rec.artist}</span>
                                            {rec.score && (
                                                <div style={{fontSize: '0.7rem', color: '#4ade80', marginTop: '4px'}}>
                                                    Match: {Math.round(rec.score * 100)}%
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        )}
    </div>
  )
}