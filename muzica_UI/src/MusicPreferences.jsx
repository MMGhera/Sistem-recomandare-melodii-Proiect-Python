import { useState, useEffect } from 'react'
import './MusicPreferences.css'

// Importăm componentele din folderul nou
import Background from './components/Background'
import HeroSection from './components/HeroSection'
import Dashboard from './components/Dashboard'

export default function MusicPreferences({ username }) {
  // --- STATE ---
  const [mySongs, setMySongs] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState([]) 
  const [recomandari, setRecomandari] = useState([]) 
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [loadingAI, setLoadingAI] = useState(false)
  const [showOverlay, setShowOverlay] = useState(false)

  // --- API CALLS & LOGIC ---
  useEffect(() => {
    if (!username) return;
    fetch(`http://127.0.0.1:8000/prefs/${username}`)
      .then(res => res.json())
      .then(data => {
         setMySongs(data.songs || [])
         if(data.songs && data.songs.length > 0) {
             getAIRecommendations(true) 
         }
      })
      .catch(err => console.error(err))
  }, [username])

  useEffect(() => {
    if (searchQuery.length < 2) {
      setSuggestions([])
      return
    }
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

  const selectSuggestion = (songData) => {
    const fullTitle = songData.full_text || `${songData.artist} - ${songData.title}`
    handleAddSong(fullTitle)
    setSearchQuery('')
    setSuggestions([])
  }

  const selectAIRecommendation = (songData) => {
    setRecomandari(prev => prev.filter(item => item.title !== songData.title));
    const fullTitle = songData.full_text || `${songData.artist} - ${songData.title}`
    handleAddSong(fullTitle)
  }

  const handleAddSong = async (textToAdd) => {
    if (!textToAdd) return
    setIsAnalyzing(true)
    setAnalysisResult(null)
    try {
        const res = await fetch(`http://127.0.0.1:8000/analyze_external?q=${textToAdd}&username=${username}`)
        const data = await res.json()
        if (data.error) { alert(data.error) } else {
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
    } catch (err) { alert("Eroare server.") } finally { setIsAnalyzing(false) }
  }

  const removeSong = async (songToDelete) => {
    const previousSongs = [...mySongs]
    setMySongs(mySongs.filter(song => song !== songToDelete))
    try {
        await fetch(`http://127.0.0.1:8000/pref?username=${username}&song=${encodeURIComponent(songToDelete)}`, { method: 'DELETE' })
    } catch (err) { setMySongs(previousSongs) }
  }

  const getAIRecommendations = async (silent = false) => {
    if (!silent) setLoadingAI(true)
    try {
        const res = await fetch(`http://127.0.0.1:8000/recommend/${username}`)
        const data = await res.json()
        if(data.recommendations) { setRecomandari(data.recommendations) }
    } catch(e) { console.error(e) }
    finally { if (!silent) setLoadingAI(false) }
  }

  const itemsDisplay = searchQuery.length >= 2 ? suggestions : []
  const isLibraryEmpty = mySongs.length === 0;

  return (
    <div className="app-wrapper">
        <Background />
        
        {isLibraryEmpty ? (
            <HeroSection 
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                onSearch={handleAddSong}
                showOverlay={showOverlay}
                setShowOverlay={setShowOverlay}
                suggestions={itemsDisplay}
                onSelectSuggestion={selectSuggestion}
                isAnalyzing={isAnalyzing}
            />
        ) : (
            <Dashboard 
                mySongs={mySongs}
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                onSearch={handleAddSong}
                showOverlay={showOverlay}
                setShowOverlay={setShowOverlay}
                suggestions={itemsDisplay}
                onSelectSuggestion={selectSuggestion}
                isAnalyzing={isAnalyzing}
                analysisResult={analysisResult}
                onRemoveSong={removeSong}
                loadingAI={loadingAI}
                onGenerateAI={getAIRecommendations}
                recomandari={recomandari}
                onSelectAI={selectAIRecommendation}
            />
        )}
    </div>
  )
}