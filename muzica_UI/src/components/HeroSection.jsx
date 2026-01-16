import React from 'react';
import SongGrid from './SongGrid';
import '../MusicPreferences.css';

export default function HeroSection({ 
    searchQuery, 
    setSearchQuery, 
    onSearch, 
    showOverlay, 
    setShowOverlay, 
    suggestions, 
    onSelectSuggestion, 
    isAnalyzing 
}) {
    return (
        <div className="hero-container">
            {/* Titlu */}
            <h1 className="hero-title">
                Discover Your Next<br />
                <span className="hero-highlight-gradient">Favorite</span> Song
            </h1>
            
            {/* --- PARAGRAFUL CERUT (ACUM E VIZIBIL PERMANENT) --- */}
            <p className="hero-subtitle">
                Explorează milioane de piese, descoperă comori ascunse și lasă muzica să te inspire. Playlist-ul tău perfect este la o căutare distanță.
            </p>

            <div className="hero-search-wrapper">
                {/* Gridul de rezultate (apare deasupra inputului) */}
                {showOverlay && suggestions.length > 0 && (
                    <SongGrid items={suggestions} onSelect={onSelectSuggestion} />
                )}
                
                {/* Input Search */}
                <input 
                    type="text" 
                    placeholder="Search for songs, artists, or albums..." 
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && onSearch(searchQuery)}
                    className="hero-input"
                    onFocus={() => setShowOverlay(true)}
                    autoFocus
                />
            </div>
            
            {isAnalyzing && (
                <p style={{marginTop: 20, color: '#94a3b8'}}>Analyzing track...</p>
            )}
        </div>
    );
}