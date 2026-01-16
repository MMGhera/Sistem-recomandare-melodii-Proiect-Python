import React from 'react';
import SongGrid from './SongGrid';
import '../MusicPreferences.css';

export default function Dashboard({ 
    mySongs, 
    searchQuery, 
    setSearchQuery, 
    onSearch, 
    showOverlay, 
    setShowOverlay, 
    suggestions, 
    onSelectSuggestion, 
    isAnalyzing, 
    analysisResult, 
    onRemoveSong, 
    loadingAI, 
    onGenerateAI, 
    recomandari, 
    onSelectAI 
}) {
    return (
        <div className="dashboard-container">
            <div className="dashboard-header">
                <p>You have {mySongs.length} tracks in your library.</p>
            </div>

            <div className="glass-card">
                {/* Search Results Grid */}
                {showOverlay && suggestions.length > 0 && (
                    <SongGrid items={suggestions} onSelect={onSelectSuggestion} />
                )}

                <div className="search-row">
                    <input
                        type="text"
                        placeholder="Search and add more tracks..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && onSearch(searchQuery)}
                        onFocus={() => setShowOverlay(true)}
                        className="dashboard-input"
                    />
                    <button onClick={() => onSearch(searchQuery)} disabled={isAnalyzing} className="add-btn">
                        {isAnalyzing ? '...' : 'Add'}
                    </button>
                </div>
                
                {analysisResult && (
                    <div className="success-msg">✅ Added: "{analysisResult.source_song}"</div>
                )}
            </div>

            {/* Library List */}
            <div className="library-grid">
                {mySongs.map((song, idx) => (
                    <div key={idx} className="chip">
                        <span>{song}</span>
                        <button onClick={() => onRemoveSong(song)} className="remove-btn">×</button>
                    </div>
                ))}
            </div>
            
            {/* AI Section */}
            <div className="ai-section">
                <button onClick={() => onGenerateAI()} disabled={loadingAI} className="generate-btn">
                    {loadingAI ? 'Generating... 🧠' : '✨ Generate AI Recommendations'}
                </button>
                
                {recomandari.length > 0 && (
                    <div style={{marginTop: '30px'}}>
                        <div className="ai-section-title">
                            Recommended For You ({recomandari.length})
                            <span className="update-link" onClick={() => onGenerateAI()}>🔄 Update</span>
                        </div>
                        {/* AI Grid */}
                        <SongGrid items={recomandari} onSelect={onSelectAI} isAi={true} />
                    </div>
                )}
            </div>
        </div>
    );
}