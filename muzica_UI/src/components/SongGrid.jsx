import React from 'react';
import '../MusicPreferences.css'; // Importăm CSS-ul din folderul părinte

export default function SongGrid({ items, onSelect, isAi = false }) {
  if (!items || items.length === 0) return null;

  return (
    <div className={isAi ? "ai-grid" : "results-area"}>
      {items.map((item, idx) => (
        <div key={idx} className={`grid-card ${isAi ? 'ai-card-wrapper' : ''}`} onClick={() => onSelect(item)}>
          {isAi && <div className="plus-overlay-btn" title="Add">+</div>}
          
          <img 
            src={item.cover || 'https://via.placeholder.com/150?text=Music'} 
            alt="art" 
            className="grid-cover"
          />
          
          <div className="grid-info">
            <span className="grid-title">{item.title}</span>
            <span className="grid-artist">{item.artist}</span>
            {item.score && (
               <div style={{fontSize: '0.75rem', color: '#4ade80', marginTop: '5px'}}>
                 Match: {Math.round(item.score * 100)}%
               </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}