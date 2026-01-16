import { useState } from 'react'
import Login from './LoginForm'
import MusicPreferences from './MusicPreferences'
import './App.css'

function App() {
  const [currentUser, setCurrentUser] = useState(null)

  const handleLoginSuccess = (username) => {
    setCurrentUser(username)
  }

  const handleLogout = () => {
    setCurrentUser(null)
  }

  return (
    <>
      {/* Dacă nu e logat, arată Login */}
      {!currentUser ? (
        <Login onLoginSuccess={handleLoginSuccess} />
      ) : (
        /* Dacă e logat, arată Dashboard-ul */
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            
            {/* --- FIX CRITIC AICI: zIndex: 1000 --- */}
            {/* Fără zIndex, stelele din fundal blochează click-ul pe buton */}
            <div style={{ 
                position: 'absolute', 
                top: '20px', 
                right: '20px', 
                zIndex: 1000,   /* <--- Asta rezolvă problema */
                display: 'flex', 
                alignItems: 'center', 
                gap: '15px' 
            }}>
              <div id="currentUser" style={{color: '#94a3b8', fontSize: '0.9rem'}}>
                  Signed in as <span id="textInCurrent" style={{color: 'white', fontWeight: 'bold'}}>{currentUser}</span>
              </div>
                <button 
                    onClick={handleLogout} 
                    style={{ 
                        background: 'rgba(255, 255, 255, 0.1)', 
                        border: '1px solid rgba(255, 255, 255, 0.2)', 
                        color: 'white', 
                        padding: '8px 16px', 
                        borderRadius: '20px',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                    }}
                    onMouseOver={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.2)'}
                    onMouseOut={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.1)'}
                >
                    Log Out
                </button>
            </div>

            {/* Componenta Principală */}
            <MusicPreferences username={currentUser} />
        </div>
      )}
    </>
  )
}

export default App