// import { useState } from 'react'
// import Login from './LoginForm'
// import './App.css'
//
// function App() {
//   // State to track the logged-in user
//   const [currentUser, setCurrentUser] = useState(null)
//
//   // Logic to handle successful login
//   const handleLoginSuccess = (username) => {
//     setCurrentUser(username)
//   }
//
//   // Logic to logout
//   const handleLogout = () => {
//     setCurrentUser(null)
//   }
//
//   return (
//     <>
//       <h1>Music App</h1>
//
//       {/* Conditional Rendering: Show Login if no user, otherwise show App content */}
//       {!currentUser ? (
//         <Login onLoginSuccess={handleLoginSuccess} />
//       ) : (
//         <div className="card">
//           <h2>Welcome, {currentUser}!</h2>
//           <p>You are now authenticated.</p>
//
//           {/* Placeholder for your music preferences components */}
//           <div style={{marginTop: '20px', padding: '20px', border: '1px dashed #666'}}>
//              Music Preferences Component will go here
//           </div>
//
//           <button onClick={handleLogout} style={{marginTop: '20px'}}>
//             Log Out
//           </button>
//         </div>
//       )}
//     </>
//   )
// }
//
// export default App

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
            {/* Butonul de Logout mic, sus în dreapta */}
            <div style={{ position: 'absolute', top: '20px', right: '20px' }}>
              <div id="currentUser">Signed in as <span id="textInCurrent">{currentUser}</span>!</div>
                <button 
                    onClick={handleLogout} 
                    style={{ 
                        background: 'rgba(0,0,0,0.3)', 
                        border: '1px solid rgba(255,255,255,0.2)', 
                        color: 'white', 
                        padding: '8px 16px', 
                        borderRadius: '20px',
                        cursor: 'pointer'
                    }}
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