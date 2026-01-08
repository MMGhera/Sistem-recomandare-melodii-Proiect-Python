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
import MusicPreferences from './MusicPreferences' // <--- 1. IMPORTĂM COMPONENTA NOUĂ
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
      <h1>Music App 🎵</h1>

      {!currentUser ? (
        <Login onLoginSuccess={handleLoginSuccess} />
      ) : (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2>Salut, {currentUser}!</h2>
            <button onClick={handleLogout} style={{ padding: '5px 10px', fontSize: '0.8rem', background: '#444' }}>
              Log Out
            </button>
          </div>

          <hr style={{ borderColor: '#444', marginBottom: '20px' }}/>

          {/* --- 2. FOLOSIM COMPONENTA NOUĂ AICI --- */}
          {/* Îi trimitem username-ul ca să știe pentru cine să salveze datele */}
          <MusicPreferences username={currentUser} />

        </div>
      )}
    </>
  )
}

export default App