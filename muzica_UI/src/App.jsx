import { useState } from 'react'
import Login from './Login'
import './App.css'

function App() {
  // State to track the logged-in user
  const [currentUser, setCurrentUser] = useState(null)

  // Logic to handle successful login
  const handleLoginSuccess = (username) => {
    setCurrentUser(username)
  }

  // Logic to logout
  const handleLogout = () => {
    setCurrentUser(null)
  }

  return (
    <>
      <h1>Music App</h1>
      
      {/* Conditional Rendering: Show Login if no user, otherwise show App content */}
      {!currentUser ? (
        <Login onLoginSuccess={handleLoginSuccess} />
      ) : (
        <div className="card">
          <h2>Welcome, {currentUser}!</h2>
          <p>You are now authenticated.</p>
          
          {/* Placeholder for your music preferences components */}
          <div style={{marginTop: '20px', padding: '20px', border: '1px dashed #666'}}>
             Music Preferences Component will go here
          </div>

          <button onClick={handleLogout} style={{marginTop: '20px'}}>
            Log Out
          </button>
        </div>
      )}
    </>
  )
}

export default App