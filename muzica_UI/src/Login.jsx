import { useState } from 'react'
import './App.css' // Reusing the default styles for simplicity

export default function Login({ onLoginSuccess }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('') // Visual only for now

  const handleSubmit = async (e) => {
    e.preventDefault()

    try {
      // We connect to your Python backend here
      const response = await fetch('http://127.0.0.1:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Your backend currently only expects 'username'
        body: JSON.stringify({ username: username }), 
      })

      if (response.ok) {
        const data = await response.json()
        // Pass the username back up to the App component
        onLoginSuccess(data.username)
      } else {
        alert('Login failed!')
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Could not connect to backend. Is it running?')
    }
  }

  return (
    <div className="card">
      <h2>Sign In</h2>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={{ padding: '8px', width: '100%' }}
          />
        </div>
        <div>
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ padding: '8px', width: '100%' }}
          />
        </div>
        <button type="submit">
          Sign In
        </button>
      </form>
    </div>
  )
}