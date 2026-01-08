// import { useState } from 'react'
// import './App.css' // Reusing the default styles for simplicity
//
// export default function Login({ onLoginSuccess }) {
//   const [username, setUsername] = useState('')
//   const [password, setPassword] = useState('') // Visual only for now
//
//   const handleSubmit = async (e) => {
//     e.preventDefault()
//
//     try {
//       // We connect to your Python backend here
//       const response = await fetch('http://127.0.0.1:8000/login', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         // Your backend currently only expects 'username'
//         body: JSON.stringify({ username: username }),
//       })
//
//       if (response.ok) {
//         const data = await response.json()
//         // Pass the username back up to the App component
//         onLoginSuccess(data.username)
//       } else {
//         alert('Login failed!')
//       }
//     } catch (error) {
//       console.error('Error:', error)
//       alert('Could not connect to backend. Is it running?')
//     }
//   }
//
//   return (
//     <div className="card">
//       <h2>Sign In</h2>
//       <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
//         <div>
//           <input
//             type="text"
//             placeholder="Username"
//             value={username}
//             onChange={(e) => setUsername(e.target.value)}
//             required
//             style={{ padding: '8px', width: '100%' }}
//           />
//         </div>
//         <div>
//           <input
//             type="password"
//             placeholder="Password"
//             value={password}
//             onChange={(e) => setPassword(e.target.value)}
//             required
//             style={{ padding: '8px', width: '100%' }}
//           />
//         </div>
//         <button type="submit">
//           Sign In
//         </button>
//       </form>
//     </div>
//   )
// }

import { useState } from 'react'

export default function Login({ onLoginSuccess }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('') // Resetăm erorile vechi

    try {
      // Conectarea la Backend
      const response = await fetch('http://127.0.0.1:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // IMPORTANT: Backend-ul nostru Python cere doar "username" în LoginRequest.
        // Ignorăm parola momentan (o trimitem doar vizual în UI).
        body: JSON.stringify({ username: username }),
      })

      if (response.ok) {
        const data = await response.json()
        // Trimitem username-ul înapoi în App.jsx pentru a schimba ecranul
        onLoginSuccess(data.username)
      } else {
        setError('Eroare la login. Încearcă alt nume.')
      }
    } catch (err) {
      console.error('Error:', err)
      setError('Nu mă pot conecta la backend (Python). Este pornit?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.card}>
      <h2 style={{ marginBottom: '20px', color: '#333' }}>Autentificare</h2>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Utilizator</label>
          <input
            type="text"
            placeholder="Numele tău"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={styles.input}
          />
        </div>

        <div style={styles.inputGroup}>
          <label style={styles.label}>Parolă (Orice)</label>
          <input
            type="password"
            placeholder="******"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={styles.input}
          />
        </div>

        {error && <p style={{ color: 'red', fontSize: '0.9rem' }}>{error}</p>}

        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? 'Se conectează...' : 'Intră în cont'}
        </button>
      </form>
    </div>
  )
}

// Stiluri simple direct în fișier pentru a arăta curat
const styles = {
  card: {
    padding: '2rem',
    borderRadius: '10px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    backgroundColor: 'white',
    maxWidth: '400px',
    width: '100%',
    margin: '0 auto',
    textAlign: 'center'
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '15px'
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    textAlign: 'left'
  },
  label: {
    fontSize: '0.9rem',
    marginBottom: '5px',
    color: '#666'
  },
  input: {
    padding: '10px',
    borderRadius: '5px',
    border: '1px solid #ccc',
    fontSize: '1rem'
  },
  button: {
    padding: '12px',
    marginTop: '10px',
    backgroundColor: '#646cff',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    fontSize: '1rem',
    cursor: 'pointer',
    transition: 'background-color 0.2s'
  }
}