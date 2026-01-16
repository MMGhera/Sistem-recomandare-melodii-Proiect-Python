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
/*
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





*/


// VERSION 3 UI LOGIN

import { useState } from 'react'

export default function Login({ onLoginSuccess }) {
  // --- LOGICA RĂMÂNE NESCHIMBATĂ ---
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const response = await fetch('http://127.0.0.1:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username }),
      })

      if (response.ok) {
        const data = await response.json()
        onLoginSuccess(data.username)
      } else {
        setError('Eroare la login. Încearcă alt nume.')
      }
    } catch (err) {
      console.error('Error:', err)
      setError('Nu mă pot conecta la backend. Este pornit?')
    } finally {
      setLoading(false)
    }
  }
  // -----------------------------------

  //
  return (
    // Acest container exterior aplică fundalul gradient pe toată pagina
    <div style={styles.pageBackground}>
        <div style={styles.glassCard}>

            <div style={{ fontSize: '3rem', marginBottom: '10px', fontWeight: '600'}}>MFinder</div>
            <h2 style={styles.heading}>Bine ai venit!</h2>
            <p style={{ color: '#a1a1aa', marginBottom: '30px' }}>Conectează-te pentru a descoperi muzică nouă!</p>

            <form onSubmit={handleSubmit} style={styles.form}>
                <div style={styles.inputGroup}>
                <label style={styles.label}>Utilizator</label>
                <input
                    type="text"
                    placeholder="Ex: Melomanu99"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                    style={styles.glassInput}
                />
                </div>

                <div style={styles.inputGroup}>
                <label style={styles.label}>Parolă</label>
                <input
                    type="password"
                    placeholder="******"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    style={styles.glassInput}
                />
                </div>

                {error && <p style={{ color: '#ff6b6b', fontSize: '0.9rem', margin: '10px 0' }}>{error}</p>}

                <button type="submit" disabled={loading} style={styles.accentButton}>
                {loading ? 'Se încarcă...' : 'Intră în cont'}
                </button>
            </form>
        </div>
    </div>
  )
}

// --- NOILE STILURI MODERN  ---
const styles = {
  // Fundalul general al paginii de login
  pageBackground: {
    minHeight: '100vh',
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    // Gradient radial
    background: 'radial-gradient(circle at top right, #1e3a8a, #0f172a 60%, #020617)',
    padding: '20px',
    position: 'fixed', 
    top: 0,
    left: 0
  },
  // Cardul cu efect de sticlă (Glassmorphism)
  glassCard: {
    padding: '3rem',
    borderRadius: '24px',
    // Fundal alb foarte transparent
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    // Efectul de blur asupra fundalului din spate
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)', // Pentru suport Safari
    // Bordură subtilă strălucitoare
    border: '1px solid rgba(255, 255, 255, 0.1)',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
    maxWidth: '420px',
    width: '100%',
    textAlign: 'center',
    color: 'white' // Text alb implicit
  },
  heading: {
    fontSize: '2rem',
    fontWeight: 'bold',
    marginBottom: '10px',
    color: '#ffffff',
    letterSpacing: '1px'
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    marginTop: '20px'
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    textAlign: 'left'
  },
  label: {
    fontSize: '0.9rem',
    marginBottom: '8px',
    color: '#e2e8f0', // Un alb-gri deschis
    fontWeight: '500'
  },
  // Input-uri 
  glassInput: {
    padding: '14px 16px',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.15)',
    backgroundColor: 'rgba(0, 0, 0, 0.2)', // Fundal întunecat transparent
    color: 'white',
    fontSize: '1rem',
    outline: 'none',
    transition: 'all 0.3s ease',
    // Notă: Pseudo-clasele ca :focus sunt greu de pus inline.
    // Ideal, în CSS normal am pune: border-color: #2563eb la focus.
  },
  // Butonul albastru
  accentButton: {
    padding: '14px',
    marginTop: '10px',
    backgroundColor: '#2563eb', 
    color: 'white',
    border: 'none',
    borderRadius: '12px',
    fontSize: '1.1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'transform 0.2s, background-color 0.2s',
    boxShadow: '0 4px 15px rgba(37, 99, 235, 0.4)'
  }
}