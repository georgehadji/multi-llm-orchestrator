import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <header className="hero">
        <h1>Untitled Project</h1>
        <p>React + Vite + TypeScript</p>
      </header>
      
      <main className="content">
        <div className="card">
          <button onClick={() => setCount((count) => count + 1)}>
            count is {count}
          </button>
          <p>Interactive React component</p>
        </div>
      </main>
    </>
  )
}

export default App
