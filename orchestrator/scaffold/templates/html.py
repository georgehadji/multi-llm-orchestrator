"""Vanilla HTML + CSS + JavaScript scaffold template.

Provides a minimal but functional static web app with:
- A single index.html entry point
- A styles/main.css stylesheet
- A scripts/main.js module script
"""
from __future__ import annotations

FILES: dict[str, str] = {
    "index.html": """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>My App</title>
    <link rel="stylesheet" href="styles/main.css" />
  </head>
  <body>
    <main id="app">
      <h1>Hello, World!</h1>
    </main>
    <script type="module" src="scripts/main.js"></script>
  </body>
</html>
""",
    "styles/main.css": """/* Reset */
*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.5;
  color: #1a1a1a;
  background: #ffffff;
}

main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

h1 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 1rem;
}
""",
    "scripts/main.js": """// Main application entry point

document.addEventListener('DOMContentLoaded', () => {
  console.log('App loaded');
});
""",
    ".gitignore": """.DS_Store
node_modules/
dist/
.env
""",
    "README.md": """# My App

A vanilla HTML/CSS/JS web application.

## Development

Open `index.html` in your browser or use a local server:

```bash
npx serve .
```
""",
}
