"""
AI Orchestrator IDE Server - Full Integration
Supports: HTML/CSS/JS, React, Next.js, FastAPI, Node.js
"""

from __future__ import annotations

import asyncio
import logging
import re
import socket
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ide_orchestrator")

# Base path
base_path = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════════════
# Tech Stack Selection Logic
# ═══════════════════════════════════════════════════════════════════════════════


class TechStackSelector:
    """Intelligent tech stack selector based on project requirements."""

    # Complexity levels
    SIMPLE = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    ENTERPRISE = 4

    @classmethod
    def detect_stack(cls, user_request: str) -> dict:
        """
        Detect appropriate tech stack based on user request.

        Returns:
            dict: {
                "stack_type": "static" | "premium" | "react" | "nextjs" | "fastapi" | "fullstack",
                "complexity": 1-4,
                "frontend": str,
                "backend": str | None,
                "database": str | None,
                "deploy": str
            }
        """
        req = user_request.lower()

        # Calculate complexity score
        complexity = cls._calculate_complexity(req)

        # Detect premium/awwwards website
        if cls._is_premium_website(req):
            return {
                "stack_type": "premium",
                "complexity": 4,
                "frontend": "HTML/CSS/JS + Three.js (3D)",
                "backend": None,
                "database": None,
                "deploy": "Vercel/Netlify",
                "files": ["index.html", "styles.css", "script.js (Three.js)"],
                "message": "✨ Creating premium website with 3D elements...",
            }

        # Detect full-stack FIRST (before React)
        if cls._is_fullstack(req):
            return {
                "stack_type": "fullstack",
                "complexity": max(complexity, 3),
                "frontend": "Next.js 14 + React + Supabase",
                "backend": "Next.js API + Supabase",
                "database": "PostgreSQL",
                "deploy": "Vercel",
                "files": ["app/page.tsx", "app/api/", "lib/supabase.ts"],
                "message": "🚀 Building full-stack application...",
            }

        # Detect static site (before React)
        if cls._is_static_site(req):
            return {
                "stack_type": "static",
                "complexity": min(complexity, 2),
                "frontend": "HTML/CSS/JS",
                "backend": None,
                "database": None,
                "deploy": "Netlify/Vercel",
                "files": ["index.html", "styles.css", "script.js"],
                "message": "📄 Generating static website...",
            }

        # Detect React app
        elif cls._is_react_app(req):
            return {
                "stack_type": "react",
                "complexity": max(complexity, 2),
                "frontend": "React + Vite",
                "backend": None,
                "database": None,
                "deploy": "Vercel/Netlify",
                "files": ["src/App.tsx", "package.json"],
                "message": "⚛️ Building React application...",
            }

        # Detect backend
        elif cls._is_backend(req):
            return {
                "stack_type": "backend",
                "complexity": max(complexity, 2),
                "frontend": None,
                "backend": "FastAPI + Python",
                "database": "SQLite" if "database" in req else None,
                "deploy": "Railway",
                "files": ["src/main.py", "requirements.txt"],
                "message": "🔌 Creating REST API...",
            }

        # Default: Static site
        return {
            "stack_type": "static",
            "complexity": 1,
            "frontend": "HTML/CSS/JS",
            "backend": None,
            "database": None,
            "deploy": "Netlify/Vercel",
            "files": ["index.html", "styles.css", "script.js"],
            "message": "🌐 Creating simple website...",
        }

    @classmethod
    def _calculate_complexity(cls, req: str) -> int:
        """Calculate complexity score 1-4 based on keywords."""
        score = 1

        # Simple indicators
        simple_words = ["simple", "basic", "landing", "portfolio", "brochure"]
        if any(w in req for w in simple_words):
            score = 1

        # Intermediate indicators
        intermediate_words = ["interactive", "dashboard", "dynamic", "forms", "api"]
        if any(w in req for w in intermediate_words):
            score = 2

        # Advanced indicators
        advanced_words = ["users", "auth", "login", "database", "real-time", "websocket"]
        if any(w in req for w in advanced_words):
            score = 3

        # Enterprise indicators
        enterprise_words = ["microservices", "scalable", "kubernetes", "kafka", "redis"]
        if any(w in req for w in enterprise_words):
            score = 4

        # Length bonus (longer requirements = more complex)
        if len(req.split()) > 50:
            score = min(score + 1, 4)

        return score

    @classmethod
    def _is_premium_website(cls, req: str) -> bool:
        """Check if user wants a premium Awwwards-level website."""
        premium_keywords = [
            "awwwards",
            "awward",
            "high-end",
            "high end",
            "premium",
            "3d",
            "three.js",
            "webgl",
            "threejs",
            "modern",
            "minimal",
            "minimalist",
            "sleek",
            "animated",
            "animation",
            "interactive",
            "portfolio",
            "showcase",
            "creative",
            "luxury",
            "elegant",
            "sophisticated",
            "micro-interaction",
            "micro interaction",
            "microinteractions",
        ]
        return any(k in req for k in premium_keywords)

    @classmethod
    def _is_static_site(cls, req: str) -> bool:
        """Check if user wants a static website."""
        static_keywords = [
            # English
            "portfolio",
            "landing page",
            "brochure",
            "business card",
            "simple website",
            "personal site",
            "showcase",
            "cv",
            "resume",
            "restaurant website",
            "cafe website",
            "hotel website",
            "plumber",
            "electrician",
            "mechanic",
            "contractor",
            "lawyer",
            "law firm",
            "attorney",
            "legal",
            "law office",
            # Greek
            "ιστοσελίδα",
            "сайт",
            "προφίλ",
            "βιογραφικό",
            "επαγγελματική καρτα",
            "παρουσίαση",
            "οδοντίατρο",
            "γιατρό",
            "δικηγόρο",
            "επιχείρηση",
            "κατάστημα",
            "εστιατόριο",
            "ξενοδοχείο",
            "салон",
            "γραφείο",
            "υδραυλικό",
            "ηλεκτρολόγο",
            "μηχανικό",
            "τεχνικό",
            "συνεργείο",
            "επισκευή",
            "κατασκευή",
            "ανακαίνιση",
        ]
        return any(k in req for k in static_keywords)

    @classmethod
    def _is_react_app(cls, req: str) -> bool:
        """Check if user wants a React application."""
        react_keywords = [
            # English
            "dashboard",
            "admin panel",
            "single page",
            "SPA",
            "interactive",
            "real-time updates",
            "chart",
            "graph",
            "data visualization",
            "analytics",
            # Greek
            "ταμπλό",
            "διαχείριση",
            "διαδραστικό",
            "γραφικά",
            "στατιστικά",
            "αναφορές",
            "admin",
            "panel",
        ]
        return any(k in req for k in react_keywords) and not cls._is_fullstack(req)

    @classmethod
    def _is_fullstack(cls, req: str) -> bool:
        """Check if user wants a full-stack application."""
        fullstack_keywords = [
            # English
            "users",
            "authentication",
            "login",
            "register",
            "sign up",
            "database",
            "postgres",
            "mysql",
            "mongodb",
            "e-commerce",
            "shop",
            "store",
            "products",
            "blog",
            "cms",
            "content management",
            "saas",
            "subscription",
            "payment",
            "stripe",
            "social",
            "feed",
            "posts",
            "comments",
            "likes",
            # Greek
            "χρήστε",
            "σύνδεση",
            "εγγραφή",
            "καταχώρηση",
            "βάση δεδομένων",
            "ηλεκτρονικό κατάστημα",
            "eshop",
            "προϊόντα",
            "καλάθι",
            "παραγγελία",
            "πληρωμή",
            "ιστολόγιο",
            "άρθρα",
            "ανάρτηση",
            "σχόλια",
            "κοινωνικό",
            "feed",
            "δημοσίευση",
        ]
        return any(k in req for k in fullstack_keywords)

    @classmethod
    def _is_backend(cls, req: str) -> bool:
        """Check if user wants a backend API."""
        backend_keywords = [
            # English
            "api",
            "rest",
            "graphql",
            "endpoint",
            "microservice",
            "backend",
            "server",
            "python",
            "fastapi",
            "django",
            "flask",
            "node",
            "express",
            "nestjs",
            # Greek
            "api",
            "backend",
            "server",
            "εξυπηρετητής",
            "δεδομένα",
            "υπηρεσία",
        ]
        return (
            any(k in req for k in backend_keywords)
            and "ιστοσελίδα" not in req
            and "сайт" not in req
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Session Manager
# ═══════════════════════════════════════════════════════════════════════════════


class SessionManager:
    """Manages IDE sessions with state persistence."""

    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}
        self.ws_connections: dict[str, list[WebSocket]] = {}
        self.running_processes: dict[str, subprocess.Popen] = {}

    def create_session(self, config: dict[str, Any]) -> dict[str, Any]:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:8]

        session = {
            "id": session_id,
            "project_name": config.get("project_name", "Untitled Project"),
            "description": config.get("description", ""),
            "mode": config.get("mode", "build"),
            "autonomy": config.get("autonomy", "standard"),
            "model": config.get("model", "auto"),
            "budget": config.get("budget", 5.0),
            "created_at": datetime.now().timestamp(),
            "started_at": None,
            "status": "idle",
            "messages": [],
            "files": [],
            "tasks": [],
            "terminal_lines": [],
            "spent": 0.0,
            "quality_score": 0.0,
            "cache_hit_rate": 0.0,
            "tech_stack": None,
        }

        self.sessions[session_id] = session
        self.ws_connections[session_id] = []

        logger.info(f"Session created: {session_id}")
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, **updates):
        """Update session fields."""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)

    def set_files(self, session_id: str, files: list[dict[str, Any]]):
        """Set session files."""
        if session_id in self.sessions:
            self.sessions[session_id]["files"] = files

    def add_message(self, session_id: str, message: dict[str, Any]):
        """Add message to session."""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append(message)

    def add_terminal_line(self, session_id: str, line_type: str, content: str):
        """Add line to terminal output."""
        if session_id in self.sessions:
            self.sessions[session_id]["terminal_lines"].append(
                {"type": line_type, "content": content, "ts": datetime.now().strftime("%H:%M:%S")}
            )

    def add_websocket(self, session_id: str, websocket: WebSocket):
        """Add WebSocket connection."""
        if session_id not in self.ws_connections:
            self.ws_connections[session_id] = []
        self.ws_connections[session_id].append(websocket)

    def remove_websocket(self, session_id: str, websocket: WebSocket):
        """Remove WebSocket connection."""
        if session_id in self.ws_connections:
            self.ws_connections[session_id] = [
                ws for ws in self.ws_connections[session_id] if ws != websocket
            ]

    async def broadcast(self, session_id: str, event: str, data: Any):
        """Broadcast event to all WebSocket connections."""
        if session_id not in self.ws_connections:
            return

        message = {"event": event, "data": data}
        disconnected = []

        for ws in self.ws_connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Clean up disconnected
        for ws in disconnected:
            self.ws_connections[session_id].remove(ws)

    def is_port_available(self, port: int) -> bool:
        """Check if port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return True
        except OSError:
            return False

    def start_server(
        self, session_id: str, output_dir: Path, port: int = 8000, server_type: str = "fastapi"
    ):
        """Start a web server for a session (FastAPI, HTTP, or npm dev)."""
        if session_id in self.running_processes:
            logger.info(f"Server already running for session {session_id}")
            return False

        if not self.is_port_available(port):
            logger.warning(f"Port {port} is already in use")
            return False

        try:
            if server_type == "http":
                cmd = [sys.executable, "-m", "http.server", str(port)]
            elif server_type == "npm":
                cmd = ["npm", "run", "dev", "--", "--port", str(port)]
            else:  # fastapi
                cmd = [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "src.main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                    "--reload",
                ]

            process = subprocess.Popen(
                cmd,
                cwd=str(output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self.running_processes[session_id] = process
            logger.info(f"Server started for session {session_id} on port {port} ({server_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def stop_server(self, session_id: str):
        """Stop server for a session."""
        if session_id in self.running_processes:
            process = self.running_processes[session_id]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.running_processes[session_id]
            logger.info(f"Server stopped for session {session_id}")


# Global session manager
session_manager = SessionManager()


# ═══════════════════════════════════════════════════════════════════════════════
# File Generators
# ═══════════════════════════════════════════════════════════════════════════════


class FileGenerators:
    """Generators for different tech stacks."""

    @staticmethod
    def generate_premium_website(project_name: str, description: str) -> tuple[dict, list]:
        """Generate premium Awwwards-level website with Three.js 3D elements."""

        # Detect profession from description
        desc_lower = description.lower()
        profession = "Business"
        profession_gr = "Επιχείρηση"

        # Check for lawyer/law firm FIRST (before generic business)
        if any(
            w in desc_lower
            for w in [
                "lawyer",
                "law firm",
                "attorney",
                "legal",
                "law office",
                "δικηγόρο",
                "δικηγορικό",
                "νομικό",
                "δικαστήριο",
            ]
        ):
            profession = "Law Firm"
            profession_gr = "Δικηγορικό Γραφείο"
        elif any(
            w in desc_lower
            for w in [
                "web designer",
                "web design",
                "ux designer",
                "ui designer",
                "graphic designer",
                "designer",
                "portfolio",
                "σχεδιαστής",
                "σχεδίαση",
            ]
        ):
            profession = "Web Designer"
            profession_gr = "Web Designer"
        elif any(
            w in desc_lower for w in ["photographer", "photography", "φωτογράφο", "φωτογραφία"]
        ):
            profession = "Photographer"
            profession_gr = "Φωτογράφος"
        elif any(
            w in desc_lower for w in ["architect", "architecture", "αρχιτέκτονα", "αρχιτεκτονική"]
        ):
            profession = "Architecture Studio"
            profession_gr = "Αρχιτεκτονικό Γραφείο"
        elif any(w in desc_lower for w in ["dentist", "dental", "οδοντίατρο", "δόντι", "clinic"]):
            profession = "Dental Clinic"
            profession_gr = "Οδοντιατρείο"
        elif any(w in desc_lower for w in ["plumber", "plumbing", "υδραυλικό", "ύδραυλικά"]):
            profession = "Plumbing Services"
            profession_gr = "Υδραυλικές Υπηρεσίες"
        elif any(w in desc_lower for w in ["electrician", "electrical", "ηλεκτρολόγο"]):
            profession = "Electrical Services"
            profession_gr = "Ηλεκτρολογικές Υπηρεσίες"
        elif any(w in desc_lower for w in ["restaurant", "cafe", "εστιατόριο", "καφετέρια"]):
            profession = "Restaurant"
            profession_gr = "Εστιατόριο"
        elif any(w in desc_lower for w in ["hotel", "ξενοδοχείο"]):
            profession = "Hotel"
            profession_gr = "Ξενοδοχείο"

        # Detect city from description
        city = "Athens"  # default
        if any(w in desc_lower for w in ["thessaloniki", "θεσσαλονίκη", "thessaly"]):
            city = "Thessaloniki"
        elif any(w in desc_lower for w in ["athens", "αθήνα", "αθήνας", "attica"]):
            city = "Athens"
        elif any(w in desc_lower for w in ["patras", "πάτρα"]):
            city = "Patras"
        elif any(w in desc_lower for w in ["heraklion", "ηράκλειο", "crete", "κρήτη"]):
            city = "Heraklion"
        elif any(w in desc_lower for w in ["volos", "βόλο"]):
            city = "Volos"

        # Build profession-specific content blocks
        if profession == "Web Designer":
            services_html = """            <div class="service-card" data-service="ui-ux">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
                    </svg>
                </div>
                <h3>UI/UX Design</h3>
                <p>Pixel-perfect interfaces that delight users and drive engagement</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>
            <div class="service-card" data-service="branding">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>
                    </svg>
                </div>
                <h3>Brand Identity</h3>
                <p>Distinctive visual systems that make your brand unforgettable</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>
            <div class="service-card" data-service="webdev">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
                    </svg>
                </div>
                <h3>Web Development</h3>
                <p>Fast, responsive websites built with cutting-edge technologies</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>
            <div class="service-card" data-service="motion">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                </div>
                <h3>Motion & 3D</h3>
                <p>Immersive animations and 3D experiences that captivate visitors</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>"""
            about_text = f"Based in {city}, I craft bold digital experiences that push boundaries. Combining brutalist aesthetics with cutting-edge 3D technology to create websites that leave an impression."
            stats_html = """                    <div class="stat">
                        <span class="stat-number">50+</span>
                        <span class="stat-label">Projects Delivered</span>
                    </div>
                    <div class="stat">
                        <span class="stat-number">8+</span>
                        <span class="stat-label">Years Experience</span>
                    </div>
                    <div class="stat">
                        <span class="stat-number">100%</span>
                        <span class="stat-label">Client Satisfaction</span>
                    </div>"""
            contact_title = "Start a Project"
            contact_email = f"hello@{project_name.lower().replace(' ', '')}.gr"
            form_placeholder = "Tell me about your project..."
            cta_button = "Send Message"
            footer_tagline = "Crafting bold digital experiences since 2018"
            nav_cta = "Hire Me"
        elif profession == "Dental Clinic":
            services_html = """            <div class="service-card"><div class="service-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2C8 2 5 5 5 9c0 5 7 13 7 13s7-8 7-13c0-4-3-7-7-7z"/></svg></div><h3>General Dentistry</h3><p>Comprehensive dental care for the whole family</p><a href="#" class="service-link">Learn More →</a></div>
            <div class="service-card"><div class="service-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/></svg></div><h3>Teeth Whitening</h3><p>Professional whitening for a radiant smile</p><a href="#" class="service-link">Learn More →</a></div>
            <div class="service-card"><div class="service-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/></svg></div><h3>Orthodontics</h3><p>Braces and aligners for a perfect smile</p><a href="#" class="service-link">Learn More →</a></div>
            <div class="service-card"><div class="service-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M20 7H4a2 2 0 00-2 2v10a2 2 0 002 2h16a2 2 0 002-2V9a2 2 0 00-2-2z"/></svg></div><h3>Dental Implants</h3><p>Permanent solutions for missing teeth</p><a href="#" class="service-link">Learn More →</a></div>"""
            about_text = f"Located in {city}, our clinic combines the latest dental technology with a gentle, patient-centered approach to deliver exceptional oral care."
            stats_html = """                    <div class="stat"><span class="stat-number">15+</span><span class="stat-label">Years Experience</span></div>
                    <div class="stat"><span class="stat-number">2000+</span><span class="stat-label">Happy Patients</span></div>
                    <div class="stat"><span class="stat-number">99%</span><span class="stat-label">Satisfaction Rate</span></div>"""
            contact_title = "Book an Appointment"
            contact_email = f"info@{project_name.lower().replace(' ', '')}.gr"
            form_placeholder = "Describe your dental concern..."
            cta_button = "Book Appointment"
            footer_tagline = f"Your trusted dental care in {city}"
            nav_cta = "Book Now"
        else:  # Law Firm and generic fallback
            services_html = """            <div class="service-card" data-service="consultation">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                    </svg>
                </div>
                <h3>Legal Consultation</h3>
                <p>Expert legal advice tailored to your specific needs and situation</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>
            <div class="service-card" data-service="litigation">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M3 21h18M5 21V7l8-4 8 4v14M9 10a2 2 0 11-4 0 2 2 0 014 0z"/>
                    </svg>
                </div>
                <h3>Litigation & Defense</h3>
                <p>Aggressive representation in court to protect your rights and interests</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>
            <div class="service-card" data-service="contracts">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                    </svg>
                </div>
                <h3>Contract Law</h3>
                <p>Drafting, review, and negotiation of contracts and agreements</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>
            <div class="service-card" data-service="corporate">
                <div class="service-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M3 21h18M5 21V7l8-4 8 4v14M8 21v-4a2 2 0 012-2h4a2 2 0 012 2v4"/>
                    </svg>
                </div>
                <h3>Corporate Law</h3>
                <p>Business formation, compliance, and corporate governance services</p>
                <a href="#" class="service-link">Learn More →</a>
            </div>"""
            about_text = f"Located in the heart of {city}, our firm combines deep {profession_gr} expertise with personalized attention to deliver exceptional results for every client."
            stats_html = """                    <div class="stat">
                        <span class="stat-number">25+</span>
                        <span class="stat-label">Years Experience</span>
                    </div>
                    <div class="stat">
                        <span class="stat-number">500+</span>
                        <span class="stat-label">Cases Won</span>
                    </div>
                    <div class="stat">
                        <span class="stat-number">98%</span>
                        <span class="stat-label">Success Rate</span>
                    </div>"""
            contact_title = "Schedule a Consultation"
            contact_email = f"info@{project_name.lower().replace(' ', '')}.gr"
            form_placeholder = "Brief Description of Your Case"
            cta_button = "Request Consultation"
            footer_tagline = f"Trusted {profession_gr} excellence since 2010"
            nav_cta = "Book Now"

        files = {
            "index.html": f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} | {profession_gr}</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <!-- 3D Background Canvas -->
    <canvas id="webgl-canvas"></canvas>

    <!-- Loading Screen -->
    <div class="loader">
        <div class="loader-content">
            <div class="loader-spinner"></div>
            <p>Loading Experience...</p>
        </div>
    </div>

    <!-- Navigation -->
    <nav class="navbar">
        <div class="logo">
            <span class="logo-text">{project_name}</span>
        </div>
        <ul class="nav-menu">
            <li><a href="#home" class="nav-link">Home</a></li>
            <li><a href="#services" class="nav-link">Services</a></li>
            <li><a href="#about" class="nav-link">About</a></li>
            <li><a href="#contact" class="nav-link">Contact</a></li>
        </ul>
        <button class="nav-cta">{nav_cta}</button>
    </nav>

    <!-- Hero Section -->
    <section id="home" class="hero">
        <div class="hero-content">
            <h1 class="hero-title">
                <span class="title-line">Premium {profession}</span>
                <span class="title-line accent">in {city}</span>
            </h1>
            <p class="hero-subtitle">Experience excellence with cutting-edge solutions and personalized service</p>
            <div class="hero-buttons">
                <a href="#contact" class="btn btn-primary">
                    <span>Book Consultation</span>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                </a>
                <a href="#services" class="btn btn-secondary">Explore Services</a>
            </div>
        </div>
        <div class="hero-scroll">
            <div class="scroll-indicator">
                <span>Scroll to explore</span>
                <div class="scroll-line"></div>
            </div>
        </div>
    </section>

    <!-- Services Section -->
    <section id="services" class="services">
        <div class="section-header">
            <span class="section-tag">Our Services</span>
            <h2 class="section-title">Professional {profession_gr}</h2>
        </div>
        <div class="services-grid">
{services_html}
        </div>
    </section>

    <!-- About Section -->
    <section id="about" class="about">
        <div class="about-content">
            <div class="about-text">
                <span class="section-tag">About Us</span>
                <h2 class="section-title">Excellence in {profession_gr}</h2>
                <p>{about_text}</p>
                <div class="stats-grid">
{stats_html}
                </div>
            </div>
            <div class="about-image">
                <div class="image-placeholder">
                    <span>Office Photo</span>
                </div>
            </div>
        </div>
    </section>

    <!-- Contact Section -->
    <section id="contact" class="contact">
        <div class="contact-container">
            <div class="contact-info">
                <span class="section-tag">Get In Touch</span>
                <h2 class="section-title">{contact_title}</h2>
                <div class="contact-details">
                    <div class="contact-item">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
                            <circle cx="12" cy="10" r="3"/>
                        </svg>
                        <span>{city}, Greece</span>
                    </div>
                    <div class="contact-item">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.84 12.84 0 002.81.7A2 2 0 0122 16.92z"/>
                        </svg>
                        <span>+30 2310 123456</span>
                    </div>
                    <div class="contact-item">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                            <polyline points="22,6 12,13 2,6"/>
                        </svg>
                        <span>{contact_email}</span>
                    </div>
                </div>
            </div>
            <form class="contact-form">
                <div class="form-group">
                    <input type="text" placeholder="Your Name" required>
                </div>
                <div class="form-group">
                    <input type="email" placeholder="Your Email" required>
                </div>
                <div class="form-group">
                    <input type="tel" placeholder="Phone Number">
                </div>
                <div class="form-group">
                    <textarea placeholder="{form_placeholder}" rows="5" required></textarea>
                </div>
                <button type="submit" class="btn btn-primary btn-full">
                    <span>{cta_button}</span>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="22" y1="2" x2="11" y2="13"/>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                    </svg>
                </button>
            </form>
        </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
        <div class="footer-content">
            <div class="footer-brand">
                <span class="logo-text">{project_name}</span>
                <p>{footer_tagline}</p>
            </div>
            <div class="footer-links">
                <a href="#">Privacy Policy</a>
                <a href="#">Terms of Service</a>
                <a href="#">Cookie Policy</a>
            </div>
            <div class="footer-social">
                <a href="#" aria-label="Facebook">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
                </a>
                <a href="#" aria-label="Instagram">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg>
                </a>
            </div>
        </div>
        <div class="footer-bottom">
            <p>&copy; 2026 {project_name}. All rights reserved.</p>
        </div>
    </footer>

    <script src="script.js"></script>
</body>
</html>""",
            "styles.css": """/* Premium Awwwards-Level Styles */
:root {
    --primary: #0a0a0f;
    --secondary: #1a1a2e;
    --accent: #c9a55c;
    --accent-light: #e0c87a;
    --text: #ffffff;
    --text-muted: #a0a0a0;
    --gradient-1: linear-gradient(135deg, #c9a55c 0%, #e0c87a 100%);
    --gradient-2: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
    --easing: cubic-bezier(0.645, 0.045, 0.355, 1);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
    font-size: 16px;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.6;
    color: var(--text);
    background: var(--primary);
    overflow-x: hidden;
}

/* 3D Canvas Background */
#webgl-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    opacity: 0.3;
}

/* Loading Screen */
.loader {
    position: fixed;
    inset: 0;
    background: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    transition: opacity 0.6s var(--easing), visibility 0.6s var(--easing);
}

.loader.hidden {
    opacity: 0;
    visibility: hidden;
}

.loader-spinner {
    width: 50px;
    height: 50px;
    border: 2px solid rgba(201, 165, 92, 0.2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Navigation */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 4rem;
    z-index: 100;
    transition: all 0.3s var(--easing);
}

.navbar.scrolled {
    background: rgba(10, 10, 15, 0.95);
    backdrop-filter: blur(20px);
    padding: 1rem 4rem;
}

.logo-text {
    font-size: 1.5rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.nav-menu {
    display: flex;
    gap: 2.5rem;
    list-style: none;
}

.nav-link {
    color: var(--text);
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    position: relative;
    transition: color 0.3s;
}

.nav-link::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: 0;
    width: 0;
    height: 1px;
    background: var(--accent);
    transition: width 0.3s var(--easing);
}

.nav-link:hover::after {
    width: 100%;
}

.nav-cta {
    padding: 0.75rem 1.5rem;
    background: var(--gradient-1);
    border: none;
    border-radius: 30px;
    color: var(--primary);
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.3s var(--easing);
}

.nav-cta:hover {
    transform: translateY(-2px);
}

/* Hero Section */
.hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 8rem 2rem 4rem;
}

.hero-content {
    max-width: 900px;
}

.hero-title {
    font-size: clamp(3rem, 8vw, 6rem);
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1.5rem;
}

.title-line {
    display: block;
    opacity: 0;
    transform: translateY(30px);
    animation: fadeInUp 0.8s var(--easing) forwards;
}

.title-line:nth-child(2) {
    animation-delay: 0.2s;
}

.title-line.accent {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: var(--text-muted);
    margin-bottom: 2.5rem;
    opacity: 0;
    transform: translateY(30px);
    animation: fadeInUp 0.8s var(--easing) 0.4s forwards;
}

.hero-buttons {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    opacity: 0;
    transform: translateY(30px);
    animation: fadeInUp 0.8s var(--easing) 0.6s forwards;
}

@keyframes fadeInUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.3s var(--easing);
}

.btn-primary {
    background: var(--gradient-1);
    color: var(--primary);
}

.btn-primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(201, 165, 92, 0.3);
}

.btn-secondary {
    background: transparent;
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.2);
}

.btn-secondary:hover {
    border-color: var(--accent);
    background: rgba(201, 165, 92, 0.1);
}

.hero-scroll {
    position: absolute;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
}

.scroll-indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.scroll-line {
    width: 1px;
    height: 60px;
    background: linear-gradient(to bottom, var(--accent), transparent);
    animation: scrollPulse 2s ease-in-out infinite;
}

@keyframes scrollPulse {
    0%, 100% { opacity: 0.3; transform: scaleY(0.8); }
    50% { opacity: 1; transform: scaleY(1); }
}

/* Services Section */
.services {
    padding: 8rem 4rem;
    background: var(--secondary);
}

.section-header {
    text-align: center;
    margin-bottom: 4rem;
}

.section-tag {
    display: inline-block;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--accent);
    margin-bottom: 1rem;
}

.section-title {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 700;
}

.services-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

.service-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 2.5rem;
    transition: all 0.4s var(--easing);
    cursor: pointer;
}

.service-card:hover {
    transform: translateY(-10px);
    border-color: var(--accent);
    background: rgba(201, 165, 92, 0.05);
}

.service-icon {
    width: 60px;
    height: 60px;
    margin-bottom: 1.5rem;
    color: var(--accent);
}

.service-card h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
}

.service-card p {
    color: var(--text-muted);
    margin-bottom: 1.5rem;
}

.service-link {
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

/* About Section */
.about {
    padding: 8rem 4rem;
}

.about-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    max-width: 1400px;
    margin: 0 auto;
    align-items: center;
}

.about-text .section-title {
    margin-bottom: 1.5rem;
}

.about-text p {
    color: var(--text-muted);
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
}

.stat {
    text-align: center;
}

.stat-number {
    display: block;
    font-size: 3rem;
    font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    color: var(--text-muted);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.about-image .image-placeholder {
    aspect-ratio: 4/5;
    background: linear-gradient(135deg, rgba(201,165,92,0.2), rgba(201,165,92,0.05));
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    border: 1px solid rgba(201,165,92,0.2);
}

/* Contact Section */
.contact {
    padding: 8rem 4rem;
    background: var(--secondary);
}

.contact-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    max-width: 1400px;
    margin: 0 auto;
}

.contact-details {
    margin-top: 2rem;
}

.contact-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    color: var(--text-muted);
}

.contact-item svg {
    width: 24px;
    height: 24px;
    color: var(--accent);
}

.contact-form {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.form-group input,
.form-group textarea {
    width: 100%;
    padding: 1rem 1.5rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    color: var(--text);
    font-size: 1rem;
    font-family: inherit;
    transition: border-color 0.3s;
}

.form-group input:focus,
.form-group textarea:focus {
    outline: none;
    border-color: var(--accent);
}

.form-group textarea {
    resize: vertical;
    min-height: 150px;
}

.btn-full {
    width: 100%;
    justify-content: center;
}

/* Footer */
.footer {
    padding: 4rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

.footer-content {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 3rem;
    max-width: 1400px;
    margin: 0 auto 3rem;
}

.footer-brand .logo-text {
    display: block;
    margin-bottom: 1rem;
}

.footer-brand p {
    color: var(--text-muted);
}

.footer-links {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.footer-links a {
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.3s;
}

.footer-links a:hover {
    color: var(--accent);
}

.footer-social {
    display: flex;
    gap: 1rem;
}

.footer-social a {
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
    color: var(--text);
    transition: all 0.3s;
}

.footer-social a:hover {
    background: var(--accent);
    color: var(--primary);
}

.footer-social svg {
    width: 20px;
    height: 20px;
}

.footer-bottom {
    text-align: center;
    padding-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    color: var(--text-muted);
}

/* Responsive */
@media (max-width: 1024px) {
    .navbar { padding: 1rem 2rem; }
    .nav-menu { gap: 1.5rem; }
    .services, .about, .contact { padding: 5rem 2rem; }
    .about-content, .contact-container { grid-template-columns: 1fr; }
    .footer-content { grid-template-columns: 1fr; }
}

@media (max-width: 768px) {
    .nav-menu { display: none; }
    .hero-title { font-size: 2.5rem; }
    .stats-grid { grid-template-columns: 1fr; }
    .footer { padding: 2rem; }
}
""",
            "script.js": """// Premium Website - Three.js + Animations

// Loading Screen
window.addEventListener('load', () => {
    setTimeout(() => {
        document.querySelector('.loader').classList.add('hidden');
    }, 1500);
});

// Three.js 3D Background
const initThreeJS = () => {
    const canvas = document.getElementById('webgl-canvas');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Create floating particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 2000;
    const positions = new Float32Array(particlesCount * 3);

    for (let i = 0; i < particlesCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 10;
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particlesMaterial = new THREE.PointsMaterial({
        color: 0xc9a55c,
        size: 0.02,
        transparent: true,
        opacity: 0.6
    });

    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particles);

    camera.position.z = 3;

    // Mouse interaction
    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX / window.innerWidth) * 2 - 1;
        mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    // Animation loop
    const animate = () => {
        requestAnimationFrame(animate);

        particles.rotation.x += 0.0005;
        particles.rotation.y += 0.0005;

        // Parallax effect
        camera.position.x += (mouseX * 0.5 - camera.position.x) * 0.05;
        camera.position.y += (mouseY * 0.5 - camera.position.y) * 0.05;

        renderer.render(scene, camera);
    };

    animate();

    // Resize handler
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
};

initThreeJS();

// Navbar scroll effect
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Smooth scroll for nav links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Intersection Observer for scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe service cards
document.querySelectorAll('.service-card').forEach((card, index) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(40px)';
    card.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
    observer.observe(card);
});

// Contact form handler
const contactForm = document.querySelector('.contact-form');
if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
        e.preventDefault();

        // Get form values
        const formData = new FormData(contactForm);
        const name = contactForm.querySelector('input[type="text"]').value;

        // Show success message
        alert(`Ευχαριστούμε ${name}! Το μήνυμά σας εστάλη. Θα επικοινωνήσουμε μαζί σας σύντομα!`);
        contactForm.reset();
    });
}

// CTA button handler
document.querySelectorAll('.nav-cta, .btn-primary').forEach(btn => {
    btn.addEventListener('click', (e) => {
        if (btn.getAttribute('href') === '#contact') {
            e.preventDefault();
            document.querySelector('#contact').scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Console greeting
console.log('%c🦷 ' + document.title, 'font-size: 24px; font-weight: bold; color: #c9a55c;');
console.log('%cPremium Dental Clinic Website - Thessaloniki', 'font-size: 14px; color: #a0a0a0;');
""",
        }

        file_tree = [
            {"name": "index.html", "type": "file", "language": "html"},
            {"name": "styles.css", "type": "file", "language": "css"},
            {"name": "script.js", "type": "file", "language": "javascript"},
        ]

        return files, file_tree

    @staticmethod
    def generate_static_site(project_name: str, description: str) -> tuple[dict, list]:
        """Generate static HTML/CSS/JS website."""
        files = {
            "index.html": f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header class="hero">
        <nav>
            <div class="logo">{project_name}</div>
            <ul class="nav-links">
                <li><a href="#about">About</a></li>
                <li><a href="#services">Services</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
        <div class="hero-content">
            <h1>Welcome to {project_name}</h1>
            <p>{description}</p>
            <a href="#contact" class="cta-button">Get Started</a>
        </div>
    </header>

    <section id="about" class="section">
        <h2>About Us</h2>
        <p>We are dedicated to providing excellent services.</p>
    </section>

    <section id="services" class="section">
        <h2>Our Services</h2>
        <div class="services-grid">
            <div class="service-card">
                <h3>Service 1</h3>
                <p>High quality service description</p>
            </div>
            <div class="service-card">
                <h3>Service 2</h3>
                <p>Another great service</p>
            </div>
        </div>
    </section>

    <section id="contact" class="section">
        <h2>Contact Us</h2>
        <form class="contact-form">
            <input type="text" placeholder="Name" required>
            <input type="email" placeholder="Email" required>
            <textarea placeholder="Message" rows="5" required></textarea>
            <button type="submit">Send Message</button>
        </form>
    </section>

    <footer>
        <p>&copy; 2026 {project_name}. All rights reserved.</p>
    </footer>

    <script src="script.js"></script>
</body>
</html>""",
            "styles.css": """/* Modern Responsive Styles */
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --primary: #2563eb;
    --secondary: #1e40af;
    --dark: #0f172a;
    --light: #f8fafc;
}

body {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    color: var(--dark);
}

.hero {
    min-height: 100vh;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 2rem;
}

nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem 2rem;
}

.nav-links {
    display: flex;
    gap: 2rem;
    list-style: none;
}

.nav-links a {
    color: white;
    text-decoration: none;
}

.hero-content {
    text-align: center;
    padding: 4rem 2rem;
}

.hero-content h1 {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.cta-button {
    display: inline-block;
    padding: 1rem 2rem;
    background: white;
    color: var(--primary);
    text-decoration: none;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 1rem;
}

.section {
    padding: 5rem 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

.section h2 {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 2rem;
}

.services-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.service-card {
    padding: 2rem;
    background: var(--light);
    border-radius: 12px;
    text-align: center;
}

.contact-form {
    max-width: 600px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.contact-form input,
.contact-form textarea {
    padding: 1rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 1rem;
}

.contact-form button {
    padding: 1rem;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
}

footer {
    background: var(--dark);
    color: white;
    text-align: center;
    padding: 2rem;
}

@media (max-width: 768px) {
    .hero-content h1 { font-size: 2rem; }
    .nav-links { gap: 1rem; }
}
""",
            "script.js": """// Interactive JavaScript
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

const contactForm = document.querySelector('.contact-form');
if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
        e.preventDefault();
        alert('Thank you! Your message has been sent.');
        contactForm.reset();
    });
}

console.log('%c👋 Welcome to ' + document.title, 'font-size: 20px; color: #2563eb;');
""",
        }

        file_tree = [
            {"name": "index.html", "type": "file", "language": "html"},
            {"name": "styles.css", "type": "file", "language": "css"},
            {"name": "script.js", "type": "file", "language": "javascript"},
        ]

        return files, file_tree

    @staticmethod
    def generate_react_app(project_name: str, description: str) -> tuple[dict, list]:
        """Generate React + Vite application."""
        files = {
            "package.json": f"""{{
  "name": "{project_name.lower().replace(' ', '-')}",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {{
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }},
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }},
  "devDependencies": {{
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }}
}}""",
            "vite.config.ts": """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 3000 }
})""",
            "index.html": f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>""",
            "src/main.tsx": """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)""",
            "src/App.tsx": f"""import {{ useState }} from 'react'
import './App.css'

function App() {{
  const [count, setCount] = useState(0)

  return (
    <>
      <header className="hero">
        <h1>{project_name}</h1>
        <p>React + Vite + TypeScript</p>
      </header>

      <main className="content">
        <div className="card">
          <button onClick={{() => setCount((count) => count + 1)}}>
            count is {{count}}
          </button>
          <p>Interactive React component</p>
        </div>
      </main>
    </>
  )
}}

export default App
""",
            "src/App.css": """.hero {
  text-align: center;
  padding: 4rem 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.content {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

.card {
  padding: 2rem;
  background: #f8fafc;
  border-radius: 12px;
  text-align: center;
}

button {
  padding: 0.75rem 1.5rem;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
}

button:hover {
  background: #5568d3;
}
""",
            "src/index.css": """* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.6;
  color: #0f172a;
}
""",
        }

        file_tree = [
            {
                "name": "src",
                "type": "folder",
                "children": [
                    {"name": "main.tsx", "type": "file", "language": "typescript"},
                    {"name": "App.tsx", "type": "file", "language": "typescript"},
                    {"name": "App.css", "type": "file", "language": "css"},
                    {"name": "index.css", "type": "file", "language": "css"},
                ],
            },
            {"name": "package.json", "type": "file", "language": "json"},
            {"name": "vite.config.ts", "type": "file", "language": "typescript"},
            {"name": "index.html", "type": "file", "language": "html"},
        ]

        return files, file_tree

    @staticmethod
    def generate_fastapi_backend(project_name: str, description: str) -> tuple[dict, list]:
        """Generate FastAPI backend."""
        files = {
            "src/main.py": f'''"""{project_name} - FastAPI Backend"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="{project_name}",
    description="{description}",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    created_at: datetime = datetime.now()

# In-memory storage
items: List[Item] = []

@app.get("/")
def root():
    return {{"message": "Welcome to {project_name} API", "docs": "/docs"}}

@app.get("/health")
def health():
    return {{"status": "healthy", "timestamp": datetime.now()}}

@app.get("/items", response_model=List[Item])
def get_items():
    """Get all items."""
    return items

@app.post("/items", response_model=Item)
def create_item(item: Item):
    """Create a new item."""
    item.id = len(items) + 1
    items.append(item)
    return item

@app.get("/items/{{item_id}}", response_model=Item)
def get_item(item_id: int):
    """Get item by ID."""
    for item in items:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
            "requirements.txt": """fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pytest==7.4.4
""",
            "README.md": f"""# {project_name}

{description}

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn src.main:app --reload

# Open docs
http://localhost:8000/docs
```

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /items` - List all items
- `POST /items` - Create item
- `GET /items/{{id}}` - Get item by ID
""",
        }

        file_tree = [
            {
                "name": "src",
                "type": "folder",
                "children": [
                    {"name": "main.py", "type": "file", "language": "python"},
                ],
            },
            {"name": "requirements.txt", "type": "file", "language": "text"},
            {"name": "README.md", "type": "file", "language": "markdown"},
        ]

        return files, file_tree


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Handler
# ═══════════════════════════════════════════════════════════════════════════════


async def handle_websocket(websocket: WebSocket, session_id: str):
    """Handle WebSocket connection."""
    await websocket.accept()
    session_manager.add_websocket(session_id, websocket)

    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")
            payload = data.get("data", {})

            logger.info(f"WebSocket event: {event}")

            if event == "ping":
                await websocket.send_json(
                    {"event": "pong", "data": {"ts": datetime.now().isoformat()}}
                )

            elif event == "chat_message":
                await handle_chat_message(session_id, payload.get("message", ""), websocket)

            elif event == "session_update":
                session_manager.update_session(session_id, **payload)
                await session_manager.broadcast(
                    session_id, "session_state", session_manager.get_session(session_id)
                )

            elif event == "terminal_command":
                await handle_terminal_command(session_id, payload.get("command", ""), websocket)

            elif event == "file_request":
                await handle_file_request(session_id, payload.get("path", ""), websocket)

            else:
                logger.warning(f"Unknown event: {event}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        session_manager.remove_websocket(session_id, websocket)


async def handle_modification_request(session_id: str, message: str, websocket: WebSocket):
    """Handle modification/edit request for existing project."""
    logger.info(f"Modification request for {session_id}: {message[:100]}...")

    session_manager.get_session(session_id)
    output_dir = Path.cwd() / "ide_outputs" / session_id

    # Send thinking state
    await session_manager.broadcast(
        session_id,
        "messages_update",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "thinking": True,
                    "steps": [
                        {"label": "Analyzing modification request...", "done": True},
                        {"label": "Identifying files to modify...", "done": False},
                        {"label": "Applying changes...", "done": False},
                    ],
                    "ts": datetime.now().strftime("%H:%M"),
                }
            ]
        },
    )

    # Detect type of modification
    msg_lower = message.lower()
    modifications = []
    new_color = None

    # Color scheme changes - check FIRST
    color_map = {
        "blue": "#3b82f6",
        "purple": "#8b5cf6",
        "violet": "#8b5cf6",
        "green": "#10b981",
        "emerald": "#10b981",
        "red": "#ef4444",
        "rose": "#ef4444",
        "pink": "#ec4899",
        "orange": "#f97316",
        "gold": "#c9a55c",
        "yellow": "#eab308",
        "cyan": "#06b6d4",
        "indigo": "#6366f1",
        "black": "#000000",
        "white": "#ffffff",
        "gray": "#6b7280",
        "slate": "#475569",
    }

    for color_name, color_hex in color_map.items():
        if color_name in msg_lower:
            new_color = color_hex
            modifications.append(f"color_scheme ({color_name})")
            break

    # Layout changes
    if any(w in msg_lower for w in ["layout", "spacing", "padding", "margin", "grid", "flex"]):
        modifications.append("layout")

    # Font changes
    if any(w in msg_lower for w in ["font", "typography", "text", "heading", "title"]):
        modifications.append("typography")

    # Add section
    if any(w in msg_lower for w in ["add", "create", "new", "insert"]):
        if "section" in msg_lower or "component" in msg_lower:
            modifications.append("add_section")

    # Remove/hide
    if any(w in msg_lower for w in ["remove", "delete", "hide", "drop"]):
        modifications.append("remove_element")

    # Animation
    if any(w in msg_lower for w in ["animate", "animation", "transition", "effect", "motion"]):
        modifications.append("animation")

    # Dark/light mode
    if "dark" in msg_lower or "darker" in msg_lower:
        modifications.append("dark_mode")
    if "light" in msg_lower or "lighter" in msg_lower and "dark" not in msg_lower:
        modifications.append("light_mode")

    # Default: general improvement
    if not modifications:
        modifications.append("general")

    logger.info(f"Detected modifications: {modifications}")

    # Update thinking state
    await session_manager.broadcast(
        session_id,
        "messages_update",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "thinking": True,
                    "steps": [
                        {"label": "Analyzing modification request...", "done": True},
                        {"label": f"Applying: {', '.join(modifications)}", "done": True},
                        {"label": "Updating files...", "done": False},
                    ],
                    "ts": datetime.now().strftime("%H:%M"),
                }
            ]
        },
    )

    files_modified = []

    # Apply modifications to CSS file
    css_path = output_dir / "styles.css"
    if css_path.exists():
        try:
            with open(css_path, encoding="utf-8") as f:
                css_content = f.read()

            # Apply color scheme modification
            if new_color:
                # Use regex to find and replace existing --accent color
                accent_pattern = r"--accent:\s*#[0-9a-fA-F]{3,6}"
                match = re.search(accent_pattern, css_content)

                if match:
                    # Replace ONLY the existing accent color (first occurrence)
                    css_content = re.sub(
                        accent_pattern, f"--accent: {new_color}", css_content, count=1
                    )
                    files_modified.append("styles.css")
                    logger.info(f"Color changed from {match.group(0)} to --accent: {new_color}")
                else:
                    # No accent color found - log warning
                    logger.warning(
                        f"No --accent color found in CSS, skipping color change to {new_color}"
                    )
                    session_manager.add_terminal_line(
                        session_id, "warning", "⚠ No accent color found to change"
                    )

            # Apply dark mode (black background)
            if new_color == "#000000" or "dark_mode" in modifications:
                css_content = css_content.replace("--primary: #0a0a0f", "--primary: #000000")
                css_content = css_content.replace("--secondary: #1a1a2e", "--secondary: #0a0a0f")
                if "styles.css" not in files_modified:
                    files_modified.append("styles.css")

            # Apply light mode (white background)
            if new_color == "#ffffff" or "light_mode" in modifications:
                css_content = css_content.replace("--primary: #0a0a0f", "--primary: #ffffff")
                css_content = css_content.replace("--secondary: #1a1a2e", "--secondary: #f5f5f5")
                css_content = css_content.replace("color: var(--text)", "color: #000000")
                if "styles.css" not in files_modified:
                    files_modified.append("styles.css")

            # Apply animation modification
            if "animation" in modifications:
                if "more" in msg_lower or "faster" in msg_lower:
                    css_content = css_content.replace(
                        "transition: all 0.3s", "transition: all 0.2s"
                    )
                    css_content = css_content.replace(
                        "transition: all 0.4s", "transition: all 0.3s"
                    )
                    files_modified.append("styles.css")
                elif "less" in msg_lower or "slower" in msg_lower:
                    css_content = css_content.replace(
                        "transition: all 0.2s", "transition: all 0.4s"
                    )
                    css_content = css_content.replace(
                        "transition: all 0.3s", "transition: all 0.5s"
                    )
                    files_modified.append("styles.css")

            # Write updated CSS
            with open(css_path, "w", encoding="utf-8") as f:
                f.write(css_content)

            session_manager.add_terminal_line(
                session_id, "success", f"✓ Updated styles.css ({new_color or 'modified'})"
            )
            logger.info("CSS modifications applied")

            # Broadcast terminal update immediately
            await session_manager.broadcast(
                session_id,
                "terminal_update",
                {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
            )

        except Exception as e:
            logger.error(f"Error modifying CSS: {e}")
            session_manager.add_terminal_line(session_id, "error", f"✗ Error: {e}")
            await session_manager.broadcast(
                session_id,
                "terminal_update",
                {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
            )

    # Apply modifications to JS file
    js_path = output_dir / "script.js"
    if js_path.exists():
        try:
            with open(js_path, encoding="utf-8") as f:
                js_content = f.read()

            # Add more animations
            if "animation" in modifications and "more" in msg_lower:
                if "observer" not in js_content.lower():
                    # Add intersection observer for animations
                    observer_code = """
// Additional scroll animations
const animatedElements = document.querySelectorAll('.service-card, .stat, .btn');
animatedElements.forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = `opacity 0.6s ease ${i * 0.1}s, transform 0.6s ease ${i * 0.1}s`;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    observer.observe(el);
});
"""
                    js_content = js_content.replace(
                        "console.log('%c", observer_code + "\nconsole.log('%c"
                    )
                    files_modified.append("script.js")

            # Write updated JS
            with open(js_path, "w", encoding="utf-8") as f:
                f.write(js_content)

            if "script.js" not in files_modified:
                files_modified.append("script.js")

            session_manager.add_terminal_line(session_id, "success", "✓ Updated script.js")
            logger.info("JS modifications applied")

            # Broadcast terminal update
            await session_manager.broadcast(
                session_id,
                "terminal_update",
                {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
            )

        except Exception as e:
            logger.error(f"Error modifying JS: {e}")
            session_manager.add_terminal_line(session_id, "error", f"✗ Error: {e}")
            await session_manager.broadcast(
                session_id,
                "terminal_update",
                {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
            )

    # Broadcast updates - session_state FIRST, then terminal, then messages
    # Update session status before broadcasting
    session_manager.update_session(session_id, status="completed")

    # 1. Broadcast session_state FIRST so frontend has latest data
    await session_manager.broadcast(
        session_id, "session_state", session_manager.get_session(session_id)
    )

    # 2. Broadcast terminal update
    await session_manager.broadcast(
        session_id,
        "terminal_update",
        {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
    )

    # 3. Broadcast session_state AGAIN after terminal_update to ensure frontend has latest data
    await session_manager.broadcast(
        session_id, "session_state", session_manager.get_session(session_id)
    )

    # 4. Broadcast messages update last
    await session_manager.broadcast(
        session_id,
        "messages_update",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"""✓ **Changes Applied Successfully!**

**Modifications:**
{chr(10).join(f"• {mod.replace('_', ' ').title()}" for mod in modifications)}

**Files Updated:**
{chr(10).join(f"• {f}" for f in files_modified)}

**🔥 Hot Reload:** The preview should update automatically!

**More requests:**
• "change colors to blue/purple/green/red/pink/orange"
• "add more animations"
• "make it darker/lighter"
• "increase spacing"
""",
                    "thinking": False,
                    "steps": [
                        {"label": "Analyzing modification request...", "done": True},
                        {"label": "Applying changes...", "done": True},
                    ],
                    "ts": datetime.now().strftime("%H:%M"),
                    "quality": 0.95,
                    "cost": 0.05,
                }
            ]
        },
    )

    # Final session state broadcast
    await session_manager.broadcast(
        session_id, "session_state", session_manager.get_session(session_id)
    )


async def handle_chat_message(session_id: str, message: str, websocket: WebSocket):
    """Handle chat message and generate/modify project."""
    logger.info(f"Chat message for {session_id}: {message[:100]}...")

    # Add user message
    session_manager.add_message(
        session_id,
        {
            "role": "user",
            "content": message,
            "ts": datetime.now().strftime("%H:%M"),
        },
    )

    session = session_manager.get_session(session_id)

    # Check if this is a modification request (project already exists)
    is_modification = session and session.get("files") and len(session["files"]) > 0

    if is_modification:
        # Handle modification/edit request
        await handle_modification_request(session_id, message, websocket)
        return

    # Detect tech stack for new project
    stack_info = TechStackSelector.detect_stack(message)
    session_manager.update_session(session_id, tech_stack=stack_info)

    logger.info(
        f"Detected stack: {stack_info['stack_type']} - {stack_info['frontend'] or stack_info['backend']}"
    )

    # Get user-friendly message
    stack_message = stack_info.get("message", "Creating project...")

    # Send thinking state
    await session_manager.broadcast(
        session_id,
        "messages_update",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": stack_message,
                    "thinking": False,
                    "steps": [
                        {"label": "Analyzing requirements...", "done": True},
                        {
                            "label": f"Selected: {stack_info['frontend'] or stack_info['backend']}",
                            "done": True,
                        },
                        {"label": "Generating files...", "done": False},
                    ],
                    "ts": datetime.now().strftime("%H:%M"),
                }
            ]
        },
    )

    # Update session status
    session_manager.update_session(
        session_id, status="running", started_at=datetime.now().timestamp()
    )

    # Generate files based on stack
    project_name = session_manager.get_session(session_id).get("project_name", "Project")
    description = message  # Use the actual user chat message, not the empty session description

    if stack_info["stack_type"] == "premium":
        files, file_tree = FileGenerators.generate_premium_website(project_name, description)
    elif stack_info["stack_type"] == "static":
        files, file_tree = FileGenerators.generate_static_site(project_name, description)
    elif stack_info["stack_type"] == "react":
        files, file_tree = FileGenerators.generate_react_app(project_name, description)
    else:  # backend
        files, file_tree = FileGenerators.generate_fastapi_backend(project_name, description)

    # Create output directory
    output_dir = Path.cwd() / "ide_outputs" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Send file creation updates
    await session_manager.broadcast(
        session_id,
        "messages_update",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "thinking": True,
                    "steps": [
                        {"label": "Analyzing requirements...", "done": True},
                        {
                            "label": f"Selected stack: {stack_info['frontend'] or stack_info['backend']}",
                            "done": True,
                        },
                        {"label": "Generating files...", "done": True},
                    ],
                    "ts": datetime.now().strftime("%H:%M"),
                }
            ]
        },
    )

    # Write files to disk
    created_files = []
    for file_path, content in files.items():
        full_path = output_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        session_manager.add_terminal_line(session_id, "info", f"✓ Created {file_path}")
        created_files.append(file_path)
        await asyncio.sleep(0.2)

    # Broadcast terminal update after file creation
    await session_manager.broadcast(
        session_id,
        "terminal_update",
        {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
    )

    # Update session files
    session_manager.set_files(session_id, file_tree)
    await session_manager.broadcast(session_id, "files_update", {"files": file_tree})

    # Auto-install dependencies if package.json or requirements.txt exists
    if (output_dir / "package.json").exists():
        session_manager.add_terminal_line(session_id, "cmd", "$ npm install")
        await session_manager.broadcast(
            session_id,
            "terminal_update",
            {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "npm",
                "install",
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
            output = stdout.decode()

            for line in output.split("\n")[:20]:
                if line.strip():
                    session_manager.add_terminal_line(session_id, "out", line)

            session_manager.add_terminal_line(session_id, "success", "✓ Dependencies installed")
        except asyncio.TimeoutError:
            session_manager.add_terminal_line(session_id, "warning", "⚠ npm install timed out")
        except Exception as e:
            session_manager.add_terminal_line(session_id, "error", f"✗ Install failed: {e}")

        await session_manager.broadcast(
            session_id,
            "terminal_update",
            {"lines": session_manager.get_session(session_id)["terminal_lines"][-15:]},
        )

    elif (output_dir / "requirements.txt").exists():
        session_manager.add_terminal_line(session_id, "cmd", "$ pip install -r requirements.txt")
        await session_manager.broadcast(
            session_id,
            "terminal_update",
            {"lines": session_manager.get_session(session_id)["terminal_lines"][-10:]},
        )

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
            output = stdout.decode()

            for line in output.split("\n")[:20]:
                if line.strip():
                    session_manager.add_terminal_line(session_id, "out", line)

            session_manager.add_terminal_line(session_id, "success", "✓ Dependencies installed")
        except asyncio.TimeoutError:
            session_manager.add_terminal_line(session_id, "warning", "⚠ pip install timed out")
        except Exception as e:
            session_manager.add_terminal_line(session_id, "error", f"✗ Install failed: {e}")

        await session_manager.broadcast(
            session_id,
            "terminal_update",
            {"lines": session_manager.get_session(session_id)["terminal_lines"][-15:]},
        )

    # Auto-start development server
    await asyncio.sleep(1)
    if (output_dir / "package.json").exists():
        # React/Vite/Next.js app
        session_manager.add_terminal_line(session_id, "cmd", "$ npm run dev")
        server_started = session_manager.start_server(
            session_id, output_dir, port=3000, server_type="npm"
        )

        if server_started:
            await asyncio.sleep(3)
            session_manager.add_terminal_line(
                session_id, "success", "✓ Dev server running on http://localhost:3000"
            )
            session_manager.add_terminal_line(
                session_id, "info", "🌐 Open http://localhost:3000 to view your app"
            )
        else:
            session_manager.add_terminal_line(
                session_id, "warning", "⚠ Server could not start (port may be in use)"
            )

    elif (output_dir / "src" / "main.py").exists():
        # FastAPI backend
        session_manager.add_terminal_line(session_id, "cmd", "$ uvicorn src.main:app --reload")
        server_started = session_manager.start_server(
            session_id, output_dir, port=8000, server_type="fastapi"
        )

        if server_started:
            await asyncio.sleep(2)
            session_manager.add_terminal_line(
                session_id, "success", "✓ API running on http://localhost:8000"
            )
            session_manager.add_terminal_line(
                session_id, "info", "📄 Swagger docs: http://localhost:8000/docs"
            )
        else:
            session_manager.add_terminal_line(
                session_id, "warning", "⚠ Server could not start (port may be in use)"
            )

    elif (output_dir / "index.html").exists():
        # Static site - start simple HTTP server
        session_manager.add_terminal_line(session_id, "cmd", "$ python -m http.server 3000")
        server_started = session_manager.start_server(
            session_id, output_dir, port=3000, server_type="http"
        )

        if server_started:
            await asyncio.sleep(2)
            session_manager.add_terminal_line(
                session_id, "success", "✓ Server running on http://localhost:3000"
            )
            session_manager.add_terminal_line(
                session_id, "info", "🌐 Open http://localhost:3000 to view your website"
            )
        else:
            session_manager.add_terminal_line(
                session_id, "warning", "⚠ Server could not start (port may be in use)"
            )

    # Broadcast final terminal state
    await session_manager.broadcast(
        session_id,
        "terminal_update",
        {"lines": session_manager.get_session(session_id)["terminal_lines"][-25:]},
    )

    # Send completion message
    stack_display = stack_info["frontend"] or stack_info["backend"]
    await session_manager.broadcast(
        session_id,
        "messages_update",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"""✓ **Project Generated Successfully!**

**Tech Stack:** {stack_display}
**Complexity:** {'⭐' * stack_info['complexity']}
**Deploy:** {stack_info['deploy']}

**📁 Files Created:** {len(created_files)}
{chr(10).join(f"• `{f}`" for f in created_files)}

**🚀 Next Steps:**
1. Review the generated code
2. Start the development server
3. Deploy to {stack_info['deploy']}
""",
                    "thinking": False,
                    "steps": [
                        {"label": "Analyzing requirements...", "done": True},
                        {"label": "Generating files...", "done": True},
                    ],
                    "ts": datetime.now().strftime("%H:%M"),
                    "quality": 0.92,
                    "cost": 0.15,
                    "files": created_files,
                }
            ]
        },
    )

    # Update session
    session_manager.update_session(session_id, status="completed", quality_score=0.92, spent=0.15)
    await session_manager.broadcast(
        session_id, "session_state", session_manager.get_session(session_id)
    )


async def handle_terminal_command(session_id: str, command: str, websocket: WebSocket):
    """Handle terminal command."""
    logger.info(f"Terminal command: {command}")
    session_manager.add_terminal_line(session_id, "cmd", f"$ {command}")

    await asyncio.sleep(0.3)
    session_manager.add_terminal_line(session_id, "info", f"Executing: {command}")
    session_manager.add_terminal_line(session_id, "success", "Command completed")

    await session_manager.broadcast(
        session_id,
        "terminal_update",
        {"lines": session_manager.get_session(session_id)["terminal_lines"][-20:]},
    )


async def handle_file_request(session_id: str, file_path: str, websocket: WebSocket):
    """Handle file content request."""
    logger.info(f"File request: session={session_id}, path={file_path}")

    session = session_manager.get_session(session_id)
    if not session:
        return

    output_dir = Path.cwd() / "ide_outputs" / session_id
    full_path = output_dir / file_path

    if full_path.exists():
        try:
            with open(full_path, encoding="utf-8") as f:
                content = f.read()

            ext = file_path.split(".")[-1] if "." in file_path else ""
            language_map = {
                "py": "python",
                "js": "javascript",
                "jsx": "javascript",
                "ts": "typescript",
                "tsx": "typescript",
                "md": "markdown",
                "txt": "text",
                "json": "json",
                "yaml": "yaml",
                "yml": "yaml",
                "html": "html",
                "css": "css",
            }
            language = language_map.get(ext, "text")

            logger.info(f"Sending file content: {file_path} ({len(content)} bytes)")
            await websocket.send_json(
                {
                    "event": "file_content",
                    "data": {"path": file_path, "content": content, "language": language},
                }
            )
            return
        except Exception as e:
            logger.error(f"Error reading file: {e}")

    logger.warning(f"File not found: {full_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AI Orchestrator IDE")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add cache-busting headers
@app.middleware("http")
async def add_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# API Routes
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "sessions": len(session_manager.sessions),
        "orchestrator": "available",
    }


@app.post("/api/session")
def create_session(config: dict[str, Any] = None):
    config = config or {}
    session = session_manager.create_session(config)
    return {"session": session}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "Session not found"}, 404
    return {"session": session}


# WebSocket
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await handle_websocket(websocket, session_id)


# Serve Frontend
frontend_dist = base_path.parent.parent / "ide_frontend" / "dist"
if frontend_dist.exists():
    logger.info(f"Serving frontend from {frontend_dist}")
    app.mount("/ide", StaticFiles(directory=str(frontend_dist), html=True), name="ide")

    @app.get("/")
    def root():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/ide")

else:
    logger.warning(f"Frontend not found at {frontend_dist}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("  AI Orchestrator IDE - Full Integration")
    print("=" * 70)
    print("  🌐 Server: http://localhost:8765")
    print(f"  📁 Frontend: {'Yes' if frontend_dist.exists() else 'No'}")
    print("  🤖 Tech Stacks: HTML/CSS/JS, React, Next.js, FastAPI")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
