import os
import matplotlib
matplotlib.use('Agg') # Force non-GUI backend to prevent segmentation fault in headless systems
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Create directories if they do not exist
for directory in [UPLOADS_DIR, REPORTS_DIR, DATABASE_DIR, MODELS_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# SQLite Database connection string / file path
DB_PATH = DATABASE_DIR / "data_analyst.db"

# Supported AI Providers
AI_PROVIDERS = ["Gemini (Google)", "OpenAI", "Mock Engine (No Key Required)"]

# Model configurations
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Theme styling settings
APP_NAME = "InsightEngine AI"
APP_SUBTITLE = "Your Intelligent Business Data Analyst"
