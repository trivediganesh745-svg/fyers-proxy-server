import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fyers API Configuration
CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
SECRET_KEY = os.environ.get("FYERS_SECRET_KEY")
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI")
ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")

# Gemini AI Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

# CORS Configuration - Simplified for Google AI Studio
ALLOW_ALL_ORIGINS = os.environ.get("ALLOW_ALL_ORIGINS", "false").lower() == "true"

if ALLOW_ALL_ORIGINS:
    ALLOWED_ORIGINS = "*"
    print("⚠️ CORS: Allowing ALL origins for Google AI Studio compatibility")
else:
    # Specific origins if not allowing all
    allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "")
    ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]
    
    # Add Google domains
    GOOGLE_DOMAINS = [
        "https://aistudio.google.com",
        "https://makersuite.google.com",
        "https://ai.google.dev",
        "https://colab.research.google.com",
        "http://localhost:3000",
        "http://localhost:5000"
    ]
    
    if isinstance(ALLOWED_ORIGINS, list):
        ALLOWED_ORIGINS.extend(GOOGLE_DOMAINS)
        ALLOWED_ORIGINS = list(set(ALLOWED_ORIGINS))  # Remove duplicates
        print(f"✅ CORS configured for specific origins: {ALLOWED_ORIGINS}")

# Server Configuration
FYERS_PROXY_BASE_URL = os.environ.get("FYERS_PROXY_BASE_URL", "http://localhost:5000")
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Supported resolutions
SECOND_RESOLUTIONS = ["1S", "5S", "10S", "15S", "30S", "45S"]
MINUTE_RESOLUTIONS = ["1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240"]
DAY_RESOLUTIONS = ["1D", "D"]
