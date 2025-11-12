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

# CORS Configuration
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]
    print("⚠️ WARNING: No ALLOWED_ORIGINS set. Allowing all origins (INSECURE!)")

# Server Configuration
FYERS_PROXY_BASE_URL = os.environ.get("FYERS_PROXY_BASE_URL", "http://localhost:5000")

# Supported resolutions
SECOND_RESOLUTIONS = ["1S", "5S", "10S", "15S", "30S", "45S"]
MINUTE_RESOLUTIONS = ["1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240"]
DAY_RESOLUTIONS = ["1D", "D"]
