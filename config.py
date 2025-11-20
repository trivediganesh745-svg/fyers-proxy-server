import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask Config
    SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-here")
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ['true', '1', 'yes']
    
    # Fyers API Configuration
    FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
    FYERS_SECRET_KEY = os.environ.get("FYERS_SECRET_KEY")
    FYERS_REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI")
    FYERS_ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
    FYERS_REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")
    FYERS_PROXY_BASE_URL = os.environ.get("FYERS_PROXY_BASE_URL", "http://localhost:5000")
    
    # Gemini AI Configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    
    # CORS Configuration
    ALLOW_ALL_ORIGINS = os.environ.get("ALLOW_ALL_ORIGINS", "false").lower() in ['true', '1', 'yes']
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "")
    
    # Supported Resolutions
    SECOND_RESOLUTIONS = ["1S", "5S", "10S", "15S", "30S", "45S"]
    MINUTE_RESOLUTIONS = ["1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240"]
    DAY_RESOLUTIONS = ["1D", "D"]
    
    @classmethod
    def get_allowed_origins(cls):
        """Get CORS allowed origins"""
        if cls.ALLOW_ALL_ORIGINS:
            return "*"
        origins = [origin.strip().strip("'\"`") for origin in cls.ALLOWED_ORIGINS.split(',') if origin.strip()]
        return origins if origins else []
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not all([cls.FYERS_CLIENT_ID, cls.FYERS_SECRET_KEY, cls.FYERS_REDIRECT_URI]):
            raise ValueError("ERROR: Fyers API credentials are not fully set.")
