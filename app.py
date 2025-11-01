import os
from flask import Flask, request, jsonify, redirect, url_for
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
import time
import logging
import json
import threading
from backtesting.routes import backtesting_bp
from flask_sock import Sock
from typing import List, Dict, Any
import math
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio
import numpy as np
import pickle
import base64
from cryptography.fernet import Fernet
import schedule
import atexit
from functools import wraps
import requests

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Sock for client-facing websocket connections
sock = Sock(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# --- Fyers API Configuration (from environment variables) ---
CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
SECRET_KEY = os.environ.get("FYERS_SECRET_KEY")
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI")
ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")

# --- Enhanced Token Management Configuration ---
TOKEN_FILE = os.environ.get("TOKEN_FILE", "tokens.enc")
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
AUTO_REFRESH_ENABLED = os.environ.get("AUTO_REFRESH_ENABLED", "true").lower() == "true"
TOKEN_REFRESH_INTERVAL = int(os.environ.get("TOKEN_REFRESH_INTERVAL", "3600"))  # 1 hour default
MAX_RETRY_ATTEMPTS = int(os.environ.get("MAX_RETRY_ATTEMPTS", "3"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "5"))  # seconds

# --- Gemini AI Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

# Add a base URL for your proxy for internal calls
app.config["FYERS_PROXY_BASE_URL"] = os.environ.get("FYERS_PROXY_BASE_URL", "http://localhost:5000")

# Supported second-level resolutions
SECOND_RESOLUTIONS = ["1S", "5S", "10S", "15S", "30S", "45S"]
MINUTE_RESOLUTIONS = ["1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240"]
DAY_RESOLUTIONS = ["1D", "D"]

# Initialize Gemini AI
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # Configure the model with specific settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        app.logger.info(f"Gemini AI initialized with model: {GEMINI_MODEL}")
    except Exception as e:
        app.logger.error(f"Failed to initialize Gemini AI: {e}")
        gemini_model = None
else:
    app.logger.warning("GEMINI_API_KEY not found. AI features will be disabled.")

if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
    app.logger.error("ERROR: Fyers API credentials (CLIENT_ID, SECRET_KEY, REDIRECT_URI) are not fully set.")

# Initialize FyersModel
fyers_instance = None 

# Token management variables
token_expiry_time = None
refresh_scheduler_thread = None
token_lock = threading.Lock()
cipher_suite = None

# Initialize encryption if key is provided
if ENCRYPTION_KEY:
    if len(ENCRYPTION_KEY) < 32:
        # Pad the key if it's too short
        ENCRYPTION_KEY = ENCRYPTION_KEY.ljust(32, '0')
    cipher_suite = Fernet(base64.urlsafe_b64encode(ENCRYPTION_KEY[:32].encode()))
else:
    # Generate a new encryption key if not provided
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    cipher_suite = Fernet(ENCRYPTION_KEY.encode())
    app.logger.warning(f"Generated new encryption key. Save this in your .env file: ENCRYPTION_KEY={ENCRYPTION_KEY}")

# --- Enhanced Token Storage and Management ---

class TokenManager:
    """Enhanced token management with encryption and persistence"""
    
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        self.last_refresh = None
        self.refresh_count = 0
        
    def save_tokens(self, access_token, refresh_token, expiry_hours=24):
        """Save tokens with encryption and expiry tracking"""
        with token_lock:
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.token_expiry = datetime.datetime.now() + datetime.timedelta(hours=expiry_hours)
            self.last_refresh = datetime.datetime.now()
            self.refresh_count += 1
            
            # Save to encrypted file
            if cipher_suite:
                token_data = {
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'token_expiry': self.token_expiry.isoformat(),
                    'last_refresh': self.last_refresh.isoformat(),
                    'refresh_count': self.refresh_count
                }
                try:
                    encrypted_data = cipher_suite.encrypt(json.dumps(token_data).encode())
                    with open(TOKEN_FILE, 'wb') as f:
                        f.write(encrypted_data)
                    app.logger.info(f"Tokens saved successfully (Refresh #{self.refresh_count})")
                except Exception as e:
                    app.logger.error(f"Failed to save tokens: {e}")
    
    def load_tokens(self):
        """Load tokens from encrypted file"""
        if not os.path.exists(TOKEN_FILE):
            app.logger.info("No saved tokens found")
            return False
            
        try:
            with open(TOKEN_FILE, 'rb') as f:
                encrypted_data = f.read()
            
            if cipher_suite:
                decrypted_data = cipher_suite.decrypt(encrypted_data)
                token_data = json.loads(decrypted_data.decode())
                
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                self.token_expiry = datetime.datetime.fromisoformat(token_data.get('token_expiry'))
                self.last_refresh = datetime.datetime.fromisoformat(token_data.get('last_refresh'))
                self.refresh_count = token_data.get('refresh_count', 0)
                
                # Update global variables for backward compatibility
                global ACCESS_TOKEN, REFRESH_TOKEN
                ACCESS_TOKEN = self.access_token
                REFRESH_TOKEN = self.refresh_token
                
                app.logger.info(f"Tokens loaded successfully (Refresh count: {self.refresh_count})")
                return True
        except Exception as e:
            app.logger.error(f"Failed to load tokens: {e}")
            return False
    
    def is_token_valid(self):
        """Check if current token is valid and not expired"""
        if not self.access_token:
            return False
        
        if self.token_expiry and datetime.datetime.now() >= self.token_expiry:
            app.logger.warning("Token has expired")
            return False
            
        return True
    
    def should_refresh(self, buffer_minutes=30):
        """Check if token should be refreshed (30 minutes before expiry)"""
        if not self.token_expiry:
            return True
        
        time_until_expiry = self.token_expiry - datetime.datetime.now()
        return time_until_expiry.total_seconds() < (buffer_minutes * 60)

# Initialize token manager
token_manager = TokenManager()

def with_retry(max_attempts=MAX_RETRY_ATTEMPTS, delay=RETRY_DELAY):
    """Decorator for retrying failed operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    app.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            app.logger.error(f"All {max_attempts} attempts failed")
            raise last_exception
        return wrapper
    return decorator

@with_retry()
def initialize_fyers_model(token=None):
    """Initialize Fyers model with retry logic"""
    global fyers_instance, ACCESS_TOKEN, REFRESH_TOKEN

    token_to_use = token if token else token_manager.access_token or ACCESS_TOKEN

    if token_to_use:
        try:
            try:
                fyers_instance = fyersModel.FyersModel(token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            except TypeError:
                fyers_instance = fyersModel.FyersModel(access_token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            
            # Test the connection
            test_result = fyers_instance.get_profile()
            if test_result and test_result.get("s") == "ok":
                app.logger.info("FyersModel initialized and verified successfully")
                return True
            else:
                raise Exception(f"Token validation failed: {test_result}")
                
        except Exception as e:
            app.logger.error(f"Failed to initialize fyers_model: {e}")
            
            # Try to refresh if we have a refresh token
            if token_manager.refresh_token or REFRESH_TOKEN:
                app.logger.info("Attempting automatic token refresh...")
                if refresh_access_token():
                    return True
            raise e
            
    elif token_manager.refresh_token or REFRESH_TOKEN:
        app.logger.info("No access token found, attempting to use refresh token")
        if refresh_access_token():
            return True
        else:
            app.logger.error("Could not initialize FyersModel: Refresh token failed")
            return False
    else:
        app.logger.warning("No tokens available for initialization")
        return False

@with_retry()
def refresh_access_token():
    """Refresh access token with retry logic"""
    global ACCESS_TOKEN, REFRESH_TOKEN
    
    refresh_token_to_use = token_manager.refresh_token or REFRESH_TOKEN
    
    if not refresh_token_to_use:
        app.logger.error("Cannot refresh token: No refresh token available")
        return False

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        state="refresh_state", 
        secret_key=SECRET_KEY,
        grant_type="refresh_token"
    )
    session.set_token(refresh_token_to_use)

    try:
        app.logger.info(f"Attempting to refresh access token...")
        response = session.generate_token()

        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response.get("refresh_token", refresh_token_to_use)

            # Save tokens using token manager
            token_manager.save_tokens(new_access_token, new_refresh_token)
            
            # Update global variables
            ACCESS_TOKEN = new_access_token
            REFRESH_TOKEN = new_refresh_token
            
            # Reinitialize Fyers model
            initialize_fyers_model(new_access_token)
            
            app.logger.info(f"Access token refreshed successfully (Refresh #{token_manager.refresh_count})")
            return True
        else:
            app.logger.error(f"Failed to refresh access token: {response}")
            return False
    except Exception as e:
        app.logger.error(f"Error during access token refresh: {e}")
        raise e

def schedule_token_refresh():
    """Schedule automatic token refresh"""
    def refresh_job():
        try:
            if token_manager.should_refresh():
                app.logger.info("Scheduled token refresh triggered")
                refresh_access_token()
        except Exception as e:
            app.logger.error(f"Scheduled refresh failed: {e}")
    
    # Schedule refresh every hour
    schedule.every(TOKEN_REFRESH_INTERVAL).seconds.do(refresh_job)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    global refresh_scheduler_thread
    refresh_scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    refresh_scheduler_thread.start()
    app.logger.info(f"Token refresh scheduler started (interval: {TOKEN_REFRESH_INTERVAL} seconds)")

def automatic_login():
    """Attempt automatic login using saved credentials"""
    try:
        # First try to load saved tokens
        if token_manager.load_tokens():
            if token_manager.is_token_valid():
                app.logger.info("Using saved valid tokens")
                if initialize_fyers_model():
                    return True
            else:
                app.logger.info("Saved tokens expired, attempting refresh")
                if refresh_access_token():
                    return True
        
        # If we have credentials but no valid tokens, attempt fresh login
        if all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
            app.logger.info("No valid tokens found, automatic login requires manual authentication")
            # You could implement automated browser login here using selenium if needed
            return False
            
    except Exception as e:
        app.logger.error(f"Automatic login failed: {e}")
        return False

# --- Health Check and Monitoring ---

@app.route('/api/health')
def health_check():
    """Health check endpoint with token status"""
    status = {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "token_status": {
            "has_access_token": bool(token_manager.access_token),
            "has_refresh_token": bool(token_manager.refresh_token),
            "is_valid": token_manager.is_token_valid(),
            "expiry": token_manager.token_expiry.isoformat() if token_manager.token_expiry else None,
            "last_refresh": token_manager.last_refresh.isoformat() if token_manager.last_refresh else None,
            "refresh_count": token_manager.refresh_count,
            "should_refresh": token_manager.should_refresh() if token_manager.token_expiry else None
        },
        "fyers_api": {
            "initialized": fyers_instance is not None,
            "client_id": CLIENT_ID[:4] + "****" if CLIENT_ID else None
        },
        "ai_status": {
            "enabled": gemini_model is not None,
            "model": GEMINI_MODEL if gemini_model else None
        }
    }
    
    # Test Fyers connection if initialized
    if fyers_instance:
        try:
            profile = fyers_instance.get_profile()
            if profile and profile.get("s") == "ok":
                status["fyers_api"]["connected"] = True
                status["fyers_api"]["user"] = profile.get("data", {}).get("name", "Unknown")
            else:
                status["fyers_api"]["connected"] = False
                status["fyers_api"]["error"] = str(profile)
        except Exception as e:
            status["fyers_api"]["connected"] = False
            status["fyers_api"]["error"] = str(e)
    
    return jsonify(status)

@app.route('/api/force-refresh', methods=['POST'])
def force_token_refresh():
    """Force token refresh endpoint"""
    try:
        if refresh_access_token():
            return jsonify({
                "success": True,
                "message": "Token refreshed successfully",
                "refresh_count": token_manager.refresh_count,
                "new_expiry": token_manager.token_expiry.isoformat() if token_manager.token_expiry else None
            })
        else:
            return jsonify({
                "success": False,
                "message": "Token refresh failed"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# --- Helper function to wrap Fyers API calls with automatic retry and refresh ---
def make_fyers_api_call(api_method, *args, **kwargs):
    """Enhanced API call wrapper with automatic token refresh and retry"""
    global fyers_instance
    
    max_attempts = 2
    for attempt in range(max_attempts):
        if not fyers_instance:
            app.logger.warning("Fyers API not initialized. Attempting automatic initialization...")
            if not automatic_login() and not initialize_fyers_model():
                return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401

        try:
            result = api_method(*args, **kwargs)
            
            # Check if token expired in response
            if isinstance(result, dict):
                error_code = result.get("code")
                error_message = str(result.get("message", "")).lower()
                
                if error_code in [-2, -3, -10, -100] or any(x in error_message for x in ["token", "expired", "invalid", "authenticate"]):
                    if attempt == 0:  # First attempt, try to refresh
                        app.logger.warning(f"Token error detected: {result}. Attempting refresh...")
                        if refresh_access_token():
                            continue  # Retry with new token
                    
                    return jsonify({"error": "Authentication failed. Please re-login."}), 401
            
            return result
            
        except Exception as e:
            error_message = str(e).lower()
            
            if any(x in error_message for x in ["token", "authenticated", "login", "invalid_access_token", "expired"]):
                if attempt == 0:  # First attempt, try to refresh
                    app.logger.warning(f"Token error in API call: {e}. Attempting refresh...")
                    if refresh_access_token():
                        continue  # Retry with new token
                
                return jsonify({"error": "Authentication failed. Please re-login."}), 401
            else:
                app.logger.error(f"API call failed: {e}")
                return jsonify({"error": f"API error: {str(e)}"}), 500
    
    return jsonify({"error": "Failed after maximum retry attempts"}), 500

# --- AI Analysis Functions ---

def analyze_market_data_with_ai(data: Dict[str, Any], analysis_type: str = "general") -> Dict[str, Any]:
    """
    Use Gemini AI to analyze market data and provide insights.
    
    Args:
        data: Market data to analyze
        analysis_type: Type of analysis (general, technical, sentiment, risk)
    
    Returns:
        AI analysis response
    """
    if not gemini_model:
        return {"error": "AI model not initialized"}

    try:
        prompts = {
            "general": f"""
                Analyze the following market data and provide insights:
                {json.dumps(data, indent=2)}
                
                Please provide:
                1. Key observations
                2. Trend analysis
                3. Risk factors
                4. Potential opportunities
                5. Recommended actions
                
                Format the response in a clear, structured manner.
            """,
            "technical": f"""
                Perform technical analysis on the following market data:
                {json.dumps(data, indent=2)}
                
                Include:
                1. Support and resistance levels
                2. Trend direction and strength
                3. Volume analysis
                4. Key technical indicators interpretation
                5. Entry and exit points
                
                Be specific with price levels and percentages.
            """,
            "sentiment": f"""
                Analyze market sentiment based on the following data:
                {json.dumps(data, indent=2)}
                
                Provide:
                1. Overall market sentiment (bullish/bearish/neutral)
                2. Sentiment strength (1-10 scale)
                3. Key sentiment drivers
                4. Potential sentiment shifts
                5. Contrarian opportunities
            """,
            "risk": f"""
                Perform risk analysis on the following market data:
                {json.dumps(data, indent=2)}
                
                Include:
                1. Risk level assessment (low/medium/high)
                2. Specific risk factors
                3. Volatility analysis
                4. Risk mitigation strategies
                5. Position sizing recommendations
                
                Provide specific percentages and thresholds.
            """
        }

        prompt = prompts.get(analysis_type, prompts["general"])
        response = gemini_model.generate_content(prompt)

        return {
            "analysis_type": analysis_type,
            "analysis": response.text,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        app.logger.error(f"AI analysis error: {e}")
        return {"error": str(e)}

def generate_trading_signals_with_ai(symbol: str, historical_data: List, current_price: float) -> Dict[str, Any]:
    """
    Generate trading signals using AI based on historical data and current price.
    """
    if not gemini_model:
        return {"error": "AI model not initialized"}

    try:
        # Prepare data for AI analysis
        recent_candles = historical_data[-20:] if len(historical_data) > 20 else historical_data

        prompt = f"""
        As a trading analyst, generate trading signals for {symbol} based on the following data:
        
        Current Price: {current_price}
        Recent Historical Data (last {len(recent_candles)} candles):
        {json.dumps(recent_candles, indent=2)}
        
        Provide:
        1. Signal: BUY/SELL/HOLD
        2. Confidence Level: (0-100%)
        3. Entry Price
        4. Stop Loss
        5. Target Price(s) - provide 3 targets
        6. Risk-Reward Ratio
        7. Time Horizon
        8. Key Reasoning
        
        Format as JSON for easy parsing.
        """

        response = gemini_model.generate_content(prompt)

        # Try to parse the response as JSON
        try:
            # Extract JSON from the response text
            response_text = response.text
            # Find JSON content between ```json and ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                signal_data = json.loads(json_match.group(1))
            else:
                # Try to parse the entire response as JSON
                signal_data = json.loads(response_text)
        except:
            # If JSON parsing fails, return the raw text
            signal_data = {"raw_analysis": response.text}

        return {
            "symbol": symbol,
            "current_price": current_price,
            "signals": signal_data,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        app.logger.error(f"AI signal generation error: {e}")
        return {"error": str(e)}

def calculate_advanced_indicators(candles: List) -> Dict[str, Any]:
    """
    Calculate advanced technical indicators for AI analysis.
    """
    if not candles or len(candles) < 2:
        return {}

    try:
        # Extract OHLCV data
        closes = [c[4] for c in candles]  # Close prices
        highs = [c[2] for c in candles]   # High prices
        lows = [c[3] for c in candles]    # Low prices
        volumes = [c[5] for c in candles] if len(candles[0]) > 5 else []

        # Calculate basic indicators
        indicators = {
            "sma_10": np.mean(closes[-10:]) if len(closes) >= 10 else None,
            "sma_20": np.mean(closes[-20:]) if len(closes) >= 20 else None,
            "sma_50": np.mean(closes[-50:]) if len(closes) >= 50 else None,
            "current_price": closes[-1],
            "price_change": closes[-1] - closes[-2] if len(closes) >= 2 else 0,
            "price_change_pct": ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 and closes[-2] != 0 else 0,
            "high_20": max(highs[-20:]) if len(highs) >= 20 else max(highs),
            "low_20": min(lows[-20:]) if len(lows) >= 20 else min(lows),
            "avg_volume": np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes) if volumes else None,
            "volume_ratio": volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 and np.mean(volumes[-20:]) != 0 else None
        }

        # Calculate RSI (14-period)
        if len(closes) >= 15:
            price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [change if change > 0 else 0 for change in price_changes]
            losses = [-change if change < 0 else 0 for change in price_changes]

            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                indicators["rsi"] = 100 - (100 / (1 + rs))
            else:
                indicators["rsi"] = 100 if avg_gain > 0 else 50

        return indicators

    except Exception as e:
        app.logger.error(f"Error calculating indicators: {e}")
        return {}

# --- Fyers Authentication Flow Endpoints ---

@app.route('/fyers-login')
def fyers_login():
    """Initiates the Fyers authentication flow."""
    if not CLIENT_ID or not REDIRECT_URI or not SECRET_KEY:
        app.logger.error("Fyers API credentials not fully configured for login.")
        return jsonify({"error": "Fyers API credentials not fully configured on the server."}), 500

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        state="fyers_proxy_state",
        secret_key=SECRET_KEY,
        grant_type="authorization_code"
    )
    generate_token_url = session.generate_authcode()
    app.logger.info(f"Redirecting to Fyers login: {generate_token_url}")
    return redirect(generate_token_url)

@app.route('/fyers-auth-callback')
def fyers_auth_callback():
    """Callback endpoint after the user logs in on Fyers."""
    auth_code = request.args.get('auth_code')
    state = request.args.get('state')
    error = request.args.get('error')

    if error:
        app.logger.error(f"Fyers authentication failed: {error}")
        return jsonify({"error": f"Fyers authentication failed: {error}"}), 400
    if not auth_code:
        app.logger.error("No auth_code received from Fyers.")
        return jsonify({"error": "No auth_code received from Fyers."}), 400

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        state=state,
        secret_key=SECRET_KEY,
        grant_type="authorization_code"
    )
    session.set_token(auth_code)
    try:
        response = session.generate_token()

        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response["refresh_token"]

            # Save tokens using token manager
            token_manager.save_tokens(new_access_token, new_refresh_token)
            
            # Update global variables
            global ACCESS_TOKEN, REFRESH_TOKEN
            ACCESS_TOKEN = new_access_token
            REFRESH_TOKEN = new_refresh_token
            
            # Initialize Fyers model
            initialize_fyers_model(new_access_token)

            app.logger.info("Fyers tokens generated successfully!")
            return jsonify({
                "message": "Fyers tokens generated successfully!",
                "access_token_available": True,
                "token_expiry": token_manager.token_expiry.isoformat() if token_manager.token_expiry else None
            })
        else:
            app.logger.error(f"Failed to generate Fyers tokens. Response: {response}")
            return jsonify({"error": f"Failed to generate Fyers tokens. Response: {response}"}), 500

    except Exception as e:
        app.logger.error(f"Error generating Fyers access token: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate Fyers access token: {str(e)}"}), 500

# --- Fyers Data Endpoints (keeping all existing endpoints) ---

@app.route('/api/fyers/profile')
def get_profile():
    result = make_fyers_api_call(fyers_instance.get_profile)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/funds')
def get_funds():
    result = make_fyers_api_call(fyers_instance.funds)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/holdings')
def get_holdings():
    result = make_fyers_api_call(fyers_instance.holdings)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

# [Keep all other existing endpoints as they are - I'm not repeating them to save space]
# This includes: history, quotes, market_depth, option_chain, etc.

# --- Startup and Cleanup ---

def cleanup():
    """Cleanup function to run on app shutdown"""
    app.logger.info("Performing cleanup...")
    # Save current tokens if valid
    if token_manager.access_token:
        token_manager.save_tokens(token_manager.access_token, token_manager.refresh_token)
    
    # Close websocket connections
    global _fyers_data_socket
    if _fyers_data_socket:
        try:
            _fyers_data_socket.close()
        except:
            pass

# Register cleanup
atexit.register(cleanup)

# --- WebSocket Integration Section (keeping existing implementation) ---
_fyers_data_socket = None
_fyers_socket_thread = None
_connected_clients = []
_subscribed_symbols = set()
_socket_lock = threading.Lock()
_socket_running = False

# [Keep all existing WebSocket functions as they are]

# --- Gemini AI Endpoints (keeping existing implementation) ---
# [Keep all existing AI endpoints as they are]

# --- Main Routes ---

@app.route('/')
def home():
    return jsonify({
        "message": "Fyers API Proxy Server with AI Integration and Automatic Login is running!",
        "endpoints": {
            "authentication": {
                "login": "/fyers-login",
                "health": "/api/health",
                "force_refresh": "/api/force-refresh"
            },
            "historical_data": "/api/fyers/history",
            "ai_endpoints": {
                "analyze": "/api/ai/analyze",
                "trading_signals": "/api/ai/trading-signals",
                "chat": "/api/ai/chat",
                "portfolio_analysis": "/api/ai/portfolio-analysis",
                "market_summary": "/api/ai/market-summary"
            },
            "supported_resolutions": {
                "second": SECOND_RESOLUTIONS,
                "minute": MINUTE_RESOLUTIONS,
                "day": DAY_RESOLUTIONS
            },
            "ai_status": "enabled" if gemini_model else "disabled (set GEMINI_API_KEY)",
            "auto_login_status": {
                "enabled": AUTO_REFRESH_ENABLED,
                "token_valid": token_manager.is_token_valid(),
                "refresh_interval": TOKEN_REFRESH_INTERVAL
            }
        }
    })

# Register the backtesting blueprint
app.register_blueprint(backtesting_bp)

if __name__ == '__main__':
    # Startup initialization
    app.logger.info("=" * 50)
    app.logger.info("Starting Fyers API Proxy Server")
    app.logger.info("=" * 50)
    
    # Attempt automatic login on startup
    if automatic_login():
        app.logger.info("✓ Automatic login successful")
    else:
        app.logger.warning("⚠ Automatic login failed - manual authentication required")
    
    # Start automatic token refresh scheduler if enabled
    if AUTO_REFRESH_ENABLED:
        schedule_token_refresh()
        app.logger.info("✓ Automatic token refresh enabled")
    
    # Start the Fyers data socket if we have a valid token
    if token_manager.access_token:
        try:
            _start_data_socket_in_thread(token_manager.access_token)
            app.logger.info("✓ WebSocket data stream initialized")
        except Exception as e:
            app.logger.warning(f"⚠ Could not start WebSocket: {e}")
    
    app.logger.info("=" * 50)
    app.logger.info("Server ready! Access at http://localhost:5000")
    app.logger.info("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)