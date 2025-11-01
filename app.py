import os
from flask import Flask, request, jsonify, redirect, url_for, session
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
from typing import List, Dict, Any, Optional
import math
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio
import numpy as np
from functools import wraps
import secrets

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)

# Configure session with secure secret key
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = True  # Enable in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=8)

# Enable CORS with secure settings
CORS(app, supports_credentials=True, resources={
    r"/api/*": {
        "origins": os.environ.get("ALLOWED_ORIGINS", "*").split(","),
        "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize Sock for client-facing websocket connections
sock = Sock(app)

# Configure logging - DO NOT LOG SENSITIVE DATA
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
app.logger.setLevel(logging.INFO)

# --- Fyers API Configuration (from environment variables) ---
CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
SECRET_KEY = os.environ.get("FYERS_SECRET_KEY")
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI")

# DO NOT auto-load tokens - require explicit authentication
# Removed: ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
# Removed: REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")

# --- Gemini AI Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Add a base URL for your proxy for internal calls
app.config["FYERS_PROXY_BASE_URL"] = os.environ.get("FYERS_PROXY_BASE_URL", "http://localhost:5000")

# Supported second-level resolutions
SECOND_RESOLUTIONS = ["1S", "5S", "10S", "15S", "30S", "45S"]
MINUTE_RESOLUTIONS = ["1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240"]
DAY_RESOLUTIONS = ["1D", "D"]

# Session storage for user-specific Fyers instances
# Key: session_id, Value: {"fyers_instance": FyersModel, "access_token": str, "refresh_token": str, "expires_at": timestamp}
user_sessions = {}
session_lock = threading.Lock()

# Initialize Gemini AI
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

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

        app.logger.info(f"Gemini AI initialized successfully with model: {GEMINI_MODEL}")
    except Exception as e:
        app.logger.error(f"Failed to initialize Gemini AI: {e}")
        gemini_model = None
else:
    app.logger.warning("GEMINI_API_KEY not found. AI features will be disabled.")

if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
    app.logger.error("ERROR: Fyers API credentials (CLIENT_ID, SECRET_KEY, REDIRECT_URI) are not fully set.")
    app.logger.error("Please set these environment variables to enable Fyers integration.")

# --- Authentication Decorator ---

def require_authentication(f):
    """Decorator to ensure user is authenticated before accessing Fyers API endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({
                "error": "Authentication required",
                "message": "Please authenticate first by visiting /fyers-login",
                "authenticated": False
            }), 401
        
        with session_lock:
            user_session = user_sessions.get(session_id)
        
        if not user_session:
            session.clear()
            return jsonify({
                "error": "Session expired",
                "message": "Please re-authenticate by visiting /fyers-login",
                "authenticated": False
            }), 401
        
        # Check if token is expired
        expires_at = user_session.get('expires_at', 0)
        if time.time() > expires_at:
            app.logger.info(f"Token expired for session {session_id[:8]}...")
            # Attempt to refresh
            if not refresh_user_token(session_id):
                with session_lock:
                    user_sessions.pop(session_id, None)
                session.clear()
                return jsonify({
                    "error": "Token expired",
                    "message": "Token refresh failed. Please re-authenticate.",
                    "authenticated": False
                }), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

# --- Token Management Functions ---

def create_user_session(session_id: str, access_token: str, refresh_token: str) -> bool:
    """Create a new user session with Fyers credentials."""
    try:
        # Initialize FyersModel for this user
        try:
            fyers_instance = fyersModel.FyersModel(
                token=access_token,
                is_async=False,
                client_id=CLIENT_ID,
                log_path=""
            )
        except TypeError:
            fyers_instance = fyersModel.FyersModel(
                access_token=access_token,
                is_async=False,
                client_id=CLIENT_ID,
                log_path=""
            )
        
        # Calculate token expiration (typically 24 hours for Fyers)
        expires_at = time.time() + (24 * 60 * 60)  # 24 hours
        
        with session_lock:
            user_sessions[session_id] = {
                "fyers_instance": fyers_instance,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "created_at": time.time()
            }
        
        app.logger.info(f"User session created successfully for session {session_id[:8]}...")
        return True
        
    except Exception as e:
        app.logger.error(f"Failed to create user session: {e}")
        return False

def get_user_fyers_instance(session_id: str) -> Optional[fyersModel.FyersModel]:
    """Get the FyersModel instance for a specific user session."""
    with session_lock:
        user_session = user_sessions.get(session_id)
    
    if user_session:
        return user_session.get("fyers_instance")
    return None

def refresh_user_token(session_id: str) -> bool:
    """Refresh the access token for a specific user session."""
    with session_lock:
        user_session = user_sessions.get(session_id)
    
    if not user_session:
        app.logger.warning(f"No session found for refresh: {session_id[:8]}...")
        return False
    
    refresh_token = user_session.get("refresh_token")
    if not refresh_token:
        app.logger.error(f"No refresh token available for session {session_id[:8]}...")
        return False
    
    try:
        session_model = fyersModel.SessionModel(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            response_type="code",
            state="refresh_state",
            secret_key=SECRET_KEY,
            grant_type="refresh_token"
        )
        session_model.set_token(refresh_token)
        
        app.logger.info(f"Attempting to refresh token for session {session_id[:8]}...")
        response = session_model.generate_token()
        
        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response.get("refresh_token", refresh_token)
            
            # Update the session with new tokens
            return create_user_session(session_id, new_access_token, new_refresh_token)
        else:
            app.logger.error(f"Failed to refresh token. Response: {response}")
            return False
            
    except Exception as e:
        app.logger.error(f"Error during token refresh: {e}")
        return False

def cleanup_expired_sessions():
    """Remove expired sessions from memory."""
    current_time = time.time()
    with session_lock:
        expired = [
            sid for sid, data in user_sessions.items()
            if current_time > data.get('expires_at', 0) + 3600  # 1 hour grace period
        ]
        for sid in expired:
            user_sessions.pop(sid, None)
            app.logger.info(f"Cleaned up expired session: {sid[:8]}...")

# Schedule periodic cleanup
def periodic_cleanup():
    """Run periodic cleanup of expired sessions."""
    while True:
        time.sleep(3600)  # Run every hour
        try:
            cleanup_expired_sessions()
        except Exception as e:
            app.logger.error(f"Error in periodic cleanup: {e}")

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True, name="SessionCleanupThread")
cleanup_thread.start()

# --- Helper function for Fyers API calls with per-user token management ---

def make_fyers_api_call(session_id: str, api_method, *args, **kwargs):
    """
    Make a Fyers API call with automatic token refresh for a specific user session.
    """
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not fyers_instance:
        return jsonify({
            "error": "Not authenticated",
            "message": "Please authenticate first"
        }), 401
    
    try:
        # Get the method from the instance
        method = getattr(fyers_instance, api_method.__name__)
        return method(*args, **kwargs)
        
    except Exception as e:
        error_message = str(e).lower()
        
        # Check if it's a token-related error
        if any(keyword in error_message for keyword in ["token", "authenticated", "login", "invalid_access_token", "unauthorized"]):
            app.logger.warning(f"Access token issue detected for session {session_id[:8]}... Attempting refresh.")
            
            if refresh_user_token(session_id):
                app.logger.info("Token refreshed successfully, retrying request...")
                fyers_instance = get_user_fyers_instance(session_id)
                if fyers_instance:
                    method = getattr(fyers_instance, api_method.__name__)
                    return method(*args, **kwargs)
            
            # Refresh failed
            app.logger.error("Token refresh failed")
            with session_lock:
                user_sessions.pop(session_id, None)
            
            return jsonify({
                "error": "Authentication failed",
                "message": "Token expired and refresh failed. Please re-authenticate."
            }), 401
        else:
            # Non-token error
            app.logger.error(f"Fyers API error: {e}")
            return jsonify({"error": f"Fyers API error: {str(e)}"}), 500

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

        try:
            import re
            response_text = response.text
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                signal_data = json.loads(json_match.group(1))
            else:
                signal_data = json.loads(response_text)
        except:
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
        closes = [c[4] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        volumes = [c[5] for c in candles] if len(candles[0]) > 5 else []

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

# --- Authentication Status Endpoint ---

@app.route('/api/auth/status')
def auth_status():
    """Check authentication status."""
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({
            "authenticated": False,
            "message": "Not authenticated",
            "login_url": "/fyers-login"
        })
    
    with session_lock:
        user_session = user_sessions.get(session_id)
    
    if not user_session:
        session.clear()
        return jsonify({
            "authenticated": False,
            "message": "Session expired",
            "login_url": "/fyers-login"
        })
    
    expires_at = user_session.get('expires_at', 0)
    created_at = user_session.get('created_at', 0)
    
    return jsonify({
        "authenticated": True,
        "session_created": datetime.datetime.fromtimestamp(created_at).isoformat(),
        "token_expires": datetime.datetime.fromtimestamp(expires_at).isoformat(),
        "time_remaining": max(0, int(expires_at - time.time())),
        "session_id": session_id[:8] + "..."  # Partial session ID for debugging
    })

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout and clear session."""
    session_id = session.get('session_id')
    
    if session_id:
        with session_lock:
            user_sessions.pop(session_id, None)
        app.logger.info(f"User logged out: {session_id[:8]}...")
    
    session.clear()
    
    return jsonify({
        "message": "Logged out successfully",
        "authenticated": False
    })

# --- Fyers Authentication Flow Endpoints ---

@app.route('/fyers-login')
def fyers_login():
    """Initiates the Fyers authentication flow."""
    if not CLIENT_ID or not REDIRECT_URI or not SECRET_KEY:
        app.logger.error("Fyers API credentials not fully configured for login.")
        return jsonify({
            "error": "Server configuration error",
            "message": "Fyers API credentials not configured. Please contact administrator."
        }), 500

    # Generate a unique state for CSRF protection
    state = secrets.token_urlsafe(32)
    session['auth_state'] = state
    
    session_model = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        state=state,
        secret_key=SECRET_KEY,
        grant_type="authorization_code"
    )
    
    generate_token_url = session_model.generate_authcode()
    app.logger.info("Redirecting to Fyers login page")
    
    return redirect(generate_token_url)

@app.route('/fyers-auth-callback')
def fyers_auth_callback():
    """Callback endpoint after the user logs in on Fyers."""
    auth_code = request.args.get('auth_code')
    state = request.args.get('state')
    error = request.args.get('error')
    
    # Validate state for CSRF protection
    expected_state = session.get('auth_state')
    if not expected_state or state != expected_state:
        app.logger.error("State mismatch - possible CSRF attack")
        return jsonify({
            "error": "Invalid state parameter",
            "message": "Authentication failed due to security check"
        }), 400
    
    # Clear the state from session
    session.pop('auth_state', None)

    if error:
        app.logger.error(f"Fyers authentication failed: {error}")
        return jsonify({
            "error": "Authentication failed",
            "details": error
        }), 400
        
    if not auth_code:
        app.logger.error("No auth_code received from Fyers")
        return jsonify({
            "error": "Authentication failed",
            "message": "No authorization code received"
        }), 400

    try:
        session_model = fyersModel.SessionModel(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            response_type="code",
            state=state,
            secret_key=SECRET_KEY,
            grant_type="authorization_code"
        )
        session_model.set_token(auth_code)
        
        response = session_model.generate_token()

        if response and response.get("s") == "ok":
            access_token = response["access_token"]
            refresh_token = response["refresh_token"]
            
            # Generate a unique session ID
            session_id = secrets.token_urlsafe(32)
            session['session_id'] = session_id
            session.permanent = True
            
            # Create user session
            if create_user_session(session_id, access_token, refresh_token):
                app.logger.info(f"User authenticated successfully: {session_id[:8]}...")
                
                return jsonify({
                    "message": "Authentication successful",
                    "authenticated": True,
                    "session_id": session_id[:8] + "...",
                    "redirect": "/"
                })
            else:
                return jsonify({
                    "error": "Session creation failed",
                    "message": "Failed to create user session"
                }), 500
        else:
            app.logger.error(f"Failed to generate tokens. Response: {response}")
            return jsonify({
                "error": "Token generation failed",
                "details": response
            }), 500

    except Exception as e:
        app.logger.error(f"Error during authentication: {e}")
        return jsonify({
            "error": "Authentication error",
            "message": str(e)
        }), 500

# --- Gemini AI Endpoints ---

@app.route('/api/ai/analyze', methods=['POST'])
def ai_analyze():
    """
    Analyze market data using Gemini AI.
    No authentication required for AI-only features.
    """
    if not gemini_model:
        return jsonify({
            "error": "AI model not initialized",
            "message": "Please set GEMINI_API_KEY environment variable"
        }), 503

    request_data = request.json
    if not request_data or not request_data.get("data"):
        return jsonify({"error": "Missing 'data' in request body"}), 400

    analysis_type = request_data.get("analysis_type", "general")
    data = request_data.get("data")

    result = analyze_market_data_with_ai(data, analysis_type)
    return jsonify(result)

@app.route('/api/ai/trading-signals', methods=['POST'])
@require_authentication
def ai_trading_signals():
    """
    Generate trading signals using AI.
    Requires authentication if using live data.
    """
    if not gemini_model:
        return jsonify({
            "error": "AI model not initialized",
            "message": "Please set GEMINI_API_KEY"
        }), 503

    request_data = request.json
    symbol = request_data.get("symbol")
    use_live_data = request_data.get("use_live_data", False)

    if not symbol:
        return jsonify({"error": "Missing 'symbol' in request body"}), 400

    session_id = session.get('session_id')

    try:
        if use_live_data:
            fyers_instance = get_user_fyers_instance(session_id)
            if not fyers_instance:
                return jsonify({"error": "Authentication required for live data"}), 401

            # Get current quote
            quote_result = make_fyers_api_call(session_id, fyers_instance.quotes, data={"symbols": symbol})
            
            if isinstance(quote_result, tuple):
                return quote_result
            
            current_price = None
            if quote_result and isinstance(quote_result, dict):
                quote_data = quote_result.get("d", [{}])[0]
                current_price = quote_data.get("v", {}).get("lp", 0)

            # Get historical data
            end_time = int(time.time())
            start_time = end_time - (30 * 24 * 60 * 60)

            history_data = {
                "symbol": symbol,
                "resolution": "1D",
                "date_format": 0,
                "range_from": str(start_time),
                "range_to": str(end_time),
                "cont_flag": 1
            }

            history_result = make_fyers_api_call(session_id, fyers_instance.history, data=history_data)
            
            if isinstance(history_result, tuple):
                return history_result

            if history_result and isinstance(history_result, dict):
                candles = history_result.get("candles", [])
                if not current_price and candles:
                    current_price = candles[-1][4]

                signals = generate_trading_signals_with_ai(symbol, candles, current_price or 0)
                indicators = calculate_advanced_indicators(candles)
                signals["technical_indicators"] = indicators

                return jsonify(signals)
            else:
                return jsonify({"error": "Failed to fetch historical data"}), 500
        else:
            historical_data = request_data.get("historical_data", [])
            current_price = request_data.get("current_price", 0)

            if not historical_data:
                return jsonify({
                    "error": "Either enable 'use_live_data' or provide 'historical_data' and 'current_price'"
                }), 400

            signals = generate_trading_signals_with_ai(symbol, historical_data, current_price)
            return jsonify(signals)

    except Exception as e:
        app.logger.error(f"Error generating trading signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """
    Chat with Gemini AI about trading and markets.
    """
    if not gemini_model:
        return jsonify({
            "error": "AI model not initialized",
            "message": "Please set GEMINI_API_KEY"
        }), 503

    request_data = request.json
    message = request_data.get("message")
    context = request_data.get("context", {})

    if not message:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    try:
        prompt = f"""
        You are an expert trading advisor and market analyst. 
        Please provide helpful, accurate, and actionable insights.
        
        User Question: {message}
        """

        if context:
            prompt += f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"

        prompt += "\n\nProvide a clear, structured response with specific insights and recommendations where applicable."

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "question": message,
            "response": response.text,
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        app.logger.error(f"AI chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/portfolio-analysis', methods=['POST'])
@require_authentication
def ai_portfolio_analysis():
    """
    Analyze portfolio using AI and provide recommendations.
    """
    if not gemini_model:
        return jsonify({
            "error": "AI model not initialized",
            "message": "Please set GEMINI_API_KEY"
        }), 503

    request_data = request.json
    holdings = request_data.get("holdings", [])
    risk_profile = request_data.get("risk_profile", "moderate")
    
    session_id = session.get('session_id')

    if not holdings:
        fyers_instance = get_user_fyers_instance(session_id)
        if fyers_instance:
            holdings_result = make_fyers_api_call(session_id, fyers_instance.holdings)
            
            if not isinstance(holdings_result, tuple) and isinstance(holdings_result, dict):
                holdings = holdings_result.get("holdings", [])

    if not holdings:
        return jsonify({"error": "No holdings data available"}), 400

    try:
        prompt = f"""
        Analyze the following portfolio and provide comprehensive recommendations:
        
        Portfolio Holdings:
        {json.dumps(holdings, indent=2)}
        
        Risk Profile: {risk_profile}
        
        Please provide:
        1. Portfolio composition analysis
        2. Risk assessment
        3. Diversification analysis
        4. Sector allocation review
        5. Rebalancing recommendations
        6. Specific buy/sell/hold recommendations for each holding
        7. New investment opportunities based on the risk profile
        8. Expected returns and risk metrics
        
        Format the response in a clear, actionable manner.
        """

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "portfolio_analysis": response.text,
            "risk_profile": risk_profile,
            "holdings_count": len(holdings),
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        app.logger.error(f"Portfolio analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/market-summary', methods=['GET'])
def ai_market_summary():
    """
    Get AI-generated market summary and outlook.
    """
    if not gemini_model:
        return jsonify({
            "error": "AI model not initialized",
            "message": "Please set GEMINI_API_KEY"
        }), 503

    try:
        prompt = f"""
        Provide a comprehensive market summary for Indian markets as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}:
        
        Include:
        1. Overall market sentiment and trend
        2. Key indices performance outlook (NIFTY, SENSEX, BANKNIFTY)
        3. Sector rotation analysis
        4. Global market impact on Indian markets
        5. Key events and their potential impact
        6. FII/DII activity insights
        7. Currency and commodity outlook
        8. Top opportunities for the day/week
        9. Key risks to watch
        10. Recommended trading strategies for current market conditions
        
        Be specific with levels, percentages, and actionable insights.
        """

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "market_summary": response.text,
            "generated_at": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        app.logger.error(f"Market summary error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Fyers Data Endpoints (All require authentication) ---

@app.route('/api/fyers/profile')
@require_authentication
def get_profile():
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    result = make_fyers_api_call(session_id, fyers_instance.get_profile)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/funds')
@require_authentication
def get_funds():
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    result = make_fyers_api_call(session_id, fyers_instance.funds)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/holdings')
@require_authentication
def get_holdings():
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    result = make_fyers_api_call(session_id, fyers_instance.holdings)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/history', methods=['POST'])
@require_authentication
def get_history():
    """
    Fetch historical data with support for second-level, minute-level, and day-level resolutions.
    """
    data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)

    required_params = ["symbol", "resolution", "date_format", "range_from", "range_to"]
    if not data or not all(k in data for k in required_params):
        return jsonify({
            "error": f"Missing required parameters. Need: {', '.join(required_params)}"
        }), 400

    include_ai_analysis = data.get("include_ai_analysis", False)
    ai_analysis_type = data.get("ai_analysis_type", "technical")

    resolution = data["resolution"]
    if resolution not in SECOND_RESOLUTIONS and resolution not in MINUTE_RESOLUTIONS and resolution not in DAY_RESOLUTIONS:
        return jsonify({
            "error": f"Unsupported resolution: {resolution}",
            "supported_resolutions": {
                "second": SECOND_RESOLUTIONS,
                "minute": MINUTE_RESOLUTIONS,
                "day": DAY_RESOLUTIONS
            }
        }), 400

    try:
        data["date_format"] = int(data["date_format"])
    except ValueError:
        return jsonify({"error": "Invalid 'date_format'. Must be 0 or 1."}), 400

    # Handle incomplete candles
    if data["date_format"] == 0:
        current_time = int(time.time())
        requested_range_to = int(data["range_to"])
        resolution = data["resolution"]

        resolution_in_seconds = 0
        if resolution.endswith('S'):
            try:
                resolution_in_seconds = int(resolution[:-1])
            except ValueError:
                return jsonify({"error": "Invalid resolution format."}), 400
        elif resolution.isdigit():
            resolution_in_seconds = int(resolution) * 60
        elif resolution in ["D", "1D"]:
            resolution_in_seconds = 24 * 60 * 60

        if resolution_in_seconds > 0:
            current_resolution_start_epoch = (current_time // resolution_in_seconds) * resolution_in_seconds

            if requested_range_to >= current_resolution_start_epoch:
                adjusted_range_to_epoch = current_resolution_start_epoch - 1

                if adjusted_range_to_epoch < int(data["range_from"]):
                    return jsonify({
                        "candles": [],
                        "s": "ok",
                        "message": "No complete candles available for the adjusted range."
                    })

                data["range_to"] = str(adjusted_range_to_epoch)
                app.logger.info(f"Adjusted range_to to {data['range_to']} for complete candles")

    if "cont_flag" in data:
        data["cont_flag"] = int(data["cont_flag"])
    if "oi_flag" in data:
        data["oi_flag"] = int(data["oi_flag"])

    result = make_fyers_api_call(session_id, fyers_instance.history, data=data)
    
    if isinstance(result, tuple):
        return result

    if result and result.get("s") == "ok":
        candles_count = len(result.get("candles", []))
        app.logger.info(f"Fetched {candles_count} candles for {data['symbol']}")

        if include_ai_analysis and gemini_model and candles_count > 0:
            try:
                candles = result.get("candles", [])
                indicators = calculate_advanced_indicators(candles)

                analysis_data = {
                    "symbol": data["symbol"],
                    "resolution": data["resolution"],
                    "candles_count": candles_count,
                    "indicators": indicators,
                    "latest_candle": candles[-1] if candles else None
                }

                ai_analysis = analyze_market_data_with_ai(analysis_data, ai_analysis_type)
                result["ai_analysis"] = ai_analysis

            except Exception as e:
                app.logger.error(f"Failed to add AI analysis: {e}")
                result["ai_analysis"] = {"error": str(e)}

    return jsonify(result)

@app.route('/api/fyers/quotes', methods=['GET'])
@require_authentication
def get_quotes():
    symbols = request.args.get('symbols')
    include_ai_analysis = request.args.get('include_ai_analysis', 'false').lower() == 'true'
    
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)

    if not symbols:
        return jsonify({
            "error": "Missing 'symbols' parameter",
            "example": "/api/fyers/quotes?symbols=NSE:SBIN-EQ,NSE:TCS-EQ"
        }), 400

    data = {"symbols": symbols}
    result = make_fyers_api_call(session_id, fyers_instance.quotes, data=data)
    
    if isinstance(result, tuple):
        return result

    if include_ai_analysis and gemini_model and result and result.get("d"):
        try:
            quotes_data = result.get("d", [])
            ai_analysis = analyze_market_data_with_ai({"quotes": quotes_data}, "general")
            result["ai_analysis"] = ai_analysis
        except Exception as e:
            app.logger.error(f"Failed to add AI analysis: {e}")
            result["ai_analysis"] = {"error": str(e)}

    return jsonify(result)

@app.route('/api/fyers/market_depth', methods=['GET'])
@require_authentication
def get_market_depth():
    symbol = request.args.get('symbol')
    ohlcv_flag = request.args.get('ohlcv_flag')
    
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)

    if not symbol or ohlcv_flag is None:
        return jsonify({
            "error": "Missing 'symbol' or 'ohlcv_flag' parameter",
            "example": "/api/fyers/market_depth?symbol=NSE:SBIN-EQ&ohlcv_flag=1"
        }), 400

    try:
        ohlcv_flag = int(ohlcv_flag)
        if ohlcv_flag not in [0, 1]:
            raise ValueError("ohlcv_flag must be 0 or 1")
    except ValueError as ve:
        return jsonify({"error": f"Invalid 'ohlcv_flag': {ve}"}), 400

    data = {
        "symbol": symbol,
        "ohlcv_flag": ohlcv_flag
    }
    
    result = make_fyers_api_call(session_id, fyers_instance.depth, data=data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/option_chain', methods=['GET'])
@require_authentication
def get_option_chain():
    symbol = request.args.get('symbol')
    strikecount = request.args.get('strikecount')
    
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)

    if not symbol or not strikecount:
        return jsonify({
            "error": "Missing 'symbol' or 'strikecount' parameter",
            "example": "/api/fyers/option_chain?symbol=NSE:NIFTY50-INDEX&strikecount=5"
        }), 400

    try:
        strikecount = int(strikecount)
        if not (1 <= strikecount <= 50):
            return jsonify({"error": "'strikecount' must be between 1 and 50"}), 400
    except ValueError:
        return jsonify({"error": "Invalid 'strikecount'. Must be an integer."}), 400

    data = {
        "symbol": symbol,
        "strikecount": strikecount
    }
    
    result = make_fyers_api_call(session_id, fyers_instance.optionchain, data=data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

# --- Option helper functions ---

def _find_option_in_chain(resp, option_symbol=None, strike=None, opt_type=None, expiry_ts=None):
    """Helper to search option chain response."""
    if not resp or not isinstance(resp, dict):
        return None

    data = resp.get("data") if isinstance(resp, dict) else None
    candidates = []
    if data is None and "options_chain" in resp:
        data = resp

    if isinstance(data, dict):
        for key in ["optionChain", "option_chain", "optionsChain", "options", "ce", "pe"]:
            node = data.get(key)
            if node:
                if isinstance(node, list):
                    candidates.extend(node)
                elif isinstance(node, dict):
                    for v in node.values():
                        if isinstance(v, list):
                            candidates.extend(v)
                        else:
                            candidates.append(v)

    if not candidates and isinstance(resp.get("data"), list):
        candidates.extend(resp.get("data"))

    if not candidates and isinstance(resp, list):
        candidates.extend(resp)

    for item in candidates:
        try:
            symbol = item.get("symbol") or item.get("s") or item.get("name")
        except Exception:
            symbol = None
            
        if option_symbol and symbol and option_symbol == symbol:
            return item

        try:
            strike_val = item.get("strike") or item.get("strike_price") or item.get("strikePrice")
            typ = item.get("option_type") or item.get("type") or item.get("opt_type") or item.get("instrument_type")
            expiry = item.get("expiry") or item.get("expiry_date") or item.get("expiry_ts")
        except Exception:
            strike_val = None
            typ = None
            expiry = None

        if strike is not None and strike_val is not None:
            try:
                if int(float(strike_val)) == int(float(strike)):
                    if not opt_type or (typ and opt_type.upper() in str(typ).upper()):
                        return item
            except Exception:
                pass

        if expiry_ts and expiry:
            try:
                if int(expiry_ts) == int(expiry):
                    if not strike or (strike_val and int(float(strike_val)) == int(float(strike))):
                        return item
            except Exception:
                pass

    return None

@app.route('/api/fyers/option_premium', methods=['GET'])
@require_authentication
def get_option_premium():
    """Returns option premium (LTP), IV, OI, change for a single option contract."""
    symbol = request.args.get('symbol')
    underlying = request.args.get('underlying')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')
    expiry_ts = request.args.get('expiry_ts')
    
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)

    if not symbol and not (underlying and strike and opt_type):
        return jsonify({
            "error": "Provide either symbol=<OPTION_SYMBOL> OR underlying=<SYM>&strike=<STRIKE>&type=<CE|PE>"
        }), 400

    try:
        if symbol:
            depth_resp = make_fyers_api_call(session_id, fyers_instance.depth, 
                                           data={"symbol": symbol, "ohlcv_flag": 1})
            if isinstance(depth_resp, tuple):
                return depth_resp

            premium = None
            try:
                d = depth_resp.get('data') if isinstance(depth_resp, dict) else None
                if d:
                    premium = d.get('ltp') or d.get('last_price') or d.get('close')
                if not premium and isinstance(depth_resp, dict):
                    premium = depth_resp.get('ltp') or depth_resp.get('last_price')
            except Exception:
                premium = None

            if premium is None:
                oc_resp = make_fyers_api_call(session_id, fyers_instance.optionchain,
                                             data={"symbol": symbol, "strikecount": 1})
                if isinstance(oc_resp, tuple):
                    return oc_resp
                found = _find_option_in_chain(oc_resp, option_symbol=symbol)
                node = found or (oc_resp if isinstance(oc_resp, dict) else None)
            else:
                node = depth_resp if isinstance(depth_resp, dict) else None

        else:
            oc_resp = make_fyers_api_call(session_id, fyers_instance.optionchain,
                                         data={"symbol": underlying, "strikecount": 50})
            if isinstance(oc_resp, tuple):
                return oc_resp
            found = _find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
            node = found

        if not node:
            return jsonify({
                "error": "Option contract not found",
                "symbol": symbol or f"{underlying}:{strike}{opt_type if opt_type else ''}"
            }), 404

        ltp = node.get('ltp') or node.get('last_price') or node.get('close')
        oi = node.get('oi') or node.get('open_interest') or node.get('openInterest')
        iv = node.get('iv') or node.get('implied_volatility') or node.get('impliedVol')
        change = node.get('change') or node.get('price_change') or node.get('p_change')
        bid = node.get('bid') or node.get('best_bid')
        ask = node.get('ask') or node.get('best_ask')

        return jsonify({
            "symbol": symbol or node.get('symbol'),
            "ltp": ltp,
            "premium": ltp,
            "iv": iv,
            "oi": oi,
            "change": change,
            "bid": bid,
            "ask": ask,
            "raw": node
        })

    except Exception as e:
        app.logger.error(f"Error in option_premium: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fyers/option_chain_depth', methods=['GET'])
@require_authentication
def get_option_chain_depth():
    """Returns detailed market depth for a chosen option contract."""
    symbol = request.args.get('symbol')
    underlying = request.args.get('underlying')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')
    expiry_ts = request.args.get('expiry_ts')
    
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)

    try:
        if symbol:
            depth_resp = make_fyers_api_call(session_id, fyers_instance.depth,
                                           data={"symbol": symbol, "ohlcv_flag": 1})
            if isinstance(depth_resp, tuple):
                return depth_resp

            try:
                underlying_guess = symbol.split(":")[0] if ":" in symbol else None
            except Exception:
                underlying_guess = None

            oc_resp = None
            try:
                if underlying_guess:
                    oc_resp = make_fyers_api_call(session_id, fyers_instance.optionchain,
                                                 data={"symbol": underlying_guess, "strikecount": 10})
            except Exception:
                oc_resp = None

            return jsonify({"depth": depth_resp, "option_chain_context": oc_resp})

        if not (underlying and strike and opt_type):
            return jsonify({
                "error": "Provide either symbol=<OPTION_SYMBOL> OR underlying & strike & type parameters"
            }), 400

        oc_resp = make_fyers_api_call(session_id, fyers_instance.optionchain,
                                     data={"symbol": underlying, "strikecount": 50})
        if isinstance(oc_resp, tuple):
            return oc_resp

        found = _find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
        if not found:
            return jsonify({
                "error": "Option strike not found",
                "underlying": underlying,
                "strike": strike,
                "type": opt_type
            }), 404

        symbol_to_query = found.get('symbol') or request.args.get('symbol')
        if not symbol_to_query:
            return jsonify({"error": "Could not determine option symbol"}), 500

        depth_resp = make_fyers_api_call(session_id, fyers_instance.depth,
                                        data={"symbol": symbol_to_query, "ohlcv_flag": 1})
        if isinstance(depth_resp, tuple):
            return depth_resp

        return jsonify({"depth": depth_resp, "option_chain_context": oc_resp})

    except Exception as e:
        app.logger.error(f"Error in option_chain_depth: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fyers/news', methods=['GET'])
def get_news():
    """Placeholder news endpoint."""
    news_headlines = [
        {
            "id": 1,
            "title": "Market sentiments positive on Q1 earnings",
            "source": "Fyers Internal Analysis",
            "timestamp": str(datetime.datetime.now())
        },
        {
            "id": 2,
            "title": "RBI holds interest rates steady",
            "source": "Economic Times",
            "timestamp": str(datetime.datetime.now() - datetime.timedelta(hours=2))
        },
        {
            "id": 3,
            "title": "Tech stocks lead the rally",
            "source": "Reuters",
            "timestamp": str(datetime.datetime.now() - datetime.timedelta(days=1))
        },
        {
            "id": 4,
            "title": "F&O expiry expected to be volatile",
            "source": "Fyers Blog",
            "timestamp": str(datetime.datetime.now() - datetime.timedelta(days=1, hours=4))
        }
    ]
    return jsonify({
        "message": "This is a placeholder for Fyers news. Actual integration requires a dedicated news API.",
        "news": news_headlines
    })

# --- Order Management APIs (All require authentication) ---

@app.route('/api/fyers/order', methods=['POST'])
@require_authentication
def place_single_order():
    order_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not order_data:
        return jsonify({"error": "Order data is required"}), 400
        
    result = make_fyers_api_call(session_id, fyers_instance.place_order, data=order_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['POST'])
@require_authentication
def place_multi_order():
    multi_order_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not multi_order_data or not isinstance(multi_order_data, list):
        return jsonify({
            "error": "An array of order objects is required for multi-order placement"
        }), 400

    result = make_fyers_api_call(session_id, fyers_instance.multi_order, data=multi_order_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multileg', methods=['POST'])
@require_authentication
def place_multileg_order():
    multileg_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not multileg_data:
        return jsonify({"error": "Multileg order data is required"}), 400
        
    result = make_fyers_api_call(session_id, fyers_instance.multileg_order, data=multileg_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/order', methods=['POST'])
@require_authentication
def place_gtt_order():
    gtt_order_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not gtt_order_data:
        return jsonify({"error": "GTT order data is required"}), 400
        
    result = make_fyers_api_call(session_id, fyers_instance.place_gttorder, data=gtt_order_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/order', methods=['PATCH'])
@require_authentication
def modify_gtt_order():
    gtt_modify_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not gtt_modify_data or not gtt_modify_data.get("id"):
        return jsonify({"error": "GTT order ID and modification data are required"}), 400

    order_id = gtt_modify_data.pop("id")
    result = make_fyers_api_call(session_id, fyers_instance.modify_gttorder,
                                 id=order_id, data=gtt_modify_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/order', methods=['DELETE'])
@require_authentication
def cancel_gtt_order():
    gtt_cancel_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not gtt_cancel_data or not gtt_cancel_data.get("id"):
        return jsonify({"error": "GTT order ID is required for cancellation"}), 400

    order_id = gtt_cancel_data.get("id")
    result = make_fyers_api_call(session_id, fyers_instance.cancel_gttorder, id=order_id)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/orders', methods=['GET'])
@require_authentication
def get_gtt_orders():
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    result = make_fyers_api_call(session_id, fyers_instance.gtt_orders)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/order', methods=['PATCH'])
@require_authentication
def modify_single_order():
    modify_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not modify_data or not modify_data.get("id"):
        return jsonify({"error": "Order ID and modification data are required"}), 400

    result = make_fyers_api_call(session_id, fyers_instance.modify_order, data=modify_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['PATCH'])
@require_authentication
def modify_multi_orders():
    modify_basket_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not modify_basket_data or not isinstance(modify_basket_data, list):
        return jsonify({
            "error": "An array of order modification objects is required"
        }), 400

    result = make_fyers_api_call(session_id, fyers_instance.modify_basket_orders,
                                 data=modify_basket_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/order', methods=['DELETE'])
@require_authentication
def cancel_single_order():
    cancel_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not cancel_data or not cancel_data.get("id"):
        return jsonify({"error": "Order ID is required for cancellation"}), 400

    result = make_fyers_api_call(session_id, fyers_instance.cancel_order, data=cancel_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['DELETE'])
@require_authentication
def cancel_multi_orders():
    cancel_basket_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not cancel_basket_data or not isinstance(cancel_basket_data, list):
        return jsonify({
            "error": "An array of order cancellation objects (with 'id') is required"
        }), 400

    result = make_fyers_api_call(session_id, fyers_instance.cancel_basket_orders,
                                 data=cancel_basket_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/positions', methods=['DELETE'])
@require_authentication
def exit_positions():
    exit_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not exit_data:
        return jsonify({"error": "Request body for exiting positions is required"}), 400

    result = make_fyers_api_call(session_id, fyers_instance.exit_positions, data=exit_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/positions', methods=['POST'])
@require_authentication
def convert_position():
    convert_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not convert_data:
        return jsonify({"error": "Position conversion data is required"}), 400

    result = make_fyers_api_call(session_id, fyers_instance.convert_positions, data=convert_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

# --- Margin Calculator APIs ---

@app.route('/api/fyers/margin/span', methods=['POST'])
@require_authentication
def span_margin_calculator():
    margin_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not margin_data or not margin_data.get("data"):
        return jsonify({
            "error": "An array of order details for span margin calculation is required under 'data' key"
        }), 400

    result = make_fyers_api_call(session_id, fyers_instance.span_margin, data=margin_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/margin/multiorder', methods=['POST'])
@require_authentication
def multiorder_margin_calculator():
    margin_data = request.json
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    if not margin_data or not margin_data.get("data"):
        return jsonify({
            "error": "An array of order details for multiorder margin calculation is required under 'data' key"
        }), 400

    result = make_fyers_api_call(session_id, fyers_instance.multiorder_margin, data=margin_data)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

@app.route('/api/fyers/market_status', methods=['GET'])
@require_authentication
def get_market_status():
    session_id = session.get('session_id')
    fyers_instance = get_user_fyers_instance(session_id)
    
    result = make_fyers_api_call(session_id, fyers_instance.market_status)
    if isinstance(result, tuple):
        return result
    return jsonify(result)

# --- WebSocket Integration Section ---

_fyers_data_sockets = {}  # Per-user websocket connections
_socket_lock = threading.Lock()

def _create_data_socket(session_id: str, access_token: str, lite_mode=False):
    """Create a FyersDataSocket instance for a specific user."""
    
    def _on_connect():
        app.logger.info(f"Fyers DataSocket connected for session {session_id[:8]}...")

    def _on_message(message):
        """Broadcast to specific user's websocket clients."""
        try:
            if isinstance(message, str):
                payload = message
            else:
                payload = json.dumps(message, default=str)
        except Exception:
            payload = str(message)

        # This would need to be enhanced to support per-user websocket clients
        # For now, it's a simplified implementation

    def _on_error(error):
        app.logger.error(f"Fyers DataSocket error for session {session_id[:8]}...: {error}")

    def _on_close(close_msg):
        app.logger.info(f"Fyers DataSocket closed for session {session_id[:8]}...: {close_msg}")

    try:
        data_socket = data_ws.FyersDataSocket(
            access_token=access_token,
            log_path="",
            litemode=lite_mode,
            write_to_file=False,
            reconnect=True,
            on_connect=_on_connect,
            on_close=_on_close,
            on_error=_on_error,
            on_message=_on_message,
            reconnect_retry=5
        )
        return data_socket
    except Exception as e:
        app.logger.error(f"Failed to create FyersDataSocket: {e}")
        return None

@sock.route('/ws/fyers')
def ws_fyers(ws):
    """
    WebSocket endpoint for real-time data.
    Requires authentication via session.
    """
    # Get session from cookies (this might need adjustment based on your setup)
    # For now, we'll use a simplified approach
    
    app.logger.info("Frontend websocket connection attempt")
    
    try:
        # Send a welcome message
        ws.send(json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Connected to Fyers WebSocket. Please authenticate.",
            "usage": {
                "subscribe": {"action": "subscribe", "symbols": ["NSE:SBIN-EQ"], "data_type": "SymbolUpdate"},
                "unsubscribe": {"action": "unsubscribe", "symbols": ["NSE:SBIN-EQ"]},
                "status": {"action": "status"}
            }
        }))
        
        # Handle websocket messages
        while True:
            msg = ws.receive()
            if msg is None:
                app.logger.info("Frontend websocket disconnected")
                break

            try:
                data = json.loads(msg)
                action = data.get("action")
                
                if action == "status":
                    ws.send(json.dumps({
                        "type": "status",
                        "message": "WebSocket is active. Authenticate via HTTP endpoints first."
                    }))
                else:
                    ws.send(json.dumps({
                        "type": "error",
                        "message": "Please authenticate via /fyers-login first"
                    }))
                    
            except Exception as e:
                app.logger.error(f"Error processing websocket message: {e}")
                ws.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except Exception as e:
        app.logger.error(f"WebSocket error: {e}")

# --- Main Routes ---

@app.route('/')
def home():
    """API documentation endpoint."""
    return jsonify({
        "message": "Fyers API Proxy Server with AI Integration",
        "version": "2.0.0",
        "status": "running",
        "authentication": {
            "login": "/fyers-login",
            "status": "/api/auth/status",
            "logout": "/api/auth/logout (POST)"
        },
        "endpoints": {
            "authentication": {
                "login": "/fyers-login",
                "callback": "/fyers-auth-callback",
                "status": "/api/auth/status",
                "logout": "/api/auth/logout"
            },
            "market_data": {
                "profile": "/api/fyers/profile",
                "funds": "/api/fyers/funds",
                "holdings": "/api/fyers/holdings",
                "history": "/api/fyers/history (POST)",
                "quotes": "/api/fyers/quotes?symbols=NSE:SBIN-EQ",
                "market_depth": "/api/fyers/market_depth?symbol=NSE:SBIN-EQ&ohlcv_flag=1",
                "option_chain": "/api/fyers/option_chain?symbol=NSE:NIFTY50-INDEX&strikecount=5",
                "option_premium": "/api/fyers/option_premium?symbol=...",
                "market_status": "/api/fyers/market_status"
            },
            "ai_features": {
                "analyze": "/api/ai/analyze (POST)",
                "trading_signals": "/api/ai/trading-signals (POST)",
                "chat": "/api/ai/chat (POST)",
                "portfolio_analysis": "/api/ai/portfolio-analysis (POST)",
                "market_summary": "/api/ai/market-summary"
            },
            "orders": {
                "place_order": "/api/fyers/order (POST)",
                "modify_order": "/api/fyers/order (PATCH)",
                "cancel_order": "/api/fyers/order (DELETE)",
                "multi_orders": "/api/fyers/orders/multi",
                "gtt_orders": "/api/fyers/gtt/order"
            },
            "websocket": "/ws/fyers"
        },
        "supported_resolutions": {
            "second": SECOND_RESOLUTIONS,
            "minute": MINUTE_RESOLUTIONS,
            "day": DAY_RESOLUTIONS
        },
        "features": {
            "ai_enabled": gemini_model is not None,
            "websocket_enabled": True,
            "authentication_required": True,
            "session_based": True
        },
        "security": {
            "authentication": "Required for all Fyers API endpoints",
            "session_timeout": "8 hours",
            "token_refresh": "Automatic"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    with session_lock:
        active_sessions = len(user_sessions)
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "active_sessions": active_sessions,
        "ai_status": "enabled" if gemini_model else "disabled",
        "fyers_configured": all([CLIENT_ID, SECRET_KEY, REDIRECT_URI])
    })

# Register the backtesting blueprint
app.register_blueprint(backtesting_bp)

# --- Application Startup ---

@app.before_request
def before_request():
    """Setup before each request."""
    session.permanent = True

@app.after_request
def after_request(response):
    """Add security headers."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # CORS headers are handled by flask-cors, but you can add custom ones here
    
    return response

# --- Error Handlers ---

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": "Visit / for API documentation"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle uncaught exceptions."""
    app.logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        "error": "An error occurred",
        "message": str(e)
    }), 500

# --- Cleanup on shutdown ---

import atexit

def cleanup_on_shutdown():
    """Cleanup resources on application shutdown."""
    app.logger.info("Application shutting down...")
    
    with session_lock:
        session_count = len(user_sessions)
        user_sessions.clear()
    
    app.logger.info(f"Cleaned up {session_count} user sessions")

atexit.register(cleanup_on_shutdown)

# --- Main Entry Point ---

if __name__ == '__main__':
    app.logger.info("=" * 80)
    app.logger.info("Fyers API Proxy Server with AI Integration - Starting")
    app.logger.info("=" * 80)
    app.logger.info(f"Fyers Configured: {all([CLIENT_ID, SECRET_KEY, REDIRECT_URI])}")
    app.logger.info(f"AI Enabled: {gemini_model is not None}")
    app.logger.info(f"Authentication: REQUIRED (no automatic login)")
    app.logger.info(f"Login URL: http://localhost:5000/fyers-login")
    app.logger.info("=" * 80)
    
    # DO NOT auto-start websocket - only start when users authenticate
    
    app.run(host='0.0.0.0', port=5000, debug=True)