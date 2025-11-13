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

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# --- 1. Get Allowed Origins from Environment ---
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

if not allowed_origins:
    allowed_origins = ["*"]
    print("⚠️ WARNING: No ALLOWED_ORIGINS set. Allowing all origins.")
else:
    print(f"✅ CORS configured for origins: {allowed_origins}")

# --- 2. Configure Flask-CORS (The Right Way) ---
# This is the single source of truth for all CORS headers.
CORS(app, 
     resources={r"/*": {"origins": allowed_origins}},
     supports_credentials=True)

# Initialize Sock
sock = Sock(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# --- 3. Log Incoming Requests for Debugging ---
@app.before_request
def log_request_info():
    app.logger.info(f'Request: {request.method} {request.url}')
    app.logger.info(f'Headers: {dict(request.headers)}')
    app.logger.info(f'Origin: {request.headers.get("Origin", "No Origin")}')

# --- 4. Add Iframe Security Policy to All Responses ---
# This function's only job is to add the security policy to allow iframes.
@app.after_request
def after_request(response):
    # This is the critical header to allow Google AI Studio to embed your app.
    csp_value = (
        "frame-ancestors 'self' "
        "https://*.google.com "
        "https://*.googleusercontent.com "
        "https://*.scf.usercontent.goog;"
    )
    response.headers['Content-Security-Policy'] = csp_value

    # Defensively remove the X-Frame-Options header. The render.yaml file
    # from Step 3 is the primary way to do this, but this is a good backup.
    if 'X-Frame-Options' in response.headers:
        del response.headers['X-Frame-Options']

    app.logger.info(f'Response Status: {response.status}')
    app.logger.info(f'Sending Response Headers: {dict(response.headers)}')
    
    return response

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Fyers Proxy Server"
    }), 200

# --- Fyers API Configuration (from environment variables) ---
CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
SECRET_KEY = os.environ.get("FYERS_SECRET_KEY")
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI")
ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")

# --- Gemini AI Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # Default to flash model

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

# --- Storage and Token Management ---

def store_tokens(access_token, refresh_token):
    """Store tokens in memory and provide instructions for persistence"""
    global ACCESS_TOKEN, REFRESH_TOKEN
    ACCESS_TOKEN = access_token
    REFRESH_TOKEN = refresh_token

    app.logger.info("Tokens updated successfully!")
    
    # For Render deployment, you need to manually update environment variables
    # or use a database/file storage solution
    app.logger.info("IMPORTANT: For persistence on Render, update these environment variables:")
    app.logger.info(f"FYERS_ACCESS_TOKEN={ACCESS_TOKEN}")
    app.logger.info(f"FYERS_REFRESH_TOKEN={REFRESH_TOKEN}")
    
    # Save to a local file as backup (optional for Render persistent disk)
    try:
        token_data = {
            "access_token": ACCESS_TOKEN,
            "refresh_token": REFRESH_TOKEN,
            "updated_at": datetime.datetime.now().isoformat()
        }
        with open("tokens.json", "w") as f:
            json.dump(token_data, f)
        app.logger.info("Tokens saved to tokens.json file")
    except Exception as e:
        app.logger.error(f"Could not save tokens to file: {e}")

def load_tokens():
    """Load tokens from environment or file"""
    global ACCESS_TOKEN, REFRESH_TOKEN
    
    # First try environment variables
    if os.environ.get("FYERS_ACCESS_TOKEN"):
        ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
        REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")
        app.logger.info("Tokens loaded from environment variables")
        return
    
    # Try loading from file as fallback
    try:
        with open("tokens.json", "r") as f:
            token_data = json.load(f)
            ACCESS_TOKEN = token_data.get("access_token")
            REFRESH_TOKEN = token_data.get("refresh_token")
            app.logger.info(f"Tokens loaded from file (updated: {token_data.get('updated_at')})")
    except Exception as e:
        app.logger.info(f"No tokens file found or error loading: {e}")

# Load tokens at startup
load_tokens()

def initialize_fyers_model(token=None):
    """Initialize Fyers model with token or refresh token"""
    global fyers_instance, ACCESS_TOKEN, REFRESH_TOKEN

    token_to_use = token if token else ACCESS_TOKEN

    if token_to_use:
        try:
            try:
                fyers_instance = fyersModel.FyersModel(token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            except TypeError:
                fyers_instance = fyersModel.FyersModel(access_token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            app.logger.info("FyersModel initialized with access token.")
            return True
        except Exception as e:
            app.logger.error(f"Failed to initialize fyers_model with provided token: {e}", exc_info=True)
            # Try refreshing if we have a refresh token
            if REFRESH_TOKEN:
                app.logger.info("Access token failed, attempting refresh...")
                return refresh_access_token()
            return False
    elif REFRESH_TOKEN and CLIENT_ID and SECRET_KEY:
        app.logger.info("No access token provided or found, attempting to use refresh token.")
        if refresh_access_token():
            app.logger.info("FyersModel initialized with refreshed access token.")
            return True
        else:
            app.logger.error("Could not initialize FyersModel: Refresh token failed.")
            return False
    else:
        app.logger.warning("FyersModel could not be initialized: No access token or refresh token available.")
        return False

def refresh_access_token():
    """Refresh the access token using refresh token"""
    global ACCESS_TOKEN, REFRESH_TOKEN, fyers_instance
    
    if not REFRESH_TOKEN:
        app.logger.error("Cannot refresh token: No refresh token available.")
        return False

    if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
        app.logger.error("Cannot refresh token: Missing CLIENT_ID, SECRET_KEY, or REDIRECT_URI")
        return False

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        state="refresh_state", 
        secret_key=SECRET_KEY,
        grant_type="refresh_token"
    )
    session.set_token(REFRESH_TOKEN)

    try:
        app.logger.info(f"Attempting to refresh access token...")
        response = session.generate_token()

        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response.get("refresh_token", REFRESH_TOKEN)

            store_tokens(new_access_token, new_refresh_token)
            
            # Re-initialize Fyers model with new token
            try:
                fyers_instance = fyersModel.FyersModel(
                    access_token=new_access_token, 
                    is_async=False, 
                    client_id=CLIENT_ID, 
                    log_path=""
                )
                app.logger.info("Access token refreshed and FyersModel reinitialized successfully.")
                return True
            except Exception as e:
                app.logger.error(f"Failed to reinitialize FyersModel after refresh: {e}")
                return False
        else:
            app.logger.error(f"Failed to refresh access token. Response: {response}")
            return False
    except Exception as e:
        app.logger.error(f"Error during access token refresh: {e}", exc_info=True)
        return False
# Initialize Fyers model at startup - will try refresh token if access token fails
initialize_fyers_model()

# --- Helper function to wrap Fyers API calls with automatic refresh logic ---
def make_fyers_api_call(api_method, *args, **kwargs):
    """Make Fyers API call with automatic token refresh on failure"""
    global fyers_instance
    
    if not fyers_instance:
        app.logger.warning("Fyers API not initialized. Attempting to initialize.")
        if not initialize_fyers_model():
            return {"error": "Fyers API not initialized. Please authenticate first."}, 401

    try:
        # First attempt
        result = api_method(*args, **kwargs)
        
        # Check if result indicates token error
        if isinstance(result, dict):
            error_code = result.get("code") or result.get("s")
            error_message = str(result.get("message", "")).lower()
            
            if (error_code in [-16, -99, 401, "error"] or 
                any(keyword in error_message for keyword in ["token", "expired", "invalid", "authenticate", "authorization"])):
                raise Exception("Token expired or invalid")
        
        return result
        
    except Exception as e:
        error_message = str(e).lower()

        if any(keyword in error_message for keyword in ["token", "expired", "invalid", "authenticate", "authorization"]):
            app.logger.warning(f"Token issue detected. Attempting to refresh. Error: {e}")
            
            if refresh_access_token():
                app.logger.info("Token refreshed, retrying original request.")
                try:
                    # Retry the API call with refreshed token
                    return api_method(*args, **kwargs)
                except Exception as retry_error:
                    app.logger.error(f"Request failed even after token refresh: {retry_error}")
                    return {"error": f"Request failed after token refresh: {str(retry_error)}"}, 500
            else:
                app.logger.error("Token refresh failed. User needs to re-authenticate.")
                return {"error": "Authentication expired. Please login again via /fyers-login"}, 401
        else:
            app.logger.error(f"Non-token related Fyers API error: {e}", exc_info=True)
            return {"error": f"Fyers API error: {str(e)}"}, 500

# --- Gemini AI Endpoints ---

@app.route('/api/ai/analyze', methods=['POST'])
def ai_analyze():
    """
    Analyze market data using Gemini AI.
    
    Request body:
    {
        "data": {...},  # Market data to analyze
        "analysis_type": "general|technical|sentiment|risk"
    }
    """
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    if not request_data or not request_data.get("data"):
        return jsonify({"error": "Missing 'data' in request body"}), 400

    analysis_type = request_data.get("analysis_type", "general")
    data = request_data.get("data")

    result = analyze_market_data_with_ai(data, analysis_type)
    return jsonify(result)

@app.route('/api/ai/trading-signals', methods=['POST'])
def ai_trading_signals():
    """
    Generate trading signals using AI.
    
    Request body:
    {
        "symbol": "NSE:SBIN-EQ",
        "use_live_data": true  # Optional, will fetch current data if true
    }
    """
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    symbol = request_data.get("symbol")
    use_live_data = request_data.get("use_live_data", False)

    if not symbol:
        return jsonify({"error": "Missing 'symbol' in request body"}), 400

    try:
        # Fetch historical data if requested
        if use_live_data and fyers_instance:
            # Get current quote
            quote_result = make_fyers_api_call(fyers_instance.quotes, data={"symbols": symbol})
            current_price = None
            if quote_result and isinstance(quote_result, dict):
                quote_data = quote_result.get("d", [{}])[0]
                current_price = quote_data.get("v", {}).get("lp", 0)

            # Get historical data
            end_time = int(time.time())
            start_time = end_time - (30 * 24 * 60 * 60)  # 30 days of data

            history_data = {
                "symbol": symbol,
                "resolution": "1D",
                "date_format": 0,
                "range_from": str(start_time),
                "range_to": str(end_time),
                "cont_flag": 1
            }

            history_result = make_fyers_api_call(fyers_instance.history, data=history_data)

            if history_result and isinstance(history_result, dict):
                candles = history_result.get("candles", [])
                if not current_price and candles:
                    current_price = candles[-1][4]  # Last close price

                signals = generate_trading_signals_with_ai(symbol, candles, current_price or 0)

                # Add technical indicators
                indicators = calculate_advanced_indicators(candles)
                signals["technical_indicators"] = indicators

                return jsonify(signals)
            else:
                return jsonify({"error": "Failed to fetch historical data"}), 500
        else:
            # Use provided data or return error
            historical_data = request_data.get("historical_data", [])
            current_price = request_data.get("current_price", 0)

            if not historical_data:
                return jsonify({"error": "Either enable 'use_live_data' or provide 'historical_data' and 'current_price'"}), 400

            signals = generate_trading_signals_with_ai(symbol, historical_data, current_price)
            return jsonify(signals)

    except Exception as e:
        app.logger.error(f"Error generating trading signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """
    Chat with Gemini AI about trading and markets.
    
    Request body:
    {
        "message": "What's your analysis on NIFTY50?",
        "context": {...}  # Optional context data
    }
    """
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    message = request_data.get("message")
    context = request_data.get("context", {})

    if not message:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    try:
        # Prepare the prompt with trading context
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
def ai_portfolio_analysis():
    """
    Analyze portfolio using AI and provide recommendations.
    
    Request body:
    {
        "holdings": [...],  # Portfolio holdings
        "risk_profile": "conservative|moderate|aggressive"
    }
    """
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    holdings = request_data.get("holdings", [])
    risk_profile = request_data.get("risk_profile", "moderate")

    if not holdings:
        # Try to fetch from Fyers if authenticated
        if fyers_instance:
            holdings_result = make_fyers_api_call(fyers_instance.holdings)
            if holdings_result and isinstance(holdings_result, dict):
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
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    try:
        # You can enhance this by fetching real market data
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

# --- Helper function to wrap Fyers API calls with refresh logic ---
def make_fyers_api_call(api_method, *args, **kwargs):
    global fyers_instance
    if not fyers_instance:
        app.logger.warning("Fyers API not initialized. Attempting to initialize.")
        if not initialize_fyers_model():
            return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401

    try:
        return api_method(*args, **kwargs)
    except Exception as e:
        error_message = str(e).lower()

        if "token" in error_message or "authenticated" in error_message or "login" in error_message or "invalid_access_token" in error_message:
            app.logger.warning(f"Access token expired or invalid. Attempting to refresh. Original error: {e}")
            if refresh_access_token():
                app.logger.info("Token refreshed, retrying original request.")
                return api_method(*args, **kwargs)
            else:
                app.logger.error("Token refresh failed. Cannot fulfill request.")
                return jsonify({"error": "Fyers API token expired and refresh failed. Please re-authenticate."}), 401
        else:
            app.logger.error(f"Non-token related Fyers API error: {e}", exc_info=True)
            return jsonify({"error": f"Fyers API error: {str(e)}"}), 500

# --- Fyers Authentication Flow Endpoints (FIXED) ---

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

            store_tokens(new_access_token, new_refresh_token)
            
            # Initialize Fyers model with new token
            if initialize_fyers_model(new_access_token):
                app.logger.info("Authentication successful!")
                
                # Return success page with instructions
                return f"""
                <html>
                <head><title>Authentication Successful</title></head>
                <body style="font-family: Arial; padding: 20px;">
                    <h2>✅ Fyers Authentication Successful!</h2>
                    <p>Your tokens have been generated and saved.</p>
                    <h3>Important for Render Deployment:</h3>
                    <p>Update these environment variables in your Render dashboard:</p>
                    <pre style="background: #f0f0f0; padding: 10px;">
FYERS_ACCESS_TOKEN={new_access_token}
FYERS_REFRESH_TOKEN={new_refresh_token}
                    </pre>
                    <p>The refresh token will be used automatically to maintain access.</p>
                    <br>
                    <p>You can now close this window and return to your application.</p>
                </body>
                </html>
                """
            else:
                return jsonify({"error": "Failed to initialize Fyers model with new tokens"}), 500
        else:
            app.logger.error(f"Failed to generate Fyers tokens. Response: {response}")
            return jsonify({"error": f"Failed to generate Fyers tokens. Response: {response}"}), 500

    except Exception as e:
        app.logger.error(f"Error generating Fyers access token: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate Fyers access token: {str(e)}"}), 500

# --- Fyers Data Endpoints ---
# --- Check Authentication Status Endpoint ---
@app.route('/api/auth/status')
def auth_status():
    """Check current authentication status"""
    global ACCESS_TOKEN, REFRESH_TOKEN, fyers_instance
    
    has_access = bool(ACCESS_TOKEN)
    has_refresh = bool(REFRESH_TOKEN)
    fyers_initialized = bool(fyers_instance)
    
    # Try a simple API call to verify token validity
    token_valid = False
    if fyers_instance:
        try:
            result = fyers_instance.get_profile()
            if result and result.get("s") == "ok":
                token_valid = True
        except:
            pass
    
    return jsonify({
        "authenticated": token_valid,
        "has_access_token": has_access,
        "has_refresh_token": has_refresh,
        "fyers_initialized": fyers_initialized,
        "message": "Authenticated and ready" if token_valid else "Authentication required"
    })
    
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

@app.route('/api/fyers/history', methods=['POST'])
def get_history():
    """
    Fetch historical data with support for second-level, minute-level, and day-level resolutions.
    Enhanced with optional AI analysis.
    """
    data = request.json

    required_params = ["symbol", "resolution", "date_format", "range_from", "range_to"]
    if not data or not all(k in data for k in required_params):
        app.logger.warning(f"Missing required parameters for history API. Received: {data}")
        return jsonify({"error": f"Missing required parameters for history API. Need {', '.join(required_params)}."}), 400

    # Check if AI analysis is requested
    include_ai_analysis = data.get("include_ai_analysis", False)
    ai_analysis_type = data.get("ai_analysis_type", "technical")

    # Validate resolution
    resolution = data["resolution"]
    if resolution not in SECOND_RESOLUTIONS and resolution not in MINUTE_RESOLUTIONS and resolution not in DAY_RESOLUTIONS:
        app.logger.warning(f"Unsupported resolution: {resolution}")
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
        app.logger.warning(f"Invalid 'date_format' received: {data.get('date_format')}")
        return jsonify({"error": "Invalid 'date_format'. Must be 0 or 1."}), 400

    # Handle incomplete candles for real-time data
    if data["date_format"] == 0:
        current_time = int(time.time())
        requested_range_to = int(data["range_to"])
        resolution = data["resolution"]

        resolution_in_seconds = 0
        if resolution.endswith('S'):
            try:
                resolution_in_seconds = int(resolution[:-1])
                app.logger.info(f"Processing second-level resolution: {resolution} ({resolution_in_seconds} seconds)")
            except ValueError:
                app.logger.warning(f"Invalid numeric part in resolution ending with 'S': {resolution}")
                return jsonify({"error": "Invalid resolution format."}), 400
        elif resolution.isdigit():
            resolution_in_seconds = int(resolution) * 60
            app.logger.info(f"Processing minute-level resolution: {resolution} minutes ({resolution_in_seconds} seconds)")
        elif resolution in ["D", "1D"]:
            resolution_in_seconds = 24 * 60 * 60
            app.logger.info(f"Processing day-level resolution: {resolution}")
        else:
            app.logger.warning(f"Unsupported resolution format for partial candle adjustment: {resolution}")
            return jsonify({"error": "Unsupported resolution format."}), 400

        if resolution_in_seconds > 0:
            current_resolution_start_epoch = (current_time // resolution_in_seconds) * resolution_in_seconds

            if requested_range_to >= current_resolution_start_epoch:
                adjusted_range_to_epoch = current_resolution_start_epoch - 1

                if adjusted_range_to_epoch < int(data["range_from"]):
                    app.logger.info(f"Adjusted range_to ({adjusted_range_to_epoch}) is less than range_from ({data['range_from']}). No complete candles available.")
                    return jsonify({"candles": [], "s": "ok", "message": "No complete candles available for the adjusted range."})

                data["range_to"] = str(adjusted_range_to_epoch)
                app.logger.info(f"Adjusted 'range_to' for resolution '{resolution}' to ensure completed candles: {requested_range_to} -> {data['range_to']}")

    if "cont_flag" in data:
        data["cont_flag"] = int(data["cont_flag"])
    if "oi_flag" in data:
        data["oi_flag"] = int(data["oi_flag"])

    app.logger.info(f"Fetching history data: Symbol={data['symbol']}, Resolution={data['resolution']}, From={data['range_from']}, To={data['range_to']}")

    result = make_fyers_api_call(fyers_instance.history, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result

    # Log successful response
    if result and result.get("s") == "ok":
        candles_count = len(result.get("candles", []))
        app.logger.info(f"Successfully fetched {candles_count} candles for {data['symbol']} at {data['resolution']} resolution")

        # Add AI analysis if requested
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
def get_quotes():
    symbols = request.args.get('symbols')
    include_ai_analysis = request.args.get('include_ai_analysis', 'false').lower() == 'true'

    if not symbols:
        app.logger.warning("Missing 'symbols' parameter for quotes API.")
        return jsonify({"error": "Missing 'symbols' parameter. Eg: /api/fyers/quotes?symbols=NSE:SBIN-EQ,NSE:TCS-EQ"}), 400

    data = {"symbols": symbols}
    result = make_fyers_api_call(fyers_instance.quotes, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result

    # Add AI analysis if requested
    if include_ai_analysis and gemini_model and result and result.get("d"):
        try:
            quotes_data = result.get("d", [])
            ai_analysis = analyze_market_data_with_ai({"quotes": quotes_data}, "general")
            result["ai_analysis"] = ai_analysis
        except Exception as e:
            app.logger.error(f"Failed to add AI analysis to quotes: {e}")
            result["ai_analysis"] = {"error": str(e)}

    return jsonify(result)

@app.route('/api/fyers/market_depth', methods=['GET'])
def get_market_depth():
    symbol = request.args.get('symbol')
    ohlcv_flag = request.args.get('ohlcv_flag')

    if not symbol or ohlcv_flag is None:
        app.logger.warning(f"Missing 'symbol' or 'ohlcv_flag' parameter for market depth. Symbol: {symbol}, Flag: {ohlcv_flag}")
        return jsonify({"error": "Missing 'symbol' or 'ohlcv_flag' parameter. Eg: /api/fyers/market_depth?symbol=NSE:SBIN-EQ&ohlcv_flag=1"}), 400

    try:
        ohlcv_flag = int(ohlcv_flag)
        if ohlcv_flag not in [0, 1]:
            raise ValueError("ohlcv_flag must be 0 or 1.")
    except ValueError as ve:
        app.logger.warning(f"Invalid 'ohlcv_flag' received for market depth: {ohlcv_flag}")
        return jsonify({"error": f"Invalid 'ohlcv_flag': {ve}"}), 400

    data = {
        "symbol": symbol,
        "ohlcv_flag": ohlcv_flag
    }
    result = make_fyers_api_call(fyers_instance.depth, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/option_chain', methods=['GET'])
def get_option_chain():
    symbol = request.args.get('symbol')
    strikecount = request.args.get('strikecount')

    if not symbol or not strikecount:
        app.logger.warning(f"Missing 'symbol' or 'strikecount' parameter for option chain. Symbol: {symbol}, Strikecount: {strikecount}")
        return jsonify({"error": "Missing 'symbol' or 'strikecount' parameter. Eg: /api/fyers/option_chain?symbol=NSE:TCS-EQ&strikecount=1"}), 400

    try:
        strikecount = int(strikecount)
        if not (1 <= strikecount <= 50):
            app.logger.warning(f"Invalid 'strikecount' for option chain: {strikecount}. Must be between 1 and 50.")
            return jsonify({"error": "'strikecount' must be between 1 and 50."}), 400
    except ValueError:
        app.logger.warning(f"Invalid 'strikecount' received for option chain: {strikecount}. Must be an integer.")
        return jsonify({"error": "Invalid 'strikecount'. Must be an integer."}), 400

    data = {
        "symbol": symbol,
        "strikecount": strikecount
    }
    result = make_fyers_api_call(fyers_instance.optionchain, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

# --- Option helper functions ---
def _find_option_in_chain(resp, option_symbol=None, strike=None, opt_type=None, expiry_ts=None):
    """Helper to search option chain response for the requested option and return the node."""
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
def get_option_premium():
    """Returns option premium (LTP), IV, OI, change, and a small parsed payload for a single option contract."""
    symbol = request.args.get('symbol')
    underlying = request.args.get('underlying')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')
    expiry_ts = request.args.get('expiry_ts')

    if not symbol and not (underlying and strike and opt_type):
        return jsonify({"error": "Provide either symbol=<OPTION_SYMBOL> OR underlying=<SYM>&strike=<STRIKE>&type=<CE|PE>"}), 400

    try:
        if symbol:
            depth_resp = make_fyers_api_call(fyers_instance.depth, data={"symbol": symbol, "ohlcv_flag": 1})
            if isinstance(depth_resp, tuple) and len(depth_resp) == 2 and isinstance(depth_resp[1], int):
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
                oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": symbol, "strikecount": 1})
                if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
                    return oc_resp
                found = _find_option_in_chain(oc_resp, option_symbol=symbol)
                node = found or (oc_resp if isinstance(oc_resp, dict) else None)
            else:
                node = depth_resp if isinstance(depth_resp, dict) else None

        else:
            oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying, "strikecount": 50})
            if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
                return oc_resp
            found = _find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
            node = found

        if not node:
            return jsonify({"error": "Option contract not found in Fyers response.", "symbol": symbol or f"{underlying}:{strike}{opt_type if opt_type else ''}"}), 404

        ltp = node.get('ltp') or node.get('last_price') or node.get('close')
        oi = node.get('oi') or node.get('open_interest') or node.get('openInterest')
        iv = node.get('iv') or node.get('implied_volatility') or node.get('impliedVol')
        change = node.get('change') or node.get('price_change') or node.get('p_change')
        bid = node.get('bid') or node.get('best_bid')
        ask = node.get('ask') or node.get('best_ask')

        response = {
            "symbol": symbol or node.get('symbol'),
            "ltp": ltp,
            "premium": ltp,
            "iv": iv,
            "oi": oi,
            "change": change,
            "bid": bid,
            "ask": ask,
            "raw": node
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error in option_premium endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/fyers/option_chain_depth', methods=['GET'])
def get_option_chain_depth():
    """Returns detailed market depth for a chosen option contract."""
    symbol = request.args.get('symbol')
    underlying = request.args.get('underlying')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')
    expiry_ts = request.args.get('expiry_ts')

    try:
        if symbol:
            depth_resp = make_fyers_api_call(fyers_instance.depth, data={"symbol": symbol, "ohlcv_flag": 1})
            if isinstance(depth_resp, tuple) and len(depth_resp) == 2 and isinstance(depth_resp[1], int):
                return depth_resp

            try:
                underlying_guess = symbol.split(":")[0] if ":" in symbol else None
            except Exception:
                underlying_guess = None

            oc_resp = None
            try:
                if underlying_guess:
                    oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying_guess, "strikecount": 10})
            except Exception:
                oc_resp = None

            return jsonify({"depth": depth_resp, "option_chain_context": oc_resp})

        if not (underlying and strike and opt_type):
            return jsonify({"error": "Provide either symbol=<OPTION_SYMBOL> OR underlying & strike & type parameters."}), 400

        oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying, "strikecount": 50})
        if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
            return oc_resp

        found = _find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
        if not found:
            return jsonify({"error": "Option strike not found in option chain response.", "underlying": underlying, "strike": strike, "type": opt_type}), 404

        symbol_to_query = found.get('symbol') or request.args.get('symbol')
        if not symbol_to_query:
            return jsonify({"error": "Could not determine option symbol to query depth for."}), 500

        depth_resp = make_fyers_api_call(fyers_instance.depth, data={"symbol": symbol_to_query, "ohlcv_flag": 1})
        if isinstance(depth_resp, tuple) and len(depth_resp) == 2 and isinstance(depth_resp[1], int):
            return depth_resp

        return jsonify({"depth": depth_resp, "option_chain_context": oc_resp})

    except Exception as e:
        app.logger.error(f"Error in option_chain_depth endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/fyers/news', methods=['GET'])
def get_news():
    app.logger.info("Accessing placeholder news endpoint.")

    news_headlines = [
        {"id": 1, "title": "Market sentiments positive on Q1 earnings", "source": "Fyers Internal Analysis", "timestamp": str(datetime.datetime.now())},
        {"id": 2, "title": "RBI holds interest rates steady", "source": "Economic Times", "timestamp": str(datetime.datetime.now() - datetime.timedelta(hours=2))},
        {"id": 3, "title": "Tech stocks lead the rally", "source": "Reuters", "timestamp": str(datetime.datetime.now() - datetime.timedelta(days=1))},
        {"id": 4, "title": "F&O expiry expected to be volatile", "source": "Fyers Blog", "timestamp": str(datetime.datetime.now() - datetime.timedelta(days=1, hours=4))}
    ]
    return jsonify({
        "message": "This is a placeholder for Fyers news. Actual integration requires a dedicated news API or scraping.",
        "news": news_headlines
    })

# --- Order Management APIs ---

@app.route('/api/fyers/order', methods=['POST'])
def place_single_order():
    order_data = request.json
    if not order_data:
        return jsonify({"error": "Order data is required."}), 400
    result = make_fyers_api_call(fyers_instance.place_order, data=order_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['POST'])
def place_multi_order():
    multi_order_data = request.json
    if not multi_order_data or not isinstance(multi_order_data, list):
        return jsonify({"error": "An array of order objects is required for multi-order placement."}), 400

    result = make_fyers_api_call(fyers_instance.multi_order, data=multi_order_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multileg', methods=['POST'])
def place_multileg_order():
    multileg_data = request.json
    if not multileg_data:
        return jsonify({"error": "Multileg order data is required."}), 400
    result = make_fyers_api_call(fyers_instance.multileg_order, data=multileg_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/order', methods=['POST'])
def place_gtt_order():
    gtt_order_data = request.json
    if not gtt_order_data:
        return jsonify({"error": "GTT order data is required."}), 400
    result = make_fyers_api_call(fyers_instance.place_gttorder, data=gtt_order_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/order', methods=['PATCH'])
def modify_gtt_order():
    gtt_modify_data = request.json
    if not gtt_modify_data or not gtt_modify_data.get("id"):
        return jsonify({"error": "GTT order ID and modification data are required."}), 400

    order_id = gtt_modify_data.pop("id")
    result = make_fyers_api_call(fyers_instance.modify_gttorder, id=order_id, data=gtt_modify_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/order', methods=['DELETE'])
def cancel_gtt_order():
    gtt_cancel_data = request.json
    if not gtt_cancel_data or not gtt_cancel_data.get("id"):
        return jsonify({"error": "GTT order ID is required for cancellation."}), 400

    order_id = gtt_cancel_data.get("id")
    result = make_fyers_api_call(fyers_instance.cancel_gttorder, id=order_id)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/gtt/orders', methods=['GET'])
def get_gtt_orders():
    result = make_fyers_api_call(fyers_instance.gtt_orders)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/order', methods=['PATCH'])
def modify_single_order():
    modify_data = request.json
    if not modify_data or not modify_data.get("id"):
        return jsonify({"error": "Order ID and modification data are required."}), 400

    result = make_fyers_api_call(fyers_instance.modify_order, data=modify_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['PATCH'])
def modify_multi_orders():
    modify_basket_data = request.json
    if not modify_basket_data or not isinstance(modify_basket_data, list):
        return jsonify({"error": "An array of order modification objects is required."}), 400

    result = make_fyers_api_call(fyers_instance.modify_basket_orders, data=modify_basket_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/order', methods=['DELETE'])
def cancel_single_order():
    cancel_data = request.json
    if not cancel_data or not cancel_data.get("id"):
        return jsonify({"error": "Order ID is required for cancellation."}), 400

    result = make_fyers_api_call(fyers_instance.cancel_order, data=cancel_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['DELETE'])
def cancel_multi_orders():
    cancel_basket_data = request.json
    if not cancel_basket_data or not isinstance(cancel_basket_data, list):
        return jsonify({"error": "An array of order cancellation objects (with 'id') is required."}), 400

    result = make_fyers_api_call(fyers_instance.cancel_basket_orders, data=cancel_basket_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/positions', methods=['DELETE'])
def exit_positions():
    exit_data = request.json
    if not exit_data:
        return jsonify({"error": "Request body for exiting positions is required."}), 400

    result = make_fyers_api_call(fyers_instance.exit_positions, data=exit_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/positions', methods=['POST'])
def convert_position():
    convert_data = request.json
    if not convert_data:
        return jsonify({"error": "Position conversion data is required."}), 400

    result = make_fyers_api_call(fyers_instance.convert_positions, data=convert_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

# --- Margin Calculator APIs ---

@app.route('/api/fyers/margin/span', methods=['POST'])
def span_margin_calculator():
    margin_data = request.json
    if not margin_data or not margin_data.get("data"):
        return jsonify({"error": "An array of order details for span margin calculation is required under 'data' key."}), 400

    result = make_fyers_api_call(fyers_instance.span_margin, data=margin_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/margin/multiorder', methods=['POST'])
def multiorder_margin_calculator():
    margin_data = request.json
    if not margin_data or not margin_data.get("data"):
        return jsonify({"error": "An array of order details for multiorder margin calculation is required under 'data' key."}), 400

    result = make_fyers_api_call(fyers_instance.multiorder_margin, data=margin_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/market_status', methods=['GET'])
def get_market_status():
    result = make_fyers_api_call(fyers_instance.market_status)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

# --- WebSocket Integration Section ---

_fyers_data_socket = None
_fyers_socket_thread = None
_connected_clients = []
_subscribed_symbols = set()
_socket_lock = threading.Lock()
_socket_running = False

def _create_data_socket(access_token, lite_mode=False, write_to_file=False, reconnect=True, reconnect_retry=10):
    """Create a FyersDataSocket instance with callbacks wired to broadcast incoming messages."""
    global _fyers_data_socket

    def _on_connect():
        app.logger.info("Fyers DataSocket connected.")
        if _subscribed_symbols:
            try:
                _fyers_data_socket.subscribe(symbols=list(_subscribed_symbols), data_type="SymbolUpdate")
                app.logger.info(f"Re-subscribed to symbols on connect: {_subscribed_symbols}")
            except Exception as e:
                app.logger.error(f"Error re-subscribing on connect: {e}", exc_info=True)

    def _on_message(message):
        """Called for incoming socket messages from Fyers. Broadcast to all connected clients."""
        try:
            if isinstance(message, str):
                payload = message
            else:
                payload = json.dumps(message, default=str)
        except Exception:
            payload = str(message)

        with _socket_lock:
            to_remove = []
            for client in _connected_clients:
                try:
                    client.send(payload)
                except Exception as e:
                    app.logger.debug(f"Failed to send to one client: {e}")
                    to_remove.append(client)
            for r in to_remove:
                try:
                    _connected_clients.remove(r)
                except ValueError:
                    pass

    def _on_error(error):
        app.logger.error(f"Fyers DataSocket error: {error}")

    def _on_close(close_msg):
        app.logger.info(f"Fyers DataSocket closed: {close_msg}")

    try:
        _fyers_data_socket = data_ws.FyersDataSocket(
            access_token=access_token,
            log_path="",
            litemode=lite_mode,
            write_to_file=write_to_file,
            reconnect=reconnect,
            on_connect=_on_connect,
            on_close=_on_close,
            on_error=_on_error,
            on_message=_on_message,
            reconnect_retry=reconnect_retry
        )
        return _fyers_data_socket
    except Exception as e:
        app.logger.error(f"Failed to create FyersDataSocket: {e}", exc_info=True)
        return None

def _start_data_socket_in_thread(access_token):
    """Starts the Fyers DataSocket in a dedicated daemon thread if not already running."""
    global _fyers_socket_thread, _socket_running, _fyers_data_socket

    with _socket_lock:
        if _socket_running:
            app.logger.info("Fyers DataSocket already running; skip start.")
            return True

        _fyers_data_socket = _create_data_socket(access_token)
        if not _fyers_data_socket:
            app.logger.error("Could not create FyersDataSocket instance.")
            return False

        def _run():
            global _socket_running
            try:
                _socket_running = True
                app.logger.info("Starting Fyers DataSocket.connect() (blocking call in thread).")
                _fyers_data_socket.connect()
            except Exception as e:
                app.logger.error(f"Fyers DataSocket thread crashed: {e}", exc_info=True)
            finally:
                _socket_running = False
                app.logger.info("Fyers DataSocket thread stopped.")

        _fyers_socket_thread = threading.Thread(target=_run, daemon=True, name="FyersDataSocketThread")
        _fyers_socket_thread.start()
        app.logger.info("Fyers DataSocket thread started.")
        return True

def _ensure_data_socket_running():
    """Ensures the socket is running; if not, try to start it with current ACCESS_TOKEN."""
    global ACCESS_TOKEN
    if not ACCESS_TOKEN:
        app.logger.warning("ACCESS_TOKEN is not available; cannot start data socket.")
        return False
    return _start_data_socket_in_thread(ACCESS_TOKEN)

@sock.route('/ws/fyers')
def ws_fyers(ws):
    """
    Frontend connects here: wss://<host>/ws/fyers
    Supported incoming JSON messages from frontend:
      - {"action": "subscribe", "symbols": ["NSE:SBIN-EQ","NSE:TCS-EQ"], "data_type":"SymbolUpdate"}
      - {"action": "unsubscribe", "symbols": ["NSE:SBIN-EQ"]}
      - {"action": "mode", "lite": true}
      - {"action": "status"}
    """
    global _connected_clients, _subscribed_symbols, _fyers_data_socket, ACCESS_TOKEN

    with _socket_lock:
        _connected_clients.append(ws)
    app.logger.info(f"Frontend websocket connected. Total clients: {len(_connected_clients)}")

    try:
        if not _socket_running:
            started = _ensure_data_socket_running()
            if not started:
                err = {"error": "Fyers DataSocket could not be started. Ensure ACCESS_TOKEN is set and valid."}
                try:
                    ws.send(json.dumps(err))
                except Exception:
                    pass
                ws.close()
                return

        while True:
            msg = ws.receive()
            if msg is None:
                app.logger.info("Frontend websocket disconnected (receive returned None).")
                break

            try:
                data = json.loads(msg)
            except Exception:
                try:
                    ws.send(json.dumps({"error":"Invalid JSON command"}))
                except Exception:
                    pass
                continue

            action = data.get("action")
            if action == "subscribe":
                symbols = data.get("symbols", [])
                data_type = data.get("data_type", "SymbolUpdate")
                if not symbols:
                    try:
                        ws.send(json.dumps({"error":"No symbols provided for subscribe"}))
                    except Exception:
                        pass
                    continue

                with _socket_lock:
                    for s in symbols:
                        _subscribed_symbols.add(s)

                try:
                    if _fyers_data_socket:
                        try:
                            _fyers_data_socket.subscribe(symbols=symbols, data_type=data_type)
                        except TypeError:
                            _fyers_data_socket.subscribe(symbols)
                        try:
                            ws.send(json.dumps({"status":"subscribed", "symbols": symbols}))
                        except Exception:
                            pass
                    else:
                        ws.send(json.dumps({"error":"Internal server: data socket not available"}))
                except Exception as e:
                    app.logger.error(f"Error subscribing via fyers socket: {e}", exc_info=True)
                    try:
                        ws.send(json.dumps({"error":str(e)}))
                    except Exception:
                        pass

            elif action == "unsubscribe":
                symbols = data.get("symbols", [])
                data_type = data.get("data_type", "SymbolUpdate")
                if not symbols:
                    try:
                        ws.send(json.dumps({"error":"No symbols provided for unsubscribe"}))
                    except Exception:
                        pass
                    continue

                with _socket_lock:
                    for s in symbols:
                        _subscribed_symbols.discard(s)

                try:
                    if _fyers_data_socket:
                        try:
                            _fyers_data_socket.unsubscribe(symbols=symbols, data_type=data_type)
                        except TypeError:
                            _fyers_data_socket.unsubscribe(symbols)
                        try:
                            ws.send(json.dumps({"status":"unsubscribed", "symbols": symbols}))
                        except Exception:
                            pass
                    else:
                        ws.send(json.dumps({"error":"Internal server: data socket not available"}))
                except Exception as e:
                    app.logger.error(f"Error unsubscribing via fyers socket: {e}", exc_info=True)
                    try:
                        ws.send(json.dumps({"error":str(e)}))
                    except Exception:
                        pass

            elif action == "mode":
                lite = data.get("lite", False)
                try:
                    with _socket_lock:
                        current_symbols = list(_subscribed_symbols)
                        try:
                            if _fyers_data_socket:
                                try:
                                    _fyers_data_socket.close()
                                except Exception:
                                    try:
                                        _fyers_data_socket.stop()
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        _create_data_socket(access_token=ACCESS_TOKEN, lite_mode=bool(lite))
                        if not _socket_running:
                            _start_data_socket_in_thread(ACCESS_TOKEN)

                        if current_symbols and _fyers_data_socket:
                            try:
                                _fyers_data_socket.subscribe(symbols=current_symbols, data_type="SymbolUpdate")
                            except Exception:
                                try:
                                    _fyers_data_socket.subscribe(current_symbols)
                                except Exception:
                                    app.logger.debug("Failed to re-subscribe after mode switch.")
                    ws.send(json.dumps({"status":"mode_changed", "lite": bool(lite)}))
                except Exception as e:
                    app.logger.error(f"Error switching mode: {e}", exc_info=True)
                    try:
                        ws.send(json.dumps({"error":str(e)}))
                    except Exception:
                        pass

            elif action == "status":
                try:
                    status = {
                        "socket_running": bool(_socket_running),
                        "connected_clients": len(_connected_clients),
                        "subscribed_symbols": list(_subscribed_symbols)
                    }
                    ws.send(json.dumps({"status": status}))
                except Exception:
                    pass

            else:
                try:
                    ws.send(json.dumps({
                        "error":"Unknown action",
                        "usage": [
                            {"action":"subscribe", "symbols":["NSE:SBIN-EQ"], "data_type":"SymbolUpdate"},
                            {"action":"unsubscribe", "symbols":["NSE:SBIN-EQ"]},
                            {"action":"mode", "lite": True},
                            {"action":"status"}
                        ]
                    }))
                except Exception:
                    pass

    except Exception as e:
        app.logger.error(f"Error in ws_fyers handling: {e}", exc_info=True)
    finally:
        with _socket_lock:
            try:
                _connected_clients.remove(ws)
            except ValueError:
                pass
        app.logger.info(f"Frontend websocket disconnected. Remaining clients: {len(_connected_clients)}")

# --- Main Routes ---

@app.route('/')
def home():
    return jsonify({
        "message": "Fyers API Proxy Server with AI Integration is running!",
        "endpoints": {
            "authentication": "/fyers-login",
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
            "ai_status": "enabled" if gemini_model else "disabled (set GEMINI_API_KEY)"
        }
    })

# Register the backtesting blueprint
app.register_blueprint(backtesting_bp)

if __name__ == '__main__':
    # Start the Fyers data socket at startup if an access token is present
    if ACCESS_TOKEN:
        try:
            _start_data_socket_in_thread(ACCESS_TOKEN)
        except Exception as e:
            app.logger.warning(f"Could not start Fyers DataSocket at startup: {e}")

    app.run(host='0.0.0.0', port=5000, debug=True)
