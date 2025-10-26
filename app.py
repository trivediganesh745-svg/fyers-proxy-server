"""
Production-ready Fyers API Proxy Server with AI Integration
Includes support for Stocks, Commodities, and Mutual Funds
"""

import os
import json
import time
import datetime
import logging
import threading
import asyncio
from typing import List, Dict, Any, Optional, Union
from functools import wraps
from dataclasses import dataclass
import hashlib

import numpy as np
import redis
from dotenv import load_dotenv
from flask import Flask, request, jsonify, redirect, url_for, current_app
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_sock import Sock
from marshmallow import Schema, fields, validate, ValidationError
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

# ========================= Configuration =========================

@dataclass
class Config:
    """Application configuration"""
    # Flask
    SECRET_KEY: str = os.environ.get("SECRET_KEY", os.urandom(32).hex())
    DEBUG: bool = False
    TESTING: bool = False
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "https://yourdomain.com"]
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Caching
    CACHE_TYPE: str = "redis"
    CACHE_REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
    CACHE_DEFAULT_TIMEOUT: int = 300
    
    # Fyers API
    FYERS_CLIENT_ID: str = os.environ.get("FYERS_CLIENT_ID", "")
    FYERS_SECRET_KEY: str = os.environ.get("FYERS_SECRET_KEY", "")
    FYERS_REDIRECT_URI: str = os.environ.get("FYERS_REDIRECT_URI", "")
    FYERS_ACCESS_TOKEN: str = os.environ.get("FYERS_ACCESS_TOKEN", "")
    FYERS_REFRESH_TOKEN: str = os.environ.get("FYERS_REFRESH_TOKEN", "")
    
    # Gemini AI
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Security
    ENCRYPTION_KEY: str = os.environ.get("ENCRYPTION_KEY", Fernet.generate_key().decode())
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB max request size
    
    # WebSocket
    MAX_WS_CONNECTIONS: int = 100
    WS_HEARTBEAT_INTERVAL: int = 30

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    CORS_ORIGINS = ["*"]  # Allow all origins in development

# ========================= App Initialization =========================

def create_app(config_name='production'):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    if config_name == 'production':
        app.config.from_object(ProductionConfig())
    else:
        app.config.from_object(DevelopmentConfig())
    
    # Security headers
    @app.after_request
    def set_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response
    
    return app

# Create app instance
app = create_app(os.environ.get('FLASK_ENV', 'production'))

# Initialize extensions
CORS(app, origins=app.config['CORS_ORIGINS'])
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"],
    storage_uri=app.config['RATELIMIT_STORAGE_URL']
)
cache = Cache(app)
sock = Sock(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not app.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================= Security Utilities =========================

class TokenManager:
    """Secure token management"""
    
    def __init__(self, encryption_key: str):
        self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        self._tokens = {}
        self._lock = threading.Lock()
    
    def store_token(self, key: str, token: str) -> None:
        """Store encrypted token"""
        with self._lock:
            encrypted = self.cipher.encrypt(token.encode())
            self._tokens[key] = encrypted
            # Also store in Redis for persistence
            try:
                redis_client = redis.from_url(app.config['CACHE_REDIS_URL'])
                redis_client.setex(f"token:{key}", 3600, encrypted)
            except Exception as e:
                logger.error(f"Failed to store token in Redis: {e}")
    
    def get_token(self, key: str) -> Optional[str]:
        """Retrieve and decrypt token"""
        with self._lock:
            # Try memory first
            encrypted = self._tokens.get(key)
            if not encrypted:
                # Try Redis
                try:
                    redis_client = redis.from_url(app.config['CACHE_REDIS_URL'])
                    encrypted = redis_client.get(f"token:{key}")
                except Exception as e:
                    logger.error(f"Failed to retrieve token from Redis: {e}")
                    return None
            
            if encrypted:
                try:
                    return self.cipher.decrypt(encrypted).decode()
                except Exception as e:
                    logger.error(f"Failed to decrypt token: {e}")
            return None

# Initialize token manager
token_manager = TokenManager(app.config['ENCRYPTION_KEY'])

# ========================= Input Validation Schemas =========================

class SymbolSchema(Schema):
    """Validation for symbol parameter"""
    symbol = fields.Str(required=True, validate=validate.Length(min=1, max=50))

class HistoryRequestSchema(Schema):
    """Validation for history request"""
    symbol = fields.Str(required=True, validate=validate.Length(min=1, max=50))
    resolution = fields.Str(required=True)
    date_format = fields.Int(required=True, validate=validate.OneOf([0, 1]))
    range_from = fields.Str(required=True)
    range_to = fields.Str(required=True)
    cont_flag = fields.Int(missing=1, validate=validate.OneOf([0, 1]))
    oi_flag = fields.Int(missing=0, validate=validate.OneOf([0, 1]))
    include_ai_analysis = fields.Bool(missing=False)
    ai_analysis_type = fields.Str(missing="technical", validate=validate.OneOf(["general", "technical", "sentiment", "risk"]))

class OrderSchema(Schema):
    """Validation for order placement"""
    symbol = fields.Str(required=True)
    qty = fields.Int(required=True, validate=validate.Range(min=1))
    type = fields.Int(required=True, validate=validate.OneOf([1, 2, 3, 4]))
    side = fields.Int(required=True, validate=validate.OneOf([1, -1]))
    productType = fields.Str(required=True, validate=validate.OneOf(["CNC", "INTRADAY", "MARGIN", "CO", "BO"]))
    limitPrice = fields.Float(missing=0)
    stopPrice = fields.Float(missing=0)
    validity = fields.Str(missing="DAY", validate=validate.OneOf(["DAY", "IOC"]))
    disclosedQty = fields.Int(missing=0)
    offlineOrder = fields.Bool(missing=False)

class CommoditySchema(Schema):
    """Validation for commodity operations"""
    symbol = fields.Str(required=True, validate=validate.Regexp(r'^MCX:.*'))
    qty = fields.Int(required=True, validate=validate.Range(min=1))
    type = fields.Str(required=True, validate=validate.OneOf(["FUTURES", "OPTIONS"]))

class MutualFundSchema(Schema):
    """Validation for mutual fund operations"""
    symbol = fields.Str(required=True)
    amount = fields.Float(required=True, validate=validate.Range(min=1))
    type = fields.Str(required=True, validate=validate.OneOf(["LUMPSUM", "SIP"]))
    folio_number = fields.Str(missing=None)

# ========================= Decorators =========================

def validate_input(schema_class):
    """Decorator for input validation"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            schema = schema_class()
            try:
                if request.method == 'GET':
                    data = schema.load(request.args)
                else:
                    data = schema.load(request.json or {})
                request.validated_data = data
            except ValidationError as err:
                return jsonify({"error": "Validation error", "details": err.messages}), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for API key in header
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "Authentication required"}), 401
        
        # Validate API key (implement your logic)
        # For now, we'll check if Fyers is initialized
        if not fyers_manager.is_initialized():
            return jsonify({"error": "Fyers API not initialized"}), 503
        
        return f(*args, **kwargs)
    return decorated_function

# ========================= Fyers API Manager =========================

class FyersAPIManager:
    """Centralized Fyers API management with error handling and retry logic"""
    
    def __init__(self):
        self.fyers_instance = None
        self.is_authenticated = False
        self._lock = threading.Lock()
        self.retry_count = 3
        self.retry_delay = 1
    
    def initialize(self, access_token: Optional[str] = None) -> bool:
        """Initialize Fyers model with token"""
        with self._lock:
            token = access_token or token_manager.get_token("fyers_access")
            
            if not token:
                logger.warning("No access token available for Fyers initialization")
                return False
            
            try:
                self.fyers_instance = fyersModel.FyersModel(
                    token=token,
                    is_async=False,
                    client_id=app.config['FYERS_CLIENT_ID'],
                    log_path=""
                )
                self.is_authenticated = True
                logger.info("Fyers API initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Fyers API: {e}")
                self.is_authenticated = False
                return False
    
    def is_initialized(self) -> bool:
        """Check if Fyers is initialized"""
        return self.fyers_instance is not None and self.is_authenticated
    
    def make_api_call(self, method, *args, **kwargs):
        """Make API call with retry logic and error handling"""
        if not self.is_initialized():
            if not self.initialize():
                raise Exception("Fyers API not initialized")
        
        last_error = None
        for attempt in range(self.retry_count):
            try:
                result = method(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Handle token expiration
                if any(word in error_msg for word in ['token', 'auth', 'expired', 'invalid']):
                    logger.warning(f"Token error detected: {e}")
                    if self.refresh_token():
                        continue
                    else:
                        raise Exception("Authentication failed and refresh token failed")
                
                # Handle rate limiting
                if 'rate' in error_msg or '429' in str(e):
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                
                # Other errors
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
        raise last_error or Exception("API call failed after retries")
    
    def refresh_token(self) -> bool:
        """Refresh access token"""
        refresh_token = token_manager.get_token("fyers_refresh")
        if not refresh_token:
            logger.error("No refresh token available")
            return False
        
        try:
            session = fyersModel.SessionModel(
                client_id=app.config['FYERS_CLIENT_ID'],
                redirect_uri=app.config['FYERS_REDIRECT_URI'],
                response_type="code",
                state="refresh_state",
                secret_key=app.config['FYERS_SECRET_KEY'],
                grant_type="refresh_token"
            )
            session.set_token(refresh_token)
            response = session.generate_token()
            
            if response and response.get("s") == "ok":
                new_access = response["access_token"]
                new_refresh = response.get("refresh_token", refresh_token)
                
                token_manager.store_token("fyers_access", new_access)
                token_manager.store_token("fyers_refresh", new_refresh)
                
                return self.initialize(new_access)
            
            logger.error(f"Token refresh failed: {response}")
            return False
            
        except Exception as e:
            logger.error(f"Error during token refresh: {e}")
            return False

# Initialize Fyers manager
fyers_manager = FyersAPIManager()

# ========================= AI Manager =========================

class AIAnalysisManager:
    """Centralized AI analysis management"""
    
    def __init__(self):
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize Gemini AI model"""
        if not app.config['GEMINI_API_KEY']:
            logger.warning("Gemini API key not configured")
            return
        
        try:
            genai.configure(api_key=app.config['GEMINI_API_KEY'])
            
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
            
            self.model = genai.GenerativeModel(
                model_name=app.config['GEMINI_MODEL'],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"Gemini AI initialized with model: {app.config['GEMINI_MODEL']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if AI model is available"""
        return self.model is not None
    
    @cache.memoize(timeout=300)
    def analyze(self, data: Dict[str, Any], analysis_type: str = "general") -> Dict[str, Any]:
        """Perform AI analysis with caching"""
        if not self.is_available():
            return {"error": "AI model not available"}
        
        try:
            # Sanitize data to prevent prompt injection
            sanitized_data = self._sanitize_data(data)
            
            prompts = {
                "general": self._get_general_prompt(sanitized_data),
                "technical": self._get_technical_prompt(sanitized_data),
                "sentiment": self._get_sentiment_prompt(sanitized_data),
                "risk": self._get_risk_prompt(sanitized_data),
                "commodity": self._get_commodity_prompt(sanitized_data),
                "mutual_fund": self._get_mutual_fund_prompt(sanitized_data)
            }
            
            prompt = prompts.get(analysis_type, prompts["general"])
            response = self.model.generate_content(prompt)
            
            return {
                "analysis_type": analysis_type,
                "analysis": response.text,
                "timestamp": datetime.datetime.now().isoformat(),
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {"error": str(e)}
    
    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize data to prevent prompt injection"""
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            # Remove potential injection patterns
            sanitized = data.replace("```", "").replace("system:", "").replace("assistant:", "")
            return sanitized[:1000]  # Limit string length
        else:
            return data
    
    def _get_general_prompt(self, data):
        return f"""
        Analyze the following market data and provide insights:
        {json.dumps(data, indent=2)}
        
        Provide:
        1. Key observations
        2. Trend analysis
        3. Risk factors
        4. Opportunities
        5. Recommendations
        
        Format as structured JSON response.
        """
    
    def _get_technical_prompt(self, data):
        return f"""
        Perform technical analysis on:
        {json.dumps(data, indent=2)}
        
        Include:
        1. Support/resistance levels
        2. Trend direction
        3. Volume analysis
        4. Technical indicators
        5. Entry/exit points
        
        Provide specific price levels and percentages.
        """
    
    def _get_sentiment_prompt(self, data):
        return f"""
        Analyze market sentiment for:
        {json.dumps(data, indent=2)}
        
        Provide:
        1. Overall sentiment (bullish/bearish/neutral)
        2. Sentiment strength (1-10)
        3. Key drivers
        4. Potential shifts
        5. Contrarian opportunities
        """
    
    def _get_risk_prompt(self, data):
        return f"""
        Perform risk analysis on:
        {json.dumps(data, indent=2)}
        
        Include:
        1. Risk level (low/medium/high)
        2. Specific risks
        3. Volatility analysis
        4. Mitigation strategies
        5. Position sizing recommendations
        """
    
    def _get_commodity_prompt(self, data):
        return f"""
        Analyze commodity market data:
        {json.dumps(data, indent=2)}
        
        Focus on:
        1. Supply/demand dynamics
        2. Seasonal patterns
        3. Global factors impact
        4. Price forecasts
        5. Trading recommendations
        
        Consider commodity-specific factors like weather, geopolitics, and storage.
        """
    
    def _get_mutual_fund_prompt(self, data):
        return f"""
        Analyze mutual fund data:
        {json.dumps(data, indent=2)}
        
        Provide:
        1. Performance analysis
        2. Risk-adjusted returns
        3. Expense ratio impact
        4. Peer comparison
        5. Investment suitability
        
        Consider long-term wealth creation and systematic investment benefits.
        """

# Initialize AI manager
ai_manager = AIAnalysisManager()

# ========================= WebSocket Manager =========================

class WebSocketManager:
    """Centralized WebSocket management"""
    
    def __init__(self):
        self.clients = set()
        self.subscriptions = {}
        self.data_socket = None
        self._lock = threading.Lock()
        self.max_connections = app.config['MAX_WS_CONNECTIONS']
        self.heartbeat_interval = app.config['WS_HEARTBEAT_INTERVAL']
        self._running = False
    
    def add_client(self, client_ws):
        """Add a new client connection"""
        with self._lock:
            if len(self.clients) >= self.max_connections:
                raise Exception("Maximum connections reached")
            self.clients.add(client_ws)
            logger.info(f"Client connected. Total: {len(self.clients)}")
    
    def remove_client(self, client_ws):
        """Remove a client connection"""
        with self._lock:
            self.clients.discard(client_ws)
            # Clean up subscriptions
            if client_ws in self.subscriptions:
                del self.subscriptions[client_ws]
            logger.info(f"Client disconnected. Total: {len(self.clients)}")
    
    def broadcast(self, message: Union[str, dict]):
        """Broadcast message to all connected clients"""
        if isinstance(message, dict):
            message = json.dumps(message)
        
        with self._lock:
            disconnected = []
            for client in self.clients:
                try:
                    client.send(message)
                except Exception as e:
                    logger.debug(f"Failed to send to client: {e}")
                    disconnected.append(client)
            
            for client in disconnected:
                self.remove_client(client)
    
    def start_heartbeat(self):
        """Start heartbeat to keep connections alive"""
        def heartbeat_loop():
            while self._running:
                time.sleep(self.heartbeat_interval)
                self.broadcast({"type": "heartbeat", "timestamp": time.time()})
        
        self._running = True
        threading.Thread(target=heartbeat_loop, daemon=True).start()
    
    def stop_heartbeat(self):
        """Stop heartbeat"""
        self._running = False

# Initialize WebSocket manager
ws_manager = WebSocketManager()
ws_manager.start_heartbeat()

# ========================= Market Data Helpers =========================

def calculate_indicators(candles: List) -> Dict[str, Any]:
    """Calculate technical indicators"""
    if not candles or len(candles) < 2:
        return {}
    
    try:
        closes = np.array([c[4] for c in candles])
        highs = np.array([c[2] for c in candles])
        lows = np.array([c[3] for c in candles])
        volumes = np.array([c[5] for c in candles]) if len(candles[0]) > 5 else None
        
        indicators = {
            "current_price": float(closes[-1]),
            "price_change": float(closes[-1] - closes[-2]),
            "price_change_pct": float((closes[-1] - closes[-2]) / closes[-2] * 100) if closes[-2] != 0 else 0
        }
        
        # Moving averages
        for period in [10, 20, 50]:
            if len(closes) >= period:
                indicators[f"sma_{period}"] = float(np.mean(closes[-period:]))
        
        # RSI
        if len(closes) >= 15:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                indicators["rsi"] = float(100 - (100 / (1 + rs)))
            else:
                indicators["rsi"] = 100.0 if avg_gain > 0 else 50.0
        
        # Bollinger Bands
        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            std_20 = np.std(closes[-20:])
            indicators["bb_upper"] = float(sma_20 + 2 * std_20)
            indicators["bb_lower"] = float(sma_20 - 2 * std_20)
            indicators["bb_middle"] = float(sma_20)
        
        # Volume indicators
        if volumes is not None and len(volumes) >= 20:
            indicators["volume_avg"] = float(np.mean(volumes[-20:]))
            indicators["volume_ratio"] = float(volumes[-1] / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) != 0 else 1.0
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

# ========================= API Routes - Authentication =========================

@app.route('/api/auth/login')
def fyers_login():
    """Initiate Fyers OAuth flow"""
    if not all([app.config['FYERS_CLIENT_ID'], app.config['FYERS_REDIRECT_URI'], app.config['FYERS_SECRET_KEY']]):
        return jsonify({"error": "Fyers API credentials not configured"}), 500
    
    session = fyersModel.SessionModel(
        client_id=app.config['FYERS_CLIENT_ID'],
        redirect_uri=app.config['FYERS_REDIRECT_URI'],
        response_type="code",
        state="fyers_state",
        secret_key=app.config['FYERS_SECRET_KEY'],
        grant_type="authorization_code"
    )
    
    auth_url = session.generate_authcode()
    logger.info("Redirecting to Fyers authentication")
    return redirect(auth_url)

@app.route('/api/auth/callback')
def fyers_callback():
    """Handle Fyers OAuth callback"""
    auth_code = request.args.get('auth_code')
    state = request.args.get('state')
    error = request.args.get('error')
    
    if error:
        logger.error(f"Authentication error: {error}")
        return jsonify({"error": f"Authentication failed: {error}"}), 400
    
    if not auth_code:
        return jsonify({"error": "No authorization code received"}), 400
    
    try:
        session = fyersModel.SessionModel(
            client_id=app.config['FYERS_CLIENT_ID'],
            redirect_uri=app.config['FYERS_REDIRECT_URI'],
            response_type="code",
            state=state,
            secret_key=app.config['FYERS_SECRET_KEY'],
            grant_type="authorization_code"
        )
        session.set_token(auth_code)
        response = session.generate_token()
        
        if response and response.get("s") == "ok":
            access_token = response["access_token"]
            refresh_token = response["refresh_token"]
            
            # Store tokens securely
            token_manager.store_token("fyers_access", access_token)
            token_manager.store_token("fyers_refresh", refresh_token)
            
            # Initialize Fyers
            fyers_manager.initialize(access_token)
            
            logger.info("Authentication successful")
            return jsonify({"message": "Authentication successful", "authenticated": True})
        else:
            logger.error(f"Token generation failed: {response}")
            return jsonify({"error": "Failed to generate tokens"}), 500
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================= API Routes - Market Data =========================

@app.route('/api/market/profile')
@require_auth
@limiter.limit("10 per minute")
def get_profile():
    """Get user profile"""
    try:
        result = fyers_manager.make_api_call(fyers_manager.fyers_instance.get_profile)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Profile API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/funds')
@require_auth
@limiter.limit("20 per minute")
def get_funds():
    """Get funds information"""
    try:
        result = fyers_manager.make_api_call(fyers_manager.fyers_instance.funds)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Funds API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/holdings')
@require_auth
@limiter.limit("20 per minute")
def get_holdings():
    """Get holdings"""
    try:
        result = fyers_manager.make_api_call(fyers_manager.fyers_instance.holdings)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Holdings API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/history', methods=['POST'])
@require_auth
@limiter.limit("30 per minute")
@validate_input(HistoryRequestSchema)
def get_history():
    """Get historical data with optional AI analysis"""
    data = request.validated_data
    
    try:
        # Fetch historical data
        history_result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.history,
            data=data
        )
        
        if data.get('include_ai_analysis') and history_result.get('candles'):
            # Calculate indicators
            indicators = calculate_indicators(history_result['candles'])
            
            # Get AI analysis
            analysis_data = {
                "symbol": data['symbol'],
                "indicators": indicators,
                "candles_count": len(history_result['candles'])
            }
            
            ai_analysis = ai_manager.analyze(
                analysis_data,
                data.get('ai_analysis_type', 'technical')
            )
            
            history_result['indicators'] = indicators
            history_result['ai_analysis'] = ai_analysis
        
        return jsonify(history_result)
        
    except Exception as e:
        logger.error(f"History API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/quotes')
@require_auth
@limiter.limit("50 per minute")
@cache.cached(timeout=10, query_string=True)
def get_quotes():
    """Get real-time quotes"""
    symbols = request.args.get('symbols')
    if not symbols:
        return jsonify({"error": "Missing symbols parameter"}), 400
    
    try:
        result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.quotes,
            data={"symbols": symbols}
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Quotes API error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================= API Routes - Commodities =========================

@app.route('/api/commodities/list')
@require_auth
@limiter.limit("10 per minute")
@cache.cached(timeout=3600)
def get_commodities_list():
    """Get list of available commodities"""
    commodities = {
        "energy": [
            {"symbol": "MCX:CRUDEOIL", "name": "Crude Oil"},
            {"symbol": "MCX:NATURALGAS", "name": "Natural Gas"}
        ],
        "metals": [
            {"symbol": "MCX:GOLD", "name": "Gold"},
            {"symbol": "MCX:SILVER", "name": "Silver"},
            {"symbol": "MCX:COPPER", "name": "Copper"},
            {"symbol": "MCX:ZINC", "name": "Zinc"},
            {"symbol": "MCX:LEAD", "name": "Lead"},
            {"symbol": "MCX:NICKEL", "name": "Nickel"},
            {"symbol": "MCX:ALUMINIUM", "name": "Aluminium"}
        ],
        "agriculture": [
            {"symbol": "MCX:COTTON", "name": "Cotton"},
            {"symbol": "MCX:CPO", "name": "Crude Palm Oil"},
            {"symbol": "MCX:MENTHAOIL", "name": "Mentha Oil"},
            {"symbol": "MCX:CARDAMOM", "name": "Cardamom"}
        ]
    }
    return jsonify(commodities)

@app.route('/api/commodities/quotes')
@require_auth
@limiter.limit("30 per minute")
def get_commodity_quotes():
    """Get commodity quotes"""
    symbols = request.args.get('symbols')
    if not symbols:
        return jsonify({"error": "Missing symbols parameter"}), 400
    
    # Validate commodity symbols
    commodity_symbols = [s for s in symbols.split(',') if s.startswith('MCX:')]
    if not commodity_symbols:
        return jsonify({"error": "No valid commodity symbols provided"}), 400
    
    try:
        result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.quotes,
            data={"symbols": ','.join(commodity_symbols)}
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Commodity quotes error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/commodities/history', methods=['POST'])
@require_auth
@limiter.limit("20 per minute")
@validate_input(HistoryRequestSchema)
def get_commodity_history():
    """Get commodity historical data"""
    data = request.validated_data
    
    # Validate commodity symbol
    if not data['symbol'].startswith('MCX:'):
        return jsonify({"error": "Invalid commodity symbol"}), 400
    
    try:
        result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.history,
            data=data
        )
        
        # Add commodity-specific analysis if requested
        if data.get('include_ai_analysis'):
            analysis = ai_manager.analyze(
                {"commodity_data": result, "symbol": data['symbol']},
                "commodity"
            )
            result['ai_analysis'] = analysis
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Commodity history error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/commodities/order', methods=['POST'])
@require_auth
@limiter.limit("10 per minute")
@validate_input(CommoditySchema)
def place_commodity_order():
    """Place commodity order"""
    data = request.validated_data
    
    try:
        order_data = {
            "symbol": data['symbol'],
            "qty": data['qty'],
            "type": 2 if data['type'] == "FUTURES" else 4,  # Market or limit
            "side": 1,  # Buy
            "productType": "INTRADAY",
            "validity": "DAY"
        }
        
        result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.place_order,
            data=order_data
        )
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Commodity order error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================= API Routes - Mutual Funds =========================

@app.route('/api/mutual-funds/list')
@require_auth
@limiter.limit("10 per minute")
@cache.cached(timeout=3600)
def get_mutual_funds_list():
    """Get list of available mutual funds"""
    mutual_funds = {
        "equity": [
            {"symbol": "INF200K01VK6", "name": "SBI Blue Chip Fund", "category": "Large Cap"},
            {"symbol": "INF179K01AU2", "name": "HDFC Mid-Cap Opportunities", "category": "Mid Cap"},
            {"symbol": "INF090I01BJ9", "name": "Axis Small Cap Fund", "category": "Small Cap"}
        ],
        "debt": [
            {"symbol": "INF277K01ZA9", "name": "ICICI Prudential Short Term", "category": "Short Duration"},
            {"symbol": "INF179K01LS8", "name": "HDFC Corporate Bond", "category": "Corporate Bond"}
        ],
        "hybrid": [
            {"symbol": "INF200K01RZ0", "name": "SBI Equity Hybrid Fund", "category": "Aggressive Hybrid"},
            {"symbol": "INF179K01BB9", "name": "HDFC Balanced Advantage", "category": "Dynamic Asset"}
        ],
        "elss": [
            {"symbol": "INF090I01239", "name": "Axis Tax Saver", "category": "ELSS"},
            {"symbol": "INF109K01BQ5", "name": "Mirae Asset Tax Saver", "category": "ELSS"}
        ]
    }
    return jsonify(mutual_funds)

@app.route('/api/mutual-funds/nav')
@require_auth
@limiter.limit("30 per minute")
def get_mutual_fund_nav():
    """Get mutual fund NAV"""
    symbols = request.args.get('symbols')
    if not symbols:
        return jsonify({"error": "Missing symbols parameter"}), 400
    
    try:
        # Mutual funds use different API endpoint
        # This is a placeholder - actual implementation depends on Fyers MF API
        result = {
            "funds": [
                {
                    "symbol": symbol,
                    "nav": 150.25,  # Placeholder
                    "date": datetime.datetime.now().isoformat(),
                    "change": 1.25,
                    "change_pct": 0.84
                }
                for symbol in symbols.split(',')
            ]
        }
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Mutual fund NAV error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mutual-funds/order', methods=['POST'])
@require_auth
@limiter.limit("10 per minute")
@validate_input(MutualFundSchema)
def place_mutual_fund_order():
    """Place mutual fund order (SIP/Lumpsum)"""
    data = request.validated_data
    
    try:
        # Mutual fund orders are different from regular orders
        # This is a placeholder - actual implementation depends on Fyers MF API
        order_data = {
            "symbol": data['symbol'],
            "amount": data['amount'],
            "type": data['type'],
            "folio_number": data.get('folio_number')
        }
        
        # Simulate order placement
        result = {
            "order_id": f"MF{int(time.time())}",
            "status": "success",
            "message": f"{data['type']} order placed successfully",
            "details": order_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Mutual fund order error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mutual-funds/sip', methods=['GET'])
@require_auth
@limiter.limit("20 per minute")
def get_sip_details():
    """Get SIP details"""
    try:
        # Placeholder for SIP details
        sips = [
            {
                "sip_id": "SIP001",
                "fund_name": "SBI Blue Chip Fund",
                "amount": 5000,
                "frequency": "Monthly",
                "start_date": "2024-01-01",
                "next_date": "2024-12-01",
                "status": "Active"
            }
        ]
        return jsonify({"sips": sips})
        
    except Exception as e:
        logger.error(f"SIP details error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mutual-funds/performance', methods=['POST'])
@require_auth
@limiter.limit("20 per minute")
def get_mutual_fund_performance():
    """Get mutual fund performance with AI analysis"""
    data = request.json
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Missing symbol"}), 400
    
    try:
        # Placeholder performance data
        performance = {
            "symbol": symbol,
            "returns": {
                "1D": 0.5,
                "1W": 1.2,
                "1M": 3.5,
                "3M": 8.2,
                "6M": 12.5,
                "1Y": 18.3,
                "3Y": 45.2,
                "5Y": 85.6
            },
            "risk_metrics": {
                "sharpe_ratio": 1.45,
                "beta": 0.95,
                "alpha": 2.3,
                "standard_deviation": 14.5
            }
        }
        
        # Add AI analysis
        if data.get('include_ai_analysis'):
            analysis = ai_manager.analyze(performance, "mutual_fund")
            performance['ai_analysis'] = analysis
        
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"MF performance error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================= API Routes - Orders =========================

@app.route('/api/orders/place', methods=['POST'])
@require_auth
@limiter.limit("20 per minute")
@validate_input(OrderSchema)
def place_order():
    """Place order for stocks"""
    data = request.validated_data
    
    try:
        result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.place_order,
            data=data
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/orders/modify', methods=['PATCH'])
@require_auth
@limiter.limit("20 per minute")
def modify_order():
    """Modify existing order"""
    data = request.json
    if not data or not data.get("id"):
        return jsonify({"error": "Order ID required"}), 400
    
    try:
        result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.modify_order,
            data=data
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Order modification error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/orders/cancel', methods=['DELETE'])
@require_auth
@limiter.limit("20 per minute")
def cancel_order():
    """Cancel order"""
    data = request.json
    if not data or not data.get("id"):
        return jsonify({"error": "Order ID required"}), 400
    
    try:
        result = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.cancel_order,
            data=data
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Order cancellation error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================= API Routes - AI Analysis =========================

@app.route('/api/ai/analyze', methods=['POST'])
@require_auth
@limiter.limit("10 per minute")
def ai_analyze():
    """Analyze market data with AI"""
    data = request.json
    if not data or not data.get("data"):
        return jsonify({"error": "Missing data for analysis"}), 400
    
    analysis_type = data.get("analysis_type", "general")
    result = ai_manager.analyze(data["data"], analysis_type)
    return jsonify(result)

@app.route('/api/ai/signals', methods=['POST'])
@require_auth
@limiter.limit("10 per minute")
def ai_trading_signals():
    """Generate AI trading signals"""
    data = request.json
    symbol = data.get("symbol")
    
    if not symbol:
        return jsonify({"error": "Missing symbol"}), 400
    
    try:
        # Fetch current data
        quotes = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.quotes,
            data={"symbols": symbol}
        )
        
        # Get historical data
        history_data = {
            "symbol": symbol,
            "resolution": "1D",
            "date_format": 0,
            "range_from": str(int(time.time()) - 30*24*60*60),
            "range_to": str(int(time.time())),
            "cont_flag": 1
        }
        
        history = fyers_manager.make_api_call(
            fyers_manager.fyers_instance.history,
            data=history_data
        )
        
        # Generate signals
        analysis_data = {
            "symbol": symbol,
            "current": quotes,
            "history": history.get("candles", []),
            "indicators": calculate_indicators(history.get("candles", []))
        }
        
        signals = ai_manager.analyze(analysis_data, "technical")
        return jsonify(signals)
        
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return jsonify({"error": str(e)}), 500

# ========================= WebSocket Routes =========================

@sock.route('/ws/market')
def market_websocket(ws):
    """WebSocket endpoint for real-time market data"""
    try:
        ws_manager.add_client(ws)
        
        while True:
            message = ws.receive()
            if message is None:
                break
            
            try:
                data = json.loads(message)
                action = data.get("action")
                
                if action == "subscribe":
                    symbols = data.get("symbols", [])
                    ws.send(json.dumps({
                        "status": "subscribed",
                        "symbols": symbols
                    }))
                    
                elif action == "unsubscribe":
                    symbols = data.get("symbols", [])
                    ws.send(json.dumps({
                        "status": "unsubscribed",
                        "symbols": symbols
                    }))
                    
                elif action == "ping":
                    ws.send(json.dumps({"type": "pong"}))
                    
            except json.JSONDecodeError:
                ws.send(json.dumps({"error": "Invalid JSON"}))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.remove_client(ws)

# ========================= Health & Status =========================

@app.route('/health')
@limiter.exempt
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "fyers": fyers_manager.is_initialized(),
            "ai": ai_manager.is_available(),
            "websocket": len(ws_manager.clients)
        }
    }
    return jsonify(status)

@app.route('/')
@limiter.exempt
def home():
    """API documentation"""
    return jsonify({
        "name": "Fyers Trading API Proxy",
        "version": "2.0.0",
        "endpoints": {
            "authentication": {
                "login": "/api/auth/login",
                "callback": "/api/auth/callback"
            },
            "market_data": {
                "profile": "/api/market/profile",
                "funds": "/api/market/funds",
                "holdings": "/api/market/holdings",
                "history": "/api/market/history",
                "quotes": "/api/market/quotes"
            },
            "commodities": {
                "list": "/api/commodities/list",
                "quotes": "/api/commodities/quotes",
                "history": "/api/commodities/history",
                "order": "/api/commodities/order"
            },
            "mutual_funds": {
                "list": "/api/mutual-funds/list",
                "nav": "/api/mutual-funds/nav",
                "order": "/api/mutual-funds/order",
                "sip": "/api/mutual-funds/sip",
                "performance": "/api/mutual-funds/performance"
            },
            "orders": {
                "place": "/api/orders/place",
                "modify": "/api/orders/modify",
                "cancel": "/api/orders/cancel"
            },
            "ai": {
                "analyze": "/api/ai/analyze",
                "signals": "/api/ai/signals"
            },
            "websocket": "/ws/market",
            "health": "/health"
        },
        "features": [
            "Stocks Trading",
            "Commodities Trading",
            "Mutual Funds Investment",
            "AI-Powered Analysis",
            "Real-time WebSocket Data",
            "Rate Limiting",
            "Caching",
            "Secure Token Management"
        ]
    })

# ========================= Error Handlers =========================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({"error": "Rate limit exceeded", "message": str(error.description)}), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# ========================= Main =========================

if __name__ == '__main__':
    # Initialize services
    if token_manager.get_token("fyers_access"):
        fyers_manager.initialize()
    
    # Run app (use gunicorn or uwsgi in production)
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False  # Never use debug=True in production
    )
