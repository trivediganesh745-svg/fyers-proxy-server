import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_sock import Sock
import datetime
import logging
import json

# Import configuration
import config
from auth import load_tokens, initialize_fyers_model

# Import blueprints
from api.fyers_routes import fyers_bp
from api.ai_routes import ai_bp
from api.order_routes import order_bp
from api.market_routes import market_bp
from backtesting.routes import backtesting_bp

# Import websocket handler
from websocket_handler import handle_websocket, start_data_socket_in_thread

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Simplified for Google AI Studio
if config.ALLOW_ALL_ORIGINS:
    # Allow all origins for Google AI Studio
    CORS(app, 
         resources={r"/*": {"origins": "*"}},
         allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
         methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
         supports_credentials=False,
         max_age=86400
    )
else:
    # Use specific origins
    CORS(app,
         resources={r"/*": {"origins": config.ALLOWED_ORIGINS}},
         allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
         methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
         supports_credentials=True,
         max_age=86400
    )

# Initialize Sock for websocket connections
sock = Sock(app)

# Configure logging
app.logger.setLevel(logging.INFO)

# Add request logging middleware
@app.before_request
def log_request_info():
    """Log incoming requests for debugging"""
    app.logger.info(f'📨 Request: {request.method} {request.path}')
    app.logger.info(f'Origin: {request.headers.get("Origin", "No Origin")}')
    
    # Log request body for POST/PUT/PATCH
    if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
        app.logger.debug(f'Body: {request.json}')

# Handle preflight requests
@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    origin = request.headers.get('Origin', '*')
    
    # For Google AI Studio, always allow
    if config.ALLOW_ALL_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = '*'
    else:
        # Check if origin is in allowed list
        if origin in config.ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'
    
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    
    # Log response
    app.logger.info(f'📤 Response: {response.status}')
    
    return response

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "status": 404,
        "path": request.path
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal error: {error}")
    return jsonify({
        "error": "Internal server error",
        "status": 500,
        "message": str(error)
    }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Fyers Proxy Server",
        "cors": "enabled",
        "allow_all_origins": config.ALLOW_ALL_ORIGINS
    }), 200

# Public test endpoint (no auth required)
@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint to verify CORS and connectivity"""
    return jsonify({
        "status": "success",
        "message": "Connection successful!",
        "timestamp": datetime.datetime.now().isoformat(),
        "method": request.method,
        "origin": request.headers.get('Origin', 'No origin'),
        "headers_received": dict(request.headers),
        "data_received": request.json if request.is_json else None
    }), 200

# Public echo endpoint
@app.route('/api/echo', methods=['POST'])
def echo_endpoint():
    """Echo back whatever is sent"""
    return jsonify({
        "status": "success",
        "echoed_data": request.json if request.is_json else request.data.decode('utf-8'),
        "timestamp": datetime.datetime.now().isoformat()
    }), 200

# Root endpoint
@app.route('/')
def home():
    """Root endpoint with API information"""
    from models import gemini_model
    return jsonify({
        "message": "🚀 Fyers API Proxy Server is running!",
        "status": "online",
        "version": "1.0.0",
        "cors_enabled": True,
        "allow_all_origins": config.ALLOW_ALL_ORIGINS,
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoints": {
            "public": {
                "health": "/health",
                "test": "/api/test",
                "echo": "/api/echo"
            },
            "authentication": {
                "login": "/fyers-login",
                "callback": "/fyers-auth-callback",
                "status": "/api/auth/status"
            },
            "market_data": {
                "history": "/api/fyers/history",
                "quotes": "/api/fyers/quotes",
                "market_depth": "/api/fyers/market_depth",
                "market_status": "/api/fyers/market_status"
            },
            "user": {
                "profile": "/api/fyers/profile",
                "funds": "/api/fyers/funds",
                "holdings": "/api/fyers/holdings",
                "positions": "/api/fyers/positions"
            },
            "ai": {
                "analyze": "/api/ai/analyze",
                "trading_signals": "/api/ai/trading-signals",
                "chat": "/api/ai/chat",
                "portfolio_analysis": "/api/ai/portfolio-analysis",
                "market_summary": "/api/ai/market-summary"
            } if gemini_model else "AI features disabled (set GEMINI_API_KEY)",
            "websocket": "/ws/fyers"
        },
        "documentation": "https://github.com/yourusername/yourrepo"
    }), 200

# WebSocket route
@sock.route('/ws/fyers')
def ws_fyers(ws):
    """WebSocket endpoint for real-time data"""
    handle_websocket(ws)

# Register blueprints
app.register_blueprint(fyers_bp)
app.register_blueprint(ai_bp)
app.register_blueprint(order_bp)
app.register_blueprint(market_bp)
app.register_blueprint(backtesting_bp)

# Initialize on startup
def initialize_app():
    """Initialize the application"""
    try:
        # Load tokens
        load_tokens()
        app.logger.info("✅ Tokens loaded")
        
        # Initialize Fyers model
        if initialize_fyers_model():
            app.logger.info("✅ Fyers model initialized")
            
            # Start WebSocket if token available
            if config.ACCESS_TOKEN:
                try:
                    start_data_socket_in_thread(config.ACCESS_TOKEN)
                    app.logger.info("✅ WebSocket started")
                except Exception as e:
                    app.logger.warning(f"⚠️ Could not start WebSocket: {e}")
        else:
            app.logger.warning("⚠️ Fyers model not initialized - authentication required")
            
    except Exception as e:
        app.logger.error(f"❌ Initialization error: {e}")

# Initialize app
initialize_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)
