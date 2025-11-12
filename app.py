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

# Configure CORS
CORS(
    app,
    resources={r"/*": {"origins": config.ALLOWED_ORIGINS if not config.ALLOW_ALL_ORIGINS else "*"}},
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

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    origin = request.headers.get("Origin")
    if not origin or config.ALLOW_ALL_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin in config.ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    app.logger.info(f"📤 Response: {response.status}")
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "status": 404, "path": request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {error}")
    return jsonify({
        "error": "Internal server error",
        "status": 500,
        "message": str(error)
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Fyers Proxy Server",
        "cors": "enabled",
        "allow_all_origins": config.ALLOW_ALL_ORIGINS
    }), 200

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    return jsonify({
        "status": "success",
        "message": "Connection successful!",
        "timestamp": datetime.datetime.now().isoformat(),
        "method": request.method,
        "origin": request.headers.get('Origin', 'No origin'),
        "headers_received": dict(request.headers),
        "data_received": request.json if request.is_json else None
    }), 200

@app.route('/api/echo', methods=['POST'])
def echo_endpoint():
    return jsonify({
        "status": "success",
        "echoed_data": request.json if request.is_json else request.data.decode('utf-8'),
        "timestamp": datetime.datetime.now().isoformat()
    }), 200

@app.route('/')
def home():
    from models import gemini_model
    return jsonify({
        "message": "🚀 Fyers API Proxy Server is running!",
        "status": "online",
        "version": "1.0.0",
        "allow_all_origins": config.ALLOW_ALL_ORIGINS,
        "timestamp": datetime.datetime.now().isoformat(),
        "token_status": "✅ Token loaded" if config.ACCESS_TOKEN else "❌ No valid token found",
        "endpoints": {
            "health": "/health",
            "test": "/api/test",
            "fyers_profile": "/api/fyers/profile",
            "fyers_funds": "/api/fyers/funds",
            "ai_routes": "/api/ai/*"
        }
    }), 200

@sock.route('/ws/fyers')
def ws_fyers(ws):
    handle_websocket(ws)

# Register blueprints
app.register_blueprint(fyers_bp)
app.register_blueprint(ai_bp)
app.register_blueprint(order_bp)
app.register_blueprint(market_bp)
app.register_blueprint(backtesting_bp)

def initialize_app():
    """Initialize the application"""
    try:
        load_tokens()
        app.logger.info("✅ Tokens loaded")

        if not config.ACCESS_TOKEN:
            app.logger.warning("⚠️ No ACCESS_TOKEN found — please authenticate via /fyers-login before using /api/fyers/*")
            return

        if initialize_fyers_model():
            app.logger.info("✅ Fyers model initialized")
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

initialize_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)
