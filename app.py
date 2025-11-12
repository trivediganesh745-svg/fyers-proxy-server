import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_sock import Sock
import datetime
import logging

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
CORS(app, 
     resources={
         r"/*": {
             "origins": config.ALLOWED_ORIGINS,
             "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
             "expose_headers": ["Content-Type"],
             "supports_credentials": True,
             "max_age": 3600
         }
     })

# Initialize Sock for client-facing websocket connections
sock = Sock(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# Add request logging
@app.before_request
def log_request_info():
    app.logger.info(f'Request: {request.method} {request.url}')
    app.logger.info(f'Headers: {dict(request.headers)}')
    app.logger.info(f'Origin: {request.headers.get("Origin", "No Origin")}')

# Handle OPTIONS requests explicitly
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = make_response("", 204)
        origin = request.headers.get('Origin')
        
        # Check if origin is allowed
        if origin:
            if "*" in config.ALLOWED_ORIGINS or origin in config.ALLOWED_ORIGINS:
                response.headers['Access-Control-Allow-Origin'] = origin
            else:
                response.headers['Access-Control-Allow-Origin'] = config.ALLOWED_ORIGINS[0] if config.ALLOWED_ORIGINS else "*"
        
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = request.headers.get('Access-Control-Request-Headers', 'Content-Type, Authorization')
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '3600'
        
        return response

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    
    if origin:
        if "*" in config.ALLOWED_ORIGINS or origin in config.ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
        elif config.ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = config.ALLOWED_ORIGINS[0]
    
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    
    app.logger.info(f'Response Status: {response.status}')
    app.logger.info(f'Response Headers: {dict(response.headers)}')
    
    return response

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Fyers Proxy Server"
    }), 200

# Main route
@app.route('/')
def home():
    from models import gemini_model
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
                "second": config.SECOND_RESOLUTIONS,
                "minute": config.MINUTE_RESOLUTIONS,
                "day": config.DAY_RESOLUTIONS
            },
            "ai_status": "enabled" if gemini_model else "disabled (set GEMINI_API_KEY)"
        }
    })

# WebSocket route
@sock.route('/ws/fyers')
def ws_fyers(ws):
    """Frontend connects here: wss://<host>/ws/fyers"""
    handle_websocket(ws)

# Register blueprints
app.register_blueprint(fyers_bp)
app.register_blueprint(ai_bp)
app.register_blueprint(order_bp)
app.register_blueprint(market_bp)
app.register_blueprint(backtesting_bp)

# Load tokens at startup
load_tokens()

# Initialize Fyers model at startup
initialize_fyers_model()

if __name__ == '__main__':
    # Start the Fyers data socket at startup if an access token is present
    if config.ACCESS_TOKEN:
        try:
            start_data_socket_in_thread(config.ACCESS_TOKEN)
        except Exception as e:
            app.logger.warning(f"Could not start Fyers DataSocket at startup: {e}")

    app.run(host='0.0.0.0', port=5000, debug=True)
