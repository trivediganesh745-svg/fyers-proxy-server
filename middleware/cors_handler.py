from flask import request, make_response
import logging
import config

logger = logging.getLogger(__name__)

def setup_cors(app):
    """Setup CORS configuration for the Flask app"""
    
    @app.before_request
    def log_request_info():
        logger.info(f'Request: {request.method} {request.url}')
        logger.info(f'Headers: {dict(request.headers)}')
        logger.info(f'Origin: {request.headers.get("Origin", "No Origin")}')

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
        
        logger.info(f'Response Status: {response.status}')
        logger.info(f'Response Headers: {dict(response.headers)}')
        
        return response

    # Log CORS configuration
    if not config.ALLOWED_ORIGINS or "*" in config.ALLOWED_ORIGINS:
        logger.warning("⚠️ WARNING: No ALLOWED_ORIGINS set. Allowing all origins (INSECURE!)")
    else:
        logger.info(f"✅ CORS configured for origins: {config.ALLOWED_ORIGINS}")
