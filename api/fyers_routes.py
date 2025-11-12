from flask import Blueprint, request, jsonify, redirect
from fyers_apiv3 import fyersModel
import datetime
import logging
import config
from auth import make_fyers_api_call, store_tokens, initialize_fyers_model
from models import fyers_instance

logger = logging.getLogger(__name__)

fyers_bp = Blueprint('fyers', __name__)

def get_base_url():
    """
    Dynamically determine base URL depending on environment.
    Falls back to Render URL if available, otherwise request.host_url or config setting.
    """
    # 1️⃣ Use config variable if explicitly set
    if hasattr(config, "FYERS_PROXY_BASE_URL") and config.FYERS_PROXY_BASE_URL:
        return config.FYERS_PROXY_BASE_URL.rstrip("/")

    # 2️⃣ If running on Render, detect public Render URL
    render_external_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_external_url:
        return render_external_url.rstrip("/")

    # 3️⃣ Try request.host_url (auto-detect)
    if request and request.host_url:
        return request.host_url.rstrip("/")

    # 4️⃣ Fallback to localhost only if nothing else
    return "http://localhost:5000"


@fyers_bp.route('/fyers-login')
def fyers_login():
    """Initiates the Fyers authentication flow."""
    import os

    base_url = get_base_url()

    if not all([config.CLIENT_ID, config.REDIRECT_URI, config.SECRET_KEY]):
        logger.error("Fyers API credentials not fully configured for login.")
        return jsonify({
            "error": "Fyers API credentials not configured",
            "required": ["FYERS_CLIENT_ID", "FYERS_SECRET_KEY", "FYERS_REDIRECT_URI"],
            "message": "Please set the required environment variables in Render"
        }), 500

    try:
        # Fix redirect_uri dynamically if it’s localhost (common issue)
        redirect_uri = config.REDIRECT_URI
        if "localhost" in redirect_uri:
            redirect_uri = f"{base_url}/fyers-auth-callback"
            logger.info(f"Redirect URI updated for Render: {redirect_uri}")

        session = fyersModel.SessionModel(
            client_id=config.CLIENT_ID,
            redirect_uri=redirect_uri,
            response_type="code",
            state="fyers_proxy_state",
            secret_key=config.SECRET_KEY,
            grant_type="authorization_code"
        )
        generate_token_url = session.generate_authcode()
        logger.info(f"Redirecting to Fyers login: {generate_token_url}")
        return redirect(generate_token_url)
    except Exception as e:
        logger.error(f"Error creating Fyers session: {e}")
        return jsonify({
            "error": "Failed to initiate Fyers login",
            "message": str(e)
        }), 500


@fyers_bp.route('/fyers-auth-callback')
def fyers_auth_callback():
    """Callback endpoint after the user logs in on Fyers."""
    import os
    base_url = get_base_url()

    auth_code = request.args.get('auth_code')
    state = request.args.get('state')
    error = request.args.get('error')

    if error:
        logger.error(f"Fyers authentication failed: {error}")
        return jsonify({"error": f"Fyers authentication failed: {error}"}), 400
    
    if not auth_code:
        logger.error("No auth_code received from Fyers.")
        return jsonify({"error": "No auth_code received from Fyers."}), 400

    try:
        session = fyersModel.SessionModel(
            client_id=config.CLIENT_ID,
            redirect_uri=config.REDIRECT_URI,
            response_type="code",
            state=state,
            secret_key=config.SECRET_KEY,
            grant_type="authorization_code"
        )
        session.set_token(auth_code)
        
        response = session.generate_token()

        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response["refresh_token"]

            store_tokens(new_access_token, new_refresh_token)
            
            if initialize_fyers_model(new_access_token):
                logger.info("Authentication successful!")
                
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Authentication Successful</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 40px; background: #f0f0f0; }}
                        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                        h2 {{ color: #28a745; }}
                        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                        .warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; margin: 20px 0; }}
                        .success {{ background: #d4edda; border: 1px solid #28a745; padding: 10px; border-radius: 5px; margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>✅ Fyers Authentication Successful!</h2>
                        <div class="success">
                            <strong>Success!</strong> Your tokens have been generated and saved.
                        </div>
                        
                        <h3>📋 Next Steps for Render Deployment:</h3>
                        <div class="warning">
                            <strong>Important:</strong> Update these environment variables in your Render dashboard to persist the authentication:
                        </div>
                        
                        <pre>
FYERS_ACCESS_TOKEN={new_access_token[:20]}...
FYERS_REFRESH_TOKEN={new_refresh_token[:20]}...
                        </pre>
                        
                        <p>The refresh token will be used automatically to maintain access.</p>
                        
                        <h3>🔧 Test Your Setup:</h3>
                        <ul>
                            <li><a href="{base_url}/api/auth/status" target="_blank">Check Authentication Status</a></li>
                            <li><a href="{base_url}/api/fyers/profile" target="_blank">View Profile</a></li>
                            <li><a href="{base_url}/health" target="_blank">Health Check</a></li>
                        </ul>
                        
                        <p style="margin-top: 30px; text-align: center;">
                            <button onclick="window.close()" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
                                Close Window
                            </button>
                        </p>
                    </div>
                </body>
                </html>
                """
            else:
                return jsonify({"error": "Failed to initialize Fyers model with new tokens"}), 500
        else:
            logger.error(f"Failed to generate Fyers tokens. Response: {response}")
            return jsonify({"error": "Failed to generate tokens", "response": response}), 500

    except Exception as e:
        logger.error(f"Error generating Fyers access token: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate access token: {str(e)}"}, 500)


@fyers_bp.route('/api/auth/status')
def auth_status():
    """Check current authentication status"""
    base_url = get_base_url()
    try:
        has_access = bool(config.ACCESS_TOKEN)
        has_refresh = bool(config.REFRESH_TOKEN)
        fyers_initialized = bool(fyers_instance)
        
        token_valid = False
        profile_data = None
        
        if fyers_instance:
            try:
                result = fyers_instance.get_profile()
                if result and result.get("s") == "ok":
                    token_valid = True
                    profile_data = result.get("data", {})
            except Exception as e:
                logger.error(f"Error checking profile: {e}")
        
        return jsonify({
            "authenticated": token_valid,
            "has_access_token": has_access,
            "has_refresh_token": has_refresh,
            "fyers_initialized": fyers_initialized,
            "message": "Authenticated and ready" if token_valid else "Authentication required - visit /fyers-login",
            "profile": profile_data if token_valid else None,
            "login_url": f"{base_url}/fyers-login" if not token_valid else None
        })
    except Exception as e:
        logger.error(f"Error checking auth status: {e}")
        return jsonify({
            "authenticated": False,
            "error": str(e),
            "message": "Error checking authentication status"
        }), 500


def _unauthenticated_response():
    """Helper to return a standardized unauthenticated response"""
    base_url = get_base_url()
    return jsonify({
        "error": "Not authenticated",
        "message": "Please authenticate first by visiting /fyers-login",
        "login_url": f"{base_url}/fyers-login"
    }), 401


@fyers_bp.route('/api/fyers/profile')
def get_profile():
    if not fyers_instance:
        return _unauthenticated_response()
    try:
        result = make_fyers_api_call(fyers_instance.get_profile)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            return result
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        return jsonify({"error": str(e)}), 500


@fyers_bp.route('/api/fyers/funds')
def get_funds():
    if not fyers_instance:
        return _unauthenticated_response()
    try:
        result = make_fyers_api_call(fyers_instance.funds)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            return result
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting funds: {e}")
        return jsonify({"error": str(e)}), 500


@fyers_bp.route('/api/fyers/holdings')
def get_holdings():
    if not fyers_instance:
        return _unauthenticated_response()
    try:
        result = make_fyers_api_call(fyers_instance.holdings)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            return result
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting holdings: {e}")
        return jsonify({"error": str(e)}), 500


@fyers_bp.route('/api/fyers/positions')
def get_positions():
    if not fyers_instance:
        return _unauthenticated_response()
    try:
        result = make_fyers_api_call(fyers_instance.positions)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            return result
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return jsonify({"error": str(e)}), 500
