from flask import Blueprint, request, jsonify, redirect
from fyers_apiv3 import fyersModel
import datetime
import logging
import config
from auth import make_fyers_api_call, store_tokens, initialize_fyers_model
from models import fyers_instance

logger = logging.getLogger(__name__)

fyers_bp = Blueprint('fyers', __name__)

@fyers_bp.route('/fyers-login')
def fyers_login():
    """Initiates the Fyers authentication flow."""
    if not config.CLIENT_ID or not config.REDIRECT_URI or not config.SECRET_KEY:
        logger.error("Fyers API credentials not fully configured for login.")
        return jsonify({"error": "Fyers API credentials not fully configured on the server."}), 500

    session = fyersModel.SessionModel(
        client_id=config.CLIENT_ID,
        redirect_uri=config.REDIRECT_URI,
        response_type="code",
        state="fyers_proxy_state",
        secret_key=config.SECRET_KEY,
        grant_type="authorization_code"
    )
    generate_token_url = session.generate_authcode()
    logger.info(f"Redirecting to Fyers login: {generate_token_url}")
    return redirect(generate_token_url)

@fyers_bp.route('/fyers-auth-callback')
def fyers_auth_callback():
    """Callback endpoint after the user logs in on Fyers."""
    auth_code = request.args.get('auth_code')
    state = request.args.get('state')
    error = request.args.get('error')

    if error:
        logger.error(f"Fyers authentication failed: {error}")
        return jsonify({"error": f"Fyers authentication failed: {error}"}), 400
    if not auth_code:
        logger.error("No auth_code received from Fyers.")
        return jsonify({"error": "No auth_code received from Fyers."}), 400

    session = fyersModel.SessionModel(
        client_id=config.CLIENT_ID,
        redirect_uri=config.REDIRECT_URI,
        response_type="code",
        state=state,
        secret_key=config.SECRET_KEY,
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
                logger.info("Authentication successful!")
                
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
            logger.error(f"Failed to generate Fyers tokens. Response: {response}")
            return jsonify({"error": f"Failed to generate Fyers tokens. Response: {response}"}), 500

    except Exception as e:
        logger.error(f"Error generating Fyers access token: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate Fyers access token: {str(e)}"}), 500

@fyers_bp.route('/api/auth/status')
def auth_status():
    """Check current authentication status"""
    has_access = bool(config.ACCESS_TOKEN)
    has_refresh = bool(config.REFRESH_TOKEN)
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

@fyers_bp.route('/api/fyers/profile')
def get_profile():
    result = make_fyers_api_call(fyers_instance.get_profile)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@fyers_bp.route('/api/fyers/funds')
def get_funds():
    result = make_fyers_api_call(fyers_instance.funds)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@fyers_bp.route('/api/fyers/holdings')
def get_holdings():
    result = make_fyers_api_call(fyers_instance.holdings)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@fyers_bp.route('/api/fyers/positions')
def get_positions():
    result = make_fyers_api_call(fyers_instance.positions)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)
