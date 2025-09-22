import os
from flask import Flask, request, jsonify, redirect, url_for
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
import time
import logging

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# --- Fyers API Configuration (from environment variables) ---
CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
SECRET_KEY = os.environ.get("FYERS_SECRET_KEY")
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI") # This should be your Render proxy URL, e.g., "https://your-render-app.onrender.com/fyers-auth-callback"
ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")

if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
    app.logger.warning("WARNING: Fyers API credentials (CLIENT_ID, SECRET_KEY, REDIRECT_URI) are not fully set. Some functionalities may not work.")

# Initialize FyersModel (will be re-initialized with an access token after login)
fyers_instance = None # Renamed to avoid conflict with fyersModel class

def initialize_fyers_model(token):
    global fyers_instance
    if token:
        fyers_instance = fyersModel.FyersModel(token=token, is_async=False, client_id=CLIENT_ID, log_path="")
        app.logger.info("FyersModel initialized with access token.")
    else:
        app.logger.warning("FyersModel could not be initialized: No access token provided.")

# Initialize Fyers model if access token is already available (e.g., from environment variable)
initialize_fyers_model(ACCESS_TOKEN)


# --- Fyers Authentication Flow Endpoints ---

@app.route('/fyers-login')
def fyers_login():
    """
    Initiates the Fyers authentication flow.
    Redirects the user to the Fyers login page.
    """
    if not CLIENT_ID or not REDIRECT_URI or not SECRET_KEY:
        app.logger.error("Fyers API credentials not fully configured for login.")
        return jsonify({"error": "Fyers API credentials not fully configured on the server."}), 500

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        state="fyers_proxy_state", # Use a unique state for security if needed
        secret_key=SECRET_KEY,
        grant_type="authorization_code"
    )
    generate_token_url = session.generate_authcode()
    app.logger.info(f"Redirecting to Fyers login: {generate_token_url}")
    return redirect(generate_token_url)

@app.route('/fyers-auth-callback')
def fyers_auth_callback():
    """
    Callback endpoint after the user logs in on Fyers.
    Exchanges the auth_code for an access_token.
    """
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
        new_access_token = response["access_token"]
        global ACCESS_TOKEN
        ACCESS_TOKEN = new_access_token
        initialize_fyers_model(ACCESS_TOKEN)
        
        app.logger.info("Fyers token generated successfully!")
        # Redirect to a success page or provide the token to the frontend
        return jsonify({"message": "Fyers token generated successfully!", "access_token_available": True})
        
    except Exception as e:
        app.logger.error(f"Error generating Fyers access token: {e} - Response: {response}", exc_info=True)
        return jsonify({"error": f"Failed to generate Fyers access token: {str(e)}"}), 500


# --- Fyers Data Endpoints ---

@app.route('/api/fyers/profile')
def get_profile():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        profile_data = fyers_instance.get_profile()
        return jsonify(profile_data)
    except Exception as e:
        app.logger.error(f"Failed to fetch profile: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch profile: {str(e)}"}), 500

@app.route('/api/fyers/funds')
def get_funds():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        funds_data = fyers_instance.funds()
        return jsonify(funds_data)
    except Exception as e:
        app.logger.error(f"Failed to fetch funds: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch funds: {str(e)}"}), 500

@app.route('/api/fyers/holdings')
def get_holdings():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        holdings_data = fyers_instance.holdings()
        return jsonify(holdings_data)
    except Exception as e:
        app.logger.error(f"Failed to fetch holdings: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch holdings: {str(e)}"}), 500

@app.route('/api/fyers/history', methods=['POST'])
def get_history():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        data = request.json
        
        required_params = ["symbol", "resolution", "date_format", "range_from", "range_to"]
        if not data or not all(k in data for k in required_params):
            app.logger.warning(f"Missing required parameters for history API. Received: {data}")
            return jsonify({"error": f"Missing required parameters for history API. Need {', '.join(required_params)}."}), 400
        
        # Ensure date_format is an integer as expected by the Fyers API
        try:
            data["date_format"] = int(data["date_format"])
        except ValueError:
            app.logger.warning(f"Invalid 'date_format' received: {data.get('date_format')}")
            return jsonify({"error": "Invalid 'date_format'. Must be 0 or 1."}), 400

        # Handle 'cont_flag' and 'oi_flag' if present, otherwise set defaults or omit
        if "cont_flag" in data:
            data["cont_flag"] = int(data["cont_flag"])
        if "oi_flag" in data:
            data["oi_flag"] = int(data["oi_flag"])

        history_data = fyers_instance.history(data)
        return jsonify(history_data)
    except Exception as e:
        app.logger.error(f"Failed to fetch history with data {request.json}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

@app.route('/api/fyers/quotes', methods=['GET'])
def get_quotes():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        symbols = request.args.get('symbols')
        if not symbols:
            app.logger.warning("Missing 'symbols' parameter for quotes API.")
            return jsonify({"error": "Missing 'symbols' parameter. Eg: /api/fyers/quotes?symbols=NSE:SBIN-EQ,NSE:TCS-EQ"}), 400
        
        data = {"symbols": symbols} # Fyers API expects a dict with 'symbols' key
        quotes_data = fyers_instance.quotes(data=data)
        return jsonify(quotes_data)
    except Exception as e:
        app.logger.error(f"Failed to fetch quotes for symbols {request.args.get('symbols')}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch quotes: {str(e)}"}), 500

@app.route('/api/fyers/market_depth', methods=['GET'])
def get_market_depth():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        symbol = request.args.get('symbol')
        ohlcv_flag = request.args.get('ohlcv_flag') # It's an int: 0 or 1

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
        # The Fyers API method is 'depth', not 'market_depth'
        market_depth_data = fyers_instance.depth(data=data)
        return jsonify(market_depth_data)
    except Exception as e:
        app.logger.error(f"Failed to fetch market depth for symbol {symbol}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch market depth: {str(e)}"}), 500

@app.route('/api/fyers/option_chain', methods=['GET'])
def get_option_chain():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        symbol = request.args.get('symbol')
        strikecount = request.args.get('strikecount')
        timestamp = request.args.get('timestamp') # Optional

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
        if timestamp:
            data["timestamp"] = timestamp # Add timestamp if provided

        # Corrected method name: 'optionchain' instead of 'option_chain'
        option_chain_data = fyers_instance.optionchain(data=data)
        
        return jsonify(option_chain_data)
    except Exception as e:
        app.logger.error(f"Failed to fetch option chain for symbol {symbol}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch option chain: {str(e)}"}), 500

# Example for placing an order (requires POST request with order data)
@app.route('/api/fyers/place_order', methods=['POST'])
def place_single_order():
    if not fyers_instance:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        order_data = request.json
        if not order_data:
            app.logger.warning("No order data provided for placing order.")
            return jsonify({"error": "No order data provided."}), 400
        
        response = fyers_instance.place_order(order_data)
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Failed to place order with data {request.json}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to place order: {str(e)}"}), 500


@app.route('/')
def home():
    return "Fyers API Proxy Server is running! Use /fyers-login to authenticate."


if __name__ == '__main__':
    # Use Gunicorn in production (handled by Render)
    # For local testing:
    app.run(host='0.0.0.0', port=5000)
