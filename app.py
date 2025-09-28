import os
from flask import Flask, request, jsonify, redirect, url_for
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
import time
import logging
import json

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
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI")
# These will be updated dynamically and potentially saved persistently
ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")

if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
    app.logger.error("ERROR: Fyers API credentials (CLIENT_ID, SECRET_KEY, REDIRECT_URI) are not fully set. Please check your .env file or environment variables.")
    # Exit or raise error, as core functionality won't work
    # For a web app, you might want to return a user-friendly error page/message
    # sys.exit(1) # Uncomment if you want the app to stop on missing essentials

# Initialize FyersModel (will be re-initialized with an access token after login or refresh)
fyers_instance = None 

# --- Persistent Storage Placeholder ---
# In a real application, you would save/load tokens from a database, file,
# or a more secure key-value store. For this example, we'll simulate it
# by updating the global variables, but this is NOT persistent across server restarts.
# For production, consider using Redis, a simple JSON file, or a proper DB.

def store_tokens(access_token, refresh_token):
    global ACCESS_TOKEN, REFRESH_TOKEN
    ACCESS_TOKEN = access_token
    REFRESH_TOKEN = refresh_token
    
    # In a real app, write these to a file or database.
    # For demonstration, we'll print them.
    app.logger.info("Tokens updated. For persistence, save these securely:")
    app.logger.info(f"New Access Token: {ACCESS_TOKEN}")
    app.logger.info(f"New Refresh Token: {REFRESH_TOKEN}")
    
    # Example of saving to a file (simple, but not recommended for production security)
    # with open("tokens.json", "w") as f:
    #     json.dump({"access_token": access_token, "refresh_token": refresh_token}, f)

def load_tokens():
    global ACCESS_TOKEN, REFRESH_TOKEN
    # In a real app, load from your persistent storage
    # For this example, we're loading from environment variables at startup.
    # If using a file:
    # try:
    #     with open("tokens.json", "r") as f:
    #         data = json.load(f)
    #         ACCESS_TOKEN = data.get("access_token")
    #         REFRESH_TOKEN = data.get("refresh_token")
    # except FileNotFoundError:
    #     app.logger.info("tokens.json not found, starting fresh.")
    pass

# Load tokens at startup
load_tokens()

def initialize_fyers_model(token=None):
    global fyers_instance, ACCESS_TOKEN, REFRESH_TOKEN
    
    # Prioritize provided token, then global ACCESS_TOKEN
    token_to_use = token if token else ACCESS_TOKEN

    if token_to_use:
        fyers_instance = fyersModel.FyersModel(token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
        app.logger.info("FyersModel initialized with access token.")
        return True
    elif REFRESH_TOKEN and CLIENT_ID and SECRET_KEY:
        app.logger.info("No access token provided or found, attempting to use refresh token.")
        # Attempt to refresh using the refresh token
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
    global ACCESS_TOKEN, REFRESH_TOKEN
    if not REFRESH_TOKEN:
        app.logger.error("Cannot refresh token: No refresh token available.")
        return False

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code", # Even for refresh token, this is part of SessionModel init
        state="refresh_state", 
        secret_key=SECRET_KEY,
        grant_type="refresh_token" # This is the key for refreshing
    )
    session.set_token(REFRESH_TOKEN) # Set the refresh token

    try:
        app.logger.info(f"Attempting to refresh access token using refresh token: {REFRESH_TOKEN[:5]}...")
        response = session.generate_token()
        
        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response.get("refresh_token", REFRESH_TOKEN) # Fyers might issue new refresh token, or keep old
            
            store_tokens(new_access_token, new_refresh_token)
            initialize_fyers_model(new_access_token) # Re-initialize Fyers model with new access token
            app.logger.info("Access token refreshed successfully.")
            return True
        else:
            app.logger.error(f"Failed to refresh access token. Response: {response}")
            return False
    except Exception as e:
        app.logger.error(f"Error during access token refresh: {e}", exc_info=True)
        return False

# Initialize Fyers model at startup
initialize_fyers_model()

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
    Exchanges the auth_code for an access_token and refresh_token.
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
        
        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response["refresh_token"] # Get the refresh token here
            
            store_tokens(new_access_token, new_refresh_token) # Store both tokens
            initialize_fyers_model(new_access_token) # Re-initialize with the new access token
            
            app.logger.info("Fyers tokens generated successfully!")
            return jsonify({"message": "Fyers tokens generated successfully!", "access_token_available": True})
        else:
            app.logger.error(f"Failed to generate Fyers tokens. Response: {response}")
            return jsonify({"error": f"Failed to generate Fyers tokens. Response: {response}"}), 500
        
    except Exception as e:
        app.logger.error(f"Error generating Fyers access token: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate Fyers access token: {str(e)}"}), 500


# --- Fyers Data Endpoints with Token Refresh Logic ---

# Helper function to wrap Fyers API calls with refresh logic
def make_fyers_api_call(api_method, *args, **kwargs):
    global fyers_instance
    if not fyers_instance:
        app.logger.warning("Fyers API not initialized. Attempting to initialize.")
        if not initialize_fyers_model():
            # If initialization (including refresh) fails, we can't proceed
            return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401

    try:
        return api_method(*args, **kwargs)
    except Exception as e: # Catch a general exception
        error_message = str(e).lower()
        # Common patterns for token expiry errors in messages or within the exception object
        # You might need to inspect the 'e' object directly to see if it contains
        # a response dict or specific error codes from Fyers.
        # Example: if hasattr(e, 'response') and e.response.get('code') == -100:
        
        # Check for keywords that indicate token issues
        if "token" in error_message or "authenticated" in error_message or "login" in error_message or "invalid_access_token" in error_message:
            app.logger.warning(f"Access token expired or invalid. Attempting to refresh. Original error: {e}")
            if refresh_access_token():
                app.logger.info("Token refreshed, retrying original request.")
                # After successful refresh, fyers_instance is re-initialized with new token
                return api_method(*args, **kwargs) # Retry the call
            else:
                app.logger.error("Token refresh failed. Cannot fulfill request.")
                # Instead of FyersException, raise a generic Exception or return an error response
                return jsonify({"error": "Fyers API token expired and refresh failed. Please re-authenticate."}), 401
        else:
            # Not a token error, re-raise original exception or return error
            app.logger.error(f"Non-token related Fyers API error: {e}", exc_info=True)
            return jsonify({"error": f"Fyers API error: {str(e)}"}), 500

@app.route('/api/fyers/profile')
def get_profile():
    # make_fyers_api_call now returns a tuple (response, status_code) on error
    # or the actual data on success.
    result = make_fyers_api_call(fyers_instance.get_profile)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        # This is an error response from make_fyers_api_call
        return result
    return jsonify(result) # It's the actual data

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
    data = request.json
    
    required_params = ["symbol", "resolution", "date_format", "range_from", "range_to"]
    if not data or not all(k in data for k in required_params):
        app.logger.warning(f"Missing required parameters for history API. Received: {data}")
        return jsonify({"error": f"Missing required parameters for history API. Need {', '.join(required_params)}."}), 400
    
    try:
        data["date_format"] = int(data["date_format"])
    except ValueError:
        app.logger.warning(f"Invalid 'date_format' received: {data.get('date_format')}")
        return jsonify({"error": "Invalid 'date_format'. Must be 0 or 1."}), 400

    if data["date_format"] == 0:
        current_time = int(time.time())
        requested_range_to = int(data["range_to"])
        resolution = data["resolution"]

        resolution_in_seconds = 0
        if resolution.endswith('S'):
            try:
                resolution_in_seconds = int(resolution[:-1])
            except ValueError:
                app.logger.warning(f"Invalid numeric part in resolution ending with 'S': {resolution}")
                return jsonify({"error": "Invalid resolution format."}), 400
        elif resolution.isdigit():
            # This handles minute resolutions like "1", "5", "30", etc.
            resolution_in_seconds = int(resolution) * 60
        elif resolution in ["D", "1D"]:
            resolution_in_seconds = 24 * 60 * 60 
        else:
            app.logger.warning(f"Unsupported resolution format for partial candle adjustment: {resolution}")
            return jsonify({"error": "Unsupported resolution format."}), 400

        if resolution_in_seconds > 0:
            # Calculate the start epoch of the current *incomplete* candle
            current_resolution_start_epoch = (current_time // resolution_in_seconds) * resolution_in_seconds
            
            # If the requested `range_to` includes or goes past the start of the current incomplete candle,
            # adjust `range_to` to the end of the *last complete* candle.
            # This means setting it to one second before the start of the current incomplete candle.
            if requested_range_to >= current_resolution_start_epoch:
                adjusted_range_to_epoch = current_resolution_start_epoch - 1
                
                # Ensure the adjusted range_to is not before range_from
                if adjusted_range_to_epoch < int(data["range_from"]):
                    app.logger.info(f"Adjusted range_to ({adjusted_range_to_epoch}) is less than range_from ({data['range_from']}). No complete candles available for resolution {resolution} in this range after adjustment.")
                    return jsonify({"candles": [], "s": "ok", "message": "No complete candles available for the adjusted range."})

                data["range_to"] = str(adjusted_range_to_epoch)
                app.logger.info(f"Adjusted 'range_to' for resolution '{resolution}' to ensure completed candles: {requested_range_to} -> {data['range_to']}")
        
    if "cont_flag" in data:
        data["cont_flag"] = int(data["cont_flag"])
    if "oi_flag" in data:
        data["oi_flag"] = int(data["oi_flag"])

    result = make_fyers_api_call(fyers_instance.history, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/quotes', methods=['GET'])
def get_quotes():
    symbols = request.args.get('symbols')
    if not symbols:
        app.logger.warning("Missing 'symbols' parameter for quotes API.")
        return jsonify({"error": "Missing 'symbols' parameter. Eg: /api/fyers/quotes?symbols=NSE:SBIN-EQ,NSE:TCS-EQ"}), 400
    
    data = {"symbols": symbols}
    result = make_fyers_api_call(fyers_instance.quotes, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
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


@app.route('/api/fyers/place_order', methods=['POST'])
def place_single_order():
    order_data = request.json
    if not order_data:
        app.logger.warning("No order data provided for placing order.")
        return jsonify({"error": "No order data provided."}), 400
    
    result = make_fyers_api_call(fyers_instance.place_order, order_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)


@app.route('/')
def home():
    return "Fyers API Proxy Server is running! Use /fyers-login to authenticate or /api/fyers/news for news (placeholder)."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
