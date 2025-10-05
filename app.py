import os
from flask import Flask, request, jsonify, redirect, url_for
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
import time
import logging
import json
import threading

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

TOKENS_FILE = os.environ.get("FYERS_TOKENS_FILE", "tokens.json")  # optional persistent file example

if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
    app.logger.error("ERROR: Fyers API credentials (CLIENT_ID, SECRET_KEY, REDIRECT_URI) are not fully set. Please check your .env file or environment variables.")
    # You may want to halt startup in production, but keep server running for local dev:
    # raise SystemExit("Missing Fyers credentials")

# Initialize FyersModel (will be re-initialized with an access token after login or refresh)
fyers_instance = None

# --- Persistent Storage Helpers ---
def _save_tokens_to_file(access_token, refresh_token):
    try:
        with open(TOKENS_FILE, "w") as f:
            json.dump({"access_token": access_token, "refresh_token": refresh_token}, f)
        app.logger.info(f"Tokens persisted to {TOKENS_FILE}")
    except Exception as e:
        app.logger.warning(f"Failed to persist tokens to {TOKENS_FILE}: {e}")

def _load_tokens_from_file():
    global ACCESS_TOKEN, REFRESH_TOKEN
    if os.path.exists(TOKENS_FILE):
        try:
            with open(TOKENS_FILE, "r") as f:
                data = json.load(f)
            ACCESS_TOKEN = data.get("access_token", ACCESS_TOKEN)
            REFRESH_TOKEN = data.get("refresh_token", REFRESH_TOKEN)
            app.logger.info("Loaded tokens from tokens file.")
        except Exception as e:
            app.logger.warning(f"Failed to load tokens from file: {e}")

def store_tokens(access_token, refresh_token, persist=True):
    """
    Update global tokens and optionally persist to file.
    WARNING: Storing tokens in plain files is NOT recommended in production.
    Use secure stores like AWS Secrets Manager, Vault, or encrypted DB.
    """
    global ACCESS_TOKEN, REFRESH_TOKEN
    ACCESS_TOKEN = access_token
    REFRESH_TOKEN = refresh_token

    app.logger.info("Tokens updated in memory.")
    try:
        # Log only partial tokens to avoid full secret printing
        app.logger.info(f"New Access Token (prefix): {ACCESS_TOKEN[:10]}...")
        app.logger.info(f"New Refresh Token (prefix): {REFRESH_TOKEN[:10]}...")
    except Exception:
        # Defensive: token might be None or shorter
        pass

    if persist:
        _save_tokens_to_file(access_token, refresh_token)

def load_tokens():
    """
    Load tokens from persistent file if available. Environment variables still have priority
    and are read at module load; this function augments them if file exists.
    """
    _load_tokens_from_file()
    # If environment variables were set, they remain in ACCESS_TOKEN/REFRESH_TOKEN

# Load tokens at startup (file-based fallback)
load_tokens()

def initialize_fyers_model(token=None):
    """
    Initialize the fyers_model instance with the provided token or the global ACCESS_TOKEN.
    If no access token but REFRESH_TOKEN exists, try refreshing.
    Returns True if fyers_instance is available, False otherwise.
    """
    global fyers_instance, ACCESS_TOKEN, REFRESH_TOKEN

    token_to_use = token if token else ACCESS_TOKEN

    if token_to_use:
        try:
            fyers_instance = fyersModel.FyersModel(token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            app.logger.info("FyersModel initialized with access token.")
            return True
        except Exception as e:
            app.logger.error(f"Failed to initialize FyersModel with token: {e}", exc_info=True)
            fyers_instance = None
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
    """
    Use the SDK's SessionModel to refresh access token using REFRESH_TOKEN.
    Returns True on success and updates global tokens and fyers_instance.
    """
    global ACCESS_TOKEN, REFRESH_TOKEN
    if not REFRESH_TOKEN:
        app.logger.error("Cannot refresh token: No refresh token available.")
        return False

    try:
        session = fyersModel.SessionModel(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            response_type="code",
            state="refresh_state",
            secret_key=SECRET_KEY,
            grant_type="refresh_token"
        )
        session.set_token(REFRESH_TOKEN)  # Set the refresh token

        app.logger.info(f"Attempting to refresh access token using refresh token: {REFRESH_TOKEN[:8]}...")
        response = session.generate_token()

        # Validate response; adjust depending on SDK
        if response and response.get("s") == "ok":
            new_access_token = response.get("access_token")
            new_refresh_token = response.get("refresh_token", REFRESH_TOKEN)

            if not new_access_token:
                app.logger.error(f"Refresh response did not contain access_token. Response: {response}")
                return False

            store_tokens(new_access_token, new_refresh_token)
            # Re-init fyers model
            try:
                initialize_fyers_model(new_access_token)
            except Exception:
                app.logger.warning("FyersModel init failed after refresh; continuing but API calls may fail.")
            app.logger.info("Access token refreshed successfully.")
            return True
        else:
            app.logger.error(f"Failed to refresh access token. Response: {response}")
            return False
    except Exception as e:
        app.logger.error(f"Error during access token refresh: {e}", exc_info=True)
        return False

# Try initialize at startup using whatever tokens we have
initialize_fyers_model()

# --- Helper wrapper to call fyers methods with token refresh retry ---
def make_fyers_api_call(api_method, *args, **kwargs):
    """
    Calls an SDK method (bound function) and automatically attempts token refresh if
    the error indicates authentication/token expiry. Returns the SDK result or a Flask-style
    (response, status_code) tuple for errors to be returned upstream.
    """
    global fyers_instance
    if not fyers_instance:
        app.logger.warning("Fyers API not initialized. Attempting to initialize.")
        if not initialize_fyers_model():
            return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401

    try:
        # Some SDK methods expect named argument 'data' or none; caller supplies appropriate args/kwargs.
        result = api_method(*args, **kwargs)
        return result
    except Exception as e:
        error_message = str(e).lower()
        app.logger.warning(f"Fyers API call error: {e}")
        # Token/ auth detection heuristics
        if any(k in error_message for k in ("token", "authenticated", "login", "invalid_access_token", "401")):
            app.logger.warning(f"Access token expired or invalid. Attempting to refresh. Original error: {e}")
            if refresh_access_token():
                app.logger.info("Token refreshed, retrying original request.")
                try:
                    result = api_method(*args, **kwargs)
                    return result
                except Exception as e2:
                    app.logger.error(f"Retry after token refresh failed: {e2}", exc_info=True)
                    return jsonify({"error": f"Fyers API error after token refresh: {str(e2)}"}), 500
            else:
                app.logger.error("Token refresh failed. Cannot fulfill request.")
                return jsonify({"error": "Fyers API token expired and refresh failed. Please re-authenticate."}), 401
        else:
            app.logger.error(f"Non-token related Fyers API error: {e}", exc_info=True)
            return jsonify({"error": f"Fyers API error: {str(e)}"}), 500

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
        state="fyers_proxy_state",
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
            new_access_token = response.get("access_token")
            new_refresh_token = response.get("refresh_token")

            if not new_access_token or not new_refresh_token:
                app.logger.error(f"Token generation response missing token(s): {response}")
                return jsonify({"error": "Token response incomplete from Fyers."}), 500

            store_tokens(new_access_token, new_refresh_token)
            initialize_fyers_model(new_access_token)
            app.logger.info("Fyers tokens generated successfully!")
            return jsonify({"message": "Fyers tokens generated successfully!", "access_token_available": True})
        else:
            app.logger.error(f"Failed to generate Fyers tokens. Response: {response}")
            return jsonify({"error": f"Failed to generate Fyers tokens. Response: {response}"}), 500

    except Exception as e:
        app.logger.error(f"Error generating Fyers access token: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate Fyers access token: {str(e)}"}), 500


# --- Fyers Data Endpoints with Token Refresh Logic ---

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

# --- History endpoint that supports seconds-level resolution ---
@app.route('/api/fyers/history', methods=['POST'])
def get_history():
    """
    Expected JSON body:
    {
      "symbol": "NSE:RELIANCE-EQ",
      "resolution": "1S" or "5S" or "1" or "5" or "D",
      "date_format": 0 (epoch seconds) or 1 (yyyy-mm-dd),
      "range_from": <int or string>,
      "range_to": <int or string>,
      optional: "cont_flag", "oi_flag"
    }

    This endpoint ensures that when resolution is seconds-level (ends with 'S' or 's'),
    the requested range_to will be truncated to the last *complete* candle so that
    the Fyers history call returns only full candles (avoids partial candle at current time).
    """
    data = request.json

    required_params = ["symbol", "resolution", "date_format", "range_from", "range_to"]
    if not data or not all(k in data for k in required_params):
        app.logger.warning(f"Missing required parameters for history API. Received: {data}")
        return jsonify({"error": f"Missing required parameters for history API. Need {', '.join(required_params)}."}), 400

    # Normalize types
    try:
        date_format = int(data["date_format"])
    except Exception:
        app.logger.warning(f"Invalid 'date_format' received: {data.get('date_format')}")
        return jsonify({"error": "Invalid 'date_format'. Must be 0 or 1."}), 400

    resolution = str(data["resolution"]).strip()
    # Allow numeric minute strings as well (e.g., "1", "5") and 'S' suffix for seconds ('1S','5S')
    # Also allow lowercase 's'
    try:
        range_from = int(str(data["range_from"]))
        range_to = int(str(data["range_to"]))
    except Exception:
        app.logger.warning(f"Invalid 'range_from' or 'range_to' values: {data.get('range_from')}, {data.get('range_to')}")
        return jsonify({"error": "'range_from' and 'range_to' must be epoch integers when date_format==0."}), 400

    # Only adjust partial-candle logic when date_format==0 (epoch seconds)
    if date_format == 0:
        current_time = int(time.time())
        resolution_in_seconds = None

        # Recognize seconds resolution: e.g., '1S', '5S', '30S'
        if resolution.lower().endswith('s'):
            # strip the last char and parse the numeric part
            numeric_part = resolution[:-1]
            try:
                resolution_in_seconds = int(numeric_part)
            except ValueError:
                app.logger.warning(f"Invalid numeric part in seconds resolution: {resolution}")
                return jsonify({"error": "Invalid seconds resolution format. Use e.g. '1S', '5S', '15S'."}), 400
        elif resolution.isdigit():
            # resolution in minutes, convert to seconds
            resolution_in_seconds = int(resolution) * 60
        elif resolution in ["D", "1D", "d", "1d"]:
            resolution_in_seconds = 24 * 60 * 60
        else:
            # Some SDKs accept formats like "1m", "5m" — handle potential 'm' suffix
            if resolution.lower().endswith('m'):
                try:
                    resolution_in_seconds = int(resolution[:-1]) * 60
                except ValueError:
                    app.logger.warning(f"Invalid minute resolution format: {resolution}")
                    return jsonify({"error": "Invalid minute resolution format. Use e.g. '1', '5', '1m', or seconds '1S'."}), 400
            else:
                app.logger.warning(f"Unsupported resolution format for partial candle adjustment: {resolution}")
                return jsonify({"error": "Unsupported resolution format."}), 400

        # Safety check
        if resolution_in_seconds and resolution_in_seconds > 0:
            # Calculate the epoch start of the current incomplete candle
            current_resolution_start_epoch = (current_time // resolution_in_seconds) * resolution_in_seconds

            # If the requested range_to includes or goes past the start of the current incomplete candle,
            # adjust `range_to` to the end of the last *complete* candle (one second before current candle start).
            if range_to >= current_resolution_start_epoch:
                adjusted_range_to_epoch = current_resolution_start_epoch - 1

                # If adjustment makes the range invalid, return "no complete candles"
                if adjusted_range_to_epoch < range_from:
                    app.logger.info(f"Adjusted range_to ({adjusted_range_to_epoch}) is less than range_from ({range_from}). No complete candles available for resolution {resolution} in this range after adjustment.")
                    return jsonify({"candles": [], "s": "ok", "message": "No complete candles available for the adjusted range."})
                # Update the request payload
                data["range_to"] = str(adjusted_range_to_epoch)
                app.logger.info(f"Adjusted 'range_to' for resolution '{resolution}' to ensure completed candles: {range_to} -> {data['range_to']}")

    # Convert optional flags to ints if present
    if "cont_flag" in data:
        try:
            data["cont_flag"] = int(data["cont_flag"])
        except ValueError:
            app.logger.warning(f"Invalid cont_flag received: {data.get('cont_flag')}")
            return jsonify({"error": "Invalid 'cont_flag'. Must be 0 or 1."}), 400

    if "oi_flag" in data:
        try:
            data["oi_flag"] = int(data["oi_flag"])
        except ValueError:
            app.logger.warning(f"Invalid oi_flag received: {data.get('oi_flag')}")
            return jsonify({"error": "Invalid 'oi_flag'. Must be 0 or 1."}), 400

    # Finally call the SDK's history method
    result = make_fyers_api_call(fyers_instance.history, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

# --- Quotes / Depth / Option Chain endpoints ---

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


# --- News placeholder ---

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


# --- Order Placement and Modification APIs ---

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


# --- Broker Config APIs ---

@app.route('/api/fyers/market_status', methods=['GET'])
def get_market_status():
    result = make_fyers_api_call(fyers_instance.market_status)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)


@app.route('/')
def home():
    return "Fyers API Proxy Server is running! Use /fyers-login to authenticate or /api/fyers/news for news (placeholder)."


if __name__ == '__main__':
    # Optionally run a background thread to periodically refresh token (if refresh token expiry is long-lived),
    # or to save tokens periodically. This is commented out by default.
    #
    # def periodic_refresh_task(interval_seconds=60*60):
    #     while True:
    #         time.sleep(interval_seconds)
    #         try:
    #             if REFRESH_TOKEN:
    #                 refresh_access_token()
    #         except Exception as e:
    #             app.logger.warning(f"Periodic refresh failed: {e}")
    #
    # threading.Thread(target=periodic_refresh_task, daemon=True).start()

    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
