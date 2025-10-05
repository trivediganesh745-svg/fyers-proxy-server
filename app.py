import os
from flask import Flask, request, jsonify, redirect, url_for
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
import time
import logging
import json
import math

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

if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
    app.logger.error("ERROR: Fyers API credentials (CLIENT_ID, SECRET_KEY, REDIRECT_URI) are not fully set. Please check your .env file or environment variables.")
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

    # It's generally not safe to log full tokens. We log truncated versions for debugging.
    app.logger.info("Tokens updated. For persistence, save these securely.")
    try:
        app.logger.info(f"New Access Token: {ACCESS_TOKEN[:10]}...") if ACCESS_TOKEN else None
        app.logger.info(f"New Refresh Token: {REFRESH_TOKEN[:10]}...") if REFRESH_TOKEN else None
    except Exception:
        # Defensive: in case tokens are None or not indexable
        app.logger.info("Tokens updated (could not display snippet).")

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
    """
    Initialize (or re-init) fyers_instance. If token is provided, use it,
    otherwise use global ACCESS_TOKEN, or attempt refresh via REFRESH_TOKEN.
    Returns True if initialized, False otherwise.
    """
    global fyers_instance, ACCESS_TOKEN, REFRESH_TOKEN

    # Prioritize provided token, then global ACCESS_TOKEN
    token_to_use = token if token else ACCESS_TOKEN

    if token_to_use:
        try:
            fyers_instance = fyersModel.FyersModel(token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            app.logger.info("FyersModel initialized with access token.")
            return True
        except Exception as e:
            app.logger.error(f"Failed to initialize FyersModel with token: {e}", exc_info=True)
            fyers_instance = None
            # attempt refresh if refresh token exists
            if REFRESH_TOKEN:
                return refresh_access_token()
            return False
    elif REFRESH_TOKEN and CLIENT_ID and SECRET_KEY:
        app.logger.info("No access token provided or found, attempting to use refresh token.")
        # Attempt to refresh using the refresh token
        return refresh_access_token()
    else:
        app.logger.warning("FyersModel could not be initialized: No access token or refresh token available.")
        return False

def refresh_access_token():
    """
    Refresh access token using the fyers SessionModel with grant_type=refresh_token.
    Returns True if refresh succeeded and fyers_instance re-initialized.
    """
    global ACCESS_TOKEN, REFRESH_TOKEN
    if not REFRESH_TOKEN:
        app.logger.error("Cannot refresh token: No refresh token available.")
        return False

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",  # Even for refresh token, this is part of SessionModel init
        state="refresh_state",
        secret_key=SECRET_KEY,
        grant_type="refresh_token"  # This is the key for refreshing
    )
    session.set_token(REFRESH_TOKEN)  # Set the refresh token

    try:
        app.logger.info(f"Attempting to refresh access token using refresh token: {REFRESH_TOKEN[:5]}...")
        response = session.generate_token()

        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response.get("refresh_token", REFRESH_TOKEN)  # Fyers might issue new refresh token, or keep old

            store_tokens(new_access_token, new_refresh_token)
            initialize_fyers_model(new_access_token)  # Re-initialize Fyers model with new access token
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

# --- Helper: resolution parsing and validation ---
# Acceptable resolution formats:
#  - Seconds: "1S", "5S", "10S", "15S", "30S", etc. (case-insensitive)
#  - Minutes: "1", "5", "15", "30", "60" (these are minute counts represented as digits)
#  - Days / D: "D", "1D" (daily)
#  - Weeks: "W", "1W" (weekly)
#  - Months: "M", "1M" (monthly)
#
# This function converts a given resolution string to seconds for candle-completeness logic.
SUPPORTED_SECOND_RESOLUTIONS = {1, 2, 3, 4, 5, 10, 15, 20, 30}  # commonly used second resolutions
SUPPORTED_MINUTE_RESOLUTIONS = {1, 2, 3, 5, 10, 15, 30, 60}  # minute resolutions
# We'll accept weekly and monthly for high-level groupings
def parse_resolution_to_seconds(resolution):
    """
    Parse resolution string and return resolution length in seconds.
    Returns integer seconds if recognized, otherwise None.
    Accept examples: "5S", "5s", "1", "5", "15", "D", "1D", "W", "M", "1W", "1M"
    """
    if not resolution or not isinstance(resolution, str):
        return None

    r = resolution.strip().upper()

    # Seconds ending with 'S' (e.g., '5S', '1S')
    if r.endswith("S"):
        num_part = r[:-1]
        try:
            seconds = int(num_part)
            if seconds <= 0:
                return None
            # Optionally restrict to supported seconds; allow others but warn
            if seconds not in SUPPORTED_SECOND_RESOLUTIONS:
                app.logger.debug(f"Resolution {seconds} seconds is not in the common list but will be accepted.")
            return seconds
        except ValueError:
            return None

    # Days / Weekly / Monthly
    if r in ("D", "1D"):
        return 24 * 60 * 60
    if r in ("W", "1W"):
        return 7 * 24 * 60 * 60
    if r in ("M", "1M"):
        # For monthly we use 30 days as a reasonable approximation for candle boundary calculations
        return 30 * 24 * 60 * 60

    # Minute-based numeric strings (e.g., "1", "5", "15")
    # Also handle hours provided as "60" (which is minutes)
    if r.isdigit():
        try:
            minutes = int(r)
            if minutes <= 0:
                return None
            # Convert minutes to seconds
            seconds = minutes * 60
            # Optionally validate common minute buckets
            if minutes not in SUPPORTED_MINUTE_RESOLUTIONS and minutes < 1440:
                app.logger.debug(f"Minute resolution {minutes} is not in common set but will be accepted.")
            return seconds
        except ValueError:
            return None

    # If we reach here, unknown format
    return None

def is_resolution_supported(resolution):
    """
    Returns True if the resolution string is supported for history calls.
    We accept second-level, minute-level, daily, weekly, monthly.
    """
    return parse_resolution_to_seconds(resolution) is not None

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
        state="fyers_proxy_state",  # Use a unique state for security if needed
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
            new_refresh_token = response["refresh_token"]  # Get the refresh token here

            store_tokens(new_access_token, new_refresh_token)  # Store both tokens
            initialize_fyers_model(new_access_token)  # Re-initialize with the new access token

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
    except Exception as e:  # Catch a general exception
        error_message = str(e).lower()
        # Check for keywords that indicate token issues
        if any(k in error_message for k in ("token", "authenticated", "login", "invalid_access_token", "access token")):
            app.logger.warning(f"Access token expired or invalid. Attempting to refresh. Original error: {e}")
            if refresh_access_token():
                app.logger.info("Token refreshed, retrying original request.")
                # After successful refresh, fyers_instance is re-initialized with new token
                try:
                    return api_method(*args, **kwargs)  # Retry the call
                except Exception as ex2:
                    app.logger.error(f"Retry after refresh failed: {ex2}", exc_info=True)
                    return jsonify({"error": f"Fyers API error after token refresh: {str(ex2)}"}), 500
            else:
                app.logger.error("Token refresh failed. Cannot fulfill request.")
                return jsonify({"error": "Fyers API token expired and refresh failed. Please re-authenticate."}), 401
        else:
            app.logger.error(f"Non-token related Fyers API error: {e}", exc_info=True)
            return jsonify({"error": f"Fyers API error: {str(e)}"}), 500

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

@app.route('/api/fyers/history', methods=['POST'])
def get_history():
    """
    Body JSON required keys:
      - symbol (string, e.g. "NSE:RELIANCE-EQ")
      - resolution (string, e.g. "5S", "1", "5", "15", "D", "W", "M")
      - date_format (0 for epoch seconds, 1 for yyyy-mm-dd)
      - range_from (int/string) (if date_format=0, epoch seconds; else date string)
      - range_to (int/string) (if date_format=0, epoch seconds; else date string)
      Optional:
      - cont_flag (0/1)
      - oi_flag (0/1)
    This endpoint now supports second-level candles like '5S' down to '1S'.
    """
    data = request.json

    required_params = ["symbol", "resolution", "date_format", "range_from", "range_to"]
    if not data or not all(k in data for k in required_params):
        app.logger.warning(f"Missing required parameters for history API. Received: {data}")
        return jsonify({"error": f"Missing required parameters for history API. Need {', '.join(required_params)}."}), 400

    # Normalize resolution to string
    resolution = str(data.get("resolution", "")).strip()
    if not is_resolution_supported(resolution):
        app.logger.warning(f"Unsupported or invalid resolution received: {resolution}")
        return jsonify({"error": f"Unsupported resolution '{resolution}'. Supported examples: '1S','5S','10S','1','5','15','30','60','D','W','M'."}), 400

    # date_format: 0 -> epoch seconds; 1 -> yyyy-mm-dd
    try:
        data["date_format"] = int(data["date_format"])
    except Exception:
        app.logger.warning(f"Invalid 'date_format' received: {data.get('date_format')}")
        return jsonify({"error": "Invalid 'date_format'. Must be 0 (epoch seconds) or 1 (yyyy-mm-dd)."}), 400

    # If date_format is epoch (0) we will apply candle completeness adjustment
    if data["date_format"] == 0:
        try:
            current_time = int(time.time())
            requested_range_to = int(data["range_to"])
            requested_range_from = int(data["range_from"])
        except Exception:
            app.logger.warning("When using date_format=0, 'range_from' and 'range_to' should be epoch seconds integers.")
            return jsonify({"error": "When using date_format=0, 'range_from' and 'range_to' must be epoch seconds (integers)."}), 400

        # Get resolution length in seconds
        resolution_in_seconds = parse_resolution_to_seconds(resolution)
        if resolution_in_seconds is None:
            # This should not happen because we validated earlier, but keep defensive
            app.logger.error(f"Could not parse resolution to seconds unexpectedly: {resolution}")
            return jsonify({"error": "Unexpected resolution parsing failure."}), 400

        # For resolutions that are large (weekly/monthly) we still calculate boundaries.
        # Calculate the start epoch of the current *incomplete* candle:
        if resolution_in_seconds >= (24 * 60 * 60):
            # For daily+ resolutions, align to UTC day boundaries (00:00 UTC)
            # For weekly align to Monday 00:00 UTC, and for monthly align to first of month 00:00 UTC.
            # We'll approximate by calculating the last full multiple of the resolution length.
            current_resolution_start_epoch = (current_time // resolution_in_seconds) * resolution_in_seconds
        else:
            # For seconds/minutes: compute by simple integer division
            current_resolution_start_epoch = (current_time // resolution_in_seconds) * resolution_in_seconds

        # If the requested `range_to` includes or goes past the start of the current incomplete candle,
        # adjust `range_to` to the end of the *last complete* candle (one second before current_resolution_start_epoch).
        if requested_range_to >= current_resolution_start_epoch:
            adjusted_range_to_epoch = current_resolution_start_epoch - 1

            # Ensure the adjusted range_to is not before range_from
            if adjusted_range_to_epoch < requested_range_from:
                app.logger.info(f"Adjusted range_to ({adjusted_range_to_epoch}) is less than range_from ({requested_range_from}). No complete candles available for resolution {resolution} in this range after adjustment.")
                return jsonify({"candles": [], "s": "ok", "message": "No complete candles available for the adjusted range."})

            # Set the adjusted value back into the data payload as a string (the SDK often expects string values)
            data["range_to"] = str(adjusted_range_to_epoch)
            app.logger.info(f"Adjusted 'range_to' for resolution '{resolution}' to ensure completed candles: {requested_range_to} -> {data['range_to']}")
        else:
            # No adjustment needed
            app.logger.debug("No candle completeness adjustment needed for requested range_to.")

    # If optional flags are present, ensure they're correct types
    if "cont_flag" in data:
        try:
            data["cont_flag"] = int(data["cont_flag"])
        except Exception:
            app.logger.warning(f"Invalid cont_flag provided: {data.get('cont_flag')}")
            return jsonify({"error": "Invalid 'cont_flag'. Must be 0 or 1."}), 400
    if "oi_flag" in data:
        try:
            data["oi_flag"] = int(data["oi_flag"])
        except Exception:
            app.logger.warning(f"Invalid oi_flag provided: {data.get('oi_flag')}")
            return jsonify({"error": "Invalid 'oi_flag'. Must be 0 or 1."}), 400

    # Finally call the fyers history API via the wrapper (with token refresh handling)
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


# --- New Order Placement and Modification APIs ---

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

    # The fyersModel.multi_order method takes a list of order objects directly
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

    # The SDK's modify_gttorder expects the ID and then the modification data
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

    # The SDK's modify_order expects the modification data directly
    result = make_fyers_api_call(fyers_instance.modify_order, data=modify_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['PATCH'])
def modify_multi_orders():
    modify_basket_data = request.json
    if not modify_basket_data or not isinstance(modify_basket_data, list):
        return jsonify({"error": "An array of order modification objects is required."}), 400

    # The SDK's modify_basket_orders expects a list of order modification objects
    result = make_fyers_api_call(fyers_instance.modify_basket_orders, data=modify_basket_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/order', methods=['DELETE'])
def cancel_single_order():
    cancel_data = request.json
    if not cancel_data or not cancel_data.get("id"):
        return jsonify({"error": "Order ID is required for cancellation."}), 400

    # The SDK's cancel_order expects the cancellation data directly (which contains 'id')
    result = make_fyers_api_call(fyers_instance.cancel_order, data=cancel_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/orders/multi', methods=['DELETE'])
def cancel_multi_orders():
    cancel_basket_data = request.json
    if not cancel_basket_data or not isinstance(cancel_basket_data, list):
        return jsonify({"error": "An array of order cancellation objects (with 'id') is required."}), 400

    # The SDK's cancel_basket_orders expects a list of cancellation objects
    result = make_fyers_api_call(fyers_instance.cancel_basket_orders, data=cancel_basket_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/positions', methods=['DELETE'])
def exit_positions():
    exit_data = request.json
    if not exit_data:
        return jsonify({"error": "Request body for exiting positions is required."}), 400

    # The positions API for delete can take different parameters based on intent
    # This endpoint will be flexible to handle 'exit_all', 'id', or segment/side/productType filters
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

    # The SDK's span_margin expects 'data' as the key in the payload
    result = make_fyers_api_call(fyers_instance.span_margin, data=margin_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@app.route('/api/fyers/margin/multiorder', methods=['POST'])
def multiorder_margin_calculator():
    margin_data = request.json
    if not margin_data or not margin_data.get("data"):
        return jsonify({"error": "An array of order details for multiorder margin calculation is required under 'data' key."}), 400

    # The SDK's multiorder_margin expects 'data' as the key in the payload
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
    # Use host=0.0.0.0 and port 5000 as default
    app.run(host='0.0.0.0', port=5000)
