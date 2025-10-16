import os
from flask import Flask, request, jsonify, redirect, url_for
from fyers_apiv3 import fyersModel
# New import for the Fyers websocket client
from fyers_apiv3.FyersWebsocket import data_ws
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
import time
import logging
import json
import threading
from backtesting.routes import backtesting_bp

# New imports for Flask WebSocket support
from flask_sock import Sock
from typing import List
import math

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Sock for client-facing websocket connections
sock = Sock(app)

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

# Add a base URL for your proxy for internal calls (important for backtesting data fetching)
# For local development: "http://localhost:5000"
# For Render deployment: Your Render service URL (e.g., "https://your-fyers-proxy.onrender.com")
app.config["FYERS_PROXY_BASE_URL"] = os.environ.get("FYERS_PROXY_BASE_URL", "http://localhost:5000")


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
    
    app.logger.info("Tokens updated. For persistence, save these securely:")
    try:
        app.logger.info(f"New Access Token: {ACCESS_TOKEN[:10]}...")
    except Exception:
        app.logger.info("New Access Token: (hidden)")
    try:
        app.logger.info(f"New Refresh Token: {REFRESH_TOKEN[:10]}...")
    except Exception:
        app.logger.info("New Refresh Token: (hidden)")
    
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
        # Note: fyersModel.FyersModel expects 'token' arg name in some versions
        try:
            # keep backward compatibility: some versions use token= or access_token=
            try:
                fyers_instance = fyersModel.FyersModel(token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            except TypeError:
                fyers_instance = fyersModel.FyersModel(access_token=token_to_use, is_async=False, client_id=CLIENT_ID, log_path="")
            app.logger.info("FyersModel initialized with access token.")
            return True
        except Exception as e:
            app.logger.error(f"Failed to initialize fyers_model with provided token: {e}", exc_info=True)
            return False
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
        
        # Check for keywords that indicate token issues
        if "token" in error_message or "authenticated" in error_message or "login" in error_message or "invalid_access_token" in error_message:
            app.logger.warning(f"Access token expired or invalid. Attempting to refresh. Original error: {e}")
            if refresh_access_token():
                app.logger.info("Token refreshed, retrying original request.")
                # After successful refresh, fyers_instance is re-initialized with new token
                return api_method(*args, **kwargs) # Retry the call
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

# ---------------------------
# BEGIN: New helper + endpoints
# ---------------------------

def _find_option_in_chain(resp, option_symbol=None, strike=None, opt_type=None, expiry_ts=None):
    """
    Helper to search option chain response for the requested option and return the node.
    Accepts either full `option_symbol` (like "NSE:NIFTY25DEC17500CE") OR strike/opt_type/expiry_ts.
    Returns the option dict or None.
    """
    if not resp or not isinstance(resp, dict):
        return None

    # common places SDK returns chain: data -> optionChain or data -> filtered
    data = resp.get("data") if isinstance(resp, dict) else None
    # Try multiple common shapes
    candidates = []
    if data is None and "options_chain" in resp:
        data = resp

    # Flatten common structures
    if isinstance(data, dict):
        # some SDKs return data['option_chain'] or data['options']
        for key in ["optionChain", "option_chain", "optionsChain", "options", "ce", "pe"]:
            node = data.get(key)
            if node:
                # node could be list or dict
                if isinstance(node, list):
                    candidates.extend(node)
                elif isinstance(node, dict):
                    # dict of strikes
                    for v in node.values():
                        if isinstance(v, list):
                            candidates.extend(v)
                        else:
                            candidates.append(v)

    # fallback: top-level 'data' might be list of strikes
    if not candidates and isinstance(resp.get("data"), list):
        candidates.extend(resp.get("data"))

    # final fallback: resp itself is a list
    if not candidates and isinstance(resp, list):
        candidates.extend(resp)

    # normalize and search
    for item in candidates:
        try:
            symbol = item.get("symbol") or item.get("s") or item.get("name")
        except Exception:
            symbol = None
        if option_symbol and symbol and option_symbol == symbol:
            return item
        # match by strike/type
        try:
            strike_val = item.get("strike") or item.get("strike_price") or item.get("strikePrice")
            typ = item.get("option_type") or item.get("type") or item.get("opt_type") or item.get("instrument_type")
            expiry = item.get("expiry") or item.get("expiry_date") or item.get("expiry_ts")
        except Exception:
            strike_val = None
            typ = None
            expiry = None

        if strike is not None and strike_val is not None:
            # numeric compare
            try:
                if int(float(strike_val)) == int(float(strike)):
                    if not opt_type or (typ and opt_type.upper() in str(typ).upper()):
                        return item
            except Exception:
                pass

        if expiry_ts and expiry:
            try:
                if int(expiry_ts) == int(expiry):
                    if not strike or (strike_val and int(float(strike_val)) == int(float(strike))):
                        return item
            except Exception:
                pass

    return None


@app.route('/api/fyers/option_premium', methods=['GET'])
def get_option_premium():
    """
    Returns option premium (LTP), IV, OI, change, and a small parsed payload for a single option contract.
    Query params (one of the following ways):
      - symbol=<FULL_OPTION_SYMBOL>  (e.g. NSE:NIFTY25DEC17500CE)
      - underlying=<UNDERLYING_SYMBOL>&strike=<STRIKE>&type=<CE|PE>&expiry_ts=<EPOCH>

    Example:
      GET /api/fyers/option_premium?symbol=NSE:NIFTY25DEC17500CE
      GET /api/fyers/option_premium?underlying=NSE:NIFTY50-INDEX&strike=17500&type=CE
    """
    symbol = request.args.get('symbol')
    underlying = request.args.get('underlying')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')  # CE or PE
    expiry_ts = request.args.get('expiry_ts')

    # Prefer symbol if provided
    if not symbol and not (underlying and strike and opt_type):
        return jsonify({"error": "Provide either symbol=<OPTION_SYMBOL> OR underlying=<SYM>&strike=<STRIKE>&type=<CE|PE>"}), 400

    try:
        # If full option symbol given, try depth call first (g faster for single symbol)
        if symbol:
            # Try market depth first (gives LTP, bids/asks, oi)
            depth_resp = make_fyers_api_call(fyers_instance.depth, data={"symbol": symbol, "ohlcv_flag": 1})
            # if depth_resp looks like a flask tuple error, return it
            if isinstance(depth_resp, tuple) and len(depth_resp) == 2 and isinstance(depth_resp[1], int):
                return depth_resp

            # depth_resp usually contains 'data' or direct fields
            premium = None
            try:
                # Many depth responses include 'ltp' or 'last_price' in data
                d = depth_resp.get('data') if isinstance(depth_resp, dict) else None
                if d:
                    premium = d.get('ltp') or d.get('last_price') or d.get('close')
                if not premium and isinstance(depth_resp, dict):
                    premium = depth_resp.get('ltp') or depth_resp.get('last_price')
            except Exception:
                premium = None

            # fallback: fetch option chain for symbol
            if premium is None:
                oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": symbol, "strikecount": 1})
                if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
                    return oc_resp
                found = _find_option_in_chain(oc_resp, option_symbol=symbol)
                node = found or (oc_resp if isinstance(oc_resp, dict) else None)
            else:
                node = depth_resp if isinstance(depth_resp, dict) else None

        else:
            # Build the approximate option symbol if SDK requires underlying symbol style
            # Some SDKs accept underlying in optionchain calls and return full data
            oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying, "strikecount": 50})
            if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
                return oc_resp
            # Try to find by strike & type
            found = _find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
            node = found

        if not node:
            return jsonify({"error": "Option contract not found in Fyers response.", "symbol": symbol or f"{underlying}:{strike}{opt_type if opt_type else ''}"}), 404

        # Extract common fields safely
        ltp = node.get('ltp') or node.get('last_price') or node.get('close')
        oi = node.get('oi') or node.get('open_interest') or node.get('openInterest')
        iv = node.get('iv') or node.get('implied_volatility') or node.get('impliedVol')
        change = node.get('change') or node.get('price_change') or node.get('p_change')
        bid = node.get('bid') or node.get('best_bid')
        ask = node.get('ask') or node.get('best_ask')

        response = {
            "symbol": symbol or node.get('symbol'),
            "ltp": ltp,
            "premium": ltp,
            "iv": iv,
            "oi": oi,
            "change": change,
            "bid": bid,
            "ask": ask,
            "raw": node
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error in option_premium endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/fyers/option_chain_depth', methods=['GET'])
def get_option_chain_depth():
    """
    Returns detailed market depth for a chosen option contract. Accepts either:
      - symbol=<FULL_OPTION_SYMBOL>
      - underlying & strike & type (CE/PE) & optional expiry_ts

    This endpoint will call the SDK depth API and also include a small context with the option chain around the strike.
    """
    symbol = request.args.get('symbol')
    underlying = request.args.get('underlying')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')
    expiry_ts = request.args.get('expiry_ts')

    try:
        if symbol:
            depth_resp = make_fyers_api_call(fyers_instance.depth, data={"symbol": symbol, "ohlcv_flag": 1})
            if isinstance(depth_resp, tuple) and len(depth_resp) == 2 and isinstance(depth_resp[1], int):
                return depth_resp
            # Optionally also fetch the small option chain context (nearby strikes)
            # Try derive underlying by splitting symbol if possible
            try:
                underlying_guess = symbol.split(":")[0] if ":" in symbol else None
            except Exception:
                underlying_guess = None

            oc_resp = None
            try:
                if underlying_guess:
                    oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying_guess, "strikecount": 10})
            except Exception:
                oc_resp = None

            return jsonify({"depth": depth_resp, "option_chain_context": oc_resp})

        # else try to locate the option using optionchain
        if not (underlying and strike and opt_type):
            return jsonify({"error": "Provide either symbol=<OPTION_SYMBOL> OR underlying & strike & type parameters."}), 400

        oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying, "strikecount": 50})
        if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
            return oc_resp

        found = _find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
        if not found:
            return jsonify({"error": "Option strike not found in option chain response.", "underlying": underlying, "strike": strike, "type": opt_type}), 404

        symbol_to_query = found.get('symbol') or request.args.get('symbol')
        if not symbol_to_query:
            return jsonify({"error": "Could not determine option symbol to query depth for."}), 500

        depth_resp = make_fyers_api_call(fyers_instance.depth, data={"symbol": symbol_to_query, "ohlcv_flag": 1})
        if isinstance(depth_resp, tuple) and len(depth_resp) == 2 and isinstance(depth_resp[1], int):
            return depth_resp

        return jsonify({"depth": depth_resp, "option_chain_context": oc_resp})

    except Exception as e:
        app.logger.error(f"Error in option_chain_depth endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------------------------
# END: New helper + endpoints
# ---------------------------


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


# -----------------------------
# WebSocket Integration Section
# -----------------------------

# GLOBALS for data socket management
_fyers_data_socket = None
_fyers_socket_thread = None
_connected_clients = []  # stores ws objects for broadcasting
_subscribed_symbols = set()
_socket_lock = threading.Lock()
_socket_running = False

def _create_data_socket(access_token, lite_mode=False, write_to_file=False, reconnect=True, reconnect_retry=10):
    """
    Create a FyersDataSocket instance with callbacks wired to broadcast incoming messages
    to connected frontend websocket clients.
    """
    global _fyers_data_socket

    def _on_connect():
        app.logger.info("Fyers DataSocket connected.")
        # If we already had subscriptions, re-subscribe
        if _subscribed_symbols:
            try:
                _fyers_data_socket.subscribe(symbols=list(_subscribed_symbols), data_type="SymbolUpdate")
                app.logger.info(f"Re-subscribed to symbols on connect: {_subscribed_symbols}")
            except Exception as e:
                app.logger.error(f"Error re-subscribing on connect: {e}", exc_info=True)

    def _on_message(message):
        """
        Called for incoming socket messages from Fyers. Broadcast to all connected clients.
        Message is expected to be a dict (SDK provides dict) or JSON string.
        """
        try:
            # Ensure it's JSON serializable
            if isinstance(message, str):
                payload = message
            else:
                payload = json.dumps(message, default=str)
        except Exception:
            payload = str(message)

        # Broadcast to connected frontend websocket clients
        with _socket_lock:
            to_remove = []
            for client in _connected_clients:
                try:
                    client.send(payload)
                except Exception as e:
                    app.logger.debug(f"Failed to send to one client: {e}")
                    # mark for removal
                    to_remove.append(client)
            # remove disconnected clients
            for r in to_remove:
                try:
                    _connected_clients.remove(r)
                except ValueError:
                    pass

    def _on_error(error):
        app.logger.error(f"Fyers DataSocket error: {error}")

    def _on_close(close_msg):
        app.logger.info(f"Fyers DataSocket closed: {close_msg}")

    # Instantiate the data socket using SDK's data_ws.FyersDataSocket
    try:
        _fyers_data_socket = data_ws.FyersDataSocket(
            access_token=access_token,
            log_path="",
            litemode=lite_mode,
            write_to_file=write_to_file,
            reconnect=reconnect,
            on_connect=_on_connect,
            on_close=_on_close,
            on_error=_on_error,
            on_message=_on_message,
            reconnect_retry=reconnect_retry
        )
        return _fyers_data_socket
    except Exception as e:
        app.logger.error(f"Failed to create FyersDataSocket: {e}", exc_info=True)
        return None

def _start_data_socket_in_thread(access_token):
    """
    Starts the Fyers DataSocket in a dedicated daemon thread if not already running.
    """
    global _fyers_socket_thread, _socket_running, _fyers_data_socket

    with _socket_lock:
        if _socket_running:
            app.logger.info("Fyers DataSocket already running; skip start.")
            return True

        # create socket instance
        _fyers_data_socket = _create_data_socket(access_token)
        if not _fyers_data_socket:
            app.logger.error("Could not create FyersDataSocket instance.")
            return False

        def _run():
            global _socket_running
            try:
                _socket_running = True
                app.logger.info("Starting Fyers DataSocket.connect() (blocking call in thread).")
                # This will block while socket is running; SDK will handle reconnects internally
                _fyers_data_socket.connect()
            except Exception as e:
                app.logger.error(f"Fyers DataSocket thread crashed: {e}", exc_info=True)
            finally:
                _socket_running = False
                app.logger.info("Fyers DataSocket thread stopped.")

        _fyers_socket_thread = threading.Thread(target=_run, daemon=True, name="FyersDataSocketThread")
        _fyers_socket_thread.start()
        app.logger.info("Fyers DataSocket thread started.")
        return True

def _ensure_data_socket_running():
    """
    Ensures the socket is running; if not, try to start it with current ACCESS_TOKEN.
    """
    global ACCESS_TOKEN
    if not ACCESS_TOKEN:
        app.logger.warning("ACCESS_TOKEN is not available; cannot start data socket.")
        return False
    return _start_data_socket_in_thread(ACCESS_TOKEN)

# Client-facing websocket route
@sock.route('/ws/fyers')
def ws_fyers(ws):
    """
    Frontend connects here: wss://<host>/ws/fyers
    Supported incoming JSON messages from frontend:
      - {"action": "subscribe", "symbols": ["NSE:SBIN-EQ","NSE:TCS-EQ"], "data_type":"SymbolUpdate"}
      - {"action": "unsubscribe", "symbols": ["NSE:SBIN-EQ"]}
      - {"action": "mode", "lite": true}  # switch to lite mode (ltp only)
      - {"action": "status"}  # asks server to reply with socket status
    The server will broadcast any incoming Fyers socket messages to all connected clients.
    """
    global _connected_clients, _subscribed_symbols, _fyers_data_socket, ACCESS_TOKEN

    # Register client
    with _socket_lock:
        _connected_clients.append(ws)
    app.logger.info(f"Frontend websocket connected. Total clients: {len(_connected_clients)}")

    try:
        # Ensure the server-side Fyers socket is running
        if not _socket_running:
            started = _ensure_data_socket_running()
            if not started:
                err = {"error": "Fyers DataSocket could not be started. Ensure ACCESS_TOKEN is set and valid."}
                try:
                    ws.send(json.dumps(err))
                except Exception:
                    pass
                ws.close()
                return

        # handle client messages
        while True:
            # If client disconnected, break (receive returns None)
            msg = ws.receive()
            if msg is None:
                app.logger.info("Frontend websocket disconnected (receive returned None).")
                break

            try:
                data = json.loads(msg)
            except Exception:
                # Not JSON — ignore but send warning
                try:
                    ws.send(json.dumps({"error":"Invalid JSON command"}))
                except Exception:
                    pass
                continue

            action = data.get("action")
            if action == "subscribe":
                symbols = data.get("symbols", [])
                # optional data_type param; default to SymbolUpdate
                data_type = data.get("data_type", "SymbolUpdate")
                if not symbols:
                    try:
                        ws.send(json.dumps({"error":"No symbols provided for subscribe"}))
                    except Exception:
                        pass
                    continue
                # update server subscription set
                with _socket_lock:
                    for s in symbols:
                        _subscribed_symbols.add(s)
                # instruct fyers data socket to subscribe
                try:
                    if _fyers_data_socket:
                        # SDK's subscribe signature varies; support both common forms
                        try:
                            _fyers_data_socket.subscribe(symbols=symbols, data_type=data_type)
                        except TypeError:
                            # fallback: positional
                            _fyers_data_socket.subscribe(symbols)
                        try:
                            ws.send(json.dumps({"status":"subscribed", "symbols": symbols}))
                        except Exception:
                            pass
                    else:
                        ws.send(json.dumps({"error":"Internal server: data socket not available"}))
                except Exception as e:
                    app.logger.error(f"Error subscribing via fyers socket: {e}", exc_info=True)
                    try:
                        ws.send(json.dumps({"error":str(e)}))
                    except Exception:
                        pass

            elif action == "unsubscribe":
                symbols = data.get("symbols", [])
                data_type = data.get("data_type", "SymbolUpdate")
                if not symbols:
                    try:
                        ws.send(json.dumps({"error":"No symbols provided for unsubscribe"}))
                    except Exception:
                        pass
                    continue
                # update server subscription set
                with _socket_lock:
                    for s in symbols:
                        _subscribed_symbols.discard(s)
                try:
                    if _fyers_data_socket:
                        try:
                            _fyers_data_socket.unsubscribe(symbols=symbols, data_type=data_type)
                        except TypeError:
                            _fyers_data_socket.unsubscribe(symbols)
                        try:
                            ws.send(json.dumps({"status":"unsubscribed", "symbols": symbols}))
                        except Exception:
                            pass
                    else:
                        ws.send(json.dumps({"error":"Internal server: data socket not available"}))
                except Exception as e:
                    app.logger.error(f"Error unsubscribing via fyers socket: {e}", exc_info=True)
                    try:
                        ws.send(json.dumps({"error":str(e)}))
                    except Exception:
                        pass

            elif action == "mode":
                # mode change: lite/full
                lite = data.get("lite", False)
                try:
                    # To change mode, we must recreate socket instance (SDK may not expose dynamic switch)
                    # We'll stop existing socket by disconnecting (if method available) and restart
                    # Note: SDK's disconnect method name may vary; attempt common patterns.
                    with _socket_lock:
                        # Record current symbol subscriptions so we can resubscribe after restart
                        current_symbols = list(_subscribed_symbols)
                        # try graceful stop
                        try:
                            if _fyers_data_socket:
                                try:
                                    _fyers_data_socket.close()
                                except Exception:
                                    try:
                                        _fyers_data_socket.stop()
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        # Create new socket with requested mode
                        _create_data_socket(access_token=ACCESS_TOKEN, lite_mode=bool(lite))
                        # Start the socket thread if not running
                        if not _socket_running:
                            _start_data_socket_in_thread(ACCESS_TOKEN)
                        # restore subscriptions
                        if current_symbols and _fyers_data_socket:
                            try:
                                _fyers_data_socket.subscribe(symbols=current_symbols, data_type="SymbolUpdate")
                            except Exception:
                                try:
                                    _fyers_data_socket.subscribe(current_symbols)
                                except Exception:
                                    app.logger.debug("Failed to re-subscribe after mode switch.")
                    ws.send(json.dumps({"status":"mode_changed", "lite": bool(lite)}))
                except Exception as e:
                    app.logger.error(f"Error switching mode: {e}", exc_info=True)
                    try:
                        ws.send(json.dumps({"error":str(e)}))
                    except Exception:
                        pass

            elif action == "status":
                try:
                    status = {
                        "socket_running": bool(_socket_running),
                        "connected_clients": len(_connected_clients),
                        "subscribed_symbols": list(_subscribed_symbols)
                    }
                    ws.send(json.dumps({"status": status}))
                except Exception:
                    pass

            else:
                # Unknown action: echo or provide usage help
                try:
                    ws.send(json.dumps({
                        "error":"Unknown action",
                        "usage": [
                            {"action":"subscribe", "symbols":["NSE:SBIN-EQ"], "data_type":"SymbolUpdate"},
                            {"action":"unsubscribe", "symbols":["NSE:SBIN-EQ"]},
                            {"action":"mode", "lite": True},
                            {"action":"status"}
                        ]
                    }))
                except Exception:
                    pass

    except Exception as e:
        app.logger.error(f"Error in ws_fyers handling: {e}", exc_info=True)
    finally:
        # Cleanup when client disconnects
        with _socket_lock:
            try:
                _connected_clients.remove(ws)
            except ValueError:
                pass
        app.logger.info(f"Frontend websocket disconnected. Remaining clients: {len(_connected_clients)}")


# -----------------------------
# End WebSocket Integration Section
# -----------------------------


@app.route('/')
def home():
    return "Fyers API Proxy Server is running! Use /fyers-login to authenticate or /api/fyers/news for news (placeholder)."

# Register the backtesting blueprint
app.register_blueprint(backtesting_bp)

if __name__ == '__main__':
    # Start the Fyers data socket at startup if an access token is present.
    # This is optional — if you prefer to start on-demand when a client connects, remove this.
    if ACCESS_TOKEN:
        try:
            # Attempt to start the socket in background
            _start_data_socket_in_thread(ACCESS_TOKEN)
        except Exception as e:
            app.logger.warning(f"Could not start Fyers DataSocket at startup: {e}")

    app.run(host='0.0.0.0', port=5000)
