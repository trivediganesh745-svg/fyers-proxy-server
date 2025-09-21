import os
from flask import Flask, request, jsonify, redirect, url_for
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
from flask_cors import CORS # For handling CORS with your frontend

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Fyers API Configuration (from environment variables) ---
CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
SECRET_KEY = os.environ.get("FYERS_SECRET_KEY")
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI") # This should be your Render proxy URL, e.g., "https://your-render-app.onrender.com/fyers-auth-callback"
# The initial access token will be set manually as an environment variable
# For automated refresh, you'd extend this logic.
ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")

if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI]):
    print("WARNING: Fyers API credentials (CLIENT_ID, SECRET_KEY, REDIRECT_URI) are not fully set. Some functionalities may not work.")

# Initialize FyersModel (will be re-initialized with an access token after login)
fyers = None

def initialize_fyers_model(token):
    global fyers
    if token:
        fyers = fyersModel.FyersModel(token=token, is_async=False, client_id=CLIENT_ID, log_path="")
        print("FyersModel initialized with access token.")
    else:
        print("FyersModel could not be initialized: No access token provided.")

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
        return jsonify({"error": f"Fyers authentication failed: {error}"}), 400
    if not auth_code:
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
        
        # In a real application, you'd securely store this ACCESS_TOKEN
        # and refresh it. For now, we'll just acknowledge it.
        
        # Redirect to a success page or provide the token to the frontend
        return jsonify({"message": "Fyers token generated successfully!", "access_token_available": True})
        
    except Exception as e:
        print(f"Error generating Fyers access token: {e} - Response: {response}")
        return jsonify({"error": f"Failed to generate Fyers access token: {str(e)}"}), 500


# --- Fyers Data Endpoints (example) ---

@app.route('/api/fyers/profile')
def get_profile():
    if not fyers:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        profile_data = fyers.get_profile()
        return jsonify(profile_data)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch profile: {str(e)}"}), 500

@app.route('/api/fyers/funds')
def get_funds():
    if not fyers:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        funds_data = fyers.funds()
        return jsonify(funds_data)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch funds: {str(e)}"}), 500

@app.route('/api/fyers/holdings')
def get_holdings():
    if not fyers:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        holdings_data = fyers.holdings()
        return jsonify(holdings_data)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch holdings: {str(e)}"}), 500

# You can add more endpoints for other Fyers API calls (tradebook, orderbook, positions, etc.)
# For example, for history:
@app.route('/api/fyers/history', methods=['POST'])
def get_history():
    if not fyers:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        data = request.json
        if not data or not all(k in data for k in ["symbol", "resolution", "range_from", "range_to"]):
            return jsonify({"error": "Missing required parameters for history API. Need symbol, resolution, range_from, range_to."}), 400
        
        history_data = fyers.history(data)
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

# Example for placing an order (requires POST request with order data)
@app.route('/api/fyers/place_order', methods=['POST'])
def place_single_order():
    if not fyers:
        return jsonify({"error": "Fyers API not initialized. Please authenticate first."}), 401
    try:
        order_data = request.json
        if not order_data:
            return jsonify({"error": "No order data provided."}), 400
        
        response = fyers.place_order(order_data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Failed to place order: {str(e)}"}), 500


@app.route('/')
def home():
    return "Fyers API Proxy Server is running!"


if __name__ == '__main__':
    # Use Gunicorn in production (handled by Render)
    # For local testing:
    app.run(host='0.0.0.0', port=5000)
