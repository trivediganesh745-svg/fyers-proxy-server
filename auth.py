import json
import datetime
import logging
from flask import jsonify, redirect, request
from fyers_apiv3 import fyersModel
import config

logger = logging.getLogger(__name__)

def store_tokens(access_token, refresh_token):
    """Store tokens in memory and provide instructions for persistence"""
    config.ACCESS_TOKEN = access_token
    config.REFRESH_TOKEN = refresh_token

    logger.info("Tokens updated successfully!")
    logger.info("IMPORTANT: For persistence on Render, update these environment variables:")
    logger.info(f"FYERS_ACCESS_TOKEN={config.ACCESS_TOKEN}")
    logger.info(f"FYERS_REFRESH_TOKEN={config.REFRESH_TOKEN}")
    
    # Save to a local file as backup
    try:
        token_data = {
            "access_token": config.ACCESS_TOKEN,
            "refresh_token": config.REFRESH_TOKEN,
            "updated_at": datetime.datetime.now().isoformat()
        }
        with open("tokens.json", "w") as f:
            json.dump(token_data, f)
        logger.info("Tokens saved to tokens.json file")
    except Exception as e:
        logger.error(f"Could not save tokens to file: {e}")

def load_tokens():
    """Load tokens from environment or file"""
    import os
    
    # First try environment variables
    if os.environ.get("FYERS_ACCESS_TOKEN"):
        config.ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
        config.REFRESH_TOKEN = os.environ.get("FYERS_REFRESH_TOKEN")
        logger.info("Tokens loaded from environment variables")
        return
    
    # Try loading from file as fallback
    try:
        with open("tokens.json", "r") as f:
            token_data = json.load(f)
            config.ACCESS_TOKEN = token_data.get("access_token")
            config.REFRESH_TOKEN = token_data.get("refresh_token")
            logger.info(f"Tokens loaded from file (updated: {token_data.get('updated_at')})")
    except Exception as e:
        logger.info(f"No tokens file found or error loading: {e}")

def initialize_fyers_model(token=None):
    """Initialize Fyers model with token or refresh token"""
    from models import fyers_instance
    
    token_to_use = token if token else config.ACCESS_TOKEN

    if token_to_use:
        try:
            try:
                import models
                models.fyers_instance = fyersModel.FyersModel(
                    token=token_to_use, 
                    is_async=False, 
                    client_id=config.CLIENT_ID, 
                    log_path=""
                )
            except TypeError:
                models.fyers_instance = fyersModel.FyersModel(
                    access_token=token_to_use, 
                    is_async=False, 
                    client_id=config.CLIENT_ID, 
                    log_path=""
                )
            logger.info("FyersModel initialized with access token.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize fyers_model with provided token: {e}", exc_info=True)
            if config.REFRESH_TOKEN:
                logger.info("Access token failed, attempting refresh...")
                return refresh_access_token()
            return False
    elif config.REFRESH_TOKEN and config.CLIENT_ID and config.SECRET_KEY:
        logger.info("No access token provided or found, attempting to use refresh token.")
        if refresh_access_token():
            logger.info("FyersModel initialized with refreshed access token.")
            return True
        else:
            logger.error("Could not initialize FyersModel: Refresh token failed.")
            return False
    else:
        logger.warning("FyersModel could not be initialized: No access token or refresh token available.")
        return False

def refresh_access_token():
    """Refresh the access token using refresh token"""
    import models
    
    if not config.REFRESH_TOKEN:
        logger.error("Cannot refresh token: No refresh token available.")
        return False

    if not all([config.CLIENT_ID, config.SECRET_KEY, config.REDIRECT_URI]):
        logger.error("Cannot refresh token: Missing CLIENT_ID, SECRET_KEY, or REDIRECT_URI")
        return False

    session = fyersModel.SessionModel(
        client_id=config.CLIENT_ID,
        redirect_uri=config.REDIRECT_URI,
        response_type="code",
        state="refresh_state",
        secret_key=config.SECRET_KEY,
        grant_type="refresh_token"
    )
    session.set_token(config.REFRESH_TOKEN)

    try:
        logger.info(f"Attempting to refresh access token...")
        response = session.generate_token()

        if response and response.get("s") == "ok":
            new_access_token = response["access_token"]
            new_refresh_token = response.get("refresh_token", config.REFRESH_TOKEN)

            store_tokens(new_access_token, new_refresh_token)
            
            # Re-initialize Fyers model with new token
            try:
                models.fyers_instance = fyersModel.FyersModel(
                    access_token=new_access_token,
                    is_async=False,
                    client_id=config.CLIENT_ID,
                    log_path=""
                )
                logger.info("Access token refreshed and FyersModel reinitialized successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to reinitialize FyersModel after refresh: {e}")
                return False
        else:
            logger.error(f"Failed to refresh access token. Response: {response}")
            return False
    except Exception as e:
        logger.error(f"Error during access token refresh: {e}", exc_info=True)
        return False

def make_fyers_api_call(api_method, *args, **kwargs):
    """Make Fyers API call with automatic token refresh on failure"""
    from models import fyers_instance
    
    if not fyers_instance:
        logger.warning("Fyers API not initialized. Attempting to initialize.")
        if not initialize_fyers_model():
            return {"error": "Fyers API not initialized. Please authenticate first."}, 401

    try:
        result = api_method(*args, **kwargs)
        
        # Check if result indicates token error
        if isinstance(result, dict):
            error_code = result.get("code") or result.get("s")
            error_message = str(result.get("message", "")).lower()
            
            if (error_code in [-16, -99, 401, "error"] or 
                any(keyword in error_message for keyword in ["token", "expired", "invalid", "authenticate", "authorization"])):
                raise Exception("Token expired or invalid")
        
        return result
        
    except Exception as e:
        error_message = str(e).lower()

        if any(keyword in error_message for keyword in ["token", "expired", "invalid", "authenticate", "authorization"]):
            logger.warning(f"Token issue detected. Attempting to refresh. Error: {e}")
            
            if refresh_access_token():
                logger.info("Token refreshed, retrying original request.")
                try:
                    return api_method(*args, **kwargs)
                except Exception as retry_error:
                    logger.error(f"Request failed even after token refresh: {retry_error}")
                    return {"error": f"Request failed after token refresh: {str(retry_error)}"}, 500
            else:
                logger.error("Token refresh failed. User needs to re-authenticate.")
                return {"error": "Authentication expired. Please login again via /fyers-login"}, 401
        else:
            logger.error(f"Non-token related Fyers API error: {e}", exc_info=True)
            return {"error": f"Fyers API error: {str(e)}"}, 500
