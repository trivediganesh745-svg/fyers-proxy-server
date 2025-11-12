from flask import Blueprint, jsonify
from utils.token_manager import make_fyers_api_call, get_fyers_instance
import logging

logger = logging.getLogger(__name__)

account_bp = Blueprint('account', __name__)

@account_bp.route('/api/fyers/profile')
def get_profile():
    fyers_instance = get_fyers_instance()
    result = make_fyers_api_call(fyers_instance.get_profile)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@account_bp.route('/api/fyers/funds')
def get_funds():
    fyers_instance = get_fyers_instance()
    result = make_fyers_api_call(fyers_instance.funds)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@account_bp.route('/api/fyers/holdings')
def get_holdings():
    fyers_instance = get_fyers_instance()
    result = make_fyers_api_call(fyers_instance.holdings)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)
