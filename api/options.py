from flask import Blueprint, request, jsonify
from utils.token_manager import make_fyers_api_call, get_fyers_instance
import logging

logger = logging.getLogger(__name__)

options_bp = Blueprint('options', __name__)

def _find_option_in_chain(resp, option_symbol=None, strike=None, opt_type=None, expiry_ts=None):
    """Helper to search option chain response for the requested option and return the node."""
    if not resp or not isinstance(resp, dict):
        return None

    data = resp.get("data") if isinstance(resp, dict) else None
    candidates = []
    if data is None and "options_chain" in resp:
        data = resp

    if isinstance(data, dict):
        for key in ["optionChain", "option_chain", "optionsChain", "options", "ce", "pe"]:
            node = data.get(key)
            if node:
                if isinstance(node, list):
                    candidates.extend(node)
                elif isinstance(node, dict):
                    for v in node.values():
                        if isinstance(v, list):
                            candidates.extend(v)
                        else:
                            candidates.append(v)

    if not candidates and isinstance(resp.get("data"), list):
        candidates.extend(resp.get("data"))

    if not candidates and isinstance(resp, list):
        candidates.extend(resp)

    for item in candidates:
        try:
            symbol = item.get("symbol") or item.get("s") or item.get("name")
        except Exception:
            symbol = None
        if option_symbol and symbol and option_symbol == symbol:
            return item

        try:
            strike_val = item.get("strike") or item.get("strike_price") or item.get("strikePrice")
            typ = item.get("option_type") or item.get("type") or item.get("opt_type") or item.get("instrument_type")
            expiry = item.get("expiry") or item.get("expiry_date") or item.get("expiry_ts")
        except Exception:
            strike_val = None
            typ = None
            expiry = None

        if strike is not None and strike_val is not None:
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

@options_bp.route('/api/fyers/option_chain', methods=['GET'])
def get_option_chain():
    fyers_instance = get_fyers_instance()
    symbol = request.args.get('symbol')
    strikecount = request.args.get('strikecount')

    if not symbol or not strikecount:
        logger.warning(f"Missing 'symbol' or 'strikecount' parameter for option chain. Symbol: {symbol}, Strikecount: {strikecount}")
        return jsonify({"error": "Missing 'symbol' or 'strikecount' parameter. Eg: /api/fyers/option_chain?symbol=NSE:TCS-EQ&strikecount=1"}), 400

    try:
        strikecount = int(strikecount)
        if not (1 <= strikecount <= 50):
            logger.warning(f"Invalid 'strikecount' for option chain: {strikecount}. Must be between 1 and 50.")
            return jsonify({"error": "'strikecount' must be between 1 and 50."}), 400
    except ValueError:
        logger.warning(f"Invalid 'strikecount' received for option chain: {strikecount}. Must be an integer.")
        return jsonify({"error": "Invalid 'strikecount'. Must be an integer."}), 400

    data = {
        "symbol": symbol,
        "strikecount": strikecount
    }
    result = make_fyers_api_call(fyers_instance.optionchain, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@options_bp.route('/api/fyers/option_premium', methods=['GET'])
def get_option_premium():
    """Returns option premium (LTP), IV, OI, change, and a small parsed payload for a single option contract."""
    fyers_instance = get_fyers_instance()
    symbol = request.args.get('symbol')
    underlying = request.args.get('underlying')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')
    expiry_ts = request.args.get('expiry_ts')

    if not symbol and not (underlying and strike and opt_type):
        return jsonify({"error": "Provide either symbol=<OPTION_SYMBOL> OR underlying=<SYM>&strike=<STRIKE>&type=<CE|PE>"}), 400

    try:
        if symbol:
            depth_resp = make_fyers_api_call(fyers_instance.depth, data={"symbol": symbol, "ohlcv_flag": 1})
            if isinstance(depth_resp, tuple) and len(depth_resp) == 2 and isinstance(depth_resp[1], int):
                return depth_resp

            premium = None
            try:
                d = depth_resp.get('data') if isinstance(depth_resp, dict) else None
                if d:
                    premium = d.get('ltp') or d.get('last_price') or d.get('close')
                if not premium and isinstance(depth_resp, dict):
                    premium = depth_resp.get('ltp') or depth_resp.get('last_price')
            except Exception:
                premium = None

            if premium is None:
                oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": symbol, "strikecount": 1})
                if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
                    return oc_resp
                found = _find_option_in_chain(oc_resp, option_symbol=symbol)
                node = found or (oc_resp if isinstance(oc_resp, dict) else None)
            else:
                node = depth_resp if isinstance(depth_resp, dict) else None

        else:
            oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying, "strikecount": 50})
            if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
                return oc_resp
            found = _find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
            node = found

        if not node:
            return jsonify({"error": "Option contract not found in Fyers response.", "symbol": symbol or f"{underlying}:{strike}{opt_type if opt_type else ''}"}), 404

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
        logger.error(f"Error in option_premium endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@options_bp.route('/api/fyers/option_chain_depth', methods=['GET'])
def get_option_chain_depth():
    """Returns detailed market depth for a chosen option contract."""
    fyers_instance = get_fyers_instance()
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
        logger.error(f"Error in option_chain_depth endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
