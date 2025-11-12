from flask import Blueprint, request, jsonify
import datetime
import time
import logging
from auth import make_fyers_api_call
from models import fyers_instance, gemini_model
from utils import find_option_in_chain, calculate_advanced_indicators
import config

logger = logging.getLogger(__name__)

market_bp = Blueprint('market', __name__)

@market_bp.route('/api/fyers/history', methods=['POST'])
def get_history():
    """Fetch historical data with support for second-level, minute-level, and day-level resolutions."""
    data = request.json

    required_params = ["symbol", "resolution", "date_format", "range_from", "range_to"]
    if not data or not all(k in data for k in required_params):
        logger.warning(f"Missing required parameters for history API. Received: {data}")
        return jsonify({"error": f"Missing required parameters for history API. Need {', '.join(required_params)}."}), 400

    # Check if AI analysis is requested
    include_ai_analysis = data.get("include_ai_analysis", False)
    ai_analysis_type = data.get("ai_analysis_type", "technical")

    # Validate resolution
    resolution = data["resolution"]
    if resolution not in config.SECOND_RESOLUTIONS and resolution not in config.MINUTE_RESOLUTIONS and resolution not in config.DAY_RESOLUTIONS:
        logger.warning(f"Unsupported resolution: {resolution}")
        return jsonify({
            "error": f"Unsupported resolution: {resolution}",
            "supported_resolutions": {
                "second": config.SECOND_RESOLUTIONS,
                "minute": config.MINUTE_RESOLUTIONS,
                "day": config.DAY_RESOLUTIONS
            }
        }), 400

    try:
        data["date_format"] = int(data["date_format"])
    except ValueError:
        logger.warning(f"Invalid 'date_format' received: {data.get('date_format')}")
        return jsonify({"error": "Invalid 'date_format'. Must be 0 or 1."}), 400

    # Handle incomplete candles for real-time data
    if data["date_format"] == 0:
        current_time = int(time.time())
        requested_range_to = int(data["range_to"])
        resolution = data["resolution"]

        resolution_in_seconds = 0
        if resolution.endswith('S'):
            try:
                resolution_in_seconds = int(resolution[:-1])
                logger.info(f"Processing second-level resolution: {resolution} ({resolution_in_seconds} seconds)")
            except ValueError:
                logger.warning(f"Invalid numeric part in resolution ending with 'S': {resolution}")
                return jsonify({"error": "Invalid resolution format."}), 400
        elif resolution.isdigit():
            resolution_in_seconds = int(resolution) * 60
            logger.info(f"Processing minute-level resolution: {resolution} minutes ({resolution_in_seconds} seconds)")
        elif resolution in ["D", "1D"]:
            resolution_in_seconds = 24 * 60 * 60
            logger.info(f"Processing day-level resolution: {resolution}")
        else:
            logger.warning(f"Unsupported resolution format for partial candle adjustment: {resolution}")
            return jsonify({"error": "Unsupported resolution format."}), 400

        if resolution_in_seconds > 0:
            current_resolution_start_epoch = (current_time // resolution_in_seconds) * resolution_in_seconds

            if requested_range_to >= current_resolution_start_epoch:
                adjusted_range_to_epoch = current_resolution_start_epoch - 1

                if adjusted_range_to_epoch < int(data["range_from"]):
                    logger.info(f"Adjusted range_to ({adjusted_range_to_epoch}) is less than range_from ({data['range_from']}). No complete candles available.")
                    return jsonify({"candles": [], "s": "ok", "message": "No complete candles available for the adjusted range."})

                data["range_to"] = str(adjusted_range_to_epoch)
                logger.info(f"Adjusted 'range_to' for resolution '{resolution}' to ensure completed candles: {requested_range_to} -> {data['range_to']}")

    if "cont_flag" in data:
        data["cont_flag"] = int(data["cont_flag"])
    if "oi_flag" in data:
        data["oi_flag"] = int(data["oi_flag"])

    logger.info(f"Fetching history data: Symbol={data['symbol']}, Resolution={data['resolution']}, From={data['range_from']}, To={data['range_to']}")

    result = make_fyers_api_call(fyers_instance.history, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result

    # Log successful response
    if result and result.get("s") == "ok":
        candles_count = len(result.get("candles", []))
        logger.info(f"Successfully fetched {candles_count} candles for {data['symbol']} at {data['resolution']} resolution")

        # Add AI analysis if requested
        if include_ai_analysis and gemini_model and candles_count > 0:
            try:
                from api.ai_routes import analyze_market_data_with_ai
                candles = result.get("candles", [])
                indicators = calculate_advanced_indicators(candles)

                analysis_data = {
                    "symbol": data["symbol"],
                    "resolution": data["resolution"],
                    "candles_count": candles_count,
                    "indicators": indicators,
                    "latest_candle": candles[-1] if candles else None
                }

                ai_analysis = analyze_market_data_with_ai(analysis_data, ai_analysis_type)
                result["ai_analysis"] = ai_analysis

            except Exception as e:
                logger.error(f"Failed to add AI analysis: {e}")
                result["ai_analysis"] = {"error": str(e)}

    return jsonify(result)

@market_bp.route('/api/fyers/quotes', methods=['GET'])
def get_quotes():
    symbols = request.args.get('symbols')
    include_ai_analysis = request.args.get('include_ai_analysis', 'false').lower() == 'true'

    if not symbols:
        logger.warning("Missing 'symbols' parameter for quotes API.")
        return jsonify({"error": "Missing 'symbols' parameter. Eg: /api/fyers/quotes?symbols=NSE:SBIN-EQ,NSE:TCS-EQ"}), 400

    data = {"symbols": symbols}
    result = make_fyers_api_call(fyers_instance.quotes, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result

    # Add AI analysis if requested
    if include_ai_analysis and gemini_model and result and result.get("d"):
        try:
            from api.ai_routes import analyze_market_data_with_ai
            quotes_data = result.get("d", [])
            ai_analysis = analyze_market_data_with_ai({"quotes": quotes_data}, "general")
            result["ai_analysis"] = ai_analysis
        except Exception as e:
            logger.error(f"Failed to add AI analysis to quotes: {e}")
            result["ai_analysis"] = {"error": str(e)}

    return jsonify(result)

@market_bp.route('/api/fyers/market_depth', methods=['GET'])
def get_market_depth():
    symbol = request.args.get('symbol')
    ohlcv_flag = request.args.get('ohlcv_flag')

    if not symbol or ohlcv_flag is None:
        logger.warning(f"Missing 'symbol' or 'ohlcv_flag' parameter for market depth. Symbol: {symbol}, Flag: {ohlcv_flag}")
        return jsonify({"error": "Missing 'symbol' or 'ohlcv_flag' parameter. Eg: /api/fyers/market_depth?symbol=NSE:SBIN-EQ&ohlcv_flag=1"}), 400

    try:
        ohlcv_flag = int(ohlcv_flag)
        if ohlcv_flag not in [0, 1]:
            raise ValueError("ohlcv_flag must be 0 or 1.")
    except ValueError as ve:
        logger.warning(f"Invalid 'ohlcv_flag' received for market depth: {ohlcv_flag}")
        return jsonify({"error": f"Invalid 'ohlcv_flag': {ve}"}), 400

    data = {
        "symbol": symbol,
        "ohlcv_flag": ohlcv_flag
    }
    result = make_fyers_api_call(fyers_instance.depth, data=data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@market_bp.route('/api/fyers/option_chain', methods=['GET'])
def get_option_chain():
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

@market_bp.route('/api/fyers/option_premium', methods=['GET'])
def get_option_premium():
    """Returns option premium (LTP), IV, OI, change, and a small parsed payload for a single option contract."""
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
                found = find_option_in_chain(oc_resp, option_symbol=symbol)
                node = found or (oc_resp if isinstance(oc_resp, dict) else None)
            else:
                node = depth_resp if isinstance(depth_resp, dict) else None

        else:
            oc_resp = make_fyers_api_call(fyers_instance.optionchain, data={"symbol": underlying, "strikecount": 50})
            if isinstance(oc_resp, tuple) and len(oc_resp) == 2 and isinstance(oc_resp[1], int):
                return oc_resp
            found = find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
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

@market_bp.route('/api/fyers/option_chain_depth', methods=['GET'])
def get_option_chain_depth():
    """Returns detailed market depth for a chosen option contract."""
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

        found = find_option_in_chain(oc_resp, strike=strike, opt_type=opt_type, expiry_ts=expiry_ts)
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

@market_bp.route('/api/fyers/market_status', methods=['GET'])
def get_market_status():
    result = make_fyers_api_call(fyers_instance.market_status)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@market_bp.route('/api/fyers/news', methods=['GET'])
def get_news():
    logger.info("Accessing placeholder news endpoint.")

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
