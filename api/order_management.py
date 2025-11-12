from flask import Blueprint, request, jsonify
from utils.token_manager import make_fyers_api_call, get_fyers_instance
import logging

logger = logging.getLogger(__name__)

orders_bp = Blueprint('orders', __name__)

@orders_bp.route('/api/fyers/order', methods=['POST'])
def place_single_order():
    fyers_instance = get_fyers_instance()
    order_data = request.json
    if not order_data:
        return jsonify({"error": "Order data is required."}), 400
    result = make_fyers_api_call(fyers_instance.place_order, data=order_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/orders/multi', methods=['POST'])
def place_multi_order():
    fyers_instance = get_fyers_instance()
    multi_order_data = request.json
    if not multi_order_data or not isinstance(multi_order_data, list):
        return jsonify({"error": "An array of order objects is required for multi-order placement."}), 400

    result = make_fyers_api_call(fyers_instance.multi_order, data=multi_order_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/orders/multileg', methods=['POST'])
def place_multileg_order():
    fyers_instance = get_fyers_instance()
    multileg_data = request.json
    if not multileg_data:
        return jsonify({"error": "Multileg order data is required."}), 400
    result = make_fyers_api_call(fyers_instance.multileg_order, data=multileg_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/gtt/order', methods=['POST'])
def place_gtt_order():
    fyers_instance = get_fyers_instance()
    gtt_order_data = request.json
    if not gtt_order_data:
        return jsonify({"error": "GTT order data is required."}), 400
    result = make_fyers_api_call(fyers_instance.place_gttorder, data=gtt_order_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/gtt/order', methods=['PATCH'])
def modify_gtt_order():
    fyers_instance = get_fyers_instance()
    gtt_modify_data = request.json
    if not gtt_modify_data or not gtt_modify_data.get("id"):
        return jsonify({"error": "GTT order ID and modification data are required."}), 400

    order_id = gtt_modify_data.pop("id")
    result = make_fyers_api_call(fyers_instance.modify_gttorder, id=order_id, data=gtt_modify_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/gtt/order', methods=['DELETE'])
def cancel_gtt_order():
    fyers_instance = get_fyers_instance()
    gtt_cancel_data = request.json
    if not gtt_cancel_data or not gtt_cancel_data.get("id"):
        return jsonify({"error": "GTT order ID is required for cancellation."}), 400

    order_id = gtt_cancel_data.get("id")
    result = make_fyers_api_call(fyers_instance.cancel_gttorder, id=order_id)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/gtt/orders', methods=['GET'])
def get_gtt_orders():
    fyers_instance = get_fyers_instance()
    result = make_fyers_api_call(fyers_instance.gtt_orders)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/order', methods=['PATCH'])
def modify_single_order():
    fyers_instance = get_fyers_instance()
    modify_data = request.json
    if not modify_data or not modify_data.get("id"):
        return jsonify({"error": "Order ID and modification data are required."}), 400

    result = make_fyers_api_call(fyers_instance.modify_order, data=modify_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/orders/multi', methods=['PATCH'])
def modify_multi_orders():
    fyers_instance = get_fyers_instance()
    modify_basket_data = request.json
    if not modify_basket_data or not isinstance(modify_basket_data, list):
        return jsonify({"error": "An array of order modification objects is required."}), 400

    result = make_fyers_api_call(fyers_instance.modify_basket_orders, data=modify_basket_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/order', methods=['DELETE'])
def cancel_single_order():
    fyers_instance = get_fyers_instance()
    cancel_data = request.json
    if not cancel_data or not cancel_data.get("id"):
        return jsonify({"error": "Order ID is required for cancellation."}), 400

    result = make_fyers_api_call(fyers_instance.cancel_order, data=cancel_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/orders/multi', methods=['DELETE'])
def cancel_multi_orders():
    fyers_instance = get_fyers_instance()
    cancel_basket_data = request.json
    if not cancel_basket_data or not isinstance(cancel_basket_data, list):
        return jsonify({"error": "An array of order cancellation objects (with 'id') is required."}), 400

    result = make_fyers_api_call(fyers_instance.cancel_basket_orders, data=cancel_basket_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/positions', methods=['DELETE'])
def exit_positions():
    fyers_instance = get_fyers_instance()
    exit_data = request.json
    if not exit_data:
        return jsonify({"error": "Request body for exiting positions is required."}), 400

    result = make_fyers_api_call(fyers_instance.exit_positions, data=exit_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/positions', methods=['POST'])
def convert_position():
    fyers_instance = get_fyers_instance()
    convert_data = request.json
    if not convert_data:
        return jsonify({"error": "Position conversion data is required."}), 400

    result = make_fyers_api_call(fyers_instance.convert_positions, data=convert_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

# --- Margin Calculator APIs ---

@orders_bp.route('/api/fyers/margin/span', methods=['POST'])
def span_margin_calculator():
    fyers_instance = get_fyers_instance()
    margin_data = request.json
    if not margin_data or not margin_data.get("data"):
        return jsonify({"error": "An array of order details for span margin calculation is required under 'data' key."}), 400

    result = make_fyers_api_call(fyers_instance.span_margin, data=margin_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)

@orders_bp.route('/api/fyers/margin/multiorder', methods=['POST'])
def multiorder_margin_calculator():
    fyers_instance = get_fyers_instance()
    margin_data = request.json
    if not margin_data or not margin_data.get("data"):
        return jsonify({"error": "An array of order details for multiorder margin calculation is required under 'data' key."}), 400

    result = make_fyers_api_call(fyers_instance.multiorder_margin, data=margin_data)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result
    return jsonify(result)
