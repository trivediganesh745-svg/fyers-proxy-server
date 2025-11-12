from flask_sock import Sock
from fyers_apiv3.FyersWebsocket import data_ws
import threading
import json
import logging

logger = logging.getLogger(__name__)

# Global variables for WebSocket management
_fyers_data_socket = None
_fyers_socket_thread = None
_connected_clients = []
_subscribed_symbols = set()
_socket_lock = threading.Lock()
_socket_running = False

def setup_websocket(app):
    """Setup WebSocket for the Flask app"""
    sock = Sock(app)
    
    @sock.route('/ws/fyers')
    def ws_fyers(ws):
        """
        Frontend connects here: wss://<host>/ws/fyers
        Supported incoming JSON messages from frontend:
          - {"action": "subscribe", "symbols": ["NSE:SBIN-EQ","NSE:TCS-EQ"], "data_type":"SymbolUpdate"}
          - {"action": "unsubscribe", "symbols": ["NSE:SBIN-EQ"]}
          - {"action": "mode", "lite": true}
          - {"action": "status"}
        """
        global _connected_clients, _subscribed_symbols, _fyers_data_socket, _socket_running
        from utils.token_manager import get_current_tokens

        with _socket_lock:
            _connected_clients.append(ws)
        logger.info(f"Frontend websocket connected. Total clients: {len(_connected_clients)}")

        try:
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

            while True:
                msg = ws.receive()
                if msg is None:
                    logger.info("Frontend websocket disconnected (receive returned None).")
                    break

                try:
                    data = json.loads(msg)
                except Exception:
                    try:
                        ws.send(json.dumps({"error":"Invalid JSON command"}))
                    except Exception:
                        pass
                    continue

                action = data.get("action")
                if action == "subscribe":
                    symbols = data.get("symbols", [])
                    data_type = data.get("data_type", "SymbolUpdate")
                    if not symbols:
                        try:
                            ws.send(json.dumps({"error":"No symbols provided for subscribe"}))
                        except Exception:
                            pass
                        continue

                    with _socket_lock:
                        for s in symbols:
                            _subscribed_symbols.add(s)

                    try:
                        if _fyers_data_socket:
                            try:
                                _fyers_data_socket.subscribe(symbols=symbols, data_type=data_type)
                            except TypeError:
                                _fyers_data_socket.subscribe(symbols)
                            try:
                                ws.send(json.dumps({"status":"subscribed", "symbols": symbols}))
                            except Exception:
                                pass
                        else:
                            ws.send(json.dumps({"error":"Internal server: data socket not available"}))
                    except Exception as e:
                        logger.error(f"Error subscribing via fyers socket: {e}", exc_info=True)
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
                        logger.error(f"Error unsubscribing via fyers socket: {e}", exc_info=True)
                        try:
                            ws.send(json.dumps({"error":str(e)}))
                        except Exception:
                            pass

                elif action == "mode":
                    lite = data.get("lite", False)
                    try:
                        with _socket_lock:
                            current_symbols = list(_subscribed_symbols)
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

                            ACCESS_TOKEN, _ = get_current_tokens()
                            _create_data_socket(access_token=ACCESS_TOKEN, lite_mode=bool(lite))
                            if not _socket_running:
                                _start_data_socket_in_thread(ACCESS_TOKEN)

                            if current_symbols and _fyers_data_socket:
                                try:
                                    _fyers_data_socket.subscribe(symbols=current_symbols, data_type="SymbolUpdate")
                                except Exception:
                                    try:
                                        _fyers_data_socket.subscribe(current_symbols)
                                    except Exception:
                                        logger.debug("Failed to re-subscribe after mode switch.")
                        ws.send(json.dumps({"status":"mode_changed", "lite": bool(lite)}))
                    except Exception as e:
                        logger.error(f"Error switching mode: {e}", exc_info=True)
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
                        
