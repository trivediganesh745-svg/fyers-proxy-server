import json
import threading
import logging
from fyers_apiv3.FyersWebsocket import data_ws
import config

logger = logging.getLogger(__name__)

# WebSocket state management
_fyers_data_socket = None
_fyers_socket_thread = None
_connected_clients = []
_subscribed_symbols = set()
_socket_lock = threading.Lock()
_socket_running = False

def create_data_socket(access_token, lite_mode=False, write_to_file=False, reconnect=True, reconnect_retry=10):
    """Create a FyersDataSocket instance with callbacks wired to broadcast incoming messages."""
    global _fyers_data_socket

    def _on_connect():
        logger.info("Fyers DataSocket connected.")
        if _subscribed_symbols:
            try:
                _fyers_data_socket.subscribe(symbols=list(_subscribed_symbols), data_type="SymbolUpdate")
                logger.info(f"Re-subscribed to symbols on connect: {_subscribed_symbols}")
            except Exception as e:
                logger.error(f"Error re-subscribing on connect: {e}", exc_info=True)

    def _on_message(message):
        """Called for incoming socket messages from Fyers. Broadcast to all connected clients."""
        try:
            if isinstance(message, str):
                payload = message
            else:
                payload = json.dumps(message, default=str)
        except Exception:
            payload = str(message)

        with _socket_lock:
            to_remove = []
            for client in _connected_clients:
                try:
                    client.send(payload)
                except Exception as e:
                    logger.debug(f"Failed to send to one client: {e}")
                    to_remove.append(client)
            for r in to_remove:
                try:
                    _connected_clients.remove(r)
                except ValueError:
                    pass

    def _on_error(error):
        logger.error(f"Fyers DataSocket error: {error}")

    def _on_close(close_msg):
        logger.info(f"Fyers DataSocket closed: {close_msg}")

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
        logger.error(f"Failed to create FyersDataSocket: {e}", exc_info=True)
        return None

def start_data_socket_in_thread(access_token):
    """Starts the Fyers DataSocket in a dedicated daemon thread if not already running."""
    global _fyers_socket_thread, _socket_running, _fyers_data_socket

    with _socket_lock:
        if _socket_running:
            logger.info("Fyers DataSocket already running; skip start.")
            return True

        _fyers_data_socket = create_data_socket(access_token)
        if not _fyers_data_socket:
            logger.error("Could not create FyersDataSocket instance.")
            return False

        def _run():
            global _socket_running
            try:
                _socket_running = True
                logger.info("Starting Fyers DataSocket.connect() (blocking call in thread).")
                _fyers_data_socket.connect()
            except Exception as e:
                logger.error(f"Fyers DataSocket thread crashed: {e}", exc_info=True)
            finally:
                _socket_running = False
                logger.info("Fyers DataSocket thread stopped.")

        _fyers_socket_thread = threading.Thread(target=_run, daemon=True, name="FyersDataSocketThread")
        _fyers_socket_thread.start()
        logger.info("Fyers DataSocket thread started.")
        return True

def ensure_data_socket_running():
    """Ensures the socket is running; if not, try to start it with current ACCESS_TOKEN."""
    if not config.ACCESS_TOKEN:
        logger.warning("ACCESS_TOKEN is not available; cannot start data socket.")
        return False
    return start_data_socket_in_thread(config.ACCESS_TOKEN)

def handle_websocket(ws):
    """Handle WebSocket connections from clients"""
    global _connected_clients, _subscribed_symbols, _fyers_data_socket, _socket_running

    with _socket_lock:
        _connected_clients.append(ws)
    logger.info(f"Frontend websocket connected. Total clients: {len(_connected_clients)}")

    try:
        if not _socket_running:
            started = ensure_data_socket_running()
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

                        create_data_socket(access_token=config.ACCESS_TOKEN, lite_mode=bool(lite))
                        if not _socket_running:
                            start_data_socket_in_thread(config.ACCESS_TOKEN)

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
                    ws.send(json.dumps({"status": status}))
                except Exception:
                    pass

            else:
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
        logger.error(f"Error in ws_fyers handling: {e}", exc_info=True)
    finally:
        with _socket_lock:
            try:
                _connected_clients.remove(ws)
            except ValueError:
                pass
        logger.info(f"Frontend websocket disconnected. Remaining clients: {len(_connected_clients)}")
