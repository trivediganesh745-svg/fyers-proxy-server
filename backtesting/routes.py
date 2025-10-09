# backtesting/routes.py

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import datetime
import json
from .engine import BacktestingEngine
from .strategies import SimpleMovingAverageCrossover, RSICrossover # Import your strategies
import requests # For fetching historical data from your own API proxy

backtesting_bp = Blueprint('backtesting', __name__, url_prefix='/api/backtest')

# Dictionary to hold available strategies
STRATEGIES = {
    "sma_crossover": SimpleMovingAverageCrossover,
    "rsi_crossover": RSICrossover,
    # Add more strategies here
}

def fetch_historical_data_from_proxy(symbol: str, resolution: str, range_from: int, range_to: int):
    """
    Fetches historical data for a symbol from your Fyers API proxy.
    Assumes your app is running on localhost:5000 or the appropriate URL.
    """
    proxy_url = current_app.config.get("FYERS_PROXY_BASE_URL", "http://localhost:5000")
    history_endpoint = f"{proxy_url}/api/fyers/history"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": 0,  # Epoch format
        "range_from": str(range_from),
        "range_to": str(range_to)
    }
    
    current_app.logger.info(f"Fetching history for {symbol} from proxy: {history_endpoint} with range {range_from}-{range_to}")
    try:
        response = requests.post(history_endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        if data and data.get("s") == "ok" and data.get("candles"):
            candles_df = pd.DataFrame(data["candles"], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            candles_df['date'] = pd.to_datetime(candles_df['date'], unit='s')
            candles_df = candles_df.set_index('date')
            candles_df.index = candles_df.index.tz_localize('UTC') # Assume UTC from Fyers
            return candles_df
        else:
            current_app.logger.error(f"Failed to fetch history for {symbol}. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None

@backtesting_bp.route('/run', methods=['POST'])
def run_backtest():
    data = request.json
    
    required_params = ["symbols", "resolution", "start_date", "end_date", "strategy_name", "strategy_params"]
    if not data or not all(k in data for k in required_params):
        return jsonify({"error": f"Missing required parameters. Need {', '.join(required_params)}."}), 400

    symbols = data["symbols"] # List of symbols, e.g., ["NSE:SBIN-EQ"]
    resolution = data["resolution"]
    start_date_str = data["start_date"]
    end_date_str = data["end_date"]
    strategy_name = data["strategy_name"]
    strategy_params = data.get("strategy_params", {})
    initial_capital = data.get("initial_capital", 100000.0)
    transaction_cost_percent = data.get("transaction_cost_percent", 0.001)

    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Convert to epoch for Fyers API
        range_from_epoch = int(start_date.timestamp())
        range_to_epoch = int(end_date.timestamp())
    except ValueError as e:
        return jsonify({"error": f"Invalid date format. Use YYYY-MM-DD. Error: {e}"}), 400

    if strategy_name not in STRATEGIES:
        return jsonify({"error": f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}"}), 400

    # Initialize strategy
    try:
        strategy_class = STRATEGIES[strategy_name]
        strategy_instance = strategy_class(**strategy_params)
    except Exception as e:
        current_app.logger.error(f"Error initializing strategy {strategy_name} with params {strategy_params}: {e}")
        return jsonify({"error": f"Error initializing strategy: {str(e)}"}), 400

    # Fetch historical data for all symbols
    historical_data = {}
    for symbol in symbols:
        df = fetch_historical_data_from_proxy(symbol, resolution, range_from_epoch, range_to_epoch)
        if df is not None and not df.empty:
            historical_data[symbol] = df
        else:
            current_app.logger.warning(f"No or insufficient historical data for {symbol}. Skipping for backtest.")
            # return jsonify({"error": f"Failed to get historical data for {symbol}."}), 500

    if not historical_data:
        return jsonify({"error": "No historical data available for any specified symbol to run backtest."}), 400

    # Run backtest
    engine = BacktestingEngine(initial_capital=initial_capital, transaction_cost_percent=transaction_cost_percent)
    try:
        engine.run_backtest(historical_data, strategy_instance, resolution)
        report = engine.generate_report()
        return jsonify({"message": "Backtest completed successfully", "report": report}), 200
    except Exception as e:
        current_app.logger.error(f"Error during backtest execution: {e}", exc_info=True)
        return jsonify({"error": f"Error during backtest execution: {str(e)}"}), 500

@backtesting_bp.route('/strategies', methods=['GET'])
def list_strategies():
    """Lists available backtesting strategies."""
    strategy_info = []
    for name, strategy_class in STRATEGIES.items():
        # Instantiate temporarily to get default params or description if available
        # This is a simple way, for more complex strategies, might need a 'describe' method
        try:
            temp_strategy = strategy_class()
            strategy_info.append({
                "name": name,
                "description": temp_strategy.name, # Using the name as a simple description
                "default_params": temp_strategy.__dict__ # Show default parameters
            })
        except TypeError: # If strategy_class requires args without defaults
             strategy_info.append({
                "name": name,
                "description": f"Strategy {name} (requires specific parameters)",
                "default_params": {}
            })

    return jsonify({"strategies": strategy_info}), 200
