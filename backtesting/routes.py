# In backtesting/routes.py

@backtesting_bp.route('/run', methods=['POST'])
def run_backtest():
    data = request.json
    
    required_params = ["symbols", "resolution", "start_date", "end_date", "strategy_name", "strategy_params"]
    if not data or not all(k in data for k in required_params):
        return jsonify({"error": f"Missing required parameters. Need {', '.join(required_params)}."}), 400

    symbols = data["symbols"]
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
        
        # --- FIX: Adjust end_date to include the entire day until 23:59:59 ---
        end_date = end_date + datetime.timedelta(days=1, seconds=-1)
        
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
        # Clean up 'name' parameter if it's being sent from the frontend
        strategy_params.pop('name', None)
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

    if not historical_data:
        return jsonify({"error": "No historical data available for any specified symbol to run backtest."}), 400

    # Run backtest
    engine = BacktestingEngine(initial_capital=initial_capital, transaction_cost_percent=transaction_cost_percent)
    try:
        engine.run_backtest(historical_data, strategy_instance, resolution)
        report = engine.generate_report()
        # Use json.dumps with a handler for Timestamp objects to prevent serialization errors
        report_json = json.dumps(report, default=str)
        return jsonify({"message": "Backtest completed successfully", "report": json.loads(report_json)}), 200
    except Exception as e:
        current_app.logger.error(f"Error during backtest execution: {e}", exc_info=True)
        return jsonify({"error": f"Error during backtest execution: {str(e)}"}), 500
