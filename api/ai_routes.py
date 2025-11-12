from flask import Blueprint, request, jsonify
import json
import datetime
import logging
from models import gemini_model, fyers_instance
from auth import make_fyers_api_call
from utils import calculate_advanced_indicators

logger = logging.getLogger(__name__)

ai_bp = Blueprint('ai', __name__)

def analyze_market_data_with_ai(data, analysis_type="general"):
    """Use Gemini AI to analyze market data and provide insights."""
    if not gemini_model:
        return {"error": "AI model not initialized"}

    try:
        prompts = {
            "general": f"""
                Analyze the following market data and provide insights:
                {json.dumps(data, indent=2)}
                
                Please provide:
                1. Key observations
                2. Trend analysis
                3. Risk factors
                4. Potential opportunities
                5. Recommended actions
                
                Format the response in a clear, structured manner.
            """,
            "technical": f"""
                Perform technical analysis on the following market data:
                {json.dumps(data, indent=2)}
                
                Include:
                1. Support and resistance levels
                2. Trend direction and strength
                3. Volume analysis
                4. Key technical indicators interpretation
                5. Entry and exit points
                
                Be specific with price levels and percentages.
            """,
            "sentiment": f"""
                Analyze market sentiment based on the following data:
                {json.dumps(data, indent=2)}
                
                Provide:
                1. Overall market sentiment (bullish/bearish/neutral)
                2. Sentiment strength (1-10 scale)
                3. Key sentiment drivers
                4. Potential sentiment shifts
                5. Contrarian opportunities
            """,
            "risk": f"""
                Perform risk analysis on the following market data:
                {json.dumps(data, indent=2)}
                
                Include:
                1. Risk level assessment (low/medium/high)
                2. Specific risk factors
                3. Volatility analysis
                4. Risk mitigation strategies
                5. Position sizing recommendations
                
                Provide specific percentages and thresholds.
            """
        }

        prompt = prompts.get(analysis_type, prompts["general"])
        response = gemini_model.generate_content(prompt)

        return {
            "analysis_type": analysis_type,
            "analysis": response.text,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return {"error": str(e)}

def generate_trading_signals_with_ai(symbol, historical_data, current_price):
    """Generate trading signals using AI based on historical data and current price."""
    if not gemini_model:
        return {"error": "AI model not initialized"}

    try:
        recent_candles = historical_data[-20:] if len(historical_data) > 20 else historical_data

        prompt = f"""
        As a trading analyst, generate trading signals for {symbol} based on the following data:
        
        Current Price: {current_price}
        Recent Historical Data (last {len(recent_candles)} candles):
        {json.dumps(recent_candles, indent=2)}
        
        Provide:
        1. Signal: BUY/SELL/HOLD
        2. Confidence Level: (0-100%)
        3. Entry Price
        4. Stop Loss
        5. Target Price(s) - provide 3 targets
        6. Risk-Reward Ratio
        7. Time Horizon
        8. Key Reasoning
        
        Format as JSON for easy parsing.
        """

        response = gemini_model.generate_content(prompt)

        try:
            response_text = response.text
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                signal_data = json.loads(json_match.group(1))
            else:
                signal_data = json.loads(response_text)
        except:
            signal_data = {"raw_analysis": response.text}

        return {
            "symbol": symbol,
            "current_price": current_price,
            "signals": signal_data,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"AI signal generation error: {e}")
        return {"error": str(e)}

@ai_bp.route('/api/ai/analyze', methods=['POST'])
def ai_analyze():
    """Analyze market data using Gemini AI."""
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    if not request_data or not request_data.get("data"):
        return jsonify({"error": "Missing 'data' in request body"}), 400

    analysis_type = request_data.get("analysis_type", "general")
    data = request_data.get("data")

    result = analyze_market_data_with_ai(data, analysis_type)
    return jsonify(result)

@ai_bp.route('/api/ai/trading-signals', methods=['POST'])
def ai_trading_signals():
    """Generate trading signals using AI."""
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    symbol = request_data.get("symbol")
    use_live_data = request_data.get("use_live_data", False)

    if not symbol:
        return jsonify({"error": "Missing 'symbol' in request body"}), 400

    try:
        if use_live_data and fyers_instance:
            # Get current quote
            quote_result = make_fyers_api_call(fyers_instance.quotes, data={"symbols": symbol})
            current_price = None
            if quote_result and isinstance(quote_result, dict):
                quote_data = quote_result.get("d", [{}])[0]
                current_price = quote_data.get("v", {}).get("lp", 0)

            # Get historical data
            import time
            end_time = int(time.time())
            start_time = end_time - (30 * 24 * 60 * 60)  # 30 days of data

            history_data = {
                "symbol": symbol,
                "resolution": "1D",
                "date_format": 0,
                "range_from": str(start_time),
                "range_to": str(end_time),
                "cont_flag": 1
            }

            history_result = make_fyers_api_call(fyers_instance.history, data=history_data)

            if history_result and isinstance(history_result, dict):
                candles = history_result.get("candles", [])
                if not current_price and candles:
                    current_price = candles[-1][4]  # Last close price

                signals = generate_trading_signals_with_ai(symbol, candles, current_price or 0)

                # Add technical indicators
                indicators = calculate_advanced_indicators(candles)
                signals["technical_indicators"] = indicators

                return jsonify(signals)
            else:
                return jsonify({"error": "Failed to fetch historical data"}), 500
        else:
            historical_data = request_data.get("historical_data", [])
            current_price = request_data.get("current_price", 0)

            if not historical_data:
                return jsonify({"error": "Either enable 'use_live_data' or provide 'historical_data' and 'current_price'"}), 400

            signals = generate_trading_signals_with_ai(symbol, historical_data, current_price)
            return jsonify(signals)

    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        return jsonify({"error": str(e)}), 500

@ai_bp.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """Chat with Gemini AI about trading and markets."""
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    message = request_data.get("message")
    context = request_data.get("context", {})

    if not message:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    try:
        prompt = f"""
        You are an expert trading advisor and market analyst. 
        Please provide helpful, accurate, and actionable insights.
        
        User Question: {message}
        """

        if context:
            prompt += f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"

        prompt += "\n\nProvide a clear, structured response with specific insights and recommendations where applicable."

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "question": message,
            "response": response.text,
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return jsonify({"error": str(e)}), 500

@ai_bp.route('/api/ai/portfolio-analysis', methods=['POST'])
def ai_portfolio_analysis():
    """Analyze portfolio using AI and provide recommendations."""
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    request_data = request.json
    holdings = request_data.get("holdings", [])
    risk_profile = request_data.get("risk_profile", "moderate")

    if not holdings:
        # Try to fetch from Fyers if authenticated
        if fyers_instance:
            holdings_result = make_fyers_api_call(fyers_instance.holdings)
            if holdings_result and isinstance(holdings_result, dict):
                holdings = holdings_result.get("holdings", [])

    if not holdings:
        return jsonify({"error": "No holdings data available"}), 400

    try:
        prompt = f"""
        Analyze the following portfolio and provide comprehensive recommendations:
        
        Portfolio Holdings:
        {json.dumps(holdings, indent=2)}
        
        Risk Profile: {risk_profile}
        
        Please provide:
        1. Portfolio composition analysis
        2. Risk assessment
        3. Diversification analysis
        4. Sector allocation review
        5. Rebalancing recommendations
        6. Specific buy/sell/hold recommendations for each holding
        7. New investment opportunities based on the risk profile
        8. Expected returns and risk metrics
        
        Format the response in a clear, actionable manner.
        """

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "portfolio_analysis": response.text,
            "risk_profile": risk_profile,
            "holdings_count": len(holdings),
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@ai_bp.route('/api/ai/market-summary', methods=['GET'])
def ai_market_summary():
    """Get AI-generated market summary and outlook."""
    if not gemini_model:
        return jsonify({"error": "AI model not initialized. Please set GEMINI_API_KEY."}), 503

    try:
        prompt = f"""
        Provide a comprehensive market summary for Indian markets as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}:
        
        Include:
        1. Overall market sentiment and trend
        2. Key indices performance outlook (NIFTY, SENSEX, BANKNIFTY)
        3. Sector rotation analysis
        4. Global market impact on Indian markets
        5. Key events and their potential impact
        6. FII/DII activity insights
        7. Currency and commodity outlook
        8. Top opportunities for the day/week
        9. Key risks to watch
        10. Recommended trading strategies for current market conditions
        
        Be specific with levels, percentages, and actionable insights.
        """

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "market_summary": response.text,
            "generated_at": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Market summary error: {e}")
        return jsonify({"error": str(e)}), 500
