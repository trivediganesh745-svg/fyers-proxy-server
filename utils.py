import json
import datetime
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def calculate_advanced_indicators(candles: List) -> Dict[str, Any]:
    """Calculate advanced technical indicators for AI analysis."""
    if not candles or len(candles) < 2:
        return {}

    try:
        # Extract OHLCV data
        closes = [c[4] for c in candles]  # Close prices
        highs = [c[2] for c in candles]   # High prices
        lows = [c[3] for c in candles]    # Low prices
        volumes = [c[5] for c in candles] if len(candles[0]) > 5 else []

        # Calculate basic indicators
        indicators = {
            "sma_10": np.mean(closes[-10:]) if len(closes) >= 10 else None,
            "sma_20": np.mean(closes[-20:]) if len(closes) >= 20 else None,
            "sma_50": np.mean(closes[-50:]) if len(closes) >= 50 else None,
            "current_price": closes[-1],
            "price_change": closes[-1] - closes[-2] if len(closes) >= 2 else 0,
            "price_change_pct": ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 and closes[-2] != 0 else 0,
            "high_20": max(highs[-20:]) if len(highs) >= 20 else max(highs),
            "low_20": min(lows[-20:]) if len(lows) >= 20 else min(lows),
            "avg_volume": np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes) if volumes else None,
            "volume_ratio": volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 and np.mean(volumes[-20:]) != 0 else None
        }

        # Calculate RSI (14-period)
        if len(closes) >= 15:
            price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [change if change > 0 else 0 for change in price_changes]
            losses = [-change if change < 0 else 0 for change in price_changes]

            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                indicators["rsi"] = 100 - (100 / (1 + rs))
            else:
                indicators["rsi"] = 100 if avg_gain > 0 else 50

        return indicators

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

def find_option_in_chain(resp, option_symbol=None, strike=None, opt_type=None, expiry_ts=None):
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
