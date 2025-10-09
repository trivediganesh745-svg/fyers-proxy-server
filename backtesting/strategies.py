# backtesting/strategies.py

import pandas as pd
import talib as ta

class TradingStrategy:
    def __init__(self, name="BaseStrategy"):
        self.name = name

    def generate_signals(self, historical_data: pd.DataFrame):
        """
        Generates buy/sell signals based on historical data.
        Returns a DataFrame with 'signal' column (-1 for sell, 0 for hold, 1 for buy).
        """
        raise NotImplementedError("Subclasses must implement generate_signals method.")

class SimpleMovingAverageCrossover(TradingStrategy):
    def __init__(self, short_window=10, long_window=30):
        super().__init__("SMACrossover")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, historical_data: pd.DataFrame):
        if len(historical_data) < max(self.short_window, self.long_window):
            return pd.DataFrame(index=historical_data.index, data={'signal': 0})

        historical_data['SMA_Short'] = ta.SMA(historical_data['close'], timeperiod=self.short_window)
        historical_data['SMA_Long'] = ta.SMA(historical_data['close'], timeperiod=self.long_window)

        # Generate signals
        signals = pd.DataFrame(index=historical_data.index)
        signals['signal'] = 0  # Default to hold

        # Buy signal: short SMA crosses above long SMA
        signals.loc[historical_data['SMA_Short'].shift(1) < historical_data['SMA_Long'].shift(1)] = 0 # Reset previous
        signals.loc[historical_data['SMA_Short'] > historical_data['SMA_Long'], 'signal'] = 1

        # Sell signal: short SMA crosses below long SMA
        signals.loc[historical_data['SMA_Short'].shift(1) > historical_data['SMA_Long'].shift(1)] = 0 # Reset previous
        signals.loc[historical_data['SMA_Short'] < historical_data['SMA_Long'], 'signal'] = -1

        # Drop NaNs from initial SMA calculation
        signals = signals.dropna()
        historical_data.drop(columns=['SMA_Short', 'SMA_Long'], inplace=True, errors='ignore') # Clean up temp cols
        return signals['signal']

class RSICrossover(TradingStrategy):
    def __init__(self, rsi_period=14, buy_threshold=30, sell_threshold=70):
        super().__init__("RSICrossover")
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(self, historical_data: pd.DataFrame):
        if len(historical_data) < self.rsi_period:
            return pd.DataFrame(index=historical_data.index, data={'signal': 0})

        historical_data['RSI'] = ta.RSI(historical_data['close'], timeperiod=self.rsi_period)

        signals = pd.DataFrame(index=historical_data.index)
        signals['signal'] = 0

        # Buy signal: RSI crosses below buy threshold
        signals.loc[historical_data['RSI'].shift(1) > self.buy_threshold] = 0 # Reset previous
        signals.loc[historical_data['RSI'] <= self.buy_threshold, 'signal'] = 1

        # Sell signal: RSI crosses above sell threshold
        signals.loc[historical_data['RSI'].shift(1) < self.sell_threshold] = 0 # Reset previous
        signals.loc[historical_data['RSI'] >= self.sell_threshold, 'signal'] = -1

        signals = signals.dropna()
        historical_data.drop(columns=['RSI'], inplace=True, errors='ignore')
        return signals['signal']

# You can add more strategies here
