# backtesting/engine.py

import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Any

class BacktestingEngine:
    def __init__(self, initial_capital=100000.0, transaction_cost_percent=0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transaction_cost_percent = transaction_cost_percent
        self.positions = {}  # {symbol: quantity}
        self.portfolio_history = []
        self.trades = [] # List of executed trades
        self.portfolio_value_history = []
        self.current_date = None

    def _calculate_portfolio_value(self, current_prices: Dict[str, float]):
        """Calculates current portfolio value based on positions and current prices."""
        market_value = sum(self.positions.get(symbol, 0) * current_prices.get(symbol, 0)
                           for symbol in self.positions if self.positions.get(symbol, 0) != 0)
        return self.capital + market_value

    def run_backtest(self, historical_data: Dict[str, pd.DataFrame], strategy, resolution: str):
        """
        Runs the backtest.
        :param historical_data: A dictionary like {symbol: DataFrame of OHLCV data}.
                                DataFrames must have 'open', 'high', 'low', 'close', 'volume'
                                and a DatetimeIndex.
        :param strategy: An instance of a TradingStrategy.
        :param resolution: The resolution of the data (e.g., "1", "5", "D"). Used for date ranges.
        """
        if not historical_data:
            raise ValueError("No historical data provided for backtesting.")
        
        # Ensure all dataframes have the same index for alignment
        first_symbol = next(iter(historical_data))
        aligned_index = historical_data[first_symbol].index
        for symbol, df in historical_data.items():
            if not df.index.equals(aligned_index):
                raise ValueError(f"Indices of historical data for {symbol} do not match. Ensure consistent timeframes.")

        # Generate signals for each symbol
        signals_by_symbol = {}
        for symbol, df in historical_data.items():
            app.logger.info(f"Generating signals for {symbol} using strategy {strategy.name}")
            signals_by_symbol[symbol] = strategy.generate_signals(df.copy()) # Pass a copy to avoid modifying original df

        # Main backtesting loop (iterate through time)
        for i, current_date in enumerate(aligned_index):
            self.current_date = current_date
            current_prices = {}
            for symbol, df in historical_data.items():
                if i < len(df):
                    current_prices[symbol] = df.iloc[i]['close']
                else:
                    current_prices[symbol] = df.iloc[-1]['close'] # Use last known price if data runs out for a symbol

            # Record portfolio value at the start of each period
            self.portfolio_value_history.append({
                'date': current_date,
                'portfolio_value': self._calculate_portfolio_value(current_prices),
                'capital': self.capital,
                'positions': {s: q for s, q in self.positions.items() if q != 0}
            })

            for symbol in historical_data.keys():
                if symbol not in signals_by_symbol or i >= len(signals_by_symbol[symbol]):
                    continue # No signal for this symbol at this time

                signal = signals_by_symbol[symbol].iloc[i]
                current_price = historical_data[symbol].iloc[i]['close']

                if signal == 1: # Buy signal
                    # For simplicity, buy with a fixed percentage of capital
                    # In a real system, you'd calculate exact quantity based on risk management
                    if self.capital > 0:
                        buy_amount = self.capital * 0.1 # Buy 10% of available capital
                        quantity_to_buy = int(buy_amount / current_price)
                        if quantity_to_buy > 0:
                            cost = quantity_to_buy * current_price
                            transaction_cost = cost * self.transaction_cost_percent
                            if self.capital >= (cost + transaction_cost):
                                self.capital -= (cost + transaction_cost)
                                self.positions[symbol] = self.positions.get(symbol, 0) + quantity_to_buy
                                self.trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'type': 'BUY',
                                    'price': current_price,
                                    'quantity': quantity_to_buy,
                                    'cost': cost,
                                    'transaction_cost': transaction_cost,
                                    'capital_after_trade': self.capital
                                })
                                # app.logger.info(f"BUY {quantity_to_buy} of {symbol} at {current_price} on {current_date}")

                elif signal == -1: # Sell signal
                    if self.positions.get(symbol, 0) > 0:
                        quantity_to_sell = self.positions[symbol] # Sell all
                        
                        revenue = quantity_to_sell * current_price
                        transaction_cost = revenue * self.transaction_cost_percent

                        self.capital += (revenue - transaction_cost)
                        self.positions[symbol] = 0 # Fully exited
                        self.trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'type': 'SELL',
                            'price': current_price,
                            'quantity': quantity_to_sell,
                            'revenue': revenue,
                            'transaction_cost': transaction_cost,
                            'capital_after_trade': self.capital
                        })
                        # app.logger.info(f"SELL {quantity_to_sell} of {symbol} at {current_price} on {current_date}")

        # After loop, liquidate any remaining positions to get final capital
        if any(self.positions.values()):
            app.logger.info("Liquidating remaining positions at the end of backtest.")
            for symbol, quantity in list(self.positions.items()): # Iterate over copy
                if quantity > 0:
                    last_price = historical_data[symbol].iloc[-1]['close'] # Use last available close price
                    revenue = quantity * last_price
                    transaction_cost = revenue * self.transaction_cost_percent
                    self.capital += (revenue - transaction_cost)
                    self.positions[symbol] = 0
                    self.trades.append({
                        'date': self.current_date, # Use the last processed date
                        'symbol': symbol,
                        'type': 'FINAL_SELL',
                        'price': last_price,
                        'quantity': quantity,
                        'revenue': revenue,
                        'transaction_cost': transaction_cost,
                        'capital_after_trade': self.capital
                    })

        self.portfolio_value_history.append({
            'date': self.current_date,
            'portfolio_value': self.capital,
            'capital': self.capital,
            'positions': {}
        })

        app.logger.info(f"Backtest complete for strategy {strategy.name}. Final Capital: {self.capital}")

    def generate_report(self):
        """Generates a performance report."""
        final_capital = self.capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        # Convert portfolio history to DataFrame for easier analysis
        if not self.portfolio_value_history:
            return {
                "initial_capital": self.initial_capital,
                "final_capital": final_capital,
                "total_return_percent": total_return,
                "message": "No portfolio history generated. Possibly no data or trades."
            }

        portfolio_df = pd.DataFrame(self.portfolio_value_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        # Calculate drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min() * 100 if not portfolio_df.empty else 0

        # Annualized Return (simple, assuming data covers enough period)
        start_date = portfolio_df.index.min()
        end_date = portfolio_df.index.max()
        num_years = (end_date - start_date).days / 365.25 if not portfolio_df.empty and (end_date - start_date).days > 0 else 1
        annualized_return = ((1 + total_return / 100) ** (1 / num_years) - 1) * 100 if num_years > 0 else total_return

        # Volatility (Standard deviation of daily returns)
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        annualized_volatility = returns.std() * np.sqrt(252) * 100 if not returns.empty else 0 # Assuming 252 trading days

        # Sharpe Ratio (Requires a risk-free rate, simplifying here)
        # For simplicity, risk_free_rate = 0 for now. In real-world, use bond yield.
        risk_free_rate = 0.0 # 0%
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan
        
        trades_df = pd.DataFrame(self.trades)

        return {
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return_percent": round(total_return, 2),
            "annualized_return_percent": round(annualized_return, 2),
            "max_drawdown_percent": round(max_drawdown, 2),
            "annualized_volatility_percent": round(annualized_volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else "N/A",
            "number_of_trades": len(self.trades),
            "portfolio_history": portfolio_df[['portfolio_value', 'capital']].reset_index().to_dict(orient='records'),
            "trades": trades_df.to_dict(orient='records')
        }

# You'll need to make app.logger available in this module if you want to use it here.
# For now, it's defined in app.py. We will pass it or re-import it.
# A simpler way is to have Flask log through app.logger via the request context or blueprint.
# For this example, we'll assume app.logger is accessible or print instead.
import logging
app_logger = logging.getLogger(__name__) # Use standard Python logging within this module
app_logger.setLevel(logging.INFO)
