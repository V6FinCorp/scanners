"""
EMA 9 and 15 Crossover Strategy for 15-minute timeframe

This strategy implements a simple exponential moving average crossover system:
- Buy signal: EMA 9 crosses above EMA 15
- Sell signal: EMA 15 crosses above EMA 9
- Optimized for 15-minute duration data

Based on the backtesting.py framework from https://github.com/kernc/backtesting.py
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the backtesting module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backtesting.py-master'))

from backtesting import Strategy, Backtest
from backtesting.lib import crossover

def EMA(series, period):
    """
    Calculate Exponential Moving Average
    
    Args:
        series: Price series (typically Close prices)
        period: Number of periods for EMA calculation
    
    Returns:
        EMA values as pandas Series
    """
    return pd.Series(series).ewm(span=period, adjust=False).mean()


class EmaCrossoverStrategy(Strategy):
    """
    EMA 9 and 15 Crossover Strategy
    
    This strategy uses two exponential moving averages:
    - Fast EMA: 9 periods
    - Slow EMA: 15 periods
    
    Entry Rules:
    - Long: When EMA9 crosses above EMA15
    - Short: When EMA9 crosses below EMA15
    
    Exit Rules:
    - Close position when opposite crossover occurs
    """
    
    # Strategy parameters - can be optimized
    fast_ema = 9    # Fast EMA period
    slow_ema = 15   # Slow EMA period
    
    def init(self):
        """
        Initialize the strategy by computing the EMAs
        """
        # Calculate the two EMAs using the self.I() method for proper plotting
        self.ema_fast = self.I(EMA, self.data.Close, self.fast_ema, name='EMA_9')
        self.ema_slow = self.I(EMA, self.data.Close, self.slow_ema, name='EMA_15')
    
    def next(self):
        """
        Main strategy logic executed on each new bar
        """
        # Long signal: Fast EMA crosses above Slow EMA
        if crossover(self.ema_fast, self.ema_slow):
            # Close any existing short position and go long
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy()
        
        # Short signal: Fast EMA crosses below Slow EMA
        elif crossover(self.ema_slow, self.ema_fast):
            # Close any existing long position and go short
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell()


class EmaCrossoverLongOnlyStrategy(Strategy):
    """
    EMA 9 and 15 Crossover Strategy - Long Only Version
    
    This strategy only takes long positions:
    - Buy: When EMA9 crosses above EMA15
    - Sell: When EMA9 crosses below EMA15
    """
    
    # Strategy parameters
    fast_ema = 9
    slow_ema = 15
    
    def init(self):
        """Initialize the strategy"""
        self.ema_fast = self.I(EMA, self.data.Close, self.fast_ema, name='EMA_9')
        self.ema_slow = self.I(EMA, self.data.Close, self.slow_ema, name='EMA_15')
    
    def next(self):
        """Main strategy logic - Long only"""
        # Buy signal: Fast EMA crosses above Slow EMA
        if crossover(self.ema_fast, self.ema_slow):
            if not self.position:
                self.buy()
        
        # Sell signal: Fast EMA crosses below Slow EMA
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position:
                self.position.close()


class EmaCrossoverWithStopLoss(Strategy):
    """
    EMA 9 and 15 Crossover Strategy with Stop Loss
    
    Enhanced version with risk management:
    - Stop Loss: 2% below entry price for long positions
    - Take Profit: 3% above entry price for long positions
    """
    
    # Strategy parameters
    fast_ema = 9
    slow_ema = 15
    stop_loss_pct = 0.02    # 2% stop loss
    take_profit_pct = 0.03  # 3% take profit
    
    def init(self):
        """Initialize the strategy"""
        self.ema_fast = self.I(EMA, self.data.Close, self.fast_ema, name='EMA_9')
        self.ema_slow = self.I(EMA, self.data.Close, self.slow_ema, name='EMA_15')
    
    def next(self):
        """Main strategy logic with stop loss and take profit"""
        current_price = self.data.Close[-1]
        
        # Entry signals
        if crossover(self.ema_fast, self.ema_slow):
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                # Calculate stop loss and take profit levels
                stop_loss_price = current_price * (1 - self.stop_loss_pct)
                take_profit_price = current_price * (1 + self.take_profit_pct)
                self.buy(sl=stop_loss_price, tp=take_profit_price)
        
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                # Calculate stop loss and take profit levels for short
                stop_loss_price = current_price * (1 + self.stop_loss_pct)
                take_profit_price = current_price * (1 - self.take_profit_pct)
                self.sell(sl=stop_loss_price, tp=take_profit_price)


def load_data(file_path):
    """
    Load OHLCV data from CSV file and prepare it for backtesting
    
    Args:
        file_path: Path to CSV file containing OHLCV data
    
    Returns:
        DataFrame with proper OHLCV format for backtesting
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Handle different CSV formats
        if 'timestamp' in df.columns:
            # Format from fetch_historical data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Rename columns to match backtesting.py requirements
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
        else:
            # Assume standard format with Date column or index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif df.index.dtype == 'object':
                df.index = pd.to_datetime(df.index)
        
        # Ensure all required columns exist with proper names
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column {col} not found in data")
        
        # Filter to 15-minute data if needed (depends on your data frequency)
        # If data is already 15-min, this will have no effect
        # If data is 5-min or 1-min, this will resample to 15-min
        if len(df) > 1000:  # Assume high frequency data needs resampling
            df_15min = df.resample('15T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min', 
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            print(f"Resampled data from {len(df)} to {len(df_15min)} bars (15-minute)")
            return df_15min
        
        return df
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def run_backtest(data, strategy_class=EmaCrossoverStrategy, **kwargs):
    """
    Run backtest with the given data and strategy
    
    Args:
        data: OHLCV DataFrame
        strategy_class: Strategy class to use
        **kwargs: Additional parameters for Backtest
    
    Returns:
        Backtest results and Backtest object
    """
    # Default backtest parameters
    default_params = {
        'cash': 100000,         # Starting capital
        'commission': 0.001,    # 0.1% commission
        'margin': 1.0,          # No leverage
        'trade_on_close': False, # Trade on next bar open
    }
    
    # Update with user-provided parameters
    default_params.update(kwargs)
    
    # Create and run backtest
    bt = Backtest(data, strategy_class, **default_params)
    results = bt.run()
    
    return results, bt


def optimize_strategy(data, strategy_class=EmaCrossoverStrategy, **kwargs):
    """
    Optimize strategy parameters
    
    Args:
        data: OHLCV DataFrame
        strategy_class: Strategy class to optimize
        **kwargs: Additional parameters for optimization
    
    Returns:
        Optimization results
    """
    bt = Backtest(data, strategy_class, cash=100000, commission=0.001)
    
    # Default optimization parameters
    optimization_params = {
        'fast_ema': range(5, 15, 2),     # Test EMA periods 5, 7, 9, 11, 13
        'slow_ema': range(10, 25, 3),    # Test EMA periods 10, 13, 16, 19, 22
        'maximize': 'Sharpe Ratio',       # Optimize for Sharpe ratio
        'constraint': lambda p: p.fast_ema < p.slow_ema,  # Ensure fast < slow
        'max_tries': 50,                  # Limit optimization iterations
    }
    
    # Update with user parameters
    optimization_params.update(kwargs)
    
    # Run optimization
    results = bt.optimize(**optimization_params)
    
    return results, bt


def main():
    """
    Example usage of the EMA crossover strategy
    """
    # Path to sample data file
    data_path = "../fetch_historical/historical_data/RELIANCE/RELIANCE_5_combined.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please ensure you have historical data available.")
        return
    
    # Load data
    print("Loading data...")
    data = load_data(data_path)
    
    if data is None:
        print("Failed to load data.")
        return
    
    print(f"Loaded {len(data)} bars of data")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Data columns: {data.columns.tolist()}")
    
    # Run basic strategy
    print("\n" + "="*50)
    print("Running EMA 9/15 Crossover Strategy...")
    print("="*50)
    
    results, bt = run_backtest(data, EmaCrossoverStrategy)
    print("\nBacktest Results:")
    print(results)
    
    # Run long-only strategy
    print("\n" + "="*50)
    print("Running EMA 9/15 Long-Only Strategy...")
    print("="*50)
    
    results_long, bt_long = run_backtest(data, EmaCrossoverLongOnlyStrategy)
    print("\nLong-Only Strategy Results:")
    print(results_long)
    
    # Run strategy with stop loss
    print("\n" + "="*50)
    print("Running EMA 9/15 Strategy with Stop Loss...")
    print("="*50)
    
    results_sl, bt_sl = run_backtest(data, EmaCrossoverWithStopLoss)
    print("\nStrategy with Stop Loss Results:")
    print(results_sl)
    
    # Optimization example
    print("\n" + "="*50)
    print("Optimizing Strategy Parameters...")
    print("="*50)
    
    try:
        opt_results, bt_opt = optimize_strategy(data, EmaCrossoverStrategy)
        print("\nOptimization Results:")
        print(opt_results)
        print(f"\nBest parameters: EMA Fast={opt_results._strategy.fast_ema}, EMA Slow={opt_results._strategy.slow_ema}")
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Generate plots (optional)
    try:
        print("\nGenerating plots...")
        bt.plot(filename='ema_crossover_backtest.html', open_browser=False)
        print("Plot saved as 'ema_crossover_backtest.html'")
    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()