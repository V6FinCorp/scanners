"""
Example script to run EMA 9/15 crossover strategy
Simple usage example for the EMA crossover strategy
"""

import pandas as pd
import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backtesting.py-master'))
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def EMA(series, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(series).ewm(span=period, adjust=False).mean()

class SimpleEmaCrossover(Strategy):
    """Simple EMA 9/15 crossover strategy for 15-minute timeframe"""
    
    def init(self):
        # Calculate EMAs
        self.ema9 = self.I(EMA, self.data.Close, 9, name='EMA_9')
        self.ema15 = self.I(EMA, self.data.Close, 15, name='EMA_15')
    
    def next(self):
        # Buy when EMA9 crosses above EMA15
        if crossover(self.ema9, self.ema15):
            if self.position.is_short:
                self.position.close()
            self.buy()
        
        # Sell when EMA15 crosses above EMA9
        elif crossover(self.ema15, self.ema9):
            if self.position.is_long:
                self.position.close()
            self.sell()

def load_sample_data():
    """Load sample data from the project"""
    # Try to load RELIANCE data
    data_path = "../fetch_historical/historical_data/RELIANCE/RELIANCE_5_combined.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Rename columns for backtesting.py
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low', 
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # Resample to 15-minute if data is 5-minute
        if len(df) > 500:  # Likely 5-minute data
            df_15min = df.resample('15T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            print(f"Resampled from {len(df)} to {len(df_15min)} 15-minute bars")
            return df_15min
        
        return df
    
    else:
        # Create synthetic data for demonstration
        print("Sample data not found, creating synthetic data...")
        dates = pd.date_range('2024-01-01', periods=200, freq='15T')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        open_prices = close_prices + np.random.randn(200) * 0.2
        high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(200) * 0.3)
        low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(200) * 0.3)
        volumes = np.random.randint(1000, 10000, 200)
        
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
        
        return df

def run_example():
    """Run the EMA crossover strategy example"""
    print("EMA 9/15 Crossover Strategy Example")
    print("=" * 40)
    
    # Load data
    data = load_sample_data()
    print(f"Loaded {len(data)} bars of 15-minute data")
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    
    # Setup backtest
    bt = Backtest(
        data, 
        SimpleEmaCrossover, 
        cash=100000,        # $100,000 starting capital
        commission=0.001,   # 0.1% commission
    )
    
    # Run backtest
    print("\nRunning backtest...")
    results = bt.run()
    
    # Display results
    print("\nBacktest Results:")
    print("-" * 30)
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {results['# Trades']}")
    print(f"Win Rate: {results['Win Rate [%]']:.1f}%")
    print(f"Final Equity: ${results['Equity Final [$]']:,.2f}")
    
    # Show detailed stats
    print(f"\nDetailed Statistics:")
    print(f"Start Date: {results['Start']}")
    print(f"End Date: {results['End']}")
    print(f"Duration: {results['Duration']}")
    print(f"Exposure Time: {results['Exposure Time [%]']:.1f}%")
    print(f"Equity Peak: ${results['Equity Peak [$]']:,.2f}")
    print(f"Best Trade: {results['Best Trade [%]']:.2f}%")
    print(f"Worst Trade: {results['Worst Trade [%]']:.2f}%")
    print(f"Average Trade: {results['Avg. Trade [%]']:.2f}%")
    print(f"SQN: {results['SQN']:.2f}")
    
    # Try to create plot
    try:
        print("\nGenerating plot...")
        bt.plot(filename='ema_crossover_example.html', open_browser=False)
        print("Plot saved as 'ema_crossover_example.html'")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    # Show strategy parameters
    print(f"\nStrategy Configuration:")
    print(f"Fast EMA: 9 periods")
    print(f"Slow EMA: 15 periods") 
    print(f"Timeframe: 15 minutes")
    
    return results, bt

if __name__ == "__main__":
    run_example()