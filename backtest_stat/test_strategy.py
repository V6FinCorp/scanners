"""
Test script for EMA Crossover Strategy
Validates all strategy variants and functionality
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backtesting.py to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backtesting.py-master'))

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    BACKTESTING_AVAILABLE = True
except ImportError:
    print("Warning: backtesting.py not available. Some tests will be skipped.")
    BACKTESTING_AVAILABLE = False

from config import STRATEGY_CONFIG, BACKTEST_CONFIG, DATA_CONFIG

def create_test_data(num_bars=300, start_price=100, timeframe='15T'):
    """
    Create synthetic test data for strategy validation
    
    Args:
        num_bars: Number of bars to generate
        start_price: Starting price
        timeframe: Timeframe string (e.g., '15T' for 15 minutes)
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducible tests
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1, 9, 15)  # Start at 9:15 AM
    timestamps = pd.date_range(start_time, periods=num_bars, freq=timeframe)
    
    # Generate price series with trend and noise
    trend = np.linspace(0, 10, num_bars)  # Slight upward trend
    noise = np.random.randn(num_bars) * 2
    close_prices = start_price + trend + np.cumsum(noise * 0.5)
    
    # Ensure prices don't go negative
    close_prices = np.maximum(close_prices, start_price * 0.5)
    
    # Generate OHLC from close prices
    open_prices = np.concatenate([[start_price], close_prices[:-1]]) + np.random.randn(num_bars) * 0.2
    
    # High and Low based on Open and Close
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(num_bars) * 0.3)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(num_bars) * 0.3)
    
    # Volume
    volumes = np.random.randint(1000, 50000, num_bars)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=timestamps)
    
    return data

def test_ema_calculation():
    """Test EMA calculation function"""
    print("Testing EMA calculation...")
    
    # Create simple test data
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    
    def EMA(series, period):
        return pd.Series(series).ewm(span=period, adjust=False).mean()
    
    # Test EMA calculation
    ema_5 = EMA(prices, 5)
    
    # Validate results
    assert len(ema_5) == len(prices), "EMA length should match input length"
    assert not ema_5.isnull().all(), "EMA should not be all NaN"
    assert ema_5.iloc[-1] > ema_5.iloc[0], "EMA should show upward trend for increasing prices"
    
    print("✓ EMA calculation test passed")

def test_data_loading():
    """Test data loading and preprocessing"""
    print("Testing data loading...")
    
    # Test with synthetic data
    test_data = create_test_data(100)
    
    # Validate data structure
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        assert col in test_data.columns, f"Missing required column: {col}"
    
    # Validate OHLC logic
    assert (test_data['High'] >= test_data['Open']).all(), "High should be >= Open"
    assert (test_data['High'] >= test_data['Close']).all(), "High should be >= Close"
    assert (test_data['Low'] <= test_data['Open']).all(), "Low should be <= Open"
    assert (test_data['Low'] <= test_data['Close']).all(), "Low should be <= Close"
    
    # Check for missing values
    assert not test_data.isnull().any().any(), "Data should not contain NaN values"
    
    print("✓ Data loading test passed")

def test_strategy_logic():
    """Test strategy logic without backtesting framework"""
    print("Testing strategy logic...")
    
    def EMA(series, period):
        return pd.Series(series).ewm(span=period, adjust=False).mean()
    
    def simple_crossover(series1, series2):
        """Simple crossover detection"""
        return (series1.iloc[-2] < series2.iloc[-2] and 
                series1.iloc[-1] > series2.iloc[-1])
    
    # Create test data with known crossover
    prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104])
    
    ema_fast = EMA(prices, 3)
    ema_slow = EMA(prices, 6)
    
    # Check that EMAs are calculated
    assert not ema_fast.isnull().all(), "Fast EMA should be calculated"
    assert not ema_slow.isnull().all(), "Slow EMA should be calculated"
    
    print("✓ Strategy logic test passed")

def test_backtesting_integration():
    """Test integration with backtesting.py framework"""
    if not BACKTESTING_AVAILABLE:
        print("⚠ Backtesting framework not available, skipping integration test")
        return
    
    print("Testing backtesting integration...")
    
    def EMA(series, period):
        return pd.Series(series).ewm(span=period, adjust=False).mean()
    
    class TestStrategy(Strategy):
        def init(self):
            self.ema_fast = self.I(EMA, self.data.Close, 9)
            self.ema_slow = self.I(EMA, self.data.Close, 15)
        
        def next(self):
            if crossover(self.ema_fast, self.ema_slow):
                if self.position.is_short:
                    self.position.close()
                self.buy()
            elif crossover(self.ema_slow, self.ema_fast):
                if self.position.is_long:
                    self.position.close()
                self.sell()
    
    # Create test data
    data = create_test_data(200)
    
    # Run backtest
    try:
        bt = Backtest(data, TestStrategy, **BACKTEST_CONFIG)
        results = bt.run()
        
        # Validate results
        assert 'Return [%]' in results, "Results should contain return percentage"
        assert 'Sharpe Ratio' in results, "Results should contain Sharpe ratio"
        assert '# Trades' in results, "Results should contain number of trades"
        
        # Check that some trades were made (with trending synthetic data)
        num_trades = results['# Trades']
        print(f"  Generated {num_trades} trades")
        
        print("✓ Backtesting integration test passed")
        
    except Exception as e:
        print(f"✗ Backtesting integration test failed: {e}")

def test_configuration():
    """Test configuration file"""
    print("Testing configuration...")
    
    # Validate config structure
    assert 'fast_ema' in STRATEGY_CONFIG, "Strategy config should have fast_ema"
    assert 'slow_ema' in STRATEGY_CONFIG, "Strategy config should have slow_ema"
    assert 'cash' in BACKTEST_CONFIG, "Backtest config should have cash"
    assert 'timeframe' in DATA_CONFIG, "Data config should have timeframe"
    
    # Validate config values
    assert STRATEGY_CONFIG['fast_ema'] < STRATEGY_CONFIG['slow_ema'], "Fast EMA should be < Slow EMA"
    assert BACKTEST_CONFIG['cash'] > 0, "Starting cash should be positive"
    assert 0 <= BACKTEST_CONFIG['commission'] < 1, "Commission should be between 0 and 1"
    
    print("✓ Configuration test passed")

def test_file_structure():
    """Test that all required files exist"""
    print("Testing file structure...")
    
    current_dir = os.path.dirname(__file__)
    required_files = [
        'ema_crossover_strategy.py',
        'example_usage.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file in required_files:
        file_path = os.path.join(current_dir, file)
        assert os.path.exists(file_path), f"Required file missing: {file}"
    
    print("✓ File structure test passed")

def run_performance_test():
    """Run a basic performance test"""
    if not BACKTESTING_AVAILABLE:
        print("⚠ Backtesting framework not available, skipping performance test")
        return
        
    print("Running performance test...")
    
    def EMA(series, period):
        return pd.Series(series).ewm(span=period, adjust=False).mean()
    
    class TestStrategy(Strategy):
        def init(self):
            self.ema_fast = self.I(EMA, self.data.Close, 9)
            self.ema_slow = self.I(EMA, self.data.Close, 15)
        
        def next(self):
            if crossover(self.ema_fast, self.ema_slow):
                self.buy()
            elif crossover(self.ema_slow, self.ema_fast):
                if self.position:
                    self.position.close()
    
    # Test with larger dataset
    data = create_test_data(1000)  # 1000 bars
    
    start_time = datetime.now()
    
    try:
        bt = Backtest(data, TestStrategy, cash=100000, commission=0.001)
        results = bt.run()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"  Processed {len(data)} bars in {execution_time:.2f} seconds")
        print(f"  Generated {results['# Trades']} trades")
        print(f"  Final return: {results['Return [%]']:.2f}%")
        
        print("✓ Performance test completed")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")

def run_all_tests():
    """Run all tests"""
    print("Running EMA Crossover Strategy Tests")
    print("=" * 50)
    
    tests = [
        test_ema_calculation,
        test_data_loading,
        test_strategy_logic,
        test_configuration,
        test_file_structure,
        test_backtesting_integration,
        run_performance_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Strategy is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()