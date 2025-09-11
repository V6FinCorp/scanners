"""
Configuration file for EMA Crossover Strategy
Modify these parameters to customize the strategy behavior
"""

# Strategy Parameters
STRATEGY_CONFIG = {
    # EMA Periods
    'fast_ema': 9,              # Fast EMA period
    'slow_ema': 15,             # Slow EMA period
    
    # Risk Management (for WithStopLoss variant)
    'stop_loss_pct': 0.02,      # 2% stop loss
    'take_profit_pct': 0.03,    # 3% take profit
    
    # Strategy Type
    'strategy_type': 'basic',   # Options: 'basic', 'long_only', 'with_stops'
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'cash': 100000,             # Starting capital ($)
    'commission': 0.001,        # Commission rate (0.1%)
    'margin': 1.0,              # Margin requirement (1.0 = no leverage)
    'trade_on_close': False,    # Execute trades on close vs next open
    'exclusive_orders': True,   # Only one order at a time
}

# Data Configuration
DATA_CONFIG = {
    'timeframe': '15T',         # 15-minute timeframe
    'resample_rule': {
        'Open': 'first',
        'High': 'max', 
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    },
    'min_bars': 50,             # Minimum bars required for backtest
}

# Optimization Configuration
OPTIMIZATION_CONFIG = {
    'fast_ema_range': range(5, 15, 2),    # Test 5, 7, 9, 11, 13
    'slow_ema_range': range(10, 25, 3),   # Test 10, 13, 16, 19, 22
    'maximize': 'Sharpe Ratio',           # Optimization target
    'max_tries': 50,                      # Maximum optimization iterations
    'random_state': 42,                   # For reproducible results
}

# File Paths
PATHS = {
    'data_dir': '../fetch_historical/historical_data/',
    'results_dir': './results/',
    'plots_dir': './plots/',
}

# Display Configuration
DISPLAY_CONFIG = {
    'plot_style': 'modern',     # Plot style
    'plot_width': 1200,         # Plot width in pixels
    'plot_height': 600,         # Plot height in pixels
    'show_volume': True,        # Show volume subplot
    'show_trades': True,        # Show trade markers
    'open_browser': False,      # Auto-open plots in browser
}