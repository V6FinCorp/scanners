# EMA 9/15 Crossover Strategy for 15-Minute Timeframe

This directory contains a comprehensive EMA (Exponential Moving Average) crossover trading strategy implementation using the backtesting.py framework.

## Strategy Overview

The EMA 9/15 crossover strategy is a simple yet effective momentum-based trading strategy that uses two exponential moving averages:

- **Fast EMA**: 9 periods
- **Slow EMA**: 15 periods  
- **Timeframe**: 15 minutes

### Signal Rules

**Entry Signals:**
- **Long Position**: Buy when EMA(9) crosses above EMA(15)
- **Short Position**: Sell when EMA(9) crosses below EMA(15)

**Exit Signals:**
- Close position when opposite crossover occurs

## Files Structure

```
backtest_stat/
├── ema_crossover_strategy.py    # Main strategy implementations
├── example_usage.py             # Simple usage example
├── README.md                    # This file
└── results/                     # Generated backtest results (created after running)
```

## Strategy Variants

The implementation includes three strategy variants:

### 1. EmaCrossoverStrategy (Default)
- Basic long/short strategy
- Takes both long and short positions
- No stop loss or take profit

### 2. EmaCrossoverLongOnlyStrategy  
- Long-only strategy for bull markets
- Only takes long positions
- Exits on bearish crossover

### 3. EmaCrossoverWithStopLoss
- Enhanced version with risk management
- 2% stop loss 
- 3% take profit
- Works for both long and short positions

## Requirements

Before running the strategies, ensure you have:

1. **Python 3.7+** installed
2. **Required packages**:
   ```bash
   pip install pandas numpy matplotlib bokeh
   ```

3. **Backtesting.py framework** (included in this workspace)
   - Located in `../backtesting.py-master/`
   - No additional installation needed

## Usage

### Quick Start

1. **Run the simple example**:
   ```python
   python example_usage.py
   ```

2. **Run the full strategy suite**:
   ```python
   python ema_crossover_strategy.py
   ```

### Using Your Own Data

To use your own 15-minute OHLCV data:

```python
from ema_crossover_strategy import EmaCrossoverStrategy, load_data, run_backtest

# Load your data
data = load_data("path/to/your/15min_data.csv")

# Run backtest
results, bt = run_backtest(data, EmaCrossoverStrategy)

# Display results
print(results)

# Generate plot
bt.plot()
```

### Data Format

Your CSV data should have one of these formats:

**Format 1 (Recommended):**
```csv
timestamp,open,high,low,close,volume
2024-01-01 09:15:00,100.5,101.2,100.1,100.8,12345
2024-01-01 09:30:00,100.8,101.5,100.6,101.2,15678
...
```

**Format 2 (Standard):**
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100.5,101.2,100.1,100.8,12345
2024-01-02,100.8,101.5,100.6,101.2,15678
...
```

## Strategy Optimization

The strategy includes parameter optimization capabilities:

```python
from ema_crossover_strategy import optimize_strategy

# Optimize EMA parameters
results, bt = optimize_strategy(data)
print(f"Best Fast EMA: {results._strategy.fast_ema}")
print(f"Best Slow EMA: {results._strategy.slow_ema}")
```

### Optimization Parameters

Default optimization ranges:
- Fast EMA: 5, 7, 9, 11, 13 periods
- Slow EMA: 10, 13, 16, 19, 22 periods
- Constraint: Fast EMA < Slow EMA

## Example Output

When you run the strategy, you'll see output like:

```
Backtest Results:
Start                     2024-01-01 09:15:00
End                       2024-12-31 15:00:00
Duration                   365 days 05:45:00
Exposure Time [%]                       78.45
Equity Final [$]                    112,450.00
Equity Peak [$]                     115,200.00
Return [%]                              12.45
Buy & Hold Return [%]                    8.32
Max. Drawdown [%]                       -5.67
Avg. Drawdown [%]                       -1.23
Max. Drawdown Duration      12 days 03:30:00
Avg. Drawdown Duration       1 days 14:25:00
# Trades                                   45
Win Rate [%]                            62.22
Best Trade [%]                           4.56
Worst Trade [%]                         -2.34
Avg. Trade [%]                           0.28
Max. Trade Duration          3 days 02:15:00
Avg. Trade Duration          1 days 08:30:00
Sharpe Ratio                             1.85
Sortino Ratio                            2.34
Calmar Ratio                             2.19
```

## Performance Tips

1. **Data Quality**: Use clean, gap-free 15-minute OHLCV data
2. **Timeframe**: Strategy is optimized for 15-minute bars
3. **Market Conditions**: Works best in trending markets
4. **Risk Management**: Consider using the stop-loss variant
5. **Commission**: Adjust commission rates to match your broker

## Customization

You can easily customize the strategy by modifying parameters:

```python
class CustomEmaCrossover(EmaCrossoverStrategy):
    fast_ema = 8    # Change from 9 to 8
    slow_ema = 21   # Change from 15 to 21

# Or set parameters during backtest
bt = Backtest(data, EmaCrossoverStrategy, cash=50000)
results = bt.run(fast_ema=12, slow_ema=26)
```

## Analyzing Results

Key metrics to focus on:

- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Max Drawdown**: Largest peak-to-trough loss
- **Win Rate**: Percentage of profitable trades
- **Return**: Total strategy return vs buy-and-hold

## Risk Considerations

⚠️ **Important Disclaimers:**

1. **Backtesting vs Live Trading**: Past performance doesn't guarantee future results
2. **Transaction Costs**: Include realistic commission and slippage
3. **Market Conditions**: Strategy may perform differently in various market regimes
4. **Overfitting**: Avoid over-optimizing parameters to historical data
5. **Capital Requirements**: Ensure adequate capital for drawdown periods

## Troubleshooting

**Common Issues:**

1. **Import Errors**: Ensure backtesting.py path is correct
2. **Data Format**: Check CSV column names match expected format  
3. **Empty Results**: Verify data has sufficient history (>50 bars)
4. **Plot Errors**: Install bokeh for visualization: `pip install bokeh`

**Getting Help:**

- Check the original backtesting.py documentation: https://kernc.github.io/backtesting.py/
- Review example files in `../backtesting.py-master/doc/examples/`

## License

This strategy implementation is provided for educational purposes. The underlying backtesting.py framework is licensed under AGPL 3.0.

## Next Steps

1. **Paper Trading**: Test with paper money before live trading
2. **Risk Management**: Implement position sizing and portfolio risk controls
3. **Multiple Timeframes**: Consider higher timeframe confirmation
4. **Additional Indicators**: Enhance with RSI, MACD, or volume analysis
5. **Walk-Forward Testing**: Validate strategy robustness over time

---

*Created for 15-minute timeframe EMA crossover analysis using backtesting.py framework*