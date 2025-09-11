"""
RELIANCE Stock Analysis - EMA 9/15 Crossover Strategy
Comprehensive backtest with multiple configurations
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

# Add the backtesting.py path to sys.path for imports
backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'backtesting.py-master')
if backtesting_path not in sys.path:
    sys.path.insert(0, backtesting_path)

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

@dataclass
class EMAConfig:
    """Configuration for EMA strategy parameters"""
    timeframe: str
    risk_reward_ratio: str
    stop_type: str  # 'fixed' or 'trailing'
    base_stop_pct: float = 0.01  # 1% base stop loss

class EMAFixedStopStrategy(Strategy):
    """EMA 9/15 crossover strategy with fixed stop loss"""

    # Strategy parameters
    ema_fast_period = 9
    ema_slow_period = 15
    risk_reward_ratio = 1.0
    base_stop_pct = 0.01

    def init(self):
        """Initialize strategy indicators"""
        # Calculate EMAs
        self.ema_fast = self.I(calculate_ema, pd.Series(self.data.Close), self.ema_fast_period)
        self.ema_slow = self.I(calculate_ema, pd.Series(self.data.Close), self.ema_slow_period)

    def next(self):
        """Execute strategy logic on each bar"""
        price = self.data.Close[-1]

        # Calculate stop loss and take profit levels
        stop_loss_pct = self.base_stop_pct
        take_profit_pct = stop_loss_pct * self.risk_reward_ratio

        # Long signal: EMA 9 crosses above EMA 15
        if crossover(self.ema_fast, self.ema_slow):
            if self.position.is_short:
                self.position.close()

            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)

            self.buy(sl=stop_loss, tp=take_profit)

        # Short signal: EMA 15 crosses above EMA 9 (EMA 9 crosses below EMA 15)
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position.is_long:
                self.position.close()

            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)

            self.sell(sl=stop_loss, tp=take_profit)

class EMATrailingStopStrategy(Strategy):
    """EMA 9/15 crossover strategy with trailing stop loss"""

    # Strategy parameters
    ema_fast_period = 9
    ema_slow_period = 15
    risk_reward_ratio = 1.0
    base_stop_pct = 0.01

    def init(self):
        """Initialize strategy indicators and variables"""
        # Calculate EMAs
        self.ema_fast = self.I(calculate_ema, pd.Series(self.data.Close), self.ema_fast_period)
        self.ema_slow = self.I(calculate_ema, pd.Series(self.data.Close), self.ema_slow_period)

        # Trailing stop variables
        self.trailing_stop_long = None
        self.trailing_stop_short = None
        self.entry_price = None

    def next(self):
        """Execute strategy logic with trailing stops"""
        price = self.data.Close[-1]

        # Calculate stop loss percentage
        stop_loss_pct = self.base_stop_pct

        # Long signal: EMA 9 crosses above EMA 15
        if crossover(self.ema_fast, self.ema_slow):
            if self.position.is_short:
                self.position.close()

            self.entry_price = price
            self.trailing_stop_long = price * (1 - stop_loss_pct)
            self.trailing_stop_short = None

            take_profit = price * (1 + stop_loss_pct * self.risk_reward_ratio)
            self.buy(tp=take_profit)

        # Short signal: EMA 15 crosses above EMA 9
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position.is_long:
                self.position.close()

            self.entry_price = price
            self.trailing_stop_short = price * (1 + stop_loss_pct)
            self.trailing_stop_long = None

            take_profit = price * (1 - stop_loss_pct * self.risk_reward_ratio)
            self.sell(tp=take_profit)

        # Update trailing stops
        if self.position.is_long and self.trailing_stop_long is not None:
            # Update trailing stop for long position
            new_stop = price * (1 - stop_loss_pct)
            if new_stop > self.trailing_stop_long:
                self.trailing_stop_long = new_stop

            # Check if price hits trailing stop
            if price <= self.trailing_stop_long:
                self.position.close()

        elif self.position.is_short and self.trailing_stop_short is not None:
            # Update trailing stop for short position
            new_stop = price * (1 + stop_loss_pct)
            if new_stop < self.trailing_stop_short:
                self.trailing_stop_short = new_stop

            # Check if price hits trailing stop
            if price >= self.trailing_stop_short:
                self.position.close()

class RELIANCEAnalyzer:
    """Main analyzer class for RELIANCE EMA strategy backtesting"""

    def __init__(self):
        self.data_path = r"E:\Mark\V6\code_zone\vs_ws_pycode\fetch_historical\historical_data\RELIANCE\RELIANCE_5_combined.csv"
        self.results = {}
        self.raw_data = None

        # Configuration matrix
        self.timeframes = ['15min', '30min']  # 15min, 30min
        self.risk_ratios = ['1:1', '1:2', '1:3']
        self.stop_types = ['fixed', 'trailing']

        print("🔥 RELIANCE EMA Crossover Strategy Analysis")
        print("=" * 50)
        print("📈 EMA 9/15 crossover with multiple configurations")
        print("⚙️  Timeframes: 15min, 30min")
        print("🎯 Risk ratios: 1:1, 1:2, 1:3")
        print("🛡️  Stop types: Fixed, Trailing")
        print("=" * 50)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare RELIANCE data"""
        print("📊 Loading RELIANCE data...")

        try:
            # Load CSV data
            df = pd.read_csv(self.data_path)

            # Parse timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Rename columns to match backtesting.py format
            df.columns = [col.title() for col in df.columns]

            # Store raw data
            self.raw_data = df.copy()

            print(f"✅ Loaded {len(df)} bars of 5-minute data")
            print(f"📅 Data range: {df.index[0]} to {df.index[-1]}")
            print(f"📊 Price range: {df['Close'].min()} to {df['Close'].max()}")

            return df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        resampled = df.resample(timeframe).agg(agg_dict)
        resampled = resampled.dropna()

        print(f"📈 Resampled to {timeframe}: {len(resampled)} bars")
        return resampled

    def run_backtest(self, data: pd.DataFrame, config: EMAConfig) -> Dict[str, Any]:
        """Run backtest with given configuration"""
        try:
            # Select strategy class based on stop type
            if config.stop_type == 'fixed':
                strategy_class = EMAFixedStopStrategy
            else:
                strategy_class = EMATrailingStopStrategy

            # Parse risk ratio
            risk_ratio = float(config.risk_reward_ratio.split(':')[1])

            # Create backtest instance
            bt = Backtest(
                data,
                strategy_class,
                commission=0.001,  # 0.1% commission
                cash=100000  # ₹1,00,000 starting capital
            )

            # Run backtest with parameters
            result = bt.run(
                risk_reward_ratio=risk_ratio,
                base_stop_pct=config.base_stop_pct
            )

            return {
                'config': config,
                'backtest_result': result,
                'equity_curve': result._equity_curve if hasattr(result, '_equity_curve') else None,
                'trades': result._trades if hasattr(result, '_trades') else None
            }

        except Exception as e:
            print(f"❌ Backtest failed for {config}: {e}")
            return None

    def run_comprehensive_analysis(self):
        """Run analysis across all configurations"""
        print("\n🚀 Starting Comprehensive RELIANCE Analysis...")
        print("=" * 60)

        # Load data
        df = self.load_data()
        if df is None:
            return False

        config_count = 0
        total_configs = len(self.timeframes) * len(self.risk_ratios) * len(self.stop_types)

        # Iterate through all configurations
        for timeframe in self.timeframes:
            print(f"\n📊 Analyzing {timeframe} timeframe...")

            # Resample data for current timeframe
            resampled_data = self.resample_data(df, timeframe)

            for risk_ratio in self.risk_ratios:
                for stop_type in self.stop_types:
                    config_count += 1

                    print(f"\n⚙️  Configuration {config_count}/{total_configs}: {timeframe}, Risk {risk_ratio}")

                    # Create configuration
                    config = EMAConfig(
                        timeframe=timeframe,
                        risk_reward_ratio=risk_ratio,
                        stop_type=stop_type
                    )

                    # Run backtest
                    if stop_type == 'fixed':
                        print("   🎯 Running Fixed Stop Loss strategy...")
                    else:
                        print("   🔄 Running Trailing Stop Loss strategy...")

                    result = self.run_backtest(resampled_data, config)

                    if result:
                        # Store result
                        config_key = f"{timeframe}_{stop_type}_{risk_ratio}"
                        backtest_result = result['backtest_result']

                        # Safely extract metrics with defaults
                        self.results[config_key] = {
                            'config_name': f"{timeframe} {stop_type.title()} {risk_ratio}",
                            'timeframe': timeframe,
                            'risk_ratio': risk_ratio,
                            'stop_type': stop_type.title(),
                            'total_return': backtest_result.get('Return [%]', 0),
                            'buy_hold_return': backtest_result.get('Buy & Hold Return [%]', 0),
                            'sharpe_ratio': backtest_result.get('Sharpe Ratio', 0),
                            'max_drawdown': abs(backtest_result.get('Max. Drawdown [%]', 0)),
                            'win_rate': backtest_result.get('Win Rate [%]', 0),
                            'profit_factor': backtest_result.get('Profit Factor', 1),
                            'num_trades': backtest_result.get('# Trades', 0),
                            'avg_trade': backtest_result.get('Avg. Trade [%]', 0),
                            'max_trade': backtest_result.get('Max. Trade [%]', 0),
                            'calmar_ratio': backtest_result.get('Calmar Ratio', 0),
                            'start_value': backtest_result.get('Start', 100000),
                            'end_value': backtest_result.get('End', 100000),
                            'result_object': backtest_result
                        }

                        print(f"   ✅ Return: {backtest_result.get('Return [%]', 0):.2f}%, Trades: {backtest_result.get('# Trades', 0)}")

        print(f"\n🎉 Analysis complete! {len(self.results)} configurations tested.")
        return True

    def generate_summary_tables(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive summary tables"""
        print("\n📋 Generating summary tables...")

        if not self.results:
            return {}

        # Create main results DataFrame
        results_data = []
        for key, result in self.results.items():
            results_data.append({
                'Configuration': result['config_name'],
                'Timeframe': result['timeframe'],
                'Risk Ratio': result['risk_ratio'],
                'Stop Type': result['stop_type'],
                'Return (%)': f"{result['total_return']:.2f}",
                'B&H Return (%)': f"{result['buy_hold_return']:.2f}",
                'Sharpe': f"{result['sharpe_ratio']:.2f}",
                'Max DD (%)': f"{result['max_drawdown']:.2f}",
                'Win Rate (%)': f"{result['win_rate']:.1f}",
                'Profit Factor': f"{result['profit_factor']:.2f}",
                'Trades': result['num_trades'],
                'Avg Trade (%)': f"{result['avg_trade']:.2f}",
                'Calmar': f"{result['calmar_ratio']:.2f}"
            })

        main_results = pd.DataFrame(results_data)

        # Group by different criteria for comparison tables
        results_df = pd.DataFrame([result for result in self.results.values()])

        # Timeframe comparison
        timeframe_comparison = results_df.groupby('timeframe').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'num_trades': 'mean'
        }).round(2)

        # Risk ratio comparison
        risk_ratio_comparison = results_df.groupby('risk_ratio').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'num_trades': 'mean'
        }).round(2)

        # Stop type comparison
        stop_type_comparison = results_df.groupby('stop_type').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'num_trades': 'mean'
        }).round(2)

        return {
            'main_results': main_results,
            'timeframe_comparison': timeframe_comparison,
            'risk_ratio_comparison': risk_ratio_comparison,
            'stop_type_comparison': stop_type_comparison
        }

    def analyze_by_periods(self) -> Dict[str, Dict]:
        """Analyze performance by time periods"""
        print("\n📊 Analyzing performance by time periods...")

        period_analysis = {
            'daily': {},
            'weekly': {},
            'monthly': {}
        }

        # This is a simplified version - in a real implementation,
        # you'd analyze the equity curves by different time periods
        for config_key, result in self.results.items():
            # For demonstration, we'll use overall metrics
            # In practice, you'd break down returns by periods
            period_metrics = {
                'avg_return': result['avg_trade'],
                'total_return': result['total_return'],
                'max_drawdown': result['max_drawdown'],
                'sharpe_ratio': result['sharpe_ratio'],
                'num_trades': result['num_trades']
            }

            period_analysis['daily'][config_key] = period_metrics
            period_analysis['weekly'][config_key] = period_metrics
            period_analysis['monthly'][config_key] = period_metrics

        return period_analysis

    def find_best_configurations(self) -> Dict[str, Dict]:
        """Find best performing configurations by different metrics"""
        if not self.results:
            return {}

        results_df = pd.DataFrame([result for result in self.results.values()])

        best_configs = {}

        # Find best configurations by different metrics
        metrics = [
            ('highest_return', 'total_return', 'max'),
            ('best_sharpe', 'sharpe_ratio', 'max'),
            ('lowest_drawdown', 'max_drawdown', 'min'),
            ('best_win_rate', 'win_rate', 'max'),
            ('best_profit_factor', 'profit_factor', 'max')
        ]

        for metric_name, column, direction in metrics:
            if direction == 'max':
                best_idx = results_df[column].idxmax()
            else:
                best_idx = results_df[column].idxmin()

            best_configs[metric_name] = results_df.loc[best_idx].to_dict()

        return best_configs

def main():
    """Main function to run RELIANCE analysis"""
    # Create analyzer instance
    analyzer = RELIANCEAnalyzer()

    # Run comprehensive analysis
    success = analyzer.run_comprehensive_analysis()

    if not success:
        print("❌ Analysis failed!")
        return None

    # Generate summary tables
    summary_tables = analyzer.generate_summary_tables()

    # Analyze by periods
    period_analysis = analyzer.analyze_by_periods()

    # Find best configurations
    best_configs = analyzer.find_best_configurations()

    # Display summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"📈 Configurations tested: {len(analyzer.results)}")
    print(f"📅 Data period: {analyzer.raw_data.index[0].date()} to {analyzer.raw_data.index[-1].date()}")
    print(f"📊 Total data points: {len(analyzer.raw_data)}")

    # Show best configurations
    if best_configs:
        print(f"\n🏆 BEST PERFORMING CONFIGURATIONS:")
        for metric, config in best_configs.items():
            metric_name = metric.replace('_', ' ').replace('highest', 'Best').replace('best', 'Best').replace('lowest', 'Lowest').title()

            # Get appropriate value based on metric
            if 'return' in metric:
                value = f"{config['total_return']:.2f}%"
            elif 'sharpe' in metric:
                value = f"{config['sharpe_ratio']:.2f}"
            elif 'drawdown' in metric:
                value = f"{config['max_drawdown']:.2f}%"
            elif 'win_rate' in metric:
                value = f"{config['win_rate']:.1f}%"
            elif 'profit_factor' in metric:
                value = f"{config['profit_factor']:.2f}"
            else:
                value = "N/A"

            print(f"• {metric_name}: {config['config_name']} ({value})")

    # Return data for PDF generation
    return {
        'analyzer': analyzer,
        'summary_tables': summary_tables,
        'period_analysis': period_analysis,
        'best_configs': best_configs
    }

if __name__ == "__main__":
    main()