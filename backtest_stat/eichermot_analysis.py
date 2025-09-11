"""
EICHERMOT EMA Crossover Strategy - Comprehensive Analysis
Multi-timeframe, Multi-risk Management Backtest with Detailed Reporting

Configurations:
1. EMA 9/15 crossover with 1:1, 1:2, 1:3 risk-reward ratios
2. 15-minute and 30-minute timeframes  
3. Fixed vs Trailing stop-loss
4. Daily, Weekly, Monthly performance reports
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add backtesting.py to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backtesting.py-master'))

from backtesting import Strategy, Backtest
from backtesting.lib import crossover

class EMAConfig:
    """Configuration class for EMA strategy parameters"""
    def __init__(self):
        self.fast_ema = 9
        self.slow_ema = 15
        self.timeframes = ['15T', '30T']  # 15-min and 30-min
        self.risk_ratios = [1, 2, 3]     # 1:1, 1:2, 1:3 risk:reward
        self.stop_loss_pct = 0.01        # 1% base stop loss
        self.commission = 0.001          # 0.1% commission
        self.cash = 100000               # Starting capital

def EMA(series, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(series).ewm(span=period, adjust=False).mean()

class EMAFixedStopStrategy(Strategy):
    """EMA Strategy with Fixed Stop Loss and Take Profit"""
    
    # Parameters
    fast_ema = 9
    slow_ema = 15
    risk_ratio = 1  # Risk:Reward ratio
    stop_loss_pct = 0.01
    
    def init(self):
        self.ema_fast = self.I(EMA, self.data.Close, self.fast_ema, name=f'EMA_{self.fast_ema}')
        self.ema_slow = self.I(EMA, self.data.Close, self.slow_ema, name=f'EMA_{self.slow_ema}')
        
    def next(self):
        current_price = self.data.Close[-1]
        
        # Long signal: Fast EMA crosses above Slow EMA
        if crossover(self.ema_fast, self.ema_slow):
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.stop_loss_pct * self.risk_ratio)
                self.buy(sl=stop_loss, tp=take_profit)
        
        # Short signal: Fast EMA crosses below Slow EMA  
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.stop_loss_pct * self.risk_ratio)
                self.sell(sl=stop_loss, tp=take_profit)

class EMATrailingStopStrategy(Strategy):
    """EMA Strategy with Trailing Stop Loss"""
    
    # Parameters
    fast_ema = 9
    slow_ema = 15
    risk_ratio = 1
    stop_loss_pct = 0.01
    trailing_pct = 0.015  # 1.5% trailing stop
    
    def init(self):
        self.ema_fast = self.I(EMA, self.data.Close, self.fast_ema, name=f'EMA_{self.fast_ema}')
        self.ema_slow = self.I(EMA, self.data.Close, self.slow_ema, name=f'EMA_{self.slow_ema}')
        self.highest_price = 0
        self.lowest_price = float('inf')
        
    def next(self):
        current_price = self.data.Close[-1]
        
        # Update trailing levels
        if self.position.is_long:
            if current_price > self.highest_price:
                self.highest_price = current_price
            trailing_stop = self.highest_price * (1 - self.trailing_pct)
            if current_price <= trailing_stop:
                self.position.close()
                
        elif self.position.is_short:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            trailing_stop = self.lowest_price * (1 + self.trailing_pct)
            if current_price >= trailing_stop:
                self.position.close()
        
        # Entry signals
        if crossover(self.ema_fast, self.ema_slow):
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy()
                self.highest_price = current_price
                
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position.is_long:
                self.position.close() 
            if not self.position.is_short:
                self.sell()
                self.lowest_price = current_price

class EICHERMOTAnalyzer:
    """Main analyzer class for EICHERMOT backtesting"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.config = EMAConfig()
        self.results = {}
        self.raw_data = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess EICHERMOT data"""
        print("📊 Loading EICHERMOT data...")
        
        # Load CSV data
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Rename columns for backtesting.py compatibility
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        self.raw_data = df
        print(f"✅ Loaded {len(df)} bars of 5-minute data")
        print(f"📅 Data range: {df.index[0]} to {df.index[-1]}")
        print(f"📊 Price range: {df['Close'].min():.1f} to {df['Close'].max():.1f}")
        
    def resample_data(self, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        resampled = self.raw_data.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        print(f"📈 Resampled to {timeframe}: {len(resampled)} bars")
        return resampled
    
    def run_single_backtest(self, data: pd.DataFrame, strategy_class, **params) -> Dict:
        """Run a single backtest configuration"""
        try:
            bt = Backtest(
                data, 
                strategy_class,
                cash=self.config.cash,
                commission=self.config.commission,
                exclusive_orders=True
            )
            
            results = bt.run(**params)
            
            # Extract key metrics
            metrics = {
                'start_date': results['Start'],
                'end_date': results['End'],
                'duration': results['Duration'],
                'total_return_pct': results['Return [%]'],
                'sharpe_ratio': results['Sharpe Ratio'],
                'max_drawdown_pct': results['Max. Drawdown [%]'],
                'win_rate_pct': results['Win Rate [%]'],
                'num_trades': results['# Trades'],
                'best_trade_pct': results.get('Best Trade [%]', 0),
                'worst_trade_pct': results.get('Worst Trade [%]', 0),
                'avg_trade_pct': results.get('Avg. Trade [%]', 0),
                'profit_factor': results.get('Profit Factor', 0),
                'equity_final': results['Equity Final [$]'],
                'equity_peak': results['Equity Peak [$]'],
                'buy_hold_return_pct': results.get('Buy & Hold Return [%]', 0),
                'exposure_time_pct': results.get('Exposure Time [%]', 0),
                'sqn': results.get('SQN', 0),
                'calmar_ratio': results.get('Calmar Ratio', 0),
                'trades_df': results.get('_trades', pd.DataFrame()),
                'equity_curve': results.get('_equity_curve', pd.DataFrame())
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return None
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis across all configurations"""
        print("\n🚀 Starting Comprehensive EICHERMOT Analysis...")
        print("=" * 60)
        
        total_configs = len(self.config.timeframes) * len(self.config.risk_ratios) * 2  # 2 strategy types
        current_config = 0
        
        for timeframe in self.config.timeframes:
            print(f"\n📊 Analyzing {timeframe} timeframe...")
            
            # Resample data
            tf_data = self.resample_data(timeframe)
            
            for risk_ratio in self.config.risk_ratios:
                current_config += 1
                print(f"\n⚙️  Configuration {current_config}/{total_configs}: {timeframe}, Risk 1:{risk_ratio}")
                
                # Fixed Stop Loss Strategy
                print("   🎯 Running Fixed Stop Loss strategy...")
                fixed_results = self.run_single_backtest(
                    tf_data, 
                    EMAFixedStopStrategy,
                    risk_ratio=risk_ratio,
                    stop_loss_pct=self.config.stop_loss_pct
                )
                
                if fixed_results:
                    key = f"{timeframe}_fixed_1to{risk_ratio}"
                    self.results[key] = {
                        'timeframe': timeframe,
                        'strategy_type': 'Fixed Stop Loss',
                        'risk_ratio': f"1:{risk_ratio}",
                        'stop_type': 'Fixed',
                        **fixed_results
                    }
                    print(f"   ✅ Return: {fixed_results['total_return_pct']:.2f}%, Trades: {fixed_results['num_trades']}")
                
                # Trailing Stop Loss Strategy
                current_config += 1
                print("   🔄 Running Trailing Stop Loss strategy...")
                trailing_results = self.run_single_backtest(
                    tf_data,
                    EMATrailingStopStrategy,
                    risk_ratio=risk_ratio,
                    stop_loss_pct=self.config.stop_loss_pct
                )
                
                if trailing_results:
                    key = f"{timeframe}_trailing_1to{risk_ratio}"
                    self.results[key] = {
                        'timeframe': timeframe,
                        'strategy_type': 'Trailing Stop Loss',
                        'risk_ratio': f"1:{risk_ratio}",
                        'stop_type': 'Trailing',
                        **trailing_results
                    }
                    print(f"   ✅ Return: {trailing_results['total_return_pct']:.2f}%, Trades: {trailing_results['num_trades']}")
        
        print(f"\n🎉 Analysis complete! {len(self.results)} configurations tested.")
        
    def analyze_by_periods(self) -> Dict:
        """Analyze performance by daily, weekly, monthly periods"""
        print("\n📊 Analyzing performance by time periods...")
        
        period_analysis = {
            'daily': {},
            'weekly': {},
            'monthly': {}
        }
        
        for config_key, result in self.results.items():
            if 'equity_curve' in result and not result['equity_curve'].empty:
                equity_df = result['equity_curve'].copy()
                
                try:
                    # Daily analysis
                    daily_returns = equity_df['Equity'].resample('D').last().pct_change().dropna()
                    period_analysis['daily'][config_key] = {
                        'avg_daily_return': daily_returns.mean() * 100,
                        'daily_volatility': daily_returns.std() * 100,
                        'best_day': daily_returns.max() * 100,
                        'worst_day': daily_returns.min() * 100,
                        'positive_days_pct': (daily_returns > 0).mean() * 100
                    }
                    
                    # Weekly analysis
                    weekly_returns = equity_df['Equity'].resample('W').last().pct_change().dropna()
                    period_analysis['weekly'][config_key] = {
                        'avg_weekly_return': weekly_returns.mean() * 100,
                        'weekly_volatility': weekly_returns.std() * 100,
                        'best_week': weekly_returns.max() * 100,
                        'worst_week': weekly_returns.min() * 100,
                        'positive_weeks_pct': (weekly_returns > 0).mean() * 100
                    }
                    
                    # Monthly analysis
                    monthly_returns = equity_df['Equity'].resample('M').last().pct_change().dropna()
                    period_analysis['monthly'][config_key] = {
                        'avg_monthly_return': monthly_returns.mean() * 100,
                        'monthly_volatility': monthly_returns.std() * 100,
                        'best_month': monthly_returns.max() * 100,
                        'worst_month': monthly_returns.min() * 100,
                        'positive_months_pct': (monthly_returns > 0).mean() * 100
                    }
                    
                except Exception as e:
                    print(f"⚠️  Period analysis failed for {config_key}: {e}")
                    
        return period_analysis
    
    def generate_summary_tables(self) -> Dict[str, pd.DataFrame]:
        """Generate summary tables for reporting"""
        print("\n📋 Generating summary tables...")
        
        # Main results table
        main_results = []
        for key, result in self.results.items():
            main_results.append({
                'Configuration': f"{result['timeframe']} {result['stop_type']} {result['risk_ratio']}",
                'Timeframe': result['timeframe'],
                'Stop Type': result['stop_type'],
                'Risk:Reward': result['risk_ratio'],
                'Total Return (%)': round(result['total_return_pct'], 2),
                'Sharpe Ratio': round(result['sharpe_ratio'], 2),
                'Max Drawdown (%)': round(result['max_drawdown_pct'], 2),
                'Win Rate (%)': round(result['win_rate_pct'], 1),
                'Trades': result['num_trades'],
                'Profit Factor': round(result['profit_factor'], 2),
                'Best Trade (%)': round(result['best_trade_pct'], 2),
                'Worst Trade (%)': round(result['worst_trade_pct'], 2),
                'Final Equity': int(result['equity_final'])
            })
        
        main_df = pd.DataFrame(main_results)
        
        # Timeframe comparison
        tf_comparison = main_df.groupby('Timeframe').agg({
            'Total Return (%)': 'mean',
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean',
            'Win Rate (%)': 'mean',
            'Trades': 'mean',
            'Profit Factor': 'mean'
        }).round(2)
        
        # Risk ratio comparison
        risk_comparison = main_df.groupby('Risk:Reward').agg({
            'Total Return (%)': 'mean',
            'Sharpe Ratio': 'mean', 
            'Max Drawdown (%)': 'mean',
            'Win Rate (%)': 'mean',
            'Trades': 'mean',
            'Profit Factor': 'mean'
        }).round(2)
        
        # Stop type comparison
        stop_comparison = main_df.groupby('Stop Type').agg({
            'Total Return (%)': 'mean',
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean', 
            'Win Rate (%)': 'mean',
            'Trades': 'mean',
            'Profit Factor': 'mean'
        }).round(2)
        
        return {
            'main_results': main_df,
            'timeframe_comparison': tf_comparison,
            'risk_ratio_comparison': risk_comparison,
            'stop_type_comparison': stop_comparison
        }
    
    def get_best_configurations(self) -> Dict:
        """Identify best performing configurations"""
        if not self.results:
            return {}
        
        # Convert results to DataFrame for analysis
        results_list = []
        for key, result in self.results.items():
            results_list.append({
                'config_key': key,
                'config_name': f"{result['timeframe']} {result['stop_type']} {result['risk_ratio']}",
                'total_return': result['total_return_pct'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': abs(result['max_drawdown_pct']),
                'win_rate': result['win_rate_pct'],
                'profit_factor': result['profit_factor'],
                'num_trades': result['num_trades']
            })
        
        df = pd.DataFrame(results_list)
        
        best_configs = {
            'highest_return': df.loc[df['total_return'].idxmax()],
            'best_sharpe': df.loc[df['sharpe_ratio'].idxmax()],
            'lowest_drawdown': df.loc[df['max_drawdown'].idxmin()],
            'highest_win_rate': df.loc[df['win_rate'].idxmax()],
            'best_profit_factor': df.loc[df['profit_factor'].idxmax()]
        }
        
        return best_configs

def main():
    """Main function to run EICHERMOT analysis"""
    print("🔥 EICHERMOT EMA Crossover Strategy Analysis")
    print("=" * 50)
    print("📈 EMA 9/15 crossover with multiple configurations")
    print("⚙️  Timeframes: 15min, 30min")  
    print("🎯 Risk ratios: 1:1, 1:2, 1:3")
    print("🛡️  Stop types: Fixed, Trailing")
    print("=" * 50)
    
    # Initialize analyzer
    data_path = "../fetch_historical/historical_data/EICHERMOT/EICHERMOT_5_combined.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    analyzer = EICHERMOTAnalyzer(data_path)
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()
    
    # Generate summary tables
    summary_tables = analyzer.generate_summary_tables()
    
    # Analyze by periods  
    period_analysis = analyzer.analyze_by_periods()
    
    # Get best configurations
    best_configs = analyzer.get_best_configurations()
    
    # Print summary
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
    results = main()