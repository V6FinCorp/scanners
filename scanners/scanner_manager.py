"""
Scanner Manager Module
Handles all scanner-related operations including execution, configuration, and data processing.
"""

import os
import json
import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime

class ScannerManager:
    """Manages all scanner operations"""

    def __init__(self):
        self.scanners_dir = os.path.dirname(__file__)
        # Since we're now inside scanners folder, the paths are relative to current directory
        self.results_storage = {
            'rsi': {'results': None, 'output': '', 'chart_data': None, 'last_run': None},
            'ema': {'results': None, 'output': '', 'chart_data': None, 'last_run': None},
            'dma': {'results': None, 'output': '', 'chart_data': None, 'last_run': None}
        }

    def run_scanner(self, scanner_type, symbols, base_timeframe, days_to_list, **kwargs):
        """Run a scanner and return results"""
        try:
            # Update config based on scanner type
            if scanner_type == 'rsi':
                config_file = os.path.join(self.scanners_dir, 'config', 'rsi_config.json')
                config_data = {
                    "symbols": symbols,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', 200),
                    "rsi_periods": kwargs.get('rsiPeriods', [15, 30, 60]),
                    "base_timeframe": base_timeframe,
                    "default_timeframe": base_timeframe,
                    "days_to_list": days_to_list
                }
            elif scanner_type == 'ema':
                config_file = os.path.join(self.scanners_dir, 'config', 'ema_config.json')
                config_data = {
                    "symbols": symbols,
                    "ema_periods": kwargs.get('emaPeriods', [9, 15, 65, 200]),
                    "base_timeframe": base_timeframe,
                    "days_to_list": days_to_list,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', 200)
                }
            elif scanner_type == 'dma':
                config_file = os.path.join(self.scanners_dir, 'config', 'dma_config.json')
                config_data = {
                    "symbols": symbols,
                    "dma_periods": kwargs.get('dmaPeriods', [10, 20, 50]),
                    "base_timeframe": base_timeframe,
                    "days_to_list": days_to_list,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', 200)
                }
            else:
                return {
                    'error': 'Invalid scanner type',
                    'output': 'Error: Invalid scanner type',
                    'returncode': -1
                }

            # Write config file
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)

            # Run the scanner
            scanner_script = os.path.join(self.scanners_dir, f'{scanner_type}_scanner.py')

            if not os.path.exists(scanner_script):
                return {
                    'error': f'Scanner script {scanner_script} not found',
                    'output': f'Error: Scanner script {scanner_script} not found',
                    'returncode': -1
                }

            # Ensure data directory exists before running scanner
            data_dir = os.path.join(self.scanners_dir, 'data', symbols[0])
            os.makedirs(data_dir, exist_ok=True)

            # Run scanner and capture output
            try:
                result = subprocess.run(
                    [sys.executable, scanner_script],
                    cwd=self.scanners_dir,
                    capture_output=True,
                    text=True,
                    timeout=120,  # Increased timeout to 2 minutes
                    env=dict(os.environ, PYTHONPATH=self.scanners_dir)  # Add scanners dir to Python path
                )

                print(f"Scanner execution completed with return code: {result.returncode}")
                print(f"STDOUT length: {len(result.stdout)}")
                print(f"STDERR length: {len(result.stderr)}")
                print(f"Working directory during execution: {os.getcwd()}")
                print(f"Scanner script path: {scanner_script}")
                print(f"Scanner script exists: {os.path.exists(scanner_script)}")

                # Prepare response
                response_data = {
                    'output': result.stdout or 'No output generated',
                    'error': result.stderr or '',
                    'returncode': result.returncode
                }

                # If there's stderr but no stdout, include stderr in output for visibility
                if result.stderr and not result.stdout:
                    response_data['output'] = f"Scanner execution warnings/errors:\n{result.stderr}"

                # Try to load results from CSV if scanner completed successfully
                if result.returncode == 0 and symbols:
                    symbol = symbols[0]  # Use first symbol for results

                    # Determine the correct CSV filename based on scanner type
                    if scanner_type == 'rsi':
                        csv_filename = f'{symbol}_multi_timeframe_rsi_data.csv'
                    else:
                        csv_filename = f'{symbol}_{scanner_type}_data.csv'

                    csv_file = os.path.join(self.scanners_dir, 'data', symbol, csv_filename)

                    print(f"Looking for CSV file: {csv_file}")

                    if os.path.exists(csv_file):
                        try:
                            df = pd.read_csv(csv_file)
                            print(f"CSV loaded successfully with {len(df)} rows")

                            # Replace 'N/A' strings back to None for JSON compatibility
                            df = df.replace('N/A', None)

                            # Convert to list of dicts for JSON response, handling NaN values
                            results = df.to_dict('records')
                            # Replace NaN values with None for JSON compatibility
                            for row in results:
                                for key, value in row.items():
                                    if pd.isna(value):
                                        row[key] = None
                            response_data['results'] = results

                            # Prepare chart data
                            chart_data = self.prepare_chart_data(df, scanner_type)
                            if chart_data:
                                response_data['chartData'] = chart_data

                        except Exception as e:
                            print(f"Error reading results CSV: {e}")
                            response_data['output'] += f"\nWarning: Could not read results CSV: {e}"
                    else:
                        print(f"CSV file not found: {csv_file}")
                        response_data['output'] += f"\nWarning: Results CSV not found at {csv_file}"

                # Store results in global storage
                self.results_storage[scanner_type]['results'] = response_data.get('results')
                self.results_storage[scanner_type]['output'] = response_data.get('output', '')
                self.results_storage[scanner_type]['chart_data'] = response_data.get('chartData')
                self.results_storage[scanner_type]['last_run'] = datetime.now().isoformat()

                return response_data

            except subprocess.TimeoutExpired:
                return {
                    'error': 'Scanner execution timed out',
                    'output': 'Error: Scanner execution timed out after 2 minutes',
                    'returncode': -2
                }
            except Exception as e:
                print(f"Scanner execution error: {e}")
                return {
                    'error': str(e),
                    'output': f'Error executing scanner: {str(e)}',
                    'returncode': -3
                }
        except Exception as e:
            return {'error': str(e)}

    def prepare_chart_data(self, df, scanner_type):
        """Prepare chart data for visualization"""
        try:
            if df.empty:
                return None

            # Replace 'N/A' strings with NaN for proper handling
            df = df.replace('N/A', np.nan)

            # Get the last 50 data points for chart
            chart_df = df.tail(50).copy()

            # Prepare datasets
            datasets = []

            # Price data - include OHLC for candlestick charts
            if 'open' in chart_df.columns and 'high' in chart_df.columns and 'low' in chart_df.columns:
                ohlc_data = chart_df[['open', 'high', 'low', 'close']].fillna(method='ffill').values.tolist()
                print(f"OHLC data sample: {ohlc_data[:3]}")  # Debug: print first 3 OHLC entries
                datasets.append({
                    'label': 'OHLC',
                    'data': ohlc_data,
                    'borderColor': 'rgb(59, 130, 246)',
                    'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                    'type': 'candlestick',  # This will be handled by frontend
                    'hidden': False  # Show by default for candlestick
                })

            # Also keep the close price line for line charts
            close_data = chart_df['close'].fillna(method='ffill').tolist()
            datasets.append({
                'label': 'Close Price',
                'data': close_data,
                'borderColor': 'rgb(59, 130, 246)',
                'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                'fill': False,
                'tension': 0.1,
                'type': 'line'
            })

            # Add indicator data based on scanner type
            if scanner_type == 'rsi':
                rsi_periods = [col.replace('rsi_', '') for col in df.columns if col.startswith('rsi_')]
                colors = ['rgb(34, 197, 94)', 'rgb(168, 85, 247)', 'rgb(251, 146, 60)']

                for i, period in enumerate(rsi_periods):
                    if f'rsi_{period}' in chart_df.columns:
                        rsi_data = chart_df[f'rsi_{period}'].fillna(method='ffill').tolist()
                        datasets.append({
                            'label': f'RSI({period})',
                            'data': rsi_data,
                            'borderColor': colors[i % len(colors)],
                            'backgroundColor': 'rgba(0, 0, 0, 0)',
                            'fill': False,
                            'tension': 0.1,
                            'yAxisID': 'y1'
                        })

            elif scanner_type == 'ema':
                ema_periods = [col.replace('ema_', '') for col in df.columns if col.startswith('ema_')]
                colors = ['rgb(250, 204, 21)', 'rgb(249, 115, 22)', 'rgb(6, 182, 212)', 'rgb(37, 99, 235)']

                for i, period in enumerate(ema_periods):
                    if f'ema_{period}' in chart_df.columns:
                        ema_data = chart_df[f'ema_{period}'].fillna(method='ffill').tolist()
                        datasets.append({
                            'label': f'EMA({period})',
                            'data': ema_data,
                            'borderColor': colors[i % len(colors)],
                            'backgroundColor': 'rgba(0, 0, 0, 0)',
                            'fill': False,
                            'tension': 0.1
                        })

            elif scanner_type == 'dma':
                dma_periods = [col.replace('dma_', '') for col in df.columns if col.startswith('dma_')]
                colors = ['rgb(34, 197, 94)', 'rgb(168, 85, 247)', 'rgb(251, 146, 60)']

                for i, period in enumerate(dma_periods):
                    if f'dma_{period}' in chart_df.columns:
                        dma_data = chart_df[f'dma_{period}'].fillna(method='ffill').tolist()
                        datasets.append({
                            'label': f'DMA({period})',
                            'data': dma_data,
                            'borderColor': colors[i % len(colors)],
                            'backgroundColor': 'rgba(0, 0, 0, 0)',
                            'fill': False,
                            'tension': 0.1
                        })

            # Prepare labels (timestamps)
            labels = []
            for _, row in chart_df.iterrows():
                if 'timestamp' in row:
                    try:
                        dt = pd.to_datetime(row['timestamp'])
                        labels.append(dt.strftime('%H:%M'))
                    except:
                        labels.append(str(row.name))
                else:
                    labels.append(str(row.name))

            return {
                'labels': labels,
                'datasets': datasets
            }

        except Exception as e:
            print(f"Error preparing chart data: {e}")
            return None

    def get_scanner_status(self):
        """Get status of available scanners"""
        scanners = {}

        for scanner_type in ['rsi', 'ema', 'dma']:
            config_file = os.path.join(self.scanners_dir, 'config', f'{scanner_type}_config.json')
            script_file = os.path.join(self.scanners_dir, f'{scanner_type}_scanner.py')

            scanners[scanner_type] = {
                'available': os.path.exists(script_file),
                'config_exists': os.path.exists(config_file),
                'last_modified': None
            }

            if os.path.exists(config_file):
                scanners[scanner_type]['last_modified'] = datetime.fromtimestamp(
                    os.path.getmtime(config_file)
                ).strftime('%Y-%m-%d %H:%M:%S')

        return scanners

    def get_symbols(self):
        """Get available symbols from JSON file"""
        try:
            symbols_file = os.path.join(self.scanners_dir, 'config', 'symbols_for_db.json')
            if os.path.exists(symbols_file):
                with open(symbols_file, 'r') as f:
                    data = json.load(f)
                    return data
            else:
                # Return default symbols if file doesn't exist
                return {
                    "description": "Common NSE symbols for trading analysis",
                    "symbols": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "WIPRO", "LT", "BAJFINANCE", "KOTAKBANK", "ITC"]
                }
        except Exception as e:
            print(f"Error loading symbols: {e}")
            return {"error": str(e)}

    def get_scanner_results(self, scanner_type):
        """Get stored results for a specific scanner type"""
        if scanner_type not in self.results_storage:
            return {'error': 'Invalid scanner type'}

        return self.results_storage[scanner_type]