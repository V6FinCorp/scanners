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
import logging
import traceback
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScannerManager:
    """Manages all scanner operations"""

    def __init__(self):
        self.scanners_dir = os.path.dirname(__file__)
        # Paths are relative to current directory (repository root)
        self.results_storage = {
            'rsi': {'results': None, 'output': '', 'chart_data': None, 'last_run': None},
            'ema': {'results': None, 'output': '', 'chart_data': None, 'last_run': None},
            'dma': {'results': None, 'output': '', 'chart_data': None, 'last_run': None},
            'macd': {'results': None, 'output': '', 'chart_data': None, 'last_run': None},
            'volume': {'results': None, 'output': '', 'chart_data': None, 'last_run': None}
        }
        logger.info("Scanner Manager initialized")
        
    def report_error(self, category: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a structured error report
        
        Args:
            category: Error category (validation, configuration, execution, data)
            message: Human-readable error message
            details: Additional error details
            
        Returns:
            Dictionary with structured error information
        """
        error_data = {
            "error": message,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "output": f"Error ({category}): {message}"
        }
        
        # Add system information
        error_data["system_info"] = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd()
        }
        
        # Log the error
        if category == "critical":
            logger.critical(f"{category.upper()}: {message}")
        elif category in ["execution", "configuration"]:
            logger.error(f"{category.upper()}: {message}")
        else:
            logger.warning(f"{category.upper()}: {message}")
            
        return error_data
        
    def validate_config(self, scanner_type: str, symbols: List[str], base_timeframe: str, 
                       days_to_list: int, **kwargs) -> Dict[str, Any]:
        """
        Validate scanner configuration parameters
        
        Args:
            scanner_type: Type of scanner (rsi, ema, dma)
            symbols: List of symbols to scan
            base_timeframe: Timeframe for analysis
            days_to_list: Number of days to display
            kwargs: Additional scanner-specific parameters
            
        Returns:
            Dictionary with validation results, error if invalid
        """
        # Validate scanner_type
        valid_scanners = ['rsi', 'ema', 'dma', 'macd', 'volume']
        if scanner_type not in valid_scanners:
            return self.report_error("validation", f"Invalid scanner type: {scanner_type}",
                                    {"valid_types": valid_scanners})

        # Validate symbols
        if not symbols:
            return self.report_error("validation", "No symbols provided",
                                     {"symbols": symbols})

        # Validate timeframe
        valid_timeframes = ['5mins', '15mins', '30mins', '1hour', '4hours', 'daily', 'weekly', 'monthly']
        if base_timeframe not in valid_timeframes:
            return self.report_error("validation", f"Invalid timeframe: {base_timeframe}",
                                    {"valid_timeframes": valid_timeframes})

        # Validate days_to_list
        if not isinstance(days_to_list, int) or days_to_list <= 0:
            return self.report_error("validation", f"Invalid days_to_list: {days_to_list}",
                                    {"value": days_to_list})

        # Scanner-specific validation
        if scanner_type == 'rsi':
            rsi_periods = kwargs.get('rsiPeriods', [])
            if not rsi_periods:
                return self.report_error("validation", "No RSI periods provided",
                                         {"rsiPeriods": rsi_periods})
        elif scanner_type == 'ema':
            ema_periods = kwargs.get('emaPeriods', [])
            if not ema_periods:
                return self.report_error("validation", "No EMA periods provided",
                                         {"emaPeriods": ema_periods})
        elif scanner_type == 'dma':
            dma_periods = kwargs.get('dmaPeriods', [])
            if not dma_periods:
                return self.report_error("validation", "No DMA periods provided",
                                         {"dmaPeriods": dma_periods})

        # If all validation passes
        return {"valid": True}

    def run_scanner(self, scanner_type, symbols, base_timeframe, days_to_list, **kwargs):
        """Run a scanner and return results with enhanced error handling"""
        try:
            # Validate inputs
            validation_result = self.validate_config(scanner_type, symbols, base_timeframe, days_to_list, **kwargs)
            if "valid" not in validation_result:
                return validation_result
                
            logger.info(f"Running {scanner_type} scanner for {symbols} ({base_timeframe}, {days_to_list} days)")
            
            # Load existing config file and merge with form values (without overwriting the config file)
            if scanner_type == 'rsi':
                config_file = os.path.join(self.scanners_dir, 'config', 'rsi_config.json')
                
                # Load existing config as base
                try:
                    with open(config_file, 'r') as f:
                        base_config = json.load(f)
                        logger.debug(f"Loaded base RSI config: {base_config}")
                except FileNotFoundError:
                    logger.warning(f"Config file not found: {config_file}, using empty base config")
                    base_config = {}
                
                # Create runtime config (merge base with form values, form values take precedence)
                config_data = {
                    "symbols": symbols,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', base_config.get('days_fallback_threshold', 200)),
                    "rsi_periods": kwargs.get('rsiPeriods', base_config.get('rsi_periods', [15, 30, 60])),
                    "rsi_overbought": kwargs.get('rsiOverbought', base_config.get('rsi_overbought', 70)),
                    "rsi_oversold": kwargs.get('rsiOversold', base_config.get('rsi_oversold', 30)),
                    "base_timeframe": base_timeframe,
                    "default_timeframe": base_timeframe,
                    "days_to_list": days_to_list
                }
            elif scanner_type == 'ema':
                config_file = os.path.join(self.scanners_dir, 'config', 'ema_config.json')
                
                # Load existing config as base
                try:
                    with open(config_file, 'r') as f:
                        base_config = json.load(f)
                        logger.debug(f"Loaded base EMA config: {base_config}")
                except FileNotFoundError:
                    logger.warning(f"Config file not found: {config_file}, using empty base config")
                    base_config = {}
                
                # Create runtime config (merge base with form values, form values take precedence)
                config_data = {
                    "symbols": symbols,
                    "ema_periods": kwargs.get('emaPeriods', base_config.get('ema_periods', [9, 15, 65, 200])),
                    "base_timeframe": base_timeframe,
                    "days_to_list": days_to_list,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', base_config.get('days_fallback_threshold', 200))
                }
            elif scanner_type == 'dma':
                config_file = os.path.join(self.scanners_dir, 'config', 'dma_config.json')
                
                # Load existing config as base
                try:
                    with open(config_file, 'r') as f:
                        base_config = json.load(f)
                        logger.debug(f"Loaded base DMA config: {base_config}")
                except FileNotFoundError:
                    logger.warning(f"Config file not found: {config_file}, using empty base config")
                    base_config = {}
                
                # Create runtime config (merge base with form values, form values take precedence)  
                config_data = {
                    "symbols": symbols,
                    "dma_periods": kwargs.get('dmaPeriods', base_config.get('dma_periods', [10, 20, 50])),
                    "base_timeframe": base_timeframe,
                    "days_to_list": days_to_list,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', base_config.get('days_fallback_threshold', 200))
                }
            elif scanner_type == 'macd':
                config_file = os.path.join(self.scanners_dir, 'config', 'macd_config.json')
                
                # Load existing config as base
                try:
                    with open(config_file, 'r') as f:
                        base_config = json.load(f)
                        logger.debug(f"Loaded base MACD config: {base_config}")
                except FileNotFoundError:
                    logger.warning(f"Config file not found: {config_file}, using empty base config")
                    base_config = {}
                
                # Create runtime config (merge base with form values, form values take precedence)  
                config_data = {
                    "symbols": symbols,
                    "fast_length": kwargs.get('fastLength', base_config.get('fast_length', 12)),
                    "slow_length": kwargs.get('slowLength', base_config.get('slow_length', 26)),
                    "signal_length": kwargs.get('signalLength', base_config.get('signal_length', 9)),
                    "oscillator_ma_type": kwargs.get('oscillatorMAType', base_config.get('oscillator_ma_type', 'EMA')),
                    "signal_ma_type": kwargs.get('signalMAType', base_config.get('signal_ma_type', 'EMA')),
                    "base_timeframe": base_timeframe,
                    "days_to_list": days_to_list,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', base_config.get('days_fallback_threshold', 300))
                }
            elif scanner_type == 'volume':
                config_file = os.path.join(self.scanners_dir, 'config', 'volume_config.json')

                # Load existing config as base
                try:
                    with open(config_file, 'r') as f:
                        base_config = json.load(f)
                        logger.debug(f"Loaded base VOLUME config: {base_config}")
                except FileNotFoundError:
                    logger.warning(f"Config file not found: {config_file}, using empty base config")
                    base_config = {}

                config_data = {
                    "symbols": symbols,
                    "ma_window": kwargs.get('maWindow', base_config.get('ma_window', 20)),
                    "multiplier": kwargs.get('multiplier', base_config.get('multiplier', 2.0)),
                    "min_volume": kwargs.get('minVolume', base_config.get('min_volume', 100000)),
                    "base_timeframe": base_timeframe,
                    "days_to_list": days_to_list,
                    "days_fallback_threshold": kwargs.get('daysFallbackThreshold', base_config.get('days_fallback_threshold', 365))
                }
            else:
                return self.report_error("validation", f"Invalid scanner type: {scanner_type}",
                                    {"valid_types": ['rsi', 'ema', 'dma', 'macd']})

            # Create temporary config file for this scanner execution only
            # This preserves original config files while allowing form overrides
            original_config_file = config_file
            temp_config_file = os.path.join(self.scanners_dir, 'config', f'temp_{scanner_type}_config.json')
            backup_config_file = os.path.join(self.scanners_dir, 'config', f'backup_{scanner_type}_config.json')
            
            os.makedirs(os.path.dirname(temp_config_file), exist_ok=True)
            
            # Step 1: Backup original config
            try:
                if os.path.exists(original_config_file):
                    # Remove any existing backup file first
                    if os.path.exists(backup_config_file):
                        os.remove(backup_config_file)
                    os.rename(original_config_file, backup_config_file)
                    logger.debug(f"Backed up original config to {backup_config_file}")
            except Exception as e:
                return self.report_error("configuration", f"Failed to backup configuration file: {str(e)}",
                                    {"original_file": original_config_file, "backup_file": backup_config_file, "error": str(e)})
            
            # Step 2: Create temporary config with merged values  
            try:
                with open(original_config_file, 'w') as f:
                    json.dump(config_data, f, indent=4)
                    logger.debug(f"Created temporary config at {original_config_file}")
            except Exception as e:
                # Try to restore backup if creation fails
                if os.path.exists(backup_config_file):
                    try:
                        os.rename(backup_config_file, original_config_file)
                    except:
                        pass  # If this fails too, we'll report the original error
                return self.report_error("configuration", f"Failed to create temporary configuration: {str(e)}",
                                    {"file": original_config_file, "config_data": config_data, "error": str(e)})

            # Run the scanner
            scanner_script = os.path.join(self.scanners_dir, f'{scanner_type}_scanner.py')

            if not os.path.exists(scanner_script):
                # Restore original config before returning error
                if os.path.exists(backup_config_file):
                    try:
                        os.rename(backup_config_file, original_config_file)
                        logger.debug(f"Restored original config from {backup_config_file}")
                    except Exception as e:
                        logger.error(f"Failed to restore original config: {e}")
                return self.report_error("configuration", f"Scanner script not found: {scanner_script}",
                                    {"script_path": scanner_script})

            # Ensure data directory exists before running scanner
            try:
                data_dir = os.path.join(self.scanners_dir, 'data', symbols[0])
                os.makedirs(data_dir, exist_ok=True)
                logger.debug(f"Ensured data directory exists: {data_dir}")
            except Exception as e:
                return self.report_error("system", f"Failed to create data directory: {str(e)}",
                                    {"directory": data_dir, "error": str(e)})

            # Run scanner and capture output
            try:
                logger.info(f"Executing scanner script: {scanner_script}")
                
                result = subprocess.run(
                    [sys.executable, scanner_script],
                    cwd=self.scanners_dir,
                    capture_output=True,
                    text=True,
                    timeout=120,  # Increased timeout to 2 minutes
                    # Don't modify PYTHONPATH as it can cause issues in containerized environments
                    # Instead, make sure imports work properly with proper package structure
                )

                logger.info(f"Scanner execution completed with return code: {result.returncode}")
                logger.debug(f"STDOUT length: {len(result.stdout)}")
                logger.debug(f"STDERR length: {len(result.stderr)}")
                logger.debug(f"Working directory during execution: {os.getcwd()}")
                
                # Check if there was an error
                if result.returncode != 0:
                    return self.report_error("execution", f"Scanner execution failed with code {result.returncode}",
                                        {"returncode": result.returncode, 
                                         "stdout": result.stdout, 
                                         "stderr": result.stderr})

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

                    logger.debug(f"Looking for CSV file: {csv_file}")

                    if os.path.exists(csv_file):
                        try:
                            df = pd.read_csv(csv_file)
                            logger.info(f"CSV loaded successfully with {len(df)} rows")

                            # Handle NaN and other problematic values for JSON serialization
                            try:
                                # Replace 'N/A' strings and NaN values with None for JSON compatibility
                                df = df.replace('N/A', None)
                                df = df.replace('NaN', None)
                                df = df.replace('nan', None)
                                
                                # Convert numeric columns with NaN to proper None values
                                for col in df.select_dtypes(include=['float64', 'int64']).columns:
                                    df.loc[df[col].isna(), col] = None

                                # Convert to list of dicts for JSON response, handling NaN values
                                results = df.to_dict('records')

                                # Replace any remaining NaN values with None for JSON compatibility
                                for row in results:
                                    for key, value in row.items():
                                        if pd.isna(value) or value is None or str(value).lower() in ['nan', 'nat']:
                                            row[key] = None
                                        # Also handle numpy types that might cause JSON issues
                                        elif hasattr(value, 'item'):  # numpy types
                                            row[key] = value.item()
                                        elif isinstance(value, (np.int64, np.float64)):
                                            row[key] = value.item()

                                response_data['results'] = results
                                
                                # Validate results
                                if not results:
                                    logger.warning(f"CSV file {csv_file} loaded but contained no valid results")
                                    response_data['warning'] = "Scanner completed but produced no results"

                                # Prepare chart data
                                chart_data = self.prepare_chart_data(df, scanner_type)
                                if chart_data:
                                    response_data['chartData'] = chart_data
                                else:
                                    logger.warning("Failed to prepare chart data")
                                    response_data['warning'] = response_data.get('warning', '') + "\nFailed to prepare chart data"
                                
                            except Exception as e:
                                logger.error(f"Error processing results dataframe: {e}")
                                response_data['warning'] = f"Error processing results: {str(e)}"

                        except Exception as e:
                            logger.error(f"Error reading results CSV: {e}")
                            response_data['output'] += f"\nWarning: Could not read results CSV: {e}"
                    else:
                        logger.warning(f"CSV file not found: {csv_file}")
                        response_data['output'] += f"\nWarning: Results CSV not found at {csv_file}"

                # Store results in global storage
                self.results_storage[scanner_type]['results'] = response_data.get('results')
                self.results_storage[scanner_type]['output'] = response_data.get('output', '')
                self.results_storage[scanner_type]['chart_data'] = response_data.get('chartData')
                self.results_storage[scanner_type]['last_run'] = datetime.now().isoformat()

                return response_data

            except subprocess.TimeoutExpired:
                return self.report_error("execution", "Scanner execution timed out",
                                    {"timeout_seconds": 120, "scanner_type": scanner_type, "symbols": symbols})
            except Exception as e:
                logger.exception(f"Scanner execution error: {e}")
                return self.report_error("critical", f"Unexpected error executing scanner: {str(e)}",
                                    {"error": str(e), "traceback": traceback.format_exc()})
            finally:
                # ALWAYS restore original config file to prevent config overwriting
                try:
                    if os.path.exists(backup_config_file):
                        # Remove the temporary config
                        if os.path.exists(original_config_file):
                            os.remove(original_config_file)
                        # Restore the original config
                        os.rename(backup_config_file, original_config_file)
                        logger.debug(f"Original config restored: {original_config_file}")
                except Exception as e:
                    logger.error(f"Error restoring config file: {e}")
        except Exception as e:
            logger.exception(f"Unhandled exception in run_scanner: {e}")
            return self.report_error("critical", f"Unhandled exception: {str(e)}",
                                {"error": str(e), "traceback": traceback.format_exc()})

    def prepare_chart_data(self, df, scanner_type):
        """Prepare chart data for visualization with improved error handling and data validation"""
        try:
            if df.empty:
                logger.warning("Cannot prepare chart data from empty dataframe")
                return None

            # Make a copy to avoid modifying the original dataframe
            chart_df = df.copy()
            
            # Check that required columns exist
            required_columns = ['timestamp', 'close']
            missing_columns = [col for col in required_columns if col not in chart_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for chart data: {missing_columns}")
                return None
                
            # Handle missing OHLC columns more gracefully
            if 'open' not in chart_df.columns:
                logger.warning("Missing 'open' column, using 'close' as fallback")
                chart_df['open'] = chart_df['close']
            if 'high' not in chart_df.columns:
                logger.warning("Missing 'high' column, using max of open/close as fallback")
                chart_df['high'] = chart_df[['open', 'close']].max(axis=1)
            if 'low' not in chart_df.columns:
                logger.warning("Missing 'low' column, using min of open/close as fallback")
                chart_df['low'] = chart_df[['open', 'close']].min(axis=1)

            # Replace 'N/A' strings and NaN values with None for JSON compatibility
            chart_df = chart_df.replace('N/A', None)
            chart_df = chart_df.replace('NaN', None)
            chart_df = chart_df.replace('nan', None)

            # Handle NaN values in the dataframe
            for col in chart_df.columns:
                if chart_df[col].dtype in ['float64', 'int64']:
                    chart_df[col] = chart_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
                    # Validate data for unreasonable values
                    if col in ['open', 'high', 'low', 'close']:
                        # Check for negative or extremely large values
                        invalid_mask = (chart_df[col] < 0) | (chart_df[col] > 1e6)
                        if invalid_mask.any():
                            logger.warning(f"Found {invalid_mask.sum()} invalid values in {col}, replacing with median")
                            median_value = chart_df[col].median()
                            chart_df.loc[invalid_mask, col] = median_value

            # Get the last 50 data points for chart, with validation
            try:
                rows_to_display = min(50, len(chart_df))
                chart_df = chart_df.tail(rows_to_display).copy()
                logger.debug(f"Using {rows_to_display} rows for chart visualization")
            except Exception as e:
                logger.error(f"Error selecting data points for chart: {e}")
                chart_df = chart_df.tail(min(50, len(chart_df))).copy()

            # Prepare datasets
            datasets = []

            # Price data - include OHLC for candlestick charts
            if 'open' in chart_df.columns and 'high' in chart_df.columns and 'low' in chart_df.columns:
                try:
                    ohlc_data = []
                    for _, row in chart_df.iterrows():
                        ohlc_data.append([
                            row['open'] if not pd.isna(row['open']) else 0,
                            row['high'] if not pd.isna(row['high']) else 0,
                            row['low'] if not pd.isna(row['low']) else 0,
                            row['close'] if not pd.isna(row['close']) else 0
                        ])
                    datasets.append({
                        'label': 'OHLC',
                        'data': ohlc_data,
                        'borderColor': 'rgb(59, 130, 246)',
                        'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                        'type': 'candlestick',  # This will be handled by frontend
                        'hidden': False  # Show by default for candlestick
                    })
                except Exception as e:
                    logger.error(f"Error creating OHLC dataset: {e}")
                    # Continue to line chart as fallback

            # Also keep the close price line for line charts
            try:
                close_data = []
                for _, row in chart_df.iterrows():
                    close_data.append(row['close'] if not pd.isna(row['close']) else 0)
                datasets.append({
                    'label': 'Close Price',
                    'data': close_data,
                    'borderColor': 'rgb(59, 130, 246)',
                    'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                    'fill': False,
                    'tension': 0.1,
                    'type': 'line'
                })
            except Exception as e:
                logger.error(f"Error creating close price dataset: {e}")
                # If we can't even create a close price dataset, we have a serious issue
                return None

            # Add indicator data based on scanner type
            if scanner_type == 'rsi':
                try:
                    rsi_periods = [col.replace('rsi_', '') for col in chart_df.columns if col.startswith('rsi_')]
                    colors = ['rgb(34, 197, 94)', 'rgb(168, 85, 247)', 'rgb(251, 146, 60)']

                    if not rsi_periods:
                        logger.warning("No RSI columns found in data")
                        
                    for i, period in enumerate(rsi_periods):
                        if f'rsi_{period}' in chart_df.columns:
                            rsi_data = []
                            for _, row in chart_df.iterrows():
                                rsi_val = row[f'rsi_{period}']
                                # RSI should be between 0-100, validate
                                if pd.isna(rsi_val) or rsi_val < 0 or rsi_val > 100:
                                    rsi_data.append(None)
                                else:
                                    rsi_data.append(rsi_val)
                            datasets.append({
                                'label': f'RSI({period})',
                                'data': rsi_data,
                                'borderColor': colors[i % len(colors)],
                                'backgroundColor': 'rgba(0, 0, 0, 0)',
                                'fill': False,
                                'tension': 0.1,
                                'yAxisID': 'y1'
                            })
                except Exception as e:
                    logger.error(f"Error creating RSI datasets: {e}")

            elif scanner_type == 'ema':
                try:
                    ema_periods = [col.replace('ema_', '') for col in chart_df.columns if col.startswith('ema_')]
                    colors = ['rgb(250, 204, 21)', 'rgb(249, 115, 22)', 'rgb(6, 182, 212)', 'rgb(37, 99, 235)']

                    if not ema_periods:
                        logger.warning("No EMA columns found in data")
                        
                    for i, period in enumerate(ema_periods):
                        if f'ema_{period}' in chart_df.columns:
                            ema_data = []
                            for _, row in chart_df.iterrows():
                                ema_val = row[f'ema_{period}']
                                # Basic price validation
                                if pd.isna(ema_val) or ema_val < 0 or ema_val > 1e6:
                                    ema_data.append(None)
                                else:
                                    ema_data.append(ema_val)
                            datasets.append({
                                'label': f'EMA({period})',
                                'data': ema_data,
                                'borderColor': colors[i % len(colors)],
                                'backgroundColor': 'rgba(0, 0, 0, 0)',
                                'fill': False,
                                'tension': 0.1
                            })
                except Exception as e:
                    logger.error(f"Error creating EMA datasets: {e}")

            elif scanner_type == 'dma':
                try:
                    dma_periods = [col.replace('dma_', '') for col in chart_df.columns if col.startswith('dma_')]
                    colors = ['rgb(34, 197, 94)', 'rgb(168, 85, 247)', 'rgb(251, 146, 60)']

                    if not dma_periods:
                        logger.warning("No DMA columns found in data")
                        
                    for i, period in enumerate(dma_periods):
                        if f'dma_{period}' in chart_df.columns:
                            dma_data = []
                            for _, row in chart_df.iterrows():
                                dma_val = row[f'dma_{period}']
                                # Basic price validation
                                if pd.isna(dma_val) or dma_val < 0 or dma_val > 1e6:
                                    dma_data.append(None)
                                else:
                                    dma_data.append(dma_val)
                            datasets.append({
                                'label': f'DMA({period})',
                                'data': dma_data,
                                'borderColor': colors[i % len(colors)],
                                'backgroundColor': 'rgba(0, 0, 0, 0)',
                                'fill': False,
                                'tension': 0.1
                            })
                except Exception as e:
                    logger.error(f"Error creating DMA datasets: {e}")

            # Prepare labels (timestamps)
            try:
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
            except Exception as e:
                logger.error(f"Error creating chart labels: {e}")
                # Fallback to simple index-based labels
                labels = [str(i) for i in range(len(chart_df))]

            # Final validation - need at least labels and one dataset
            if not labels or not datasets:
                logger.error("Invalid chart data: missing labels or datasets")
                return None
                
            # Check for data length mismatch
            for dataset in datasets:
                if len(dataset['data']) != len(labels):
                    logger.error(f"Data length mismatch: {dataset['label']} has {len(dataset['data'])} points, but labels has {len(labels)}")
                    return None

            return {
                'labels': labels,
                'datasets': datasets
            }

        except Exception as e:
            logger.exception(f"Error preparing chart data: {e}")
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