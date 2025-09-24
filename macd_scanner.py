"""
MACD Scanner - Calculates and displays MACD (Moving Average Convergence Divergence) values.
Follows TradingView's MACD calculation method as defined in the provided Pine Script.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import sys
import pytz  # Added for timezone handling
import logging
import time
import traceback
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_moving_average(data, period, ma_type="EMA"):
    """Calculate moving average (EMA or SMA)"""
    if len(data) < period:
        return pd.Series([float('nan')] * len(data), index=data.index)
    
    if ma_type.upper() == "SMA":
        return data.rolling(window=period).mean()
    else:  # Default to EMA
        return data.ewm(span=period, adjust=False).mean()

def calculate_macd(data, fast_length=12, slow_length=26, signal_length=9, 
                   oscillator_ma_type="EMA", signal_ma_type="EMA"):
    """
    Calculate MACD following TradingView's method
    
    Args:
        data: DataFrame with 'close' column
        fast_length: Fast moving average period (default 12)
        slow_length: Slow moving average period (default 26)
        signal_length: Signal line moving average period (default 9)
        oscillator_ma_type: Type of MA for MACD calculation ("EMA" or "SMA")
        signal_ma_type: Type of MA for signal line ("EMA" or "SMA")
    
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    if len(data) < max(fast_length, slow_length) + signal_length:
        # Not enough data for MACD calculation
        return pd.DataFrame({
            'MACD': [float('nan')] * len(data),
            'Signal': [float('nan')] * len(data),
            'Histogram': [float('nan')] * len(data)
        }, index=data.index)

    # Calculate fast and slow moving averages
    fast_ma = calculate_moving_average(data['close'], fast_length, oscillator_ma_type)
    slow_ma = calculate_moving_average(data['close'], slow_length, oscillator_ma_type)
    
    # Calculate MACD line
    macd = fast_ma - slow_ma
    
    # Calculate Signal line
    signal = calculate_moving_average(macd, signal_length, signal_ma_type)
    
    # Calculate Histogram
    histogram = macd - signal
    
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal,
        'Histogram': histogram
    }, index=data.index)

def report_error(category: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a structured error report with categorization and diagnostics
    
    Args:
        category: Error category (data, calculation, configuration, api)
        message: Human-readable error message
        details: Additional error details and diagnostic information
        
    Returns:
        Dictionary with structured error information
    """
    error_data = {
        "category": category,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "details": details or {}
    }
    
    # Add system information to help with debugging
    error_data["system_info"] = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "memory_available": "Unknown"  # Could add psutil.virtual_memory() if available
    }
    
    # Log the error with appropriate level based on category
    if category == "critical":
        logger.critical(f"{category.upper()}: {message}")
    elif category in ["data", "calculation"]:
        logger.error(f"{category.upper()}: {message}")
    else:
        logger.warning(f"{category.upper()}: {message}")
        
    return error_data

def format_date(date_str, include_time=False):
    """Format date to DD-Mon-YYYY or DD-Mon-YYYY HH:MM for intraday"""
    if isinstance(date_str, str):
        # Handle string timestamps
        if 'T' in date_str or '+' in date_str:
            # ISO format or timezone aware
            date_obj = pd.to_datetime(date_str)
        else:
            date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
    else:
        date_obj = date_str
    
    if include_time:
        return date_obj.strftime("%d-%b-%Y %H:%M")
    else:
        return date_obj.strftime("%d-%b-%Y")

def load_instrument_mapping():
    """Load instrument mapping for symbol resolution"""
    mapping_path = os.path.join(os.path.dirname(__file__), 'data_loader', 'config', 'instrument_mapping.json')
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        print("Could not load instrument mapping")
        return None

def get_instrument_key(symbol, instrument_map):
    """Get instrument key for symbol"""
    if symbol.upper() in instrument_map:
        symbol_data = instrument_map[symbol.upper()]
        if isinstance(symbol_data, dict) and 'instrument_key' in symbol_data:
            return symbol_data['instrument_key']
    return None

def fetch_timeframe_data_direct(symbol, timeframe, days_back=300):
    """Fetch data for specific timeframe using direct API calls (fallback method)"""
    instrument_map = load_instrument_mapping()
    if not instrument_map:
        return None

    instrument_key = get_instrument_key(symbol, instrument_map)
    if not instrument_key:
        print(f"Could not find instrument key for {symbol}")
        return None

    # Map timeframe to API parameters
    timeframe_mapping = {
        '5mins': ('minutes', '5'),
        '15mins': ('minutes', '15'),
        '30mins': ('minutes', '30'),
        '1hour': ('minutes', '60'),
        '2hours': ('minutes', '120'),
        '4hours': ('minutes', '240'),
        'daily': ('days', '1'),
        'weekly': ('days', '7'),
        'monthly': ('days', '30'),
        'yearly': ('days', '365')
    }

    if timeframe not in timeframe_mapping:
        print(f"Unsupported timeframe: {timeframe}")
        return None

    unit, interval = timeframe_mapping[timeframe]

    # Adjust date range based on timeframe (API limitations)
    end_date = datetime.now()

    if timeframe in ['5mins', '15mins', '30mins']:
        # Intraday data typically only available for recent periods
        days_back = min(days_back, 30)  # Max 30 days for intraday
    elif timeframe in ['1hour', '2hours', '4hours']:
        days_back = min(days_back, 90)  # Max 90 days for hourly
    elif timeframe == 'daily':
        days_back = min(days_back, 365 * 2)  # Max 2 years for daily
    elif timeframe == 'weekly':
        days_back = min(days_back, 365 * 5)  # Max 5 years for weekly
    elif timeframe == 'monthly':
        days_back = min(days_back, 365 * 10)  # Max 10 years for monthly
    elif timeframe == 'yearly':
        days_back = min(days_back, 365 * 20)  # Max 20 years for yearly

    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Fetching {timeframe} data for {symbol}: {start_str} to {end_str} (DIRECT API)")

    # Build API URL
    safe_key = instrument_key.replace('|', '%7C')
    if unit == 'days':
        url = f"https://api.upstox.com/v3/historical-candle/{safe_key}/days/{interval}/{end_str}/{start_str}"
    else:
        url = f"https://api.upstox.com/v3/historical-candle/{safe_key}/minutes/{interval}/{end_str}/{start_str}"

    headers = {'Accept': 'application/json'}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        data = response.json()

        if 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            print(f"Received {len(candles)} {timeframe} data points for {symbol} (DIRECT API)")

            # Convert to DataFrame
            rows = []
            for candle in candles:
                if len(candle) >= 6:
                    rows.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })

            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            return df
        else:
            print(f"No candles in response for {symbol}")
            return None

    except Exception as e:
        print(f"Error fetching {timeframe} data for {symbol}: {e}")
        return None

def fetch_timeframe_data(symbol, timeframe, days_back=300):
    """Fetch data for specific timeframe using data_loader.py"""
    try:
        # Import data_loader functionality
        sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader'))
        from data_loader import fetch_data_for_symbol

        print(f"Using data_loader.py to fetch {days_back} days of data for {symbol}")

        # Use data_loader to fetch data
        combined_file = fetch_data_for_symbol(symbol, days_back)

        if combined_file and os.path.exists(combined_file):
            print(f"Loading data from: {combined_file}")

            # Read the combined CSV file
            df = pd.read_csv(combined_file)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            print(f"Loaded {len(df)} data points from data_loader.py")

            # Resample data to requested timeframe if different from 5-minute data
            if timeframe != '5mins':
                print(f"Resampling 5-minute data to {timeframe} timeframe")

                # Set timestamp as index for resampling
                df = df.set_index('timestamp')

                # Define resampling rules based on timeframe with market open offset
                # Indian market opens at 9:15 AM, so we offset resampling to start from market open
                market_open_offset = '9h15min'
                
                if timeframe == '15mins':
                    resampled = df.resample('15min', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '30mins':
                    resampled = df.resample('30min', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '1hour':
                    resampled = df.resample('1h', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '2hours':
                    resampled = df.resample('2h', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '4hours':
                    resampled = df.resample('4h', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == 'daily':
                    # For daily, use market session (9:15 AM to 3:30 PM)
                    resampled = df.resample('1D', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == 'weekly':
                    # Weekly data starting from Monday
                    resampled = df.resample('W-MON').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == 'monthly':
                    # Monthly data starting from first day of month
                    resampled = df.resample('MS').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                else:
                    print(f"Unsupported timeframe for resampling: {timeframe}")
                    return None

                # Remove rows with NaN values (incomplete periods)
                resampled = resampled.dropna()

                # Reset index to get timestamp back as a column
                df = resampled.reset_index()
                
                print(f"Resampled to {len(df)} {timeframe} data points")

            return df
        else:
            print(f"No data file found, falling back to direct API")
            return fetch_timeframe_data_direct(symbol, timeframe, days_back)

    except Exception as e:
        print(f"Error using data_loader.py for {symbol}: {e}")
        print("Falling back to direct API call")
        return fetch_timeframe_data_direct(symbol, timeframe, days_back)

def calculate_macd_for_symbol(symbol, config):
    """Calculate MACD for a symbol using configuration parameters"""
    try:
        # Get configuration parameters
        fast_length = config.get('fast_length', 12)
        slow_length = config.get('slow_length', 26)
        signal_length = config.get('signal_length', 9)
        oscillator_ma_type = config.get('oscillator_ma_type', 'EMA')
        signal_ma_type = config.get('signal_ma_type', 'EMA')
        base_timeframe = config.get('base_timeframe', 'daily')
        days_fallback_threshold = config.get('days_fallback_threshold', 300)

        print(f"\n=== MACD Calculation for {symbol} ===")
        print(f"Fast Length: {fast_length}, Slow Length: {slow_length}, Signal Length: {signal_length}")
        print(f"Oscillator MA Type: {oscillator_ma_type}, Signal MA Type: {signal_ma_type}")
        print(f"Base Timeframe: {base_timeframe}")

        # Fetch data
        df = fetch_timeframe_data(symbol, base_timeframe, days_fallback_threshold)

        if df is None or df.empty:
            error = report_error("data", f"No data available for {symbol}", {
                "symbol": symbol,
                "timeframe": base_timeframe
            })
            return {"error": error}

        print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total data points: {len(df)}")

        # Calculate MACD
        macd_data = calculate_macd(df, fast_length, slow_length, signal_length,
                                   oscillator_ma_type, signal_ma_type)

        # Add timestamp and other data
        result_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        result_df = pd.concat([result_df, macd_data], axis=1)

        # Remove rows with NaN values in MACD calculations
        result_df = result_df.dropna(subset=['MACD', 'Signal', 'Histogram'])

        if result_df.empty:
            error = report_error("calculation", f"MACD calculation resulted in no valid data for {symbol}", {
                "symbol": symbol,
                "data_points": len(df),
                "fast_length": fast_length,
                "slow_length": slow_length,
                "signal_length": signal_length
            })
            return {"error": error}

        print(f"MACD calculation completed. Valid data points: {len(result_df)}")

        # Save to file
        output_dir = os.path.join(os.path.dirname(__file__), 'data', symbol)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{symbol}_macd_data.csv")
        
        # Round numeric values for better readability
        numeric_columns = ['open', 'high', 'low', 'close', 'MACD', 'Signal', 'Histogram']
        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].round(4)
        
        result_df.to_csv(output_file, index=False)
        print(f"MACD data saved to: {output_file}")

        return {
            "symbol": symbol,
            "data": result_df,
            "file_path": output_file,
            "config": {
                "fast_length": fast_length,
                "slow_length": slow_length,
                "signal_length": signal_length,
                "oscillator_ma_type": oscillator_ma_type,
                "signal_ma_type": signal_ma_type,
                "base_timeframe": base_timeframe
            }
        }

    except Exception as e:
        error = report_error("calculation", f"Error calculating MACD for {symbol}: {str(e)}", {
            "symbol": symbol,
            "traceback": traceback.format_exc()
        })
        return {"error": error}

def display_macd_summary(results, config):
    """Display MACD summary table"""
    if not results:
        print("No MACD results to display")
        return

    print("\n" + "="*110)
    print(f"{'MACD SCANNER RESULTS':^110}")
    print("="*110)
    
    # Display configuration
    fast_length = config.get('fast_length', 12)
    slow_length = config.get('slow_length', 26)
    signal_length = config.get('signal_length', 9)
    oscillator_ma_type = config.get('oscillator_ma_type', 'EMA')
    signal_ma_type = config.get('signal_ma_type', 'EMA')
    base_timeframe = config.get('base_timeframe', 'daily')
    days_to_list = config.get('days_to_list', 10)
    
    # Check if timeframe is intraday (less than daily)
    intraday_timeframes = ['5mins', '15mins', '30mins', '1hour', '2hours', '4hours']
    is_intraday = base_timeframe in intraday_timeframes
    
    print(f"Configuration: Fast({fast_length}) Slow({slow_length}) Signal({signal_length}) | MA Types: {oscillator_ma_type}/{signal_ma_type} | Timeframe: {base_timeframe}")
    print(f"Showing last {days_to_list} {'periods' if is_intraday else 'days'} of data")
    print("-"*110)

    for result in results:
        if "error" in result:
            print(f"ERROR - {result['error']['message']}")
            continue

        symbol = result["symbol"]
        df = result["data"]
        
        if df.empty:
            print(f"{symbol}: No data available")
            continue

        print(f"\n{symbol} - MACD Analysis:")
        print("-" * 60)

        # Get recent data based on days_to_list
        recent_df = df.tail(days_to_list)

        # Check if timeframe is intraday
        base_timeframe = config.get('base_timeframe', 'daily')
        intraday_timeframes = ['5mins', '15mins', '30mins', '1hour', '2hours', '4hours']
        is_intraday = base_timeframe in intraday_timeframes

        # Display recent MACD values with pipe-separated format for dashboard parsing
        if is_intraday:
            print("DateTime             | Symbol   | CMP     | MACD    | Signal  | Histogram | Status")
            print("-" * 85)
        else:
            print("Time            | Symbol   | CMP     | MACD    | Signal  | Histogram | Status")
            print("-" * 75)

        for _, row in recent_df.iterrows():
            date_str = format_date(row['timestamp'], include_time=is_intraday)
            close_val = f"{row['close']:.2f}"
            macd_val = f"{row['MACD']:.4f}"
            signal_val = f"{row['Signal']:.4f}"
            histogram_val = f"{row['Histogram']:.4f}"
            
            # Determine MACD status
            if row['MACD'] > row['Signal']:
                if row['Histogram'] > 0:
                    status = "Bullish"
                else:
                    status = "Weakening"
            else:
                if row['Histogram'] < 0:
                    status = "Bearish"
                else:
                    status = "Recovering"
            
            if is_intraday:
                print(f"{date_str:<20} | {symbol:<8} | {close_val:<7} | {macd_val:<7} | {signal_val:<7} | {histogram_val:<9} | {status}")
            else:
                print(f"{date_str:<15} | {symbol:<8} | {close_val:<7} | {macd_val:<7} | {signal_val:<7} | {histogram_val:<9} | {status}")

        # Additional analysis
        latest = recent_df.iloc[-1]
        previous = recent_df.iloc[-2] if len(recent_df) > 1 else latest

        print(f"\nLatest Analysis ({format_date(latest['timestamp'], include_time=is_intraday)}):")
        print(f"  MACD: {latest['MACD']:.4f}")
        print(f"  Signal: {latest['Signal']:.4f}")
        print(f"  Histogram: {latest['Histogram']:.4f}")
        
        # Signal analysis
        if latest['MACD'] > latest['Signal']:
            trend = "Bullish"
        else:
            trend = "Bearish"
            
        if len(recent_df) > 1:
            if latest['Histogram'] > previous['Histogram']:
                momentum = "Strengthening"
            else:
                momentum = "Weakening"
        else:
            momentum = "Unknown"
            
        print(f"  Overall Trend: {trend}")
        print(f"  Momentum: {momentum}")
        
        # Check for potential crossovers
        if len(recent_df) > 1:
            if previous['MACD'] <= previous['Signal'] and latest['MACD'] > latest['Signal']:
                print(f"  * BULLISH CROSSOVER DETECTED!")
            elif previous['MACD'] >= previous['Signal'] and latest['MACD'] < latest['Signal']:
                print(f"  * BEARISH CROSSOVER DETECTED!")
        
        # Generate pipe-separated output for dashboard table parsing
        print("\n" + "="*80)
        print("TABLE_DATA_START")
        
        # Print header
        print("Time | Symbol | CMP | MACD | Signal | Histogram | Status")
        
        # Print data rows (same recent_df used for formatted output above)
        for _, row in recent_df.iterrows():
            date_str = format_date(row['timestamp'], include_time=is_intraday)
            close_val = f"{row['close']:.2f}"
            macd_val = f"{row['MACD']:.4f}"
            signal_val = f"{row['Signal']:.4f}"
            histogram_val = f"{row['Histogram']:.4f}"
            
            # Determine MACD status (same logic as above)
            if row['MACD'] > row['Signal']:
                if row['Histogram'] > 0:
                    status = "Bullish"
                else:
                    status = "Weakening"
            else:
                if row['Histogram'] < 0:
                    status = "Bearish"
                else:
                    status = "Recovering"
            
            # Print pipe-separated row
            print(f"{date_str} | {symbol} | {close_val} | {macd_val} | {signal_val} | {histogram_val} | {status}")
        
        print("TABLE_DATA_END")
        print("="*80)

def load_config():
    """Load MACD configuration"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'macd_config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        # Return default configuration
        return {
            "symbols": ["ADANIENT"],
            "fast_length": 12,
            "slow_length": 26,
            "signal_length": 9,
            "oscillator_ma_type": "EMA",
            "signal_ma_type": "EMA",
            "base_timeframe": "daily",
            "days_to_list": 10,
            "days_fallback_threshold": 300
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return {}

def main():
    """Main function to run MACD scanner"""
    print("Starting MACD Scanner...")
    
    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        return
    
    symbols = config.get('symbols', [])
    if not symbols:
        print("No symbols configured. Exiting.")
        return
    
    print(f"Processing {len(symbols)} symbol(s): {', '.join(symbols)}")
    
    # Calculate MACD for each symbol
    results = []
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        result = calculate_macd_for_symbol(symbol, config)
        results.append(result)
        
        # Small delay to avoid API rate limits
        time.sleep(0.5)
    
    # Display results
    display_macd_summary(results, config)
    print(f"\nMACD Scanner completed. Processed {len(results)} symbol(s).")

if __name__ == "__main__":
    main()