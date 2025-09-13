"""
DMA Scanner - Calculates and displays DMA (Displaced Moving Average) values for specified periods.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import sys
import pytz  # Added for timezone handling

def calculate_dma(data, period, displacement=1):
    """Calculate DMA (Displaced Moving Average) with backward displacement"""
    if len(data) < period + displacement:
        # Not enough data for this DMA period
        return pd.Series([float('nan')] * len(data), index=data.index)

    # Calculate simple moving average
    sma = data['close'].rolling(window=period).mean()

    # Apply backward displacement (shift forward in time by displacement periods)
    dma = sma.shift(displacement)

    return dma

def format_date(date_str):
    """Format date to DD-Mon-YYYY"""
    if isinstance(date_str, str):
        date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
    else:
        date_obj = date_str
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
    """Fetch data for specific timeframe with improved accuracy"""
    try:
        # Import data_loader functionality
        sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader'))
        from data_loader import fetch_data_for_symbol

        print(f"Using data_loader.py to fetch {days_back} days of data for {symbol}")

        # Use data_loader to fetch 5-minute data
        combined_file = fetch_data_for_symbol(symbol, days_back)

        if combined_file and os.path.exists(combined_file):
            print(f"Loading data from: {combined_file}")

            # Read the combined CSV file
            df = pd.read_csv(combined_file)

            # Convert timestamp to datetime and localize to IST
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
            df = df.sort_values('timestamp')

            print(f"Loaded {len(df)} data points from data_loader.py")

            # Apply market open filter BEFORE resampling for accuracy
            if timeframe in ['5mins', '15mins', '30mins', '1hour', '4hours']:
                # Filter for market hours: 9:15 to 15:30 IST
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                market_open_mask = (
                    ((df['hour'] == 9) & (df['minute'] >= 15)) |
                    ((df['hour'] > 9) & (df['hour'] < 15)) |
                    ((df['hour'] == 15) & (df['minute'] <= 30))
                )
                df = df[market_open_mask].copy()
                df = df.drop(['hour', 'minute'], axis=1, errors='ignore')
                print(f"Filtered to {len(df)} market hours data points")
            elif timeframe in ['daily', 'weekly', 'monthly']:
                # For daily and higher, filter to trading days (exclude weekends and holidays if possible)
                df['date'] = df['timestamp'].dt.date
                # Basic weekend filter (could be enhanced with holiday API)
                df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
                trading_day_mask = df['weekday'] < 5  # Monday to Friday
                df = df[trading_day_mask].copy()
                df = df.drop(['date', 'weekday'], axis=1, errors='ignore')
                print(f"Filtered to {len(df)} trading day data points")

            # Resample to requested timeframe
            if timeframe != '5mins':
                print(f"Resampling to {timeframe} timeframe")
                if timeframe == '15mins':
                    rule = '15min'
                elif timeframe == '30mins':
                    rule = '30min'
                elif timeframe == '1hour':
                    rule = 'h'
                elif timeframe == '4hours':
                    rule = '4h'
                elif timeframe == 'daily':
                    rule = 'D'
                elif timeframe == 'weekly':
                    rule = 'W'
                elif timeframe == 'monthly':
                    rule = 'M'
                else:
                    print(f"Unsupported timeframe: {timeframe}")
                    return None

                # Resample with proper OHLC aggregation
                df = df.set_index('timestamp').resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().reset_index()

                print(f"Resampled to {len(df)} {timeframe} data points")

            # Validate data integrity
            if df.empty:
                print("No data after filtering/resampling")
                return None

            # Check for basic data quality
            invalid_mask = (df['high'] < df['low']) | (df['open'] <= 0) | (df['close'] <= 0)
            if invalid_mask.any():
                print(f"Warning: {invalid_mask.sum()} invalid OHLC records found, removing")
                df = df[~invalid_mask]

            # Remove duplicates
            df = df.drop_duplicates(subset='timestamp')

            print(f"Final data: {len(df)} records")
            return df

        else:
            print("Failed to fetch data from data_loader.py")
            return None

    except Exception as e:
        print(f"Error in fetch_timeframe_data: {e}")
        return None

def run_dma_scanner():
    """Main DMA scanner function"""
    print("DMA Scanner Starting...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'dma_config.json')
    print(f"Loading config from: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Config loaded successfully")
    except FileNotFoundError:
        print("Config file not found, creating default...")
        # Create default config
        config = {
            "symbols": ["RELIANCE"],
            "dma_periods": [10, 20, 50],
            "base_timeframe": "1hour",
            "days_to_list": 2,
            "days_fallback_threshold": 1600,
            "displacement": 1
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created default config at: {config_path}")

    symbols = config['symbols']
    dma_periods = config['dma_periods']
    base_timeframe = config.get('base_timeframe', '15mins')
    days_to_list = config.get('days_to_list', 2)
    days_fallback_threshold = config.get('days_fallback_threshold', 200)
    displacement = config.get('displacement', 1)

    print(f"Scanning symbols: {symbols}")
    print(f"DMA periods: {dma_periods}")
    print(f"Base timeframe: {base_timeframe}")
    print(f"Days to display: {days_to_list}")
    print(f"Displacement: {displacement}")

    # Process each symbol
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")

        # Calculate required days based on DMA periods
        # DMA needs more data for longer periods to stabilize
        max_dma_period = max(dma_periods)
        ideal_days = max(max_dma_period * 3, 200)  # At least 3x the longest period or 200 days for better accuracy
        actual_days = min(days_fallback_threshold, ideal_days)

        print(f"DMA needs {ideal_days} days for accuracy, but limited to max {days_fallback_threshold} days")
        print(f"NOTE: This assumes proper historical data fetching. If data spans < {ideal_days} days,")
        print(f"      the data_loader may not be configured correctly for historical data retrieval.")
        print(f"Fetching {actual_days} days of data")

        # Fetch data
        df = fetch_timeframe_data(symbol, base_timeframe, days_back=actual_days)
        if df is None or df.empty:
            print(f"CRITICAL: Failed to fetch ANY data for {symbol}")
            print("SKIPPING {symbol} - No data available")
            continue

        print(f"Fetched {len(df)} {base_timeframe} data points for {symbol}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # STRICT DATA SUFFICIENCY CHECK
        insufficient_periods = []
        sufficient_periods = []

        for period in dma_periods:
            required_points = period + displacement + 20  # period + displacement + generous buffer
            if len(df) < required_points:
                insufficient_periods.append(f"DMA{period} (needs {required_points}, has {len(df)})")
            else:
                sufficient_periods.append(f"DMA{period}")

        if insufficient_periods:
            print("CRITICAL DATA INSUFFICIENCY DETECTED!")
            print(f"Insufficient data for: {', '.join(insufficient_periods)}")
            print(f"Sufficient data for: {', '.join(sufficient_periods)}" if sufficient_periods else "None sufficient")
            print("SOLUTION: The data_loader is not fetching historical data properly.")
            print(f"         Current data has only {len(df)} points, but DMA calculations need:")
            for period in dma_periods:
                required_points = period + displacement + 20
                print(f"         - DMA{period}: {required_points} data points minimum")
            print(f"         RECOMMENDATION: Check data_loader configuration and ensure it's fetching")
            print(f"         historical data, not future data. Current data spans only {len(df)} points")
            print(f"         from {df['timestamp'].min()} to {df['timestamp'].max()}")
            print("SKIPPING CALCULATIONS FOR {symbol} - INACCURATE RESULTS WOULD BE GENERATED")
            continue  # Skip to next symbol

        print("DATA SUFFICIENCY CHECK PASSED - Proceeding with calculations...")

        # Calculate DMA for each period
        dma_results = {}
        for period in dma_periods:
            dma_series = calculate_dma(df, period, displacement)
            dma_results[f'dma_{period}'] = dma_series
            print(f"Calculated DMA({period}) with displacement {displacement}")

        # Add DMA columns to dataframe
        for dma_col, dma_series in dma_results.items():
            df[dma_col] = dma_series

        # FILTER TO VALID ROWS ONLY (NO N/A IN TABLE)
        # Find the latest index where ANY DMA value is NaN
        max_nan_index = -1
        for dma_col in dma_results.keys():
            nan_indices = df[df[dma_col].isna()].index
            if not nan_indices.empty:
                max_nan_index = max(max_nan_index, nan_indices.max())

        # Keep only rows after the last NaN
        if max_nan_index >= 0:
            df = df.iloc[max_nan_index + 1:].copy()
            print(f"Filtered to {len(df)} valid rows (removed {max_nan_index + 1} rows with NaN DMA values)")

        if df.empty:
            print("No valid DMA data after filtering - insufficient data even after fetch")
            continue

        # Filter data for the specified number of days
        if not df.empty:
            latest_date = df['timestamp'].max().date()
            start_date = latest_date - timedelta(days=days_to_list - 1)
            df = df[df['timestamp'].dt.date >= start_date]

            print(f"Showing data for last {days_to_list} days")

        # Save data with DMA calculations to CSV file
        os.makedirs(f"data/{symbol}", exist_ok=True)
        csv_path = f"data/{symbol}/{symbol}_dma_data.csv"

        # Keep NaN values as NaN for proper numerical analysis
        csv_df = df.copy()
        csv_df.to_csv(csv_path, index=False)
        print(f"Data with DMA calculations saved to: {csv_path}")
        print(f"Final dataframe shape: {df.shape}")
        print(f"Final dataframe columns: {list(df.columns)}")

        # Display results in table format
        print(f"\n{'='*100}")
        print(f"DMA ANALYSIS - {symbol.upper()}")
        print(f"{'='*100}")

        # Create headers
        headers = ['Time', 'Symbol', 'CMP'] + [f'DMA{period}' for period in dma_periods]

        # Calculate column widths
        col_widths = {}
        for header in headers:
            col_widths[header] = len(header)

        # Update widths based on data
        for _, row in df.iterrows():
            for header in headers:
                if header == 'Time':
                    dt = row['timestamp']
                    if pd.isna(dt):
                        value = 'N/A'
                    else:
                        hour = dt.hour if dt.hour != 0 else 12
                        time_str = f"{hour}:{dt.minute:02d}:{dt.second:02d}"
                        date_str = dt.strftime("%d-%m-%Y")
                        value = f"{date_str} {time_str}"
                elif header == 'Symbol':
                    value = symbol
                elif header == 'CMP':
                    close_val = row['close']
                    if pd.isna(close_val) or str(close_val).lower() in ['nat', 'nan']:
                        value = 'N/A'
                    else:
                        try:
                            value = f"{float(close_val):.2f}"
                        except (ValueError, TypeError):
                            value = 'N/A'
                else:
                    period = int(header.replace('DMA', ''))
                    dma_value = row[f'dma_{period}']
                    if pd.isna(dma_value):
                        value = 'N/A'
                    else:
                        value = f"{dma_value:.2f}"
                col_widths[header] = max(col_widths[header], len(str(value)))

        # Print table header
        header_row = ' | '.join(header.ljust(col_widths[header]) for header in headers)
        print(header_row)
        print('-' * len(header_row))

        # Print data rows
        for _, row in df.iterrows():
            row_data = []
            for header in headers:
                if header == 'Time':
                    dt = row['timestamp']
                    if pd.isna(dt):
                        value = 'N/A'
                    else:
                        hour = dt.hour if dt.hour != 0 else 12
                        time_str = f"{hour}:{dt.minute:02d}:{dt.second:02d}"
                        date_str = dt.strftime("%d-%m-%Y")
                        value = f"{date_str} {time_str}"
                elif header == 'Symbol':
                    value = symbol
                elif header == 'CMP':
                    close_val = row['close']
                    if pd.isna(close_val):
                        value = 'N/A'
                    else:
                        try:
                            value = f"{float(close_val):.2f}"
                        except (ValueError, TypeError):
                            value = 'N/A'
                else:
                    period = int(header.replace('DMA', ''))
                    dma_value = row[f'dma_{period}']
                    if pd.isna(dma_value):
                        value = 'N/A'
                    else:
                        value = f"{dma_value:.2f}"
                row_data.append(str(value).ljust(col_widths[header]))
            print(' | '.join(row_data))

        print(f"{'='*100}\n")

    print("DMA Scanner completed successfully!")

if __name__ == "__main__":
    run_dma_scanner()