"""
Parameterized data loader for fetching historical data.

Adapted from batch_download.py to accept symbol and duration parameters.
"""

import os
import time
import logging
import importlib
from typing import List, Dict, Tuple
from datetime import datetime, timedelta, timezone
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from symbol_mapper import SymbolMapper
except Exception:
    # minimal fallback mapper: returns symbol unchanged
    class SymbolMapper:
        def __init__(self):
            pass
        def map(self, s):
            return s

try:
    from download_historical_data import HistoricalDataDownloader
except Exception:
    HistoricalDataDownloader = None
    # We'll provide a simple inline downloader below if needed
try:
    import requests
except ImportError:
    requests = None
    logger.warning("requests library not available - some features will be limited")

# Import config - handle both relative and absolute imports
try:
    from .config import config as base_cfg
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
    from config import config as base_cfg

def fetch_data_for_symbol(symbol, days_back=30):
    """
    Fetch historical data for a single symbol for the specified number of days.
    
    Args:
        symbol: Symbol to download data for
        days_back: Number of trading days to fetch
        
    Returns:
        Path to the combined CSV file
    """
    # Check if data already exists and is sufficient before fetching
    symbol_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', symbol.upper())
    combined_filename = f"{symbol}_5_combined.csv"
    combined_filepath = os.path.join(symbol_dir, combined_filename)
    
    if os.path.exists(combined_filepath):
        try:
            import pandas as pd
            existing_df = pd.read_csv(combined_filepath)
            if not existing_df.empty:
                # Check if existing data covers the required date range
                existing_start = pd.to_datetime(existing_df['timestamp'].min()).date()
                existing_end = pd.to_datetime(existing_df['timestamp'].max()).date()
                
                # Calculate required date range
                today = datetime.now(timezone.utc).date()
                end_date = today
                while not is_trading_day(end_date):
                    end_date = end_date - timedelta(days=1)
                
                start_date = end_date - timedelta(days=days_back)
                while not is_trading_day(start_date):
                    start_date = start_date + timedelta(days=1)
                
                # If existing data covers the required range, return the existing file
                if existing_start <= start_date and existing_end >= end_date:
                    logger.info(f"Existing data for {symbol} already covers the required date range ({existing_start} to {existing_end})")
                    logger.info(f"Using existing combined file: {combined_filepath} ({len(existing_df)} records)")
                    
                    # Clean up any leftover chunk files if KEEP_CHUNK_FILES is False
                    keep_chunk_files = getattr(base_cfg, 'KEEP_CHUNK_FILES', True)
                    if not keep_chunk_files:
                        import glob
                        chunk_pattern = os.path.join(symbol_dir, f"{symbol}_5_*.csv")
                        chunk_files_to_clean = [f for f in glob.glob(chunk_pattern) if 'combined' not in f]
                        if chunk_files_to_clean:
                            logger.info(f"Cleaning up {len(chunk_files_to_clean)} leftover chunk files (KEEP_CHUNK_FILES=False)")
                            for chunk_file in chunk_files_to_clean:
                                try:
                                    os.remove(chunk_file)
                                    logger.debug(f"Removed leftover chunk file: {chunk_file}")
                                except Exception as e:
                                    logger.warning(f"Failed to remove chunk file {chunk_file}: {e}")
                    
                    return combined_filepath
                else:
                    logger.info(f"Existing data for {symbol} ({existing_start} to {existing_end}) doesn't cover required range ({start_date} to {end_date})")
                    logger.info("Fetching additional data...")
        except Exception as e:
            logger.warning(f"Error checking existing data for {symbol}: {e}")
    
    # Create a per-symbol copy of configuration
    import types
    custom_config = types.SimpleNamespace()
    # copy only needed settings from the module into an independent object
    for key in (
        'SYMBOL', 'UNIT', 'INTERVALS', 'INTERVAL',
        'CHUNK_DAYS', 'MAX_RETRIES', 'RETRY_BACKOFF_SECONDS',
        'API_BASE_URL', 'INTRADAY_API_URL', 'HOLIDAY_API_URL', 'OUTPUT_DIRECTORY', 'OUTPUT_FORMAT', 'ENABLE_HOLIDAY_CHECK',
        'KEEP_CHUNK_FILES'
    ):
        if hasattr(base_cfg, key):
            setattr(custom_config, key, getattr(base_cfg, key))
    # set symbol and days_back
    custom_config.SYMBOL = symbol
    custom_config.DAYS_BACK = days_back
    custom_config.USE_DAYS_BACK = True
    custom_config.OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    # Create symbol mapper once for efficiency
    symbol_mapper = SymbolMapper()
    
    # Load instrument mapping (if available) once and attempt to resolve the instrument key
    instrument_map = None
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.normpath(os.path.join(module_dir, 'config', 'instrument_mapping.json'))
        if os.path.exists(mapping_path):
            import json
            with open(mapping_path, 'r', encoding='utf-8') as mf:
                instrument_map = json.load(mf)
    except Exception:
        instrument_map = None

    # Helper: resolve instrument_key for a symbol using mapping
    def resolve_instrument_key(sym: str):
        if not instrument_map:
            return None
        key = instrument_map.get(sym.upper())
        if key and isinstance(key, dict) and 'instrument_key' in key:
            return key['instrument_key']
        # fallback: search trading_symbol fields
        for k, v in instrument_map.items():
            if isinstance(v, dict) and v.get('trading_symbol', '').upper() == sym.upper():
                return v.get('instrument_key')
        return None

    # If config requests validation, resolve and skip invalid symbols to avoid wasted retries
    if getattr(custom_config, 'VALIDATE_SYMBOLS', True):
        resolved_key = resolve_instrument_key(symbol)
        if not resolved_key:
            logger.error(f"Symbol {symbol} could not be resolved in instrument mapping — skipping")
            return None
        else:
            # Attach resolved key to custom_config so downloader can use it if needed
            custom_config.INSTRUMENT_KEY = resolved_key
    
    # Helper: split the configured date range into calendar-month aligned chunks
    def split_date_range_by_month(start_str: str, end_str: str):
        """
        Split inclusive YYYY-MM-DD start/end into list of (start,end) pairs aligned to calendar months.
        """
        start = datetime.strptime(start_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
        if end < start:
            raise ValueError("END_DATE must be >= START_DATE")

        chunks = []
        cur = start
        while cur <= end:
            last_day = calendar.monthrange(cur.year, cur.month)[1]
            month_end = datetime(cur.year, cur.month, last_day).date()
            chunk_end = month_end if month_end <= end else end
            chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            # move to first day of next month
            cur = month_end + timedelta(days=1)
        return chunks

    # Create downloader
    if HistoricalDataDownloader is not None:
        downloader = HistoricalDataDownloader(
            config_module=custom_config,
            symbol_mapper=symbol_mapper
        )
    else:
        # Inline simple downloader using Upstox API v3 as a fallback.
        import requests

        class SimpleDownloader:
            def __init__(self, config_module, symbol):
                self.cfg = config_module
                self.symbol = symbol
                # Try to load instrument mapping to resolve proper instrument_key
                try:
                    module_dir = os.path.dirname(os.path.abspath(__file__))
                    mapping_path = os.path.normpath(os.path.join(module_dir, 'config', 'instrument_mapping.json'))
                    if os.path.exists(mapping_path):
                        import json
                        with open(mapping_path, 'r', encoding='utf-8') as mf:
                            self.instrument_map = json.load(mf)
                    else:
                        self.instrument_map = None
                except Exception:
                    self.instrument_map = None

            def _build_url(self, start_date: str, end_date: str, interval: str):
                base = getattr(self.cfg, 'API_BASE_URL', 'https://api.upstox.com/v3/historical-candle')
                unit = getattr(self.cfg, 'UNIT', 'minutes')
                # Use resolved instrument_key from config, fallback to mapping lookup
                instrument_key = getattr(self.cfg, 'INSTRUMENT_KEY', None)
                if not instrument_key:
                    # Try to resolve from mapping
                    if self.instrument_map and self.symbol.upper() in self.instrument_map:
                        symbol_data = self.instrument_map[self.symbol.upper()]
                        if isinstance(symbol_data, dict) and 'instrument_key' in symbol_data:
                            instrument_key = symbol_data['instrument_key']
                    # Final fallback
                    if not instrument_key:
                        instrument_key = f"NSE_EQ|{self.symbol}"
                safe_key = instrument_key.replace('|', '%7C')
                to_date = end_date
                from_date = start_date

                if unit == 'days':
                    return f"{base}/{safe_key}/days/{interval}/{to_date}/{from_date}"
                else:
                    return f"{base}/{safe_key}/minutes/{interval}/{to_date}/{from_date}"

            def run(self):
                # Get dates from config
                start_date = getattr(self.cfg, 'START_DATE')
                end_date = getattr(self.cfg, 'END_DATE')
                out_dir = getattr(self.cfg, 'OUTPUT_DIRECTORY', 'data')
                os.makedirs(out_dir, exist_ok=True)

                intervals = getattr(self.cfg, 'INTERVALS', [getattr(self.cfg, 'INTERVAL', '5')])
                for interval_val in intervals:
                    url = self._build_url(start_date, end_date, interval=str(interval_val))

                    # auth token: prefer env var UPSTOX_ACCESS_TOKEN
                    token = os.environ.get('UPSTOX_ACCESS_TOKEN', '')
                    headers = {'Accept': 'application/json'}
                    if token:
                        headers['Authorization'] = f'Bearer {token}'

                    resp = requests.get(url, headers=headers, timeout=20)

                    try:
                        data = resp.json()
                    except Exception:
                        logger.exception("Failed to parse JSON response")
                        return None

                    candles = []
                    if isinstance(data, dict) and 'data' in data:
                        d = data['data']
                        if isinstance(d, dict) and 'candles' in d:
                            candles = d['candles']
                        elif isinstance(d, list):
                            candles = d
                    elif isinstance(data, list):
                        candles = data

                    if not candles:
                        logger.error(f"No candles returned for {self.symbol}")
                        return None

                    rows = []
                    for c in candles:
                        if isinstance(c, (list, tuple)):
                            rows.append(c[:6])
                        elif isinstance(c, dict):
                            ts = c.get('timestamp') or c.get('time') or c.get('date')
                            rows.append([
                                ts,
                                c.get('open', 0),
                                c.get('high', 0),
                                c.get('low', 0),
                                c.get('close', 0),
                                c.get('volume', 0)
                            ])

                    symbol_dir = os.path.join(out_dir, self.symbol.upper())
                    os.makedirs(symbol_dir, exist_ok=True)
                    # Create unique filename for each chunk using date range
                    start_date = getattr(self.cfg, 'START_DATE')
                    end_date = getattr(self.cfg, 'END_DATE')
                    fname = f"{self.symbol}_{interval_val}_{start_date}_{end_date}.csv"
                    fpath = os.path.join(symbol_dir, fname)
                    try:
                        with open(fpath, 'w', encoding='utf-8') as f:
                            f.write('timestamp,open,high,low,close,volume\n')
                            for r in rows:
                                line = ','.join([str(x) for x in r])
                                f.write(line + '\n')
                    except Exception:
                        logger.exception(f"Failed to write file {fpath}")
                        return None

                    return fpath

        downloader = SimpleDownloader(custom_config, symbol)

    # Use the days_back parameter passed to the function
    # This overrides any config defaults
    # For DMA scanner, we want calendar days, not trading days
    calendar_days_back = days_back

    today = datetime.now(timezone.utc).date()
    end_date = today

    # Find the most recent trading day for end_date
    while not is_trading_day(end_date):
        end_date = end_date - timedelta(days=1)

    # Calculate start date using calendar days (not trading days)
    # This ensures we get the full historical range requested
    start_date = end_date - timedelta(days=calendar_days_back)

    # Find the first trading day on or after the calculated start date
    while not is_trading_day(start_date):
        start_date = start_date + timedelta(days=1)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    custom_config.START_DATE = start_date_str
    custom_config.END_DATE = end_date_str

    logger.info(f"Fetching data for {symbol} from {start_date_str} to {end_date_str}")

    # Use monthly chunks like the original code
    all_chunks = split_date_range_by_month(start_date_str, end_date_str)
    logger.info(f"Total required chunks: {len(all_chunks)}")
    
    # Download chunks
    chunk_files = []
    for idx, (chunk_start, chunk_end) in enumerate(all_chunks, start=1):
        logger.info(f"Chunk {idx}/{len(all_chunks)} for {symbol}: {chunk_start} -> {chunk_end}")
        
        # Update config for this chunk
        custom_config.START_DATE = chunk_start
        custom_config.END_DATE = chunk_end
        
        # Create downloader for this chunk
        if HistoricalDataDownloader is not None:
            chunk_downloader = HistoricalDataDownloader(
                config_module=custom_config,
                symbol_mapper=symbol_mapper
            )
        else:
            chunk_downloader = SimpleDownloader(custom_config, symbol)
        
        chunk_output = chunk_downloader.run()
        if chunk_output:
            chunk_files.append(chunk_output)
            logger.info(f"Chunk saved: {chunk_output}")
        else:
            logger.error(f"Failed to download chunk {chunk_start} -> {chunk_end}")
    
    # FETCH CURRENT DAY'S DATA IF TODAY IS A TRADING DAY
    today = datetime.now(timezone.utc).date()
    if is_trading_day(today) and requests is not None:
        logger.info(f"Today ({today}) is a trading day - fetching current day data")
        try:
            # Get instrument key for the symbol
            instrument_key = getattr(custom_config, 'INSTRUMENT_KEY', None)
            if not instrument_key:
                # Try to resolve from mapping
                if instrument_map:
                    im = instrument_map.get(symbol.upper())
                    if not im:
                        for k, v in instrument_map.items():
                            if isinstance(v, dict) and v.get('trading_symbol', '').upper() == symbol.upper():
                                im = v
                                break
                    if im and 'instrument_key' in im:
                        instrument_key = im['instrument_key']

            if not instrument_key:
                instrument_key = f"NSE_EQ|{symbol}"

            # Build intraday API URL
            intraday_url = getattr(custom_config, 'INTRADAY_API_URL', 'https://api.upstox.com/v3/historical-candle/intraday')
            unit = getattr(custom_config, 'UNIT', 'minutes')
            interval = str(getattr(custom_config, 'INTERVAL', '5'))
            safe_key = instrument_key.replace('|', '%7C')

            if unit == 'minutes':
                url = f"{intraday_url}/{safe_key}/minutes/{interval}"
            else:
                # Fallback to minutes if unit is not supported for intraday
                url = f"{intraday_url}/{safe_key}/minutes/{interval}"

            # Make API request
            token = os.environ.get('UPSTOX_ACCESS_TOKEN', '')
            headers = {'Accept': 'application/json'}
            if token:
                headers['Authorization'] = f'Bearer {token}'

            response = requests.get(url, headers=headers, timeout=20)

            if response.status_code == 200:
                data = response.json()
                candles = []

                if isinstance(data, dict) and 'data' in data:
                    d = data['data']
                    if isinstance(d, dict) and 'candles' in d:
                        candles = d['candles']
                    elif isinstance(d, list):
                        candles = d
                elif isinstance(data, list):
                    candles = data

                if candles:
                    # Create today's data file
                    interval = str(getattr(custom_config, 'INTERVAL', '5'))
                    today_str = today.strftime("%Y-%m-%d")
                    symbol_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', symbol.upper())
                    os.makedirs(symbol_dir, exist_ok=True)
                    fname = f"{symbol}_{interval}_{today_str}_{today_str}.{getattr(custom_config, 'OUTPUT_FORMAT', 'csv')}"
                    fpath = os.path.join(symbol_dir, fname)

                    try:
                        with open(fpath, 'w', encoding='utf-8') as f:
                            f.write('timestamp,open,high,low,close,volume\n')
                            for c in candles:
                                if isinstance(c, (list, tuple)):
                                    line = ','.join([str(x) for x in c[:6]])
                                    f.write(line + '\n')
                                elif isinstance(c, dict):
                                    ts = c.get('timestamp') or c.get('time') or c.get('date')
                                    line = ','.join([
                                        str(ts),
                                        str(c.get('open', 0)),
                                        str(c.get('high', 0)),
                                        str(c.get('low', 0)),
                                        str(c.get('close', 0)),
                                        str(c.get('volume', 0))
                                    ])
                                    f.write(line + '\n')

                        logger.info(f"Current day data saved: {fpath}")
                        chunk_files.append(fpath)  # Add to chunk_files list

                    except Exception as e:
                        logger.warning(f"Failed to save current day data: {e}")
                else:
                    logger.info("No current day data available")
            else:
                logger.warning(f"Failed to fetch current day data: HTTP {response.status_code}")

        except Exception as e:
            logger.warning(f"Error fetching current day data: {e}")
    else:
        if not is_trading_day(today):
            logger.info(f"Today ({today}) is not a trading day - skipping current day data fetch")
        if requests is None:
            logger.info("requests library not available - skipping current day data fetch")
    
    # Create combined file
    if chunk_files:
        combined_data = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header + data
                        data_lines = lines[1:] if combined_data else lines
                        combined_data.extend(data_lines)
            except Exception as e:
                logger.warning(f"Failed to read chunk file {chunk_file}: {e}")
        
        if combined_data:
            # Remove duplicates and sort
            seen_timestamps = set()
            unique_data = []
            
            for line in combined_data:
                line = line.strip()
                if line and not line.startswith('timestamp,'):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        timestamp = parts[0]
                        if timestamp not in seen_timestamps:
                            seen_timestamps.add(timestamp)
                            unique_data.append(line)
            
            unique_data.sort(key=lambda x: x.split(',')[0])
            
            # Create combined file
            symbol_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', symbol.upper())
            os.makedirs(symbol_dir, exist_ok=True)
            combined_filename = f"{symbol}_5_combined.csv"
            combined_filepath = os.path.join(symbol_dir, combined_filename)
            
            with open(combined_filepath, 'w', encoding='utf-8') as f:
                f.write('timestamp,open,high,low,close,volume\n')
                for line in unique_data:
                    f.write(line + '\n')
            
            logger.info(f"Combined file created: {combined_filepath} ({len(unique_data)} records)")
            
            # Clean up chunk files if KEEP_CHUNK_FILES is False
            keep_chunk_files = getattr(custom_config, 'KEEP_CHUNK_FILES', True)
            if not keep_chunk_files:
                logger.info("Cleaning up chunk files (KEEP_CHUNK_FILES=False)")
                for chunk_file in chunk_files:
                    try:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                            logger.debug(f"Removed chunk file: {chunk_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove chunk file {chunk_file}: {e}")
                logger.info(f"Cleaned up {len(chunk_files)} chunk files")
            
            return combined_filepath
    
    return None

def is_trading_day(date):
    """Check if a date is a trading day using Upstox API or fallback to weekend check"""
    # Check if holiday checking is enabled
    enable_holiday_check = getattr(base_cfg, 'ENABLE_HOLIDAY_CHECK', True)

    if not enable_holiday_check or requests is None:
        # Fallback to weekend-only checking (original logic)
        return date.weekday() < 5  # Monday=0, Sunday=6

    try:
        date_str = date.strftime("%Y-%m-%d")
        url = f"{getattr(base_cfg, 'HOLIDAY_API_URL', 'https://api.upstox.com/v2/market/holidays')}/{date_str}"

        # Make request without authentication (holiday API works without auth)
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            holiday_data = data.get('data', [])

            if holiday_data:
                logger.info(f"Date {date_str} is a holiday: {holiday_data[0].get('description', 'Holiday')}")

            # If no holiday data returned, it's a trading day
            return not holiday_data
        else:
            # If API fails, assume it's a trading day (fallback)
            logger.warning(f"Failed to check holiday for {date_str}, assuming trading day")
            return True
    except Exception as e:
        logger.warning(f"Error checking holiday for {date_str}: {e}, assuming trading day")
        return True

if __name__ == "__main__":
    # Test the data loader with a sample symbol
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    else:
        symbol = "ITC"  # Default test symbol
        days_back = 30
    
    print(f"Testing data loader with symbol: {symbol}, days_back: {days_back}")
    
    result = fetch_data_for_symbol(symbol, days_back)
    
    if result:
        print(f"Success! Data saved to: {result}")
    else:
        print("Failed to fetch data")