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
from . import config as base_cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_data_for_symbol(symbol: str, days_back: int) -> str:
    """
    Fetch historical data for a single symbol for the specified number of days.
    
    Args:
        symbol: Symbol to download data for
        days_back: Number of trading days to fetch
        
    Returns:
        Path to the combined CSV file
    """
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

                    headers = {'Accept': 'application/json'}

                    resp = requests.get(url, headers=headers, timeout=20)

                    try:
                        data = resp.json()
                        print(f"API Response for {self.symbol}: {data}")  # Debug print
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
                    fname = f"{self.symbol}_{interval_val}_combined.csv"
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

    # Calculate date range based on DAYS_BACK
    days_back = getattr(custom_config, 'DAYS_BACK', 30)
    
    today = datetime.now(timezone.utc).date()
    end_date = today
    
    # Find the most recent trading day
    while not is_trading_day(end_date):
        end_date = end_date - timedelta(days=1)
    
    # Calculate start date
    start_date = end_date
    trading_days_count = 0
    
    while trading_days_count < days_back:
        start_date = start_date - timedelta(days=1)
        if is_trading_day(start_date):
            trading_days_count += 1
    
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
            return combined_filepath
    
    return None

def is_trading_day(date):
    """Check if a date is a trading day (simplified)"""
    return date.weekday() < 5  # Monday=0, Sunday=6