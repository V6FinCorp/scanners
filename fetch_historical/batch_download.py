"""
Batch download of historical data for multiple symbols.

This script demonstrates how to download historical data for multiple symbols
in a batch process using the HistoricalDataDownloader.
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
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_multiple_symbols(symbols: List[str], config_module=None) -> Dict[str, Dict[str, object]]:
    """
    Download historical data for multiple symbols.
    
    Args:
        symbols: List of symbols to download data for
        config_module: Config module to use for download settings
        
    Returns:
        Dictionary mapping symbols to output filepaths
    """
    if config_module is None:
        config_module = config
        
    # Create symbol mapper once for efficiency
    symbol_mapper = SymbolMapper()
    
    # results: symbol -> { 'chunks': [filepaths], 'missing': [(start,end), ...] }
    results: Dict[str, Dict[str, object]] = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Processing symbol: {symbol}")
            
            # Create a per-symbol copy of configuration so we don't mutate the imported
            # module (importlib returns the same module object) which caused later
            # symbols to inherit modified dates from prior runs.
            import types
            from config import config as base_cfg
            custom_config = types.SimpleNamespace()
            # copy only needed settings from the module into an independent object
            for key in (
                'SYMBOL', 'UNIT', 'INTERVALS', 'INTERVAL',
                'CHUNK_DAYS', 'MAX_RETRIES', 'RETRY_BACKOFF_SECONDS',
                'API_BASE_URL', 'INTRADAY_API_URL', 'HOLIDAY_API_URL', 'OUTPUT_DIRECTORY', 'OUTPUT_FORMAT', 'DAYS_BACK', 'ENABLE_HOLIDAY_CHECK',
                'USE_DAYS_BACK', 'START_DATE', 'END_DATE', 'KEEP_CHUNK_FILES'
            ):
                if hasattr(base_cfg, key):
                    setattr(custom_config, key, getattr(base_cfg, key))
            # set symbol for this run
            custom_config.SYMBOL = symbol

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
            from config import config as import_cfg
            if getattr(import_cfg, 'VALIDATE_SYMBOLS', True):
                resolved_key = resolve_instrument_key(symbol)
                if not resolved_key:
                    logger.error(f"Symbol {symbol} could not be resolved in instrument mapping — skipping to avoid repeated failures")
                    results[symbol] = {'chunks': [], 'missing': [], 'invalid': True}
                    continue
                else:
                    # Attach resolved key to custom_config so downloader can use it if needed
                    custom_config.INSTRUMENT_KEY = resolved_key
            
            # Helper: split the configured date range into calendar-month aligned chunks
            def split_date_range_by_month(start_str: str, end_str: str):
                """
                Split inclusive YYYY-MM-DD start/end into list of (start,end) pairs aligned to calendar months.
                Example: 2025-01-15 -> 2025-03-10 yields:
                  [ (2025-01-15,2025-01-31), (2025-02-01,2025-02-28), (2025-03-01,2025-03-10) ]
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

            # Create downloader (we will reuse it across chunks by updating the config module)
            if HistoricalDataDownloader is not None:
                downloader = HistoricalDataDownloader(
                    config_module=custom_config,
                    symbol_mapper=symbol_mapper
                )
            else:
                # Inline simple downloader using Upstox API v3 as a fallback.
                # It writes CSV files per chunk named {SYMBOL}_{start}_{end}.csv and returns the path.
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
                        # The user requested the URL format to be: .../:to_date/:from_date
                        # i.e. to_date first then from_date in the path
                        base = getattr(self.cfg, 'API_BASE_URL', 'https://api.upstox.com/v3/historical-candle')
                        unit = getattr(self.cfg, 'UNIT', 'minutes')
                        # interval passed explicitly
                        # Resolve instrument_key from mapping if available, else fallback
                        instrument_key = None
                        try:
                            if self.instrument_map:
                                # mapping keys are usually trading symbols
                                im = self.instrument_map.get(self.symbol.upper())
                                if not im:
                                    # also try trading_symbol match
                                    for k, v in self.instrument_map.items():
                                        if isinstance(v, dict) and v.get('trading_symbol', '').upper() == self.symbol.upper():
                                            im = v
                                            break
                                if im and 'instrument_key' in im:
                                    instrument_key = im['instrument_key']
                        except Exception:
                            instrument_key = None

                        if not instrument_key:
                            instrument_key = f"NSE_EQ|{self.symbol}"

                        safe_key = instrument_key.replace('|', '%7C')
                        # Per your exact requirement, put to_date then from_date
                        to_date = end_date
                        from_date = start_date

                        if unit == 'days':
                            return f"{base}/{safe_key}/days/{interval}/{to_date}/{from_date}"
                        elif unit == 'weeks':
                            return f"{base}/{safe_key}/weeks/{interval}/{to_date}/{from_date}"
                        elif unit == 'months':
                            return f"{base}/{safe_key}/months/{interval}/{to_date}/{from_date}"
                        else:
                            # default to minutes
                            return f"{base}/{safe_key}/minutes/{interval}/{to_date}/{from_date}"

                    def run(self):
                        # Get dates from config (now calculated based on DAYS_BACK)
                        start_date = getattr(self.cfg, 'START_DATE')
                        end_date = getattr(self.cfg, 'END_DATE')
                        out_dir = getattr(self.cfg, 'OUTPUT_DIRECTORY', 'historical_data')
                        os.makedirs(out_dir, exist_ok=True)

                        # We'll iterate over configured INTERVALS at the caller level; for a single
                        # run call this with a single interval value. Build URL using to_date/from_date order.
                        url = self._build_url(start_date, end_date, interval=str(getattr(self.cfg, 'INTERVAL', '5')))

                        # auth token: prefer env var UPSTOX_ACCESS_TOKEN
                        token = os.environ.get('UPSTOX_ACCESS_TOKEN', '')
                        headers = {'Accept': 'application/json'}
                        if token:
                            headers['Authorization'] = f'Bearer {token}'

                        def _do_request(u):
                            try:
                                return requests.get(u, headers=headers, timeout=20)
                            except Exception as e:
                                logger.exception(f"HTTP request failed for {u}: {e}")
                                return None

                        # The caller will iterate intervals and call run() for each. Here, assume
                        # the URL already follows to_date/from_date order. Just attempt it and
                        # return the resp if successful. Do NOT attempt to swap order here because
                        # the user explicitly requested the to_date/from_date ordering in the path.
                        resp = _do_request(url)
                        chosen_order = 'to_date/from_date'

                        try:
                            data = resp.json()
                        except Exception:
                            logger.exception("Failed to parse JSON response")
                            return None

                        # Extract candles (various possible shapes)
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
                            logger.error(f"No candles returned for {self.symbol} {start_date}->{end_date}")
                            return None

                        # Normalize to list of rows [timestamp, open, high, low, close, volume]
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

                        # Place files under per-symbol subfolder and include interval in filename
                        symbol_dir = os.path.join(out_dir, self.symbol.upper())
                        os.makedirs(symbol_dir, exist_ok=True)
                        interval = str(getattr(self.cfg, 'INTERVAL', '5'))
                        fname = f"{self.symbol}_{interval}_{start_date}_{end_date}.{getattr(self.cfg, 'OUTPUT_FORMAT', 'csv')}"
                        fpath = os.path.join(symbol_dir, fname)
                        try:
                            with open(fpath, 'w', encoding='utf-8') as f:
                                f.write('timestamp,open,high,low,close,volume\n')
                                for r in rows:
                                    # ensure values are simple
                                    line = ','.join([str(x) for x in r])
                                    f.write(line + '\n')
                        except Exception:
                            logger.exception(f"Failed to write chunk file {fpath}")
                            return None

                        return fpath

                downloader = SimpleDownloader(custom_config, symbol)

            # Calculate date range based on configuration method
            use_days_back = getattr(custom_config, 'USE_DAYS_BACK', True)
            
            # Track holidays found in our date range
            holidays_found = []

            # Function to check if a date is a trading day using Upstox API
            def is_trading_day(date):
                """Check if a date is a trading day using Upstox holiday API or fallback to weekend check"""
                # Check if holiday checking is enabled
                enable_holiday_check = getattr(custom_config, 'ENABLE_HOLIDAY_CHECK', True)

                if not enable_holiday_check:
                    # Fallback to weekend-only checking (original logic)
                    return date.weekday() < 5  # Monday=0, Sunday=6

                try:
                    date_str = date.strftime("%Y-%m-%d")
                    url = f"{getattr(custom_config, 'HOLIDAY_API_URL', 'https://api.upstox.com/v2/market/holidays')}/{date_str}"

                    # Make request without authentication (holiday API works without auth)
                    headers = {'Accept': 'application/json'}
                    response = requests.get(url, headers=headers, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        holiday_data = data.get('data', [])

                        if holiday_data:
                            # This is a holiday - collect the information
                            for holiday in holiday_data:
                                holiday_info = {
                                    'date': date_str,
                                    'description': holiday.get('description', 'Holiday'),
                                    'holiday_type': holiday.get('holiday_type', 'Unknown')
                                }
                                holidays_found.append(holiday_info)

                        # If no holiday data returned, it's a trading day
                        return not holiday_data
                    else:
                        # If API fails, assume it's a trading day (fallback)
                        logger.warning(f"Failed to check holiday for {date_str}, assuming trading day")
                        return True
                except Exception as e:
                    logger.warning(f"Error checking holiday for {date_str}: {e}, assuming trading day")
                    return True
            
            if use_days_back:
                # Method 1: DAYS_BACK - calculate from current date
                days_back = getattr(custom_config, 'DAYS_BACK', 30)
                
                # Find the end date (last trading day)
                today = datetime.now(timezone.utc).date()
                end_date = today
                
                # Find the most recent trading day (walking backwards from today)
                while not is_trading_day(end_date):
                    end_date = end_date - timedelta(days=1)
                
                # Calculate start date by going back N trading days from end_date
                start_date = end_date
                trading_days_count = 0
                
                while trading_days_count < days_back:
                    start_date = start_date - timedelta(days=1)
                    if is_trading_day(start_date):
                        trading_days_count += 1
                
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                
                logger.info(f"Using DAYS_BACK method: fetching {days_back} trading days from {start_date_str} to {end_date_str}")
            else:
                # Method 2: Fixed date range
                start_date_str = getattr(custom_config, 'START_DATE', None)
                end_date_str = getattr(custom_config, 'END_DATE', None)
                
                if not start_date_str or not end_date_str:
                    raise ValueError("START_DATE and END_DATE must be specified when USE_DAYS_BACK=False")
                
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                
                logger.info(f"Using fixed date range: {start_date_str} to {end_date_str}")

            custom_config.START_DATE = start_date_str
            custom_config.END_DATE = end_date_str

            logger.info(f"Fetching data for {symbol} from {start_date_str} to {end_date_str}")

            # Log holidays found in the date range
            enable_holiday_check = getattr(custom_config, 'ENABLE_HOLIDAY_CHECK', True)
            if not enable_holiday_check:
                logger.info("Holiday checking disabled - using weekend-only logic")
            elif holidays_found:
                logger.info(f"Holidays in date range: {len(holidays_found)} found")
                for holiday in holidays_found:
                    logger.info(f"  - {holiday['date']}: {holiday['description']} ({holiday['holiday_type']})")
            else:
                logger.info("No holidays found in the date range")

            # Check for existing files and determine which chunks need to be downloaded
            out_dir = getattr(custom_config, 'OUTPUT_DIRECTORY', 'historical_data')
            symbol_dir = os.path.join(out_dir, symbol.upper())
            os.makedirs(symbol_dir, exist_ok=True)

            # Parse existing files to determine what date ranges are already covered
            existing_ranges = []
            if os.path.exists(symbol_dir):
                for filename in os.listdir(symbol_dir):
                    if filename.startswith(f"{symbol}_") and filename.endswith(".csv"):
                        try:
                            # Parse filename format: {SYMBOL}_{INTERVAL}_{START}_{END}.csv
                            parts = filename.split('_')
                            if len(parts) >= 4:
                                file_start = parts[-2]
                                file_end = parts[-1].replace('.csv', '')
                                existing_ranges.append((file_start, file_end))
                        except Exception:
                            continue

            # Clean up files that are outside the current date range or trim files that extend beyond the range
            files_to_remove = []
            files_to_trim = []  # (filepath, new_start, new_end)

            for existing_start, existing_end in existing_ranges:
                existing_start_date = datetime.strptime(existing_start, "%Y-%m-%d").date()
                existing_end_date = datetime.strptime(existing_end, "%Y-%m-%d").date()

                # Case 1: File is completely outside the current range - remove it
                if existing_end_date < start_date or existing_start_date > end_date:
                    interval = str(getattr(custom_config, 'INTERVAL', '5'))
                    filename = f"{symbol}_{interval}_{existing_start}_{existing_end}.csv"
                    filepath = os.path.join(symbol_dir, filename)
                    if os.path.exists(filepath):
                        files_to_remove.append(filepath)

                # Case 2: File extends before the start_date - trim it
                elif existing_start_date < start_date and existing_end_date >= start_date:
                    if existing_end_date <= end_date:
                        # File starts too early but ends within range - trim start
                        new_start = start_date.strftime("%Y-%m-%d")
                        new_end = existing_end
                        interval = str(getattr(custom_config, 'INTERVAL', '5'))
                        filename = f"{symbol}_{interval}_{existing_start}_{existing_end}.csv"
                        filepath = os.path.join(symbol_dir, filename)
                        if os.path.exists(filepath):
                            files_to_trim.append((filepath, new_start, new_end))
                    else:
                        # File spans across the entire range - trim both ends
                        new_start = start_date.strftime("%Y-%m-%d")
                        new_end = end_date.strftime("%Y-%m-%d")
                        interval = str(getattr(custom_config, 'INTERVAL', '5'))
                        filename = f"{symbol}_{interval}_{existing_start}_{existing_end}.csv"
                        filepath = os.path.join(symbol_dir, filename)
                        if os.path.exists(filepath):
                            files_to_trim.append((filepath, new_start, new_end))

                # Case 3: File extends after the end_date - trim it
                elif existing_end_date > end_date and existing_start_date <= end_date:
                    if existing_start_date >= start_date:
                        # File ends too late but starts within range - trim end
                        new_start = existing_start
                        new_end = end_date.strftime("%Y-%m-%d")
                        interval = str(getattr(custom_config, 'INTERVAL', '5'))
                        filename = f"{symbol}_{interval}_{existing_start}_{existing_end}.csv"
                        filepath = os.path.join(symbol_dir, filename)
                        if os.path.exists(filepath):
                            files_to_trim.append((filepath, new_start, new_end))

            # Remove files that are completely outside the range
            for old_file in files_to_remove:
                try:
                    os.remove(old_file)
                    logger.info(f"Removed outdated file: {os.path.basename(old_file)}")
                    # Remove from existing_ranges
                    filename = os.path.basename(old_file)
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        old_start = parts[-2]
                        old_end = parts[-1].replace('.csv', '')
                        existing_ranges = [(s, e) for s, e in existing_ranges if not (s == old_start and e == old_end)]
                except Exception as e:
                    logger.warning(f"Failed to remove outdated file {old_file}: {e}")

            # Trim files that extend beyond the current range
            for filepath, new_start, new_end in files_to_trim:
                try:
                    # Read the existing file
                    rows = []
                    with open(filepath, 'r', encoding='utf-8') as f:
                        header = f.readline().strip()  # Read header
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) >= 6:
                                timestamp = parts[0]
                                try:
                                    # Parse timestamp (assuming YYYY-MM-DD format or ISO format)
                                    if 'T' in timestamp:
                                        row_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                                    else:
                                        row_date = datetime.strptime(timestamp.split()[0], "%Y-%m-%d").date()

                                    # Keep only rows within the new date range
                                    if new_start <= row_date.strftime("%Y-%m-%d") <= new_end:
                                        rows.append(line.strip())
                                except Exception:
                                    # If we can't parse the date, keep the row
                                    rows.append(line.strip())

                    # Write the trimmed file with new date range in filename
                    if rows:
                        old_filename = os.path.basename(filepath)
                        parts = old_filename.split('_')
                        if len(parts) >= 4:
                            interval = parts[1]
                            new_filename = f"{symbol}_{interval}_{new_start}_{new_end}.csv"
                            new_filepath = os.path.join(os.path.dirname(filepath), new_filename)

                            with open(new_filepath, 'w', encoding='utf-8') as f:
                                f.write(header + '\n')
                                for row in rows:
                                    f.write(row + '\n')

                            # Remove old file and update existing_ranges
                            os.remove(filepath)
                            logger.info(f"Trimmed file {old_filename} to {new_filename}")

                            # Update existing_ranges
                            old_start = parts[-2]
                            old_end = parts[-1].replace('.csv', '')
                            existing_ranges = [(s, e) for s, e in existing_ranges if not (s == old_start and e == old_end)]
                            existing_ranges.append((new_start, new_end))
                    else:
                        # No rows left after trimming, remove the file
                        os.remove(filepath)
                        logger.info(f"Removed file {os.path.basename(filepath)} - no data in range after trimming")

                except Exception as e:
                    logger.warning(f"Failed to trim file {filepath}: {e}")

            # Use calendar-month aligned chunks so each API call stays within a single month
            all_chunks = split_date_range_by_month(start_date_str, end_date_str)
            logger.info(f"Total required chunks: {len(all_chunks)}")

            # Filter out chunks that are already covered by existing files
            chunks_to_download = []
            for chunk_start, chunk_end in all_chunks:
                chunk_needed = True
                for existing_start, existing_end in existing_ranges:
                    # Check if this chunk is fully covered by an existing file
                    if existing_start <= chunk_start and existing_end >= chunk_end:
                        logger.info(f"Chunk {chunk_start} -> {chunk_end} already exists in {existing_start}_{existing_end}")
                        chunk_needed = False
                        break
                if chunk_needed:
                    chunks_to_download.append((chunk_start, chunk_end))

            if not chunks_to_download:
                logger.info(f"All chunks for {symbol} already exist. Skipping download.")
                # Return existing files for consistency
                existing_files = []
                for existing_start, existing_end in existing_ranges:
                    interval = str(getattr(custom_config, 'INTERVAL', '5'))
                    filename = f"{symbol}_{interval}_{existing_start}_{existing_end}.csv"
                    filepath = os.path.join(symbol_dir, filename)
                    if os.path.exists(filepath):
                        existing_files.append(filepath)
                results[symbol] = {'chunks': existing_files, 'missing': []}
                # Still perform post-download cleanup even when no downloads occurred
                perform_post_download_cleanup = True
                chunk_files = []  # No new files downloaded
            else:
                perform_post_download_cleanup = True

            logger.info(f"Downloading {len(chunks_to_download)} new chunk(s) for {symbol}: {chunks_to_download}")

            chunk_files: List[str] = []
            # Track missing chunks for final reporting
            missing_chunks: List[Tuple[str, str]] = []

            for idx, (chunk_start, chunk_end) in enumerate(chunks_to_download, start=1):
                logger.info(f"Chunk {idx}/{len(chunks_to_download)} for {symbol}: {chunk_start} -> {chunk_end}")
                # Update config module dates for this chunk
                custom_config.START_DATE = chunk_start
                custom_config.END_DATE = chunk_end
                # Run the download for this chunk with retries and exponential backoff
                max_retries = getattr(custom_config, 'MAX_RETRIES', 3)
                backoff = getattr(custom_config, 'RETRY_BACKOFF_SECONDS', 2)
                attempt = 0
                chunk_output = None

                # We'll try each configured interval for this chunk (config.INTERVALS)
                intervals = getattr(custom_config, 'INTERVALS', [getattr(custom_config, 'INTERVAL', '5')])
                # Attempt logic applies per interval; stop when any interval yields a file for this chunk
                interval_success = False
                for interval_val in intervals:
                    attempt = 0
                    chunk_output = None
                    while attempt < max_retries:
                        attempt += 1
                        try:
                            # Before running, capture existing files in output dir so we can detect new files
                            out_dir = getattr(custom_config, 'OUTPUT_DIRECTORY', 'historical_data')
                            os.makedirs(out_dir, exist_ok=True)
                            before_files = set(os.listdir(out_dir))

                            # Set the interval in custom_config so downloader can include it in the URL/filename
                            custom_config.INTERVAL = interval_val

                            chunk_output = downloader.run()
                            after_files = set(os.listdir(out_dir))
                            new_files = list(after_files - before_files)

                            # If downloader.run() didn't return a path, try to infer newly created files
                            if not chunk_output and new_files:
                                # prefer files that include the symbol name if available
                                chosen = None
                                symbol_upper = symbol.upper()
                                for nf in new_files:
                                    if symbol_upper in nf.upper():
                                        chosen = nf
                                        break
                                if chosen is None:
                                    # fallback to most recently modified new file
                                    new_paths = [os.path.join(out_dir, nf) for nf in new_files]
                                    new_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                                    chosen = os.path.basename(new_paths[0])
                                chunk_output = os.path.join(out_dir, chosen)

                            if chunk_output:
                                chunk_files.append(chunk_output)
                                logger.info(f"Chunk saved: {chunk_output} (attempt {attempt})")
                                interval_success = True
                                break
                            else:
                                logger.warning(f"Chunk attempt {attempt} failed for {symbol}: {chunk_start} -> {chunk_end}")

                        except Exception as e:
                            logger.exception(f"Downloader error for {symbol} chunk {chunk_start}->{chunk_end} (attempt {attempt}): {e}")
                            chunk_output = None

                        # If we reach here, schedule a backoff before retrying
                        if attempt < max_retries:
                            sleep_for = backoff * (2 ** (attempt - 1))
                            logger.info(f"Retrying chunk in {sleep_for} seconds...")
                            time.sleep(sleep_for)

                    if interval_success:
                        # we got a chunk file for this interval; stop trying other intervals
                        break

                    # if we exhausted attempts for this interval and didn't get a file, continue to next interval

                # After trying all intervals, if we still don't have any chunk_output, mark missing
                if not interval_success:
                    logger.error(f"Chunk ultimately failed after {max_retries} attempts for {symbol}: {chunk_start} -> {chunk_end}")
                    missing_chunks.append((chunk_start, chunk_end))

                # Small delay between successful chunk calls to respect rate limits
                time.sleep(1)

            # POST-DOWNLOAD CLEANUP: Handle duplicates and merges after all downloads are complete
            if perform_post_download_cleanup:
                logger.info("Performing post-download cleanup of duplicates...")

                # Re-scan files after downloads to catch any new duplicates
                if os.path.exists(symbol_dir):
                    files_by_range = {}  # (start, end) -> [filepaths]
                    for filename in os.listdir(symbol_dir):
                        if filename.startswith(f"{symbol}_") and filename.endswith(".csv"):
                            try:
                                # Parse filename format: {SYMBOL}_{INTERVAL}_{START}_{END}.csv
                                parts = filename.split('_')
                                if len(parts) >= 4:
                                    file_start = parts[-2]
                                    file_end = parts[-1].replace('.csv', '')
                                    file_range = (file_start, file_end)
                                    filepath = os.path.join(symbol_dir, filename)

                                    if file_range not in files_by_range:
                                        files_by_range[file_range] = []
                                    files_by_range[file_range].append(filepath)
                            except Exception:
                                continue

                    # Smart duplicate detection - prefer merging over deleting
                    processed_ranges = []
                    files_to_merge = []  # (keep_file, merge_files)

                    # Sort ranges by start date for better processing
                    sorted_ranges = sorted(files_by_range.keys(), key=lambda x: x[0])

                    for file_range in sorted_ranges:
                        filepaths = files_by_range[file_range]
                        range_start = datetime.strptime(file_range[0], "%Y-%m-%d").date()
                        range_end = datetime.strptime(file_range[1], "%Y-%m-%d").date()

                        # Check if this range overlaps with any already processed range
                        is_overlapping = False
                        overlapping_processed_range = None

                        for processed_range in processed_ranges:
                            processed_start = datetime.strptime(processed_range[0], "%Y-%m-%d").date()
                            processed_end = datetime.strptime(processed_range[1], "%Y-%m-%d").date()

                            # Check for overlap: ranges overlap if one starts before the other ends
                            if (range_start <= processed_end and range_end >= processed_start):
                                is_overlapping = True
                                overlapping_processed_range = processed_range
                                break

                        if is_overlapping:
                            # Instead of deleting, mark for merging
                            processed_start = datetime.strptime(overlapping_processed_range[0], "%Y-%m-%d").date()
                            processed_end = datetime.strptime(overlapping_processed_range[1], "%Y-%m-%d").date()

                            current_range_days = (range_end - range_start).days
                            processed_range_days = (processed_end - processed_start).days

                            if current_range_days > processed_range_days:
                                # Current range is more complete - it becomes the primary, others get merged
                                processed_ranges.remove(overlapping_processed_range)
                                processed_ranges.append(file_range)
                                # Mark for merging into the current file
                                files_to_merge.append((file_range, [overlapping_processed_range]))
                            else:
                                # Keep the processed range as primary, merge current into it
                                files_to_merge.append((overlapping_processed_range, [file_range]))
                        else:
                            # No overlap, add this range
                            processed_ranges.append(file_range)

                    # Handle merging of overlapping files
                    for primary_range, merge_ranges in files_to_merge:
                        primary_filepath = None
                        merge_filepaths = []

                        # Find filepaths for primary and merge ranges
                        for file_range, filepaths in files_by_range.items():
                            if file_range == primary_range:
                                primary_filepath = filepaths[0]  # Use first file as primary
                            elif file_range in merge_ranges:
                                merge_filepaths.extend(filepaths)

                        if primary_filepath and merge_filepaths:
                            try:
                                # Read primary file
                                primary_data = []
                                with open(primary_filepath, 'r', encoding='utf-8') as f:
                                    header = f.readline().strip()
                                    for line in f:
                                        primary_data.append(line.strip())

                                # Read and merge data from overlapping files
                                for merge_filepath in merge_filepaths:
                                    with open(merge_filepath, 'r', encoding='utf-8') as f:
                                        f.readline()  # Skip header
                                        for line in f:
                                            primary_data.append(line.strip())

                                # Remove duplicates by timestamp and sort
                                seen_timestamps = set()
                                unique_data = []
                                for line in primary_data:
                                    parts = line.split(',')
                                    if len(parts) >= 6:
                                        timestamp = parts[0]
                                        if timestamp not in seen_timestamps:
                                            seen_timestamps.add(timestamp)
                                            unique_data.append(line)

                                # Sort by timestamp
                                unique_data.sort(key=lambda x: x.split(',')[0])

                                # Write merged data back to primary file
                                with open(primary_filepath, 'w', encoding='utf-8') as f:
                                    f.write(header + '\n')
                                    for line in unique_data:
                                        f.write(line + '\n')

                                # Remove merged files
                                for merge_filepath in merge_filepaths:
                                    try:
                                        os.remove(merge_filepath)
                                        logger.info(f"Merged and removed duplicate file: {os.path.basename(merge_filepath)}")
                                    except Exception as e:
                                        logger.warning(f"Failed to remove merged file {merge_filepath}: {e}")

                            except Exception as e:
                                logger.warning(f"Failed to merge files for range {primary_range}: {e}")

                    # Remove exact duplicates (same range, multiple files) - keep newest
                    for file_range, filepaths in files_by_range.items():
                        if len(filepaths) > 1:
                            # Sort by modification time, keep newest
                            filepaths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            for old_file in filepaths[1:]:
                                try:
                                    os.remove(old_file)
                                    logger.info(f"Removed exact duplicate file: {os.path.basename(old_file)}")
                                except Exception as e:
                                    logger.warning(f"Failed to remove exact duplicate {old_file}: {e}")

            # FETCH CURRENT DAY'S DATA IF TODAY IS A TRADING DAY
            today = datetime.now(timezone.utc).date()
            if is_trading_day(today):
                logger.info(f"Today ({today}) is a trading day - fetching current day data")
                try:
                    # Get instrument key for the symbol
                    instrument_key = None
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
                logger.info(f"Today ({today}) is not a trading day - skipping current day data fetch")

            # CREATE COMBINED CSV FILE
            logger.info(f"Creating combined CSV file for {symbol}")
            combined_filepath = None
            try:
                combined_data = []
                interval = str(getattr(custom_config, 'INTERVAL', '5'))

                # Read all CSV files for this symbol
                for filename in sorted(os.listdir(symbol_dir)):
                    if filename.startswith(f"{symbol}_{interval}_") and filename.endswith(".csv") and "_combined" not in filename:
                        filepath = os.path.join(symbol_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                if len(lines) > 1:  # Has header + data
                                    # Skip header for all files except the first
                                    data_lines = lines[1:] if combined_data else lines
                                    combined_data.extend(data_lines)
                        except Exception as e:
                            logger.warning(f"Failed to read {filename}: {e}")

                if combined_data:
                    # Remove duplicates based on timestamp and sort by timestamp
                    seen_timestamps = set()
                    unique_data = []

                    for line in combined_data:
                        line = line.strip()
                        if line and not line.startswith('timestamp,'):  # Skip header lines
                            parts = line.split(',')
                            if len(parts) >= 6:
                                timestamp = parts[0]
                                if timestamp not in seen_timestamps:
                                    seen_timestamps.add(timestamp)
                                    unique_data.append(line)

                    # Sort by timestamp
                    unique_data.sort(key=lambda x: x.split(',')[0])

                    # Create combined file
                    combined_filename = f"{symbol}_{interval}_combined.csv"
                    combined_filepath = os.path.join(symbol_dir, combined_filename)

                    with open(combined_filepath, 'w', encoding='utf-8') as f:
                        f.write('timestamp,open,high,low,close,volume\n')  # Header
                        for line in unique_data:
                            f.write(line + '\n')

                    logger.info(f"Combined file created: {combined_filepath} ({len(unique_data)} records)")

                    # Delete chunk files if KEEP_CHUNK_FILES is False
                    keep_chunk_files = getattr(custom_config, 'KEEP_CHUNK_FILES', True)
                    if not keep_chunk_files:
                        logger.info(f"Deleting individual chunk files for {symbol} (KEEP_CHUNK_FILES=False)")
                        deleted_count = 0
                        for filename in os.listdir(symbol_dir):
                            if (filename.startswith(f"{symbol}_{interval}_") and 
                                filename.endswith(".csv") and 
                                "_combined" not in filename):
                                chunk_filepath = os.path.join(symbol_dir, filename)
                                try:
                                    os.remove(chunk_filepath)
                                    deleted_count += 1
                                    logger.debug(f"Deleted chunk file: {filename}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete chunk file {filename}: {e}")
                        logger.info(f"Deleted {deleted_count} chunk files for {symbol}")

                else:
                    logger.warning(f"No data found to create combined file for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to create combined file for {symbol}: {e}")

            # Collect all files (existing + newly downloaded) for the final result
            all_files = []
            if os.path.exists(symbol_dir):
                for filename in os.listdir(symbol_dir):
                    if filename.startswith(f"{symbol}_") and filename.endswith(".csv"):
                        filepath = os.path.join(symbol_dir, filename)
                        all_files.append(filepath)

            # Do not combine chunk files into a single _full file. Keep chunk files as canonical
            # Instead, report which chunks are missing (if any) and return the list of downloaded chunk files
            if not all_files:
                logger.error(f"No files available for {symbol}")
                results[symbol] = {'chunks': [], 'missing': missing_chunks, 'holidays': holidays_found}
            else:
                results[symbol] = {'chunks': all_files, 'missing': missing_chunks, 'holidays': holidays_found}
                if missing_chunks:
                    logger.warning(f"Symbol {symbol} has missing chunks: {missing_chunks}")
                else:
                    logger.info(f"Symbol {symbol} has {len(all_files)} total files ({len(chunk_files)} new, {len(all_files) - len(chunk_files)} existing)")
            
        except Exception as e:
            logger.exception(f"Error processing {symbol}: {str(e)}")
            results[symbol] = ""
            
    return results

def download_from_list_file(file_path: str, config_module=None) -> Dict[str, str]:
    """
    Download historical data for symbols listed in a text file.
    
    Args:
        file_path: Path to a text file with one symbol per line
        config_module: Config module to use for download settings
        
    Returns:
        Dictionary mapping symbols to output filepaths
    """
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
            
        return download_multiple_symbols(symbols, config_module)
    
    except Exception as e:
        logger.exception(f"Error reading symbol list file: {str(e)}")
        return {}

if __name__ == "__main__":
    # Use the SYMBOLS list from config.py by default
    from config import config as cfg
    # Use SYMBOLS from config.py exclusively. If empty, fall back to single SYMBOL.
    symbols_to_download = getattr(cfg, 'SYMBOLS', None)
    if not symbols_to_download:
        symbols_to_download = [getattr(cfg, 'SYMBOL', 'SBIN')]

    print(f"Starting batch download for {len(symbols_to_download)} symbols")
    results = download_multiple_symbols(symbols_to_download, config_module=cfg)

    # Print summary with per-symbol detail
    success_count = sum(1 for v in results.values() if v and v.get('chunks'))
    print(f"Download summary: {success_count}/{len(results)} symbols have downloaded chunks")
    for symbol, info in results.items():
        chunks = info.get('chunks', []) if isinstance(info, dict) else []
        missing = info.get('missing', []) if isinstance(info, dict) else []
        holidays = info.get('holidays', []) if isinstance(info, dict) else []
        if not chunks:
            print(f"  {symbol}: FAILED (no chunks downloaded)")
        else:
            print(f"  {symbol}: {len(chunks)} chunk files -> {chunks[0]}{'...' if len(chunks)>1 else ''}")
            if missing:
                print(f"    WARNING: missing chunks: {missing}")

    # Write a run summary JSON to config folder
    try:
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        os.makedirs(config_dir, exist_ok=True)
        summary_path = os.path.join(config_dir, 'run_summary.json')
        import json
        # Use timezone-aware UTC timestamp to avoid DeprecationWarning
        generated_at = datetime.now(timezone.utc).isoformat()
        with open(summary_path, 'w', encoding='utf-8') as sf:
            json.dump({'generated_at': generated_at, 'results': results}, sf, indent=2)
        print(f"Run summary written to {summary_path}")
    except Exception:
        logger.exception("Failed to write run summary JSON")
