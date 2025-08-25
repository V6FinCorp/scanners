def get_last_n_working_days(n):
    """Return a list of last n working days (date objects), most recent first"""
    days = []
    current = datetime.now().date()
    while len(days) < n:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)
    return days[::-1]

def fetch_intraday_data(instrument_key, target_date):
    """Fetch 1-hour interval intraday data for a given date"""
    safe_key = instrument_key.replace('|', '%7C')
    url = f'https://api.upstox.com/v2/historical-candle/intraday/{safe_key}/1minute'
    headers = {'Accept': 'application/json'}
    print(f"[DEBUG] Fetching intraday data: {url}")
    response = requests.get(url, headers=headers, timeout=30)
    print(f"[DEBUG] Response status: {response.status_code}")
    print(f"[DEBUG] Response body: {response.text[:500]}")
    if response.status_code == 200:
        data = response.json().get('data', {})
        candles = data.get('candles', [])
        print(f"[DEBUG] Parsed {len(candles)} candles")
        target_date_str = target_date.strftime('%Y-%m-%d')
        filtered = [c for c in candles if c[0].split('T')[0] == target_date_str]
        print(f"[DEBUG] Filtered {len(filtered)} candles for date {target_date_str}")
        for c in filtered:
            print(f"[DEBUG] Candle timestamp: {c[0]}")
        return filtered
    return []
def is_working_day(date):
    """Check if given date is a working day (Monday to Friday)"""
    return date.weekday() < 5

def get_last_working_day():
    """Get the last working day before today"""
    today = datetime.now().date()
    current_date = today
    while not is_working_day(current_date):
        current_date -= timedelta(days=1)
    return current_date
#!/usr/bin/env python3
"""
DMA Scanner - Calculates Daily Moving Averages (DMA) for configurable periods
"""
import json
import requests
import numpy as np
from datetime import datetime, timedelta

def load_config():
    """Load DMA scanner configuration from JSON file"""
    try:
        with open('config/dma_scanner_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Config Error: {e}")
        return None

def load_instrument_mapping():
    """Load instrument mapping from JSON file"""
    try:
        with open('config/instrument_mapping.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Instrument Mapping Error: {e}")
        return None

def get_instrument_key(symbol, instrument_mapping):
    if not instrument_mapping:
        return None
    instrument = instrument_mapping.get(symbol)
    if instrument:
        return instrument.get('instrument_key')
    return None

def fetch_market_data(instrument_key, target_date):
    """Fetch daily market data from Upstox API for DMA calculation"""
    try:
        from inspect import currentframe
        frame = currentframe()
        config = frame.f_back.f_locals.get('config') if frame and frame.f_back else None
        api_token = None
        if config and 'upstox_api_token' in config:
            api_token = config['upstox_api_token']
        safe_key = instrument_key.replace('|', '%7C')
        interval = 1  # daily
        end_date = target_date.strftime('%Y-%m-%d')
        start_date = (target_date - timedelta(days=250)).strftime('%Y-%m-%d')
        url = f'https://api.upstox.com/v3/historical-candle/{safe_key}/days/{interval}/{end_date}/{start_date}'
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json().get('data', {})
            candles = data.get('candles', [])
            return candles, "SUCCESS"
        else:
            return None, f"API_ERROR_{response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException: {e}")
        return None, "NETWORK_ERROR"
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        return None, "UNKNOWN_ERROR"

def calculate_dma(closes, period):
    """Calculate Simple Moving Average (DMA) for given period"""
    if len(closes) < period:
        return None
    return np.convolve(closes, np.ones(period)/period, mode='valid')

def main():
    print("DMA SCANNER - Daily Moving Averages")
    print("=" * 60)
    config = load_config()
    if not config:
        print("Failed to load configuration")
        return
    instrument_mapping = load_instrument_mapping()
    if not instrument_mapping:
        print("Failed to load instrument mapping")
        return
    symbols = config['scanner_settings']['symbols_to_scan']
    dma_periods = [int(x.strip()) for x in config['scanner_settings']['dma_periods'].split(',') if x.strip().isdigit()]
    today = datetime.now().date()
    print(f"Scanning {len(symbols)} symbols for DMA periods: {dma_periods}")
    print("=" * 60)
    table_rows = []
    header = ['Timestamp', 'Symbol', 'Price'] + [f'DMA-{p}' for p in dma_periods]
    # Get last 5 working days (excluding today)
    last_days = []
    current = today - timedelta(days=1)
    while len(last_days) < 5:
        if is_working_day(current):
            last_days.append(current)
        current -= timedelta(days=1)
    last_days = last_days[::-1]
    for target_date in last_days:
        for symbol in symbols:
            instrument_key = get_instrument_key(symbol, instrument_mapping)
            if not instrument_key:
                table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(dma_periods))
                continue
            candles, status = fetch_market_data(instrument_key, target_date)
            if candles is None or not candles:
                table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(dma_periods))
                continue
            candles.reverse()
            closes = [float(c[4]) for c in candles]
            day_candle = next((c for c in candles if c[0].split('T')[0] == target_date.strftime('%Y-%m-%d')), None)
            if not day_candle:
                table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(dma_periods))
                continue
            idx = candles.index(day_candle)
            price = float(day_candle[4])
            raw_timestamp = day_candle[0]
            try:
                if 'T' in raw_timestamp:
                    dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                    timestamp = dt.strftime('%Y-%m-%d')
                else:
                    timestamp = raw_timestamp.split('T')[0] if 'T' in raw_timestamp else raw_timestamp
            except Exception:
                timestamp = raw_timestamp.split('T')[0] if 'T' in raw_timestamp else raw_timestamp
            dma_values = []
            for period in dma_periods:
                dma_arr = calculate_dma(closes, period)
                if dma_arr is not None and idx - (period - 1) >= 0:
                    dma_val = dma_arr[idx - (period - 1)]
                    diff_pct = ((price - dma_val) / dma_val * 100) if dma_val != 0 else 0
                    dma_str = f"{dma_val:.2f} ({diff_pct:+.2f}%)"
                else:
                    dma_str = 'NA'
                dma_values.append(dma_str)
            table_rows.append([timestamp, symbol, f"Rs.{price:.2f}"] + dma_values)
    # Display table
    print("\n" + "=" * (15 * len(header)))
    print('  ' + '  '.join(f'{h:<15}' for h in header))
    print('  ' + '-' * (15 * len(header)))
    for row in table_rows:
        print('  ' + '  '.join(f'{str(v):<15}' for v in row))
    print('  ' + '-' * (15 * len(header)))
    print("=" * (15 * len(header)))

if __name__ == '__main__':
    main()
