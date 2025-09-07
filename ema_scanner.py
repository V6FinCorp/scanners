"""
EMA Scanner - Calculates Exponential Moving Averages (EMA) for configurable periods
"""
import json
import requests
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
def load_config():
    try:
        with open('config/ema_scanner_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return None
def load_instrument_mapping():
    try:
        with open('config/instrument_mapping.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Instrument Mapping Error: {e}")
        return None
def get_instrument_key(symbol, instrument_mapping):
    if not instrument_mapping:
        return None
    instrument = instrument_mapping.get(symbol)
    if instrument:
        return instrument.get('instrument_key')
    return None
def fetch_market_data(instrument_key, target_date, interval):
    safe_key = instrument_key.replace('|', '%7C')
    end_date = target_date.strftime('%Y-%m-%d')
    start_date = (target_date - timedelta(days=5)).strftime('%Y-%m-%d')
    headers = {'Accept': 'application/json'}
    # Use v2 intraday API for 1min/30min intervals if target_date is today
    today = datetime.now().date()
    if interval in ["1", "30"] and target_date == today:
        url = f'https://api.upstox.com/v2/historical-candle/intraday/{safe_key}/{interval}minute'
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json().get('data', {})
            candles = data.get('candles', [])
            return candles, "SUCCESS"
        else:
            return None, f"API_ERROR_{response.status_code}"
    # If interval is not available, aggregate 1min data to requested interval
    if interval not in ["1", "30"] and target_date == today:
        # Fetch 1min data
        url = f'https://api.upstox.com/v2/historical-candle/intraday/{safe_key}/1minute'
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json().get('data', {})
            one_min_candles = data.get('candles', [])
            # Aggregate 1min candles to requested interval
            agg_candles = []
            interval_minutes = int(interval)
            if one_min_candles:
                # Group by interval
                for i in range(0, len(one_min_candles), interval_minutes):
                    group = one_min_candles[i:i+interval_minutes]
                    if not group:
                        continue
                    open_ = group[0][1]
                    high_ = max(float(c[2]) for c in group)
                    low_ = min(float(c[3]) for c in group)
                    close_ = group[-1][4]
                    volume_ = sum(float(c[5]) for c in group)
                    ts_ = group[-1][0]
                    agg_candles.append([ts_, open_, high_, low_, close_, volume_])
                return agg_candles, "AGGREGATED"
            else:
                return None, "NO_1MIN_DATA"
        else:
            return None, f"API_ERROR_{response.status_code}"
    # Otherwise use v3 historical API
    if interval in ["5", "15", "30"]:
        url = f'https://api.upstox.com/v3/historical-candle/{safe_key}/minutes/{interval}/{end_date}/{start_date}'
    elif interval == "day":
        url = f'https://api.upstox.com/v3/historical-candle/{safe_key}/days/1/{end_date}/{start_date}'
    elif interval == "week":
        url = f'https://api.upstox.com/v3/historical-candle/{safe_key}/weeks/1/{end_date}/{start_date}'
    elif interval == "month":
        url = f'https://api.upstox.com/v3/historical-candle/{safe_key}/months/1/{end_date}/{start_date}'
    else:
        url = f'https://api.upstox.com/v3/historical-candle/{safe_key}/days/1/{end_date}/{start_date}'
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        data = response.json().get('data', {})
        candles = data.get('candles', [])
        return candles, "SUCCESS"
    else:
        return None, f"API_ERROR_{response.status_code}"
def calculate_ema(closes, period):
    if len(closes) < period:
        return None
    return list(np.round(np.array(pd.Series(closes).ewm(span=period, adjust=False).mean()), 2))
def is_working_day(date):
    return date.weekday() < 5
def get_last_working_day():
    today = datetime.now().date()
    current_date = today
    while not is_working_day(current_date):
        current_date -= timedelta(days=1)
    return current_date
def main():
    print("EMA SCANNER - Exponential Moving Averages")
    print("=" * 60)
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return
    instrument_mapping = load_instrument_mapping()
    if not instrument_mapping:
        print("❌ Failed to load instrument mapping")
        return
    symbols = config['scanner_settings']['symbols_to_scan']
    ema_periods = [int(x.strip()) for x in config['scanner_settings']['ema_periods'].split(',') if x.strip().isdigit()]
    ema_intervals = config['scanner_settings'].get('ema_intervals', ["day"])
    today = datetime.now().date()
    num_days = config['scanner_settings'].get('num_days', 2)
    last_days = []
    current = today
    # Collect up to num_days most recent working days (including today if working day)
    while len(last_days) < num_days:
        if is_working_day(current):
            last_days.append(current)
        current -= timedelta(days=1)
    last_days = last_days[::-1]
    for interval in ema_intervals:
        print(f"\n{'='*40}\nEMA Interval: {interval} min\n{'='*40}")
        table_rows = []
    header = ['Timestamp', 'Symbol', 'Price'] + [f'EMA-{p}' for p in ema_periods] + ['Summary']
    crossover_summary = []
    for idx_day, target_date in enumerate(last_days):
            for symbol in symbols:
                instrument_key = get_instrument_key(symbol, instrument_mapping)
                if not instrument_key:
                    table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(ema_periods))
                    continue
                candles, status = fetch_market_data(instrument_key, target_date, interval)
                # Robust API validation
                if candles is None or not candles:
                    # If today is working day and API failed, fallback to previous working day
                    if idx_day == 1 and is_working_day(target_date):
                        print(f"⚠️ No data for today ({target_date}) for {symbol} [{interval}min], trying previous working day...")
                        prev_day = last_days[0]
                        candles, status = fetch_market_data(instrument_key, prev_day, interval)
                        if candles is None or not candles:
                            table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(ema_periods))
                            continue
                        target_date = prev_day
                    else:
                        table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(ema_periods))
                        continue
                candles.reverse()
                closes = [float(c[4]) for c in candles]
                if interval in ["5", "15", "30"]:
                    day_candles = [c for c in candles if c[0].split('T')[0] == target_date.strftime('%Y-%m-%d')]
                    if not day_candles:
                        table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(ema_periods))
                        continue
                    for candle in day_candles:
                        price = float(candle[4])
                        raw_timestamp = candle[0]
                        try:
                            if 'T' in raw_timestamp:
                                dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                                timestamp = dt.strftime('%Y-%m-%d %H:%M')
                            else:
                                timestamp = raw_timestamp.split('T')[0] if 'T' in raw_timestamp else raw_timestamp
                        except Exception:
                            timestamp = raw_timestamp.split('T')[0] if 'T' in raw_timestamp else raw_timestamp
                        ema_values = []
                        ema_latest = []
                        for period in ema_periods:
                            ema_arr = calculate_ema(closes, period)
                            candle_idx = candles.index(candle)
                            if ema_arr is not None and candle_idx >= 0 and candle_idx < len(ema_arr):
                                ema_val = ema_arr[candle_idx]
                                diff_pct = ((price - ema_val) / ema_val * 100) if ema_val != 0 else 0
                                ema_str = f"{ema_val:.2f} ({diff_pct:+.2f}%)"
                                ema_latest.append(ema_val)
                            else:
                                ema_str = 'NA'
                                ema_latest.append(None)
                            ema_values.append(ema_str)
                        # Determine summary for this row
                        if all(e is not None for e in ema_latest):
                            if ema_latest[0] > ema_latest[1]:
                                summary = 'Bullish'
                                crossover_summary.append(f"{symbol} [{timestamp}]: EMA-{ema_periods[0]} > EMA-{ema_periods[1]} (Bullish)")
                            elif ema_latest[0] < ema_latest[1]:
                                summary = 'Bearish'
                                crossover_summary.append(f"{symbol} [{timestamp}]: EMA-{ema_periods[0]} < EMA-{ema_periods[1]} (Bearish)")
                            else:
                                summary = 'Neutral'
                                crossover_summary.append(f"{symbol} [{timestamp}]: EMA-{ema_periods[0]} = EMA-{ema_periods[1]} (Neutral)")
                        else:
                            summary = 'NA'
                        table_rows.append([timestamp, symbol, f"Rs.{price:.2f}"] + ema_values + [summary])
                else:
                    day_candle = next((c for c in candles if c[0].split('T')[0] == target_date.strftime('%Y-%m-%d')), None)
                    if not day_candle:
                        table_rows.append(['NA', symbol, 'NA'] + ['NA'] * len(ema_periods))
                        continue
                    idx = candles.index(day_candle)
                    price = float(day_candle[4])
                    raw_timestamp = day_candle[0]
                    try:
                        if 'T' in raw_timestamp:
                            dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%d %H:%M')
                        else:
                            timestamp = raw_timestamp.split('T')[0] if 'T' in raw_timestamp else raw_timestamp
                    except Exception:
                        timestamp = raw_timestamp.split('T')[0] if 'T' in raw_timestamp else raw_timestamp
                    ema_values = []
                    ema_latest = []
                    for period in ema_periods:
                        ema_arr = calculate_ema(closes, period)
                        if ema_arr is not None and idx >= 0 and idx < len(ema_arr):
                            ema_val = ema_arr[idx]
                            diff_pct = ((price - ema_val) / ema_val * 100) if ema_val != 0 else 0
                            ema_str = f"{ema_val:.2f} ({diff_pct:+.2f}%)"
                            ema_latest.append(ema_val)
                        else:
                            ema_str = 'NA'
                            ema_latest.append(None)
                        ema_values.append(ema_str)
                    # Determine summary for this row
                    if all(e is not None for e in ema_latest):
                        if ema_latest[0] > ema_latest[1]:
                            summary = 'Bullish'
                            crossover_summary.append(f"{symbol} [{timestamp}]: EMA-{ema_periods[0]} > EMA-{ema_periods[1]} (Bullish)")
                        elif ema_latest[0] < ema_latest[1]:
                            summary = 'Bearish'
                            crossover_summary.append(f"{symbol} [{timestamp}]: EMA-{ema_periods[0]} < EMA-{ema_periods[1]} (Bearish)")
                        else:
                            summary = 'Neutral'
                            crossover_summary.append(f"{symbol} [{timestamp}]: EMA-{ema_periods[0]} = EMA-{ema_periods[1]} (Neutral)")
                    else:
                        summary = 'NA'
                    table_rows.append([timestamp, symbol, f"Rs.{price:.2f}"] + ema_values + [summary])
    # Show summary for last record above the table
    if table_rows:
        last_row = table_rows[-1]
        last_summary = last_row[-1]
        print(f"\nCurrent EMA Trend: {last_summary}\n")
    print("=" * (15 * len(header)))
    print('  ' + '  '.join(f'{h:<15}' for h in header))
    print('  ' + '-' * (15 * len(header)))
    for row in table_rows:
        print('  ' + '  '.join(f'{str(v):<15}' for v in row))
    print('  ' + '-' * (15 * len(header)))
    print("=" * (15 * len(header)))

if __name__ == '__main__':
    import pandas as pd
    main()
