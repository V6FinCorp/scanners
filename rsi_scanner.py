#!/usr/bin/env python3
"""
RSI Scanner - Exact TradingView RSI Calculation
Uses ta.rma() method (Wilder's smoothing) exactly like TradingView Pine Script
"""
import json
import requests
import numpy as np
from datetime import datetime, timedelta

def load_config():
    try:
        with open('config/rsi_scanner_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Config Error: {e}")
        return None

def load_instrument_mapping():
    try:
        with open('config/instrument_mapping.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Instrument Mapping Error: {e}")
        return None

def tradingview_rma(values, length):
    if len(values) < length:
        return None
    rma_values = []
    alpha = 1.0 / length
    first_rma = np.mean(values[:length])
    rma_values.append(first_rma)
    for i in range(length, len(values)):
        prev_rma = rma_values[-1]
        current_value = values[i]
        new_rma = alpha * current_value + (1 - alpha) * prev_rma
        rma_values.append(new_rma)
    return rma_values

def calculate_tradingview_rsi(closes, length=14):
    if len(closes) < length + 1:
        return None
    closes = np.array(closes)
    changes = np.diff(closes)
    gains = np.maximum(changes, 0)
    losses = -np.minimum(changes, 0)
    up_rma = tradingview_rma(gains, length)
    down_rma = tradingview_rma(losses, length)
    if up_rma is None or down_rma is None:
        return None
    rsi_values = []
    for i in range(len(up_rma)):
        up = up_rma[i]
        down = down_rma[i]
        if down == 0:
            rsi = 100.0
        elif up == 0:
            rsi = 0.0
        else:
            rsi = 100 - (100 / (1 + up / down))
        rsi_values.append(rsi)
    return rsi_values

def is_working_day(date):
    return date.weekday() < 5

def get_last_working_day():
    today = datetime.now().date()
    current_date = today
    while not is_working_day(current_date):
        current_date -= timedelta(days=1)
    return current_date

def get_target_date():
    today = datetime.now().date()
    if is_working_day(today):
        return today, "historical"
    else:
        last_working = get_last_working_day()
        return last_working, "historical"

def get_instrument_key(symbol, instrument_mapping):
    if not instrument_mapping:
        return None
    instrument = instrument_mapping.get(symbol)
    if instrument:
        return instrument.get('instrument_key')
    return None

def fetch_market_data(instrument_key, timeframe, target_date, data_type):
    try:
        safe_key = instrument_key.replace('|', '%7C')
        date_str = target_date.strftime('%Y-%m-%d')
        today = datetime.now().date()
        if target_date == today:
            # Always fetch 1min candles for today
            url = f'https://api.upstox.com/v2/historical-candle/intraday/{safe_key}/1minute'
            headers = {'Accept': 'application/json'}
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json().get('data', {})
                candles = data.get('candles', [])
                return candles, "SUCCESS"
            else:
                return None, f"API_ERROR_{response.status_code}"
        else:
            url = f'https://api.upstox.com/v3/historical-candle/{safe_key}/minutes/{timeframe}/{date_str}/{date_str}'
            headers = {'Accept': 'application/json'}
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json().get('data', {})
                candles = data.get('candles', [])
                return candles, "SUCCESS"
            else:
                return None, f"API_ERROR_{response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"NETWORK_ERROR"
    except Exception as e:
        return None, f"UNKNOWN_ERROR"

def format_timestamp(timestamp_str):
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str

def scan_symbols(config, instrument_mapping):
    symbols = config['scanner_settings']['symbols_to_scan']
    rsi_period = config['rsi_parameters']['rsi_period']
    tf_str = config['scanner_settings'].get('timeframes', '5')
    timeframes = [int(tf.strip()) for tf in tf_str.split(',') if tf.strip().isdigit()]
    target_date, data_type = get_target_date()
    print(f"Target Date: {target_date} ({data_type} data)")
    print(f"Scanning {len(symbols)} symbols with timeframes: {timeframes}")
    print(f"Using TradingView RSI calculation (ta.rma method)")
    print("=" * 80)
    results = []
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        instrument_key = get_instrument_key(symbol, instrument_mapping)
        if not instrument_key:
            print(f"{symbol}: Instrument key not found")
            continue
        symbol_result = {'symbol': symbol, 'timeframes': {}}
        for timeframe in timeframes:
            candles, status = fetch_market_data(instrument_key, timeframe, target_date, data_type)
            print(f"[DEBUG] {symbol} [{timeframe}m]: API status: {status}, candles fetched: {len(candles) if candles else 0}")
            if candles is None or not candles:
                print(f"{symbol} [{timeframe}m]: {status}")
                continue
            candles.reverse()
            # For today, aggregate 1min candles into requested interval
            if target_date == datetime.now().date() and timeframe != 1:
                interval = int(timeframe)
                grouped = []
                group = []
                for c in candles:
                    timestamp = c[0]
                    # Only include times at or after 09:15
                    try:
                        time_part = timestamp.split('T')[1][:5] if 'T' in timestamp else timestamp[11:16]
                    except Exception:
                        time_part = ''
                    if time_part >= '09:15':
                        group.append(c)
                        if len(group) == interval:
                            grouped.append(group)
                            group = []
                if group:
                    grouped.append(group)
                all_closes = [float(g[-1][4]) for g in grouped if len(g) == interval]
                rsi_values = calculate_tradingview_rsi(all_closes, rsi_period)
                if rsi_values is None:
                    print(f"{symbol} [{timeframe}m]: Insufficient data for RSI calculation")
                    continue
                print(f"[DEBUG] {symbol} [{timeframe}m]: {len(rsi_values)} RSI values calculated")
                tf_results = []
                for i in range(rsi_period, len(grouped)):
                    g = grouped[i]
                    timestamp = g[-1][0]
                    close_price = float(g[-1][4])
                    rsi = rsi_values[i - rsi_period]
                    tf_results.append({
                        'time': format_timestamp(timestamp),
                        'current_price': f"Rs.{close_price:.2f}",
                        'rsi_value': f"{rsi:.2f}",
                        'raw_timestamp': timestamp,
                        'raw_price': close_price,
                        'raw_rsi': rsi
                    })
                symbol_result['timeframes'][str(timeframe)] = tf_results
            else:
                all_closes = [float(c[4]) for c in candles]
                rsi_values = calculate_tradingview_rsi(all_closes, rsi_period)
                if rsi_values is None:
                    print(f"{symbol} [{timeframe}m]: Insufficient data for RSI calculation")
                    continue
                print(f"[DEBUG] {symbol} [{timeframe}m]: {len(rsi_values)} RSI values calculated")
                tf_results = []
                for i in range(rsi_period, len(candles)):
                    candle = candles[i]
                    timestamp = candle[0]
                    try:
                        time_part = timestamp.split('T')[1][:5] if 'T' in timestamp else timestamp[11:16]
                    except Exception:
                        time_part = ''
                    if time_part >= '09:15':
                        close_price = float(candle[4])
                        rsi = rsi_values[i - rsi_period]
                        tf_results.append({
                            'time': format_timestamp(timestamp),
                            'current_price': f"Rs.{close_price:.2f}",
                            'rsi_value': f"{rsi:.2f}",
                            'raw_timestamp': timestamp,
                            'raw_price': close_price,
                            'raw_rsi': rsi
                        })
                symbol_result['timeframes'][str(timeframe)] = tf_results
        results.append(symbol_result)
    return results

def display_results_table(results):
    if not results:
        print("No data to display")
        return
    timeframes = []
    if results and 'timeframes' in results[0]:
        timeframes = [int(tf) for tf in results[0]['timeframes'].keys()]
        timeframes.sort()
    print("\n" + "=" * 90)
    print("RSI SCANNER RESULTS - MULTI-TIMEFRAME (TradingView Method)")
    print("=" * 90)
    for symbol_result in results:
        symbol = symbol_result['symbol']
        all_times = set()
        tf_data = {}
        for timeframe in timeframes:
            tf_results = symbol_result['timeframes'].get(str(timeframe), [])
            tf_data[timeframe] = {r['time']: r for r in tf_results}
            all_times.update(r['time'] for r in tf_results)
        all_times = sorted(all_times)
        print(f"\nSymbol: {symbol}")
        header = ['Time', 'Current Price'] + [f'RSI{tf}' for tf in timeframes]
        print('  ' + '  '.join(f'{h:<15}' for h in header))
        print('  ' + '-' * (15 * len(header)))
        for t in all_times:
            price = 'NA'
            for tf in sorted(timeframes):
                if t in tf_data[tf]:
                    price = tf_data[tf][t]['current_price']
                    break
            row = [t, price]
            for tf in timeframes:
                rsi = tf_data[tf][t]['rsi_value'] if t in tf_data[tf] else 'NA'
                row.append(rsi)
            print('  ' + '  '.join(f'{v:<15}' for v in row))
        print('  ' + '-' * (15 * len(header)))
        latest_time = all_times[-1] if all_times else None
        if latest_time:
                print(f"  Latest: {latest_time} | " + ' | '.join([f'RSI{tf}: {tf_data[tf][latest_time]['rsi_value'] if latest_time in tf_data[tf] else 'NA'}' for tf in timeframes]))
    print("=" * 90)

def main():
    print("RSI SCANNER - TRADINGVIEW CALCULATION METHOD")
    print("=" * 50)
    config = load_config()
    if not config:
        print("Failed to load configuration")
        return
    instrument_mapping = load_instrument_mapping()
    if not instrument_mapping:
        print("Failed to load instrument mapping")
        return
    results = scan_symbols(config, instrument_mapping)
    display_results_table(results)

if __name__ == '__main__':
    main()
