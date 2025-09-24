"""
Volume Scanner - Detects volume spikes based on moving average and multiplier.

Produces a CSV in data/<SYMBOL>/<SYMBOL>_volume_data.csv and prints a legacy output block
that the dashboard can parse. Designed to follow the pattern used by other scanners.
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd


def fetch_data_via_loader(symbol, days_back=365):
    """Use data_loader.fetch_data_for_symbol to get combined CSV path and load it"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader'))
        from data_loader import fetch_data_for_symbol

        combined_file = fetch_data_for_symbol(symbol, days_back)
        if combined_file and os.path.exists(combined_file):
            df = pd.read_csv(combined_file)
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            print(f"data_loader did not return a file for {symbol}")
            return None
    except Exception as e:
        print(f"Error fetching data via data_loader for {symbol}: {e}")
        return None


def run_volume_scanner():
    print("Volume Scanner Starting...")

    config_path = os.path.join(os.path.dirname(__file__), 'config', 'volume_config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config: {config_path}")
    except FileNotFoundError:
        print("Config file not found. Using built-in defaults.")
        config = {
            "symbols": ["ITC"],
            "ma_window": 20,
            "multiplier": 2.0,
            "min_volume": 100000,
            "base_timeframe": "daily",
            "days_to_list": 30,
            "days_fallback_threshold": 365
        }

    symbols = config.get('symbols', [])
    ma_window = int(config.get('ma_window', 20))
    multiplier = float(config.get('multiplier', 2.0))
    min_volume = int(config.get('min_volume', 100000))
    base_timeframe = config.get('base_timeframe', 'daily')
    days_to_list = int(config.get('days_to_list', 30))
    days_fallback = int(config.get('days_fallback_threshold', 365))

    print(f"Scanning symbols: {symbols}")
    print(f"MA Window: {ma_window}, Multiplier: {multiplier}, MinVolume: {min_volume}")

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        df = fetch_data_via_loader(symbol, days_back=days_fallback)
        if df is None or df.empty:
            print(f"No data for {symbol}, skipping")
            continue

        # Ensure volume and close columns exist
        if 'volume' not in df.columns:
            print(f"No 'volume' column for {symbol}, skipping")
            continue

        # Use the dataframe as-is; if timeframe resampling is required it should already be done
        # Compute rolling average of volume
        df = df.sort_values('timestamp')
        df['avg_volume'] = df['volume'].rolling(window=ma_window, min_periods=1).mean()
        # Avoid division by zero
        df['vol_ratio'] = df.apply(lambda r: (float(r['volume']) / float(r['avg_volume'])) if r['avg_volume'] and r['avg_volume'] > 0 else None, axis=1)

        # Determine status
        def status_row(r):
            if r['volume'] is None:
                return 'N/A'
            if r['avg_volume'] is None or r['avg_volume'] <= 0:
                return 'LOW'
            try:
                if float(r['volume']) >= max(min_volume, multiplier * float(r['avg_volume'])):
                    return 'SPIKE'
            except Exception:
                pass
            try:
                if float(r['volume']) > float(r['avg_volume']):
                    return 'HIGH'
            except Exception:
                pass
            return 'NORMAL'

        df['status'] = df.apply(status_row, axis=1)

        # Select last N rows for output
        out_df = df.tail(days_to_list).copy()

        # Save CSV in data/<symbol>/<symbol>_volume_data.csv
        data_dir = os.path.join(os.path.dirname(__file__), 'data', symbol)
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, f'{symbol}_volume_data.csv')

        # Try to include close price if present
        columns_to_save = ['timestamp']
        if 'close' in out_df.columns:
            columns_to_save.append('close')
        columns_to_save += ['volume', 'avg_volume', 'vol_ratio', 'status']

        try:
            out_df.to_csv(csv_path, index=False, columns=columns_to_save)
            print(f"Saved volume CSV: {csv_path}")
        except Exception as e:
            print(f"Failed to save CSV for {symbol}: {e}")

        # Print legacy output block for dashboard parsing
        try:
            print('\n--- VOLUME_SCANNER_OUTPUT_START ---')
            headers = ['Time', 'Symbol']
            if 'close' in out_df.columns:
                headers.append('CMP')
            headers += ['Volume', 'AvgVolume', 'VolRatio', 'Status', 'Recommendation']
            print(','.join(headers))

            for _, row in out_df.iterrows():
                time_str = ''
                try:
                    time_str = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    time_str = str(row['timestamp'])

                symbol_val = symbol
                cmp_val = f"{row['close']:.2f}" if 'close' in row and not pd.isna(row.get('close')) else ''
                vol = int(row['volume']) if not pd.isna(row['volume']) else ''
                avgv = f"{row['avg_volume']:.2f}" if not pd.isna(row['avg_volume']) else ''
                vr = f"{row['vol_ratio']:.4f}" if not pd.isna(row.get('vol_ratio')) else ''
                status = row.get('status', '')
                rec = 'WATCH' if status == 'SPIKE' else 'NONE'

                parts = [time_str, symbol_val]
                if 'close' in out_df.columns:
                    parts.append(cmp_val)
                parts += [str(vol), str(avgv), str(vr), status, rec]
                print(','.join(parts))

            print('--- VOLUME_SCANNER_OUTPUT_END ---')
        except Exception as e:
            print(f"Error printing legacy output block: {e}")

    print("Volume Scanner completed.")


if __name__ == '__main__':
    run_volume_scanner()
