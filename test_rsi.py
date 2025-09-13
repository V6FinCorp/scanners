import sys
sys.path.append('data_loader')
from rsi_scanner import fetch_timeframe_data
import pandas as pd

print('Testing RSI scanner data_loader integration...')
result = fetch_timeframe_data('ITC', '15mins', days_back=5)
if result is not None:
    print(f'Success! Loaded {len(result)} records')
    print(f'Date range: {result.timestamp.min()} to {result.timestamp.max()}')
    print('RSI scanner data_loader integration working!')
else:
    print('Failed to fetch data')