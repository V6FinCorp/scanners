# RSI Scanner Configuration Guide

## Configuration Options

### `symbols`
- **Type**: Array of strings
- **Description**: List of stock symbols to analyze
- **Example**: `["RELIANCE", "LT", "TCS"]`

### `days_fallback_threshold`
- **Type**: Integer
- **Description**: MAXIMUM number of days to fetch from the API (hard limit)
- **Behavior**: Caps the data fetching to prevent excessive API calls and control costs
- **Example**: `10` (fetch maximum 10 days, even if RSI needs more for perfect accuracy)
- **Purpose**: Controls API usage and prevents the scanner from fetching too much historical data

### `rsi_periods`
- **Type**: Array of integers
- **Description**: RSI calculation periods (number of candles to use)
- **Example**: `[15, 30, 60]` (calculate RSI-15, RSI-30, and RSI-60)

### `base_timeframe`
- **Type**: String
- **Description**: Primary timeframe for data fetching and analysis
- **Supported Values**:
  - `"5mins"` - 5-minute candles
  - `"15mins"` - 15-minute candles
  - `"30mins"` - 30-minute candles
  - `"1hour"` - 1-hour candles
  - `"2hours"` - 2-hour candles
  - `"4hours"` - 4-hour candles
  - `"daily"` - Daily candles
  - `"weekly"` - Weekly candles
  - `"monthly"` - Monthly candles
  - `"yearly"` - Yearly candles
- **Example**: `"15mins"`

### `default_timeframe`
- **Type**: String
- **Description**: Fallback timeframe if base_timeframe is not set
- **Example**: `"15mins"`

### `days_to_list`
- **Type**: Integer
- **Description**: Number of days to display in output
- **Behavior**: Scanner shows data for these many days, always starting from 9:15 AM market open
- **Example**: `2` (show data for last 2 trading days from 9:15 AM)

## Timeframe Examples

### For Intraday Analysis (Minutes/Hours)
```json
{
  "base_timeframe": "15mins",
  "rsi_periods": [15, 30, 60],
  "days_to_list": 2
}
```

### For Daily Analysis
```json
{
  "base_timeframe": "daily",
  "rsi_periods": [14, 21, 28],
  "days_to_list": 30
}
```

### For Weekly Analysis
```json
{
  "base_timeframe": "weekly",
  "rsi_periods": [4, 8, 12],
  "days_to_list": 52
}
```

## Important Notes

1. **Market Hours**: Data always starts from 9:15 AM (market open time)
2. **Data Availability**: Intraday data is typically limited to recent periods (30-90 days)
3. **RSI Accuracy**: More historical data = more accurate RSI calculations
4. **Dynamic Fetching**: `days_fallback_threshold` acts as hard maximum limit for API control
5. **Display Control**: `days_to_list` controls how many days to show, independent of fetching limits

## Example Scenarios

### Scenario 1: RSI needs 15 days, but limited to 10 days max
```json
{
  "rsi_periods": [60],
  "days_fallback_threshold": 10,
  "days_to_list": 5
}
```
**Result**: Fetches 10 days (capped by days_fallback_threshold), displays last 5 days

### Scenario 2: RSI needs 8 days, well within limit
```json
{
  "rsi_periods": [30], 
  "days_fallback_threshold": 20,
  "days_to_list": 7
}
```
**Result**: Fetches 8 days (RSI requirement), displays last 7 days