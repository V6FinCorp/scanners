# Trading Scanners Dashboard

A comprehensive web dashboard for running and visualizing technical analysis scanners including RSI, EMA, and DMA calculations.

## 🚀 Features

- **Interactive Web Interface**: Modern, responsive dashboard built with HTML, CSS, and JavaScript
- **Multiple Scanners**: Support for RSI, EMA, and DMA analysis
- **Real-time Execution**: Run scanners directly from the web interface
- **Parameter Customization**: Configure scanner parameters through the UI
- **Data Visualization**: Chart visualization of price and indicator data
- **CSV Export**: Export results to CSV files
- **API Backend**: RESTful API for scanner execution

## 📊 Available Scanners

### RSI Scanner
- Calculates RSI using TradingView-compatible RMA (Running Moving Average) method
- Supports multiple periods (default: 15, 30, 60)
- Multi-timeframe analysis capability

### EMA Scanner
- Calculates EMA using TradingView's exact method
- Supports multiple periods (default: 9, 15, 65, 200)
- Exponential smoothing with proper alpha calculation

### DMA Scanner
- Calculates Displaced Moving Averages
- Configurable displacement (default: 1 period)
- Supports multiple periods (default: 10, 20, 50)

## 🛠️ Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Dashboard Server:**
   ```bash
   python dashboard_server.py
   ```

3. **Access the Dashboard:**
   Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
├── dashboard.html              # Main dashboard HTML interface
├── dashboard_server.py        # Flask server with API endpoints
├── requirements.txt            # Python dependencies
├── scanners/                   # Scanner implementations
│   ├── rsi_scanner.py         # RSI scanner
│   ├── ema_scanner.py         # EMA scanner
│   ├── dma_scanner.py         # DMA scanner
│   └── config/                # Scanner configurations
│       ├── rsi_config.json
│       ├── ema_config.json
│       └── dma_config.json
└── data/                      # Output data directory
    └── [SYMBOL]/              # Symbol-specific data
        ├── [SYMBOL]_rsi_data.csv
        ├── [SYMBOL]_ema_data.csv
        └── [SYMBOL]_dma_data.csv
```

## 🔧 Configuration

Each scanner has its own configuration file in `scanners/config/`:

### RSI Config (`rsi_config.json`)
```json
{
    "symbols": ["RELIANCE"],
    "rsi_periods": [15, 30, 60],
    "base_timeframe": "15mins",
    "days_to_list": 2,
    "days_fallback_threshold": 200
}
```

### EMA Config (`ema_config.json`)
```json
{
    "symbols": ["RELIANCE"],
    "ema_periods": [9, 15, 65, 200],
    "base_timeframe": "15mins",
    "days_to_list": 2,
    "days_fallback_threshold": 200
}
```

### DMA Config (`dma_config.json`)
```json
{
    "symbols": ["RELIANCE"],
    "dma_periods": [10, 20, 50],
    "base_timeframe": "15mins",
    "days_to_list": 2,
    "days_fallback_threshold": 200,
    "displacement": 1
}
```

## 🌐 API Endpoints

### Run Scanner
```http
POST /api/run-scanner
Content-Type: application/json

{
    "scanner": "rsi|ema|dma",
    "symbols": ["RELIANCE"],
    "rsi_periods": [15, 30, 60],
    "baseTimeframe": "15mins",
    "daysToList": 2,
    "daysFallbackThreshold": 200
}
```

### Get Scanner Status
```http
GET /api/scanner-status
```

## 📈 Usage Examples

### Running RSI Scanner
```javascript
fetch('/api/run-scanner', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        scanner: 'rsi',
        symbols: ['RELIANCE'],
        rsi_periods: [15, 30, 60],
        baseTimeframe: '15mins',
        daysToList: 2
    })
});
```

### Running EMA Scanner
```javascript
fetch('/api/run-scanner', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        scanner: 'ema',
        symbols: ['RELIANCE'],
        ema_periods: [9, 15, 65, 200],
        baseTimeframe: '15mins',
        daysToList: 2
    })
});
```

## 🎨 Dashboard Features

- **Parameter Selection**: Interactive forms for configuring scanner parameters
- **Real-time Feedback**: Loading indicators and status messages
- **Data Visualization**: Chart.js integration for price and indicator visualization
- **Export Functionality**: Download results as CSV files
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, professional interface with Tailwind CSS

## 🔍 Data Sources

The scanners fetch data from Upstox API with support for:
- Multiple timeframes (5min, 15min, 30min, 1hour, daily, etc.)
- Historical data up to 2 years for daily data
- Real-time market data filtering (9:15 AM market open)

## 📊 Output Formats

### Console Output
Professional table format with:
- Timestamp, Symbol, Close Price
- Indicator values (RSI/EMA/DMA)
- Proper column alignment
- Market hours filtering

### CSV Export
Structured data files containing:
- OHLCV data
- Calculated indicators
- Timestamps
- Ready for further analysis

### Chart Visualization
Interactive charts showing:
- Price action
- Indicator overlays
- Multiple timeframe support
- Zoom and pan capabilities

## 🚀 Deployment

For production deployment:

1. **Railway/Render/Heroku:**
   ```bash
   # Set environment variables
   export FLASK_ENV=production
   export PORT=5000

   # Start server
   gunicorn dashboard_server:app
   ```

2. **Docker:**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "dashboard_server.py"]
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the configuration files
2. Verify API connectivity
3. Check scanner logs
4. Review error messages in the dashboard

---

**Built with ❤️ for technical analysis enthusiasts**