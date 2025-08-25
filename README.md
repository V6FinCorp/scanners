# Scanners

This repository contains simple scanner scripts (RSI, DMA, EMA) and a lightweight Flask wrapper `run_scanners_server.py` that runs each scanner and shows their stdout in a small web dashboard.

Usage

- Locally:

```powershell
python run_scanners_server.py
```

- On Railway / Heroku:
  - Railway will use the `Procfile` to start the app: `web: python run_scanners_server.py`.
  - Ensure the following environment variables are set in your Railway project (if required by the scanners):
    - `UPSTOX_ACCESS_TOKEN` (if any of the scanners call Upstox APIs requiring auth)

Dependencies

- See `requirements.txt`. Railway / Nixpacks will install these automatically.

Notes

- The Flask server binds to `0.0.0.0` and reads the `PORT` environment variable.
- The scanners expect their configuration files to live under the `config/` directory.
