from flask import Flask, render_template_string
import subprocess
import os
app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scanners Dashboard</title>
    <style>
        body { background: #181a20; color: #eee; font-family: Arial, sans-serif; }
        .tabs { display: flex; margin-bottom: 1em; }
        .tab { padding: 1em 2em; cursor: pointer; background: #222; border: none; color: #eee; margin-right: 2px; }
        .tab.active { background: #007bff; color: #fff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        pre { background: #222; color: #eee; padding: 1em; border-radius: 6px; overflow-x: auto; }
    </style>
    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(e => e.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            document.getElementById(tabId + '-btn').classList.add('active');
        }
        window.onload = function() { showTab('rsi'); };
    </script>
</head>
<body>
    <h2>Scanners Dashboard (Live Output)</h2>
    <div class="tabs">
        <button class="tab" id="rsi-btn" onclick="showTab('rsi')">RSI Scanner</button>
        <button class="tab" id="dma-btn" onclick="showTab('dma')">DMA Scanner</button>
        <button class="tab" id="ema-btn" onclick="showTab('ema')">EMA Scanner</button>
    </div>
    <div id="rsi" class="tab-content">
        <h3>RSI Scanner Output</h3>
        <pre>{{ rsi_output }}</pre>
    </div>
    <div id="dma" class="tab-content">
        <h3>DMA Scanner Output</h3>
        <pre>{{ dma_output }}</pre>
    </div>
    <div id="ema" class="tab-content">
        <h3>EMA Scanner Output</h3>
        <pre>{{ ema_output }}</pre>
    </div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    rsi_path = os.path.join(base_dir, 'rsi_scanner.py')
    dma_path = os.path.join(base_dir, 'dma_scanner.py')
    ema_path = os.path.join(base_dir, 'ema_scanner.py')
    rsi_output = subprocess.getoutput(f'python "{rsi_path}"')
    dma_output = subprocess.getoutput(f'python "{dma_path}"')
    ema_output = subprocess.getoutput(f'python "{ema_path}"')
    return render_template_string(TEMPLATE, rsi_output=rsi_output, dma_output=dma_output, ema_output=ema_output)

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use PORT environment variable for platform compatibility (Railway, Heroku, etc.)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
