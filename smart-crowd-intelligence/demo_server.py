from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import json
import random
from datetime import datetime

app = FastAPI(title="Smart Crowd Intelligence Demo")

connections = []

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html>
<head><title>Smart Crowd Intelligence Demo</title>
<style>
body { font-family: Arial; margin: 20px; background: #1a1a2e; color: white; }
.header { text-align: center; margin-bottom: 30px; }
.card { background: #16213e; padding: 20px; margin: 10px; border-radius: 10px; }
.metric { display: flex; justify-content: space-between; margin: 10px 0; }
.value { font-weight: bold; color: #4ade80; }
</style></head>
<body>
<div class="header">
<h1>🎥 Smart Crowd Intelligence System</h1>
<p>Real-time AI-powered crowd analysis demo</p>
</div>
<div class="card">
<h3>📊 Live Data</h3>
<div class="metric"><span>Crowd Count:</span><span class="value" id="count">-</span></div>
<div class="metric"><span>Density:</span><span class="value" id="density">-</span></div>
<div class="metric"><span>Status:</span><span class="value" id="status">Loading...</span></div>
</div>
<script>
const ws = new WebSocket(`ws://${window.location.host}/ws`);
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    document.getElementById('count').textContent = data.crowd_count;
    document.getElementById('density').textContent = data.density_level;
    document.getElementById('status').textContent = 'LIVE ● ' + new Date().toLocaleTimeString();
};
</script>
</body>
</html>
    """

@app.get("/health")
async def health():
    return {"status": "healthy", "connections": len(connections)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        connections.remove(websocket)

async def generate_data():
    while True:
        if connections:
            data = {
                "crowd_count": random.randint(10, 300),
                "density_level": random.choice(["low", "medium", "high"]),
                "timestamp": datetime.now().isoformat()
            }
            for conn in connections.copy():
                try:
                    await conn.send_text(json.dumps(data))
                except:
                    connections.remove(conn)
        await asyncio.sleep(3)

@app.on_event("startup")
async def startup():
    asyncio.create_task(generate_data())
    print("🚀 Smart Crowd Intelligence Demo ready at http://localhost:8000")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
