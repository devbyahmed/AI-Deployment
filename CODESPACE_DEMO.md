# Smart Crowd Intelligence System - GitHub Codespaces Demo

## 🚀 Quick Start Guide

This guide shows you how to run the Smart Event Crowd Intelligence System demo in GitHub Codespaces.

### 📋 Prerequisites

- GitHub account with Codespaces access
- No additional software needed (everything runs in the browser!)

### 🎯 One-Click Setup

1. **Open GitHub Codespace:**
   - Go to [AI-Deployment Repository](https://github.com/devbyahmed/AI-Deployment)
   - Click **"<> Code"** → **"Codespaces"** → **"Create codespace on main"**

2. **Navigate to Project:**
   ```bash
   cd smart-crowd-intelligence
   ```

3. **Set up Python Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn websockets
   ```

4. **Start the Demo:**
   ```bash
   python demo_server.py
   ```

5. **Access the Dashboard:**
   - Open http://localhost:8000 in your browser
   - Or use the Codespace Ports tab to access the application

## 🎥 What You'll See

### Live Dashboard Features:
- **📊 Real-time crowd data** updating every 3 seconds
- **🧠 AI behavior analysis** with 5 behavior categories
- **🚨 Smart alerts** triggered by crowd conditions
- **📝 Live system logs** showing real-time activity

### Demo Data:
- **Crowd Count**: 10-300 people (simulated)
- **Density Levels**: Low, Medium, High, Critical
- **Movement Speed**: 0.5-3.0 m/s
- **Behavior Analysis**: Normal, Aggressive, Panic, Gathering, Dispersing
- **Smart Alerts**: Triggered when crowd > 200 or panic > 5%

## 🛠️ API Endpoints

Test these endpoints while the demo is running:

```bash
# Health check
curl http://localhost:8000/health

# Analytics data
curl http://localhost:8000/analytics

# Interactive API docs
# Visit: http://localhost:8000/docs
```

## 🎉 Success Indicators

✅ **Server Started**: Console shows "Smart Crowd Intelligence Demo ready"  
✅ **Dashboard Loaded**: Browser displays the dark-themed interface  
✅ **Real-time Updates**: Crowd count changes every 3 seconds  
✅ **WebSocket Connected**: System logs show "Connected to Smart Crowd Intelligence System"  
✅ **Alerts Working**: Warnings appear when crowd > 200 people  

---

**�� Congratulations!** You've successfully deployed and tested the Smart Event Crowd Intelligence System in GitHub Codespaces.
