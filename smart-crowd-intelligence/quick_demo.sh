#!/bin/bash

echo "🚀 Smart Crowd Intelligence System - Quick Demo Setup"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "demo_server.py" ]; then
    echo "📁 Navigating to smart-crowd-intelligence directory..."
    cd smart-crowd-intelligence 2>/dev/null || {
        echo "❌ Error: demo_server.py not found. Make sure you're in the right directory."
        echo "💡 Try: cd smart-crowd-intelligence"
        exit 1
    }
fi

echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv

echo "📦 Activating virtual environment..."
source venv/bin/activate

echo "⬇️  Installing dependencies..."
pip install --quiet fastapi uvicorn websockets

echo "🎯 Starting Smart Crowd Intelligence Demo..."
echo ""
echo "🌐 Dashboard will be available at:"
echo "   → http://localhost:8000"
echo ""
echo "📋 API Endpoints:"
echo "   → Health: http://localhost:8000/health"
echo "   → Analytics: http://localhost:8000/analytics"
echo "   → API Docs: http://localhost:8000/docs"
echo ""
echo "🔄 Press Ctrl+C to stop the server"
echo "================================================="

python demo_server.py
