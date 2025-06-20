#!/bin/bash
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install streamlit pandas kiteconnect gspread oauth2client requests
echo "✅ Dependencies installed successfully."
