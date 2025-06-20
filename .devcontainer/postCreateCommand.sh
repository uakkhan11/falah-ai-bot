#!/bin/bash
echo "Running post-create setup..."

# Optional: Upgrade pip and install dependencies
pip install --upgrade pip

# Install Python packages (adjust as per your needs)
pip install streamlit pandas gspread oauth2client kiteconnect requests

echo "✅ Post-create setup complete."
