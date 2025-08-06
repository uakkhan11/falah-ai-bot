# main.py
from model_training import model_training
from monitor import monitor_positions

if __name__ == "__main__":
    print("🚀 Starting AI Trading System...")
    
    print("\n📊 Training model...")
    model_training()

    print("\n👀 Starting position monitoring...")
    monitor_positions()

    print("\n✅ AI Trading System started successfully.")
