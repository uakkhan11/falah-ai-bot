import time
import subprocess

while True:
    try:
        print("✅ Running monitor.py...")
        subprocess.run(["python3", "/root/falah-ai-bot/monitor.py"])
    except Exception as e:
        print(f"❌ Error running monitor: {e}")
    time.sleep(900)  # Sleep 15 min
