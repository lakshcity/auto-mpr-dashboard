import schedule
import time
from pipeline_manager import run_full_pipeline

def job():
    print("\n⏰ Scheduler triggered job...")
    run_full_pipeline()

# Configure Schedule
schedule.every(30).minutes.do(job)

# Also run once immediately on startup
print("🚀 Scheduler Started. Running first job immediately...")
run_full_pipeline()

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Scheduler stopped by user.")
        break