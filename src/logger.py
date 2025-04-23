import logging
import os
from datetime import datetime, timedelta
import time
import threading

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def cleanup_old_logs(directory: str, days: int = 7):
    cutoff_time = datetime.now() - timedelta(days=days)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_mod_time < cutoff_time:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted old log file: {filename}")
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {str(e)}")

def periodic_cleanup(interval: int, directory: str, days: int):
    while True:
        cleanup_old_logs(directory, days)
        time.sleep(interval)

if __name__ == "__main__":
    logging.info("Logging has started")
    
    
    cleanup_thread = threading.Thread(
        target=periodic_cleanup,
        args=(24 * 60 * 60, LOGS_DIR, 7),
        daemon=True
    )
    cleanup_thread.start()

    
    while True:
        logging.info("Simulating Streamlit app logging")
        time.sleep(10)