import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOGS_PATH=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(LOGS_PATH, exist_ok=True)

LOG_FILE_PATH=os.path.join(LOGS_PATH,LOG_FILE)

os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)   
