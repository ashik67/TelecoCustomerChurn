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

# Add StreamHandler for console output (stdout)
import sys
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# Add CloudWatch logging if AWS credentials and log group are set
try:
    import boto3
    from watchtower import CloudWatchLogHandler
    AWS_LOG_GROUP = os.getenv('CLOUDWATCH_LOG_GROUP', 'TelecoCustomerChurnLogs')
    AWS_LOG_STREAM = os.getenv('CLOUDWATCH_LOG_STREAM', 'pipeline')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    if os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('AWS_REGION') or os.getenv('AWS_PROFILE'):
        cw_handler = CloudWatchLogHandler(
            log_group=AWS_LOG_GROUP,
            stream_name=AWS_LOG_STREAM,
            boto3_session=boto3.Session(region_name=AWS_REGION)
        )
        cw_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        cw_handler.setFormatter(formatter)
        logging.getLogger().addHandler(cw_handler)
        logging.info(f"CloudWatch logging enabled: group={AWS_LOG_GROUP}, stream={AWS_LOG_STREAM}, region={AWS_REGION}")
except Exception as e:
    logging.warning(f"CloudWatch logging setup failed: {e}")
