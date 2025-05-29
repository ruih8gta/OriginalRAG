import sys
import logging
from logging.handlers import RotatingFileHandler

# ログの設定
def setup_logger(log_path):
    logger = logging.getLogger('Logger')

    fh= RotatingFileHandler(
        filename = log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,  # バックアップファイルの数
        encoding='utf-8'
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[fh]
    )
    return logger

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    import traceback
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(f"Uncaught exception:\n{tb_str}", )
def log_message(logger, message,to_stdout=False ,level='INFO'):
    if level == 'INFO':
        logger.info(message)
    elif level == 'ERROR':
        logger.error(message)
    if to_stdout:
        print(message,flush=True)