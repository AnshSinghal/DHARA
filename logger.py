import logging
from logging.handlers import RotatingFileHandler
import os

class DharaLogger:
    def __init__(self, name=__name__, log_dir="logs", log_file="dhara.log", level=logging.INFO):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        log_path = os.path.join(log_dir, log_file)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger 