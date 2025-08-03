import logging
from logging.handlers import RotatingFileHandler
import os

class DharaLogger:
    def __init__(self, name=__name__, log_dir="logs", log_file="dhara.log", level=logging.INFO, console_output=True):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(name)
        
        # Clear existing handlers to prevent duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        self.logger.setLevel(level)
        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False
        
        log_path = os.path.join(log_dir, log_file)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler - always add
        file_handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)

        # Console handler - add based on console_output parameter
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

    def test_logging(self):
        """Test method to verify both console and file logging work"""
        logger = self.get_logger()
        logger.info("🔍 Testing DharaLogger - this should appear in both console and log file")
        logger.warning("⚠️ This is a warning message")
        logger.error("❌ This is an error message")
        logger.debug("🐛 This is a debug message (may not appear if level is INFO)")

# Example usage and test
if __name__ == "__main__":
    print("=== DharaLogger Test ===")
    
    # Test with console output (default)
    dhara_logger = DharaLogger(name="test_logger", log_file="test_dhara.log")
    dhara_logger.test_logging()
    
    print("\n=== Check the 'logs/test_dhara.log' file to see the same messages ===")
    
    # Test without console output (file only)
    print("\n=== Testing file-only logging ===")
    file_only_logger = DharaLogger(name="file_only_test", log_file="file_only.log", console_output=False)
    file_only_logger.get_logger().info("This message should only appear in the log file, not console")
    print("The above message was logged to file only - check logs/file_only.log") 