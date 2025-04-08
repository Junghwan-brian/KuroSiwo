import logging
import os


class FileLogger:
    def __init__(self, log_file, log_level=logging.INFO):
        self.log_file = log_file
        self.log_level = log_level
        self.logger = logging.getLogger(os.path.basename(log_file))
        self._configure_logger()

    def _configure_logger(self):
        # Ensure the logger is not configured multiple times
        if not self.logger.hasHandlers():
            self.logger.setLevel(self.log_level)
            file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, message):
        if self.log_level == 'info':
            self.logger.info(message)
        elif self.log_level == 'debug':
            self.logger.debug(message)
        elif self.log_level == 'warning':
            self.logger.warning(message)
        elif self.log_level == 'error':
            self.logger.error(message)
        elif self.log_level == 'critical':
            self.logger.critical(message)
        else:
            raise ValueError(
                f"Unsupported log self.log_level: {self.log_level}")

    def print(self, message):
        self.log(message)

# Example usage:
# logger = FileLogger('/path/to/logfile.log')
# logger.log('info', 'This is an info message.')
