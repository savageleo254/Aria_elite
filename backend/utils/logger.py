import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import gzip
import shutil

def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = os.path.join(log_dir, f'{name}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

class CompressedFileHandler(logging.Handler):
    """
    Custom handler that writes to compressed log files
    """
    
    def __init__(self, filename: str, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.current_file = None
        self.current_size = 0
        self._open_file()
    
    def _open_file(self):
        """Open current log file"""
        if self.current_file:
            self.current_file.close()
        
        self.current_file = open(self.filename, 'a')
        self.current_size = os.path.getsize(self.filename) if os.path.exists(self.filename) else 0
    
    def emit(self, record):
        """Emit a log record"""
        try:
            msg = self.format(record)
            msg += '\n'
            
            # Check if we need to rotate
            if self.current_size + len(msg) > self.max_bytes:
                self._rotate_files()
            
            # Write message
            self.current_file.write(msg)
            self.current_file.flush()
            self.current_size += len(msg)
            
        except Exception:
            self.handleError(record)
    
    def _rotate_files(self):
        """Rotate log files and compress old ones"""
        self.current_file.close()
        
        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            old_file = f"{self.filename}.{i}"
            new_file = f"{self.filename}.{i + 1}"
            
            if os.path.exists(old_file):
                if i == self.backup_count - 1:
                    # Compress and remove oldest file
                    self._compress_file(old_file)
                    os.remove(old_file)
                else:
                    # Rename file
                    if os.path.exists(new_file):
                        os.remove(new_file)
                    os.rename(old_file, new_file)
        
        # Move current file to .1
        if os.path.exists(self.filename):
            os.rename(self.filename, f"{self.filename}.1")
        
        # Open new file
        self._open_file()
    
    def _compress_file(self, filename: str):
        """Compress a file"""
        try:
            compressed_file = f"{filename}.gz"
            with open(filename, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"Error compressing file {filename}: {e}")
    
    def close(self):
        """Close the handler"""
        if self.current_file:
            self.current_file.close()

def setup_compressed_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Set up logger with compressed file rotation
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Compressed file handler
    log_file = os.path.join(log_dir, f'{name}.log')
    compressed_handler = CompressedFileHandler(
        log_file,
        max_bytes=10*1024*1024,  # 10MB
        backup_count=5
    )
    compressed_handler.setLevel(logging.INFO)
    compressed_handler.setFormatter(formatter)
    logger.addHandler(compressed_handler)
    
    return logger
