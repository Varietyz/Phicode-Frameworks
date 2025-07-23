"""
Systematically Refactored Python Code - PROTOCOL_RUN Guided
Addresses all identified anti-patterns with structured approach
"""
import sys
import json
import threading
import logging
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import re

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Enumeration for validation results."""
    VALID = "valid"
    INVALID_LENGTH = "invalid_length"
    INVALID_CHARACTERS = "invalid_characters"
    INVALID_TYPE = "invalid_type"
    EMPTY_INPUT = "empty_input"

@dataclass
class ProcessingResult:
    """Data class for processing results."""
    success: bool
    message: str
    data: Optional[Any] = None
    error_code: Optional[str] = None

@dataclass
class CalculationParams:
    """Data class for calculation parameters."""
    base: float
    multiplier: float
    multiplicand: float
    divisor: float
    subtrahend: float
    modulus: float

class SecurityUtils:
    """Security utility functions."""
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not isinstance(user_input, str):
            raise TypeError("Input must be a string")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[^\w\s\-_.]', '', user_input)
        return sanitized.strip()
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data for secure storage."""
        salt = secrets.token_hex(16)
        return hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000).hex()

class ConfigurationManager:
    """Manages application configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        self._config = {
            "debug": False,
            "version": "2.0",
            "max_input_length": 500,
            "database_path": "application.db",
            "log_level": "INFO"
        }
        if config_file:
            self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> None:
        """Load configuration from file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._config.update(file_config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config file {config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

class DatabaseManager:
    """Manages database operations safely."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database schema."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data_hash TEXT
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_user_data(self, username: str, data_hash: str) -> bool:
        """Save user data securely using parameterized queries."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO users (username, data_hash) VALUES (?, ?)",
                    (username, data_hash)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save user data: {e}")
            return False

class InputValidator:
    """Validates user input with comprehensive checks."""
    
    def __init__(self, max_length: int = 500):
        self.max_length = max_length
        self.valid_pattern = re.compile(r'^[\w\s\-_.]+$')
    
    def validate(self, user_input: Any) -> ValidationResult:
        """Comprehensive input validation."""
        if user_input is None:
            return ValidationResult.EMPTY_INPUT
        
        if not isinstance(user_input, str):
            return ValidationResult.INVALID_TYPE
        
        if not user_input.strip():
            return ValidationResult.EMPTY_INPUT
        
        if len(user_input) > self.max_length:
            return ValidationResult.INVALID_LENGTH
        
        if not self.valid_pattern.match(user_input):
            return ValidationResult.INVALID_CHARACTERS
        
        return ValidationResult.VALID

class CalculationEngine:
    """Handles mathematical operations with proper error handling."""
    
    def calculate_score(self, params: CalculationParams) -> Union[float, str]:
        """Calculate score with comprehensive error handling."""
        try:
            if params.divisor == 0:
                return "division_by_zero"
            if params.modulus == 0:
                return "modulus_by_zero"
            
            result = (params.base + 
                     params.multiplier * params.multiplicand / params.divisor - 
                     params.subtrahend % params.modulus)
            
            return self._categorize_result(result)
            
        except (TypeError, ValueError, OverflowError) as e:
            logger.error(f"Calculation error: {e}")
            return "calculation_error"
    
    def _categorize_result(self, result: float) -> str:
        """Categorize calculation result."""
        if result <= 0:
            return "negative"
        elif result == 50:
            return "fifty"
        elif result >= 100:
            return "very_high"
        elif result % 2 != 0:
            return "odd"
        elif result <= 10:
            return "too_low"
        elif result >= 90:
            return "too_high"
        else:
            return "good"
    
    @staticmethod
    def safe_divide(dividend: float, divisor: float) -> Optional[float]:
        """Perform safe division operation."""
        try:
            if divisor == 0:
                logger.warning("Division by zero attempted")
                return None
            return dividend / divisor
        except (TypeError, ValueError, OverflowError) as e:
            logger.error(f"Division error: {e}")
            return None

class FileOperationManager:
    """Manages file operations with proper error handling."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def write_user_data(self, filename: str, data: str) -> bool:
        """Write user data to file safely."""
        file_path = self.base_path / filename
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"{data}\n")
            logger.info(f"Data written to {file_path}")
            return True
        except (IOError, OSError) as e:
            logger.error(f"File operation error: {e}")
            return False

class StringOperations:
    """Efficient string operations."""
    
    @staticmethod
    def build_string_efficiently(items: List[Any], separator: str = ", ") -> str:
        """Build string from list using efficient join operation."""
        return separator.join(str(item) for item in items)
    
    @staticmethod
    def add_item_to_new_list(item: Any, existing_list: Optional[List[Any]] = None) -> List[Any]:
        """Add item to list avoiding mutable default argument issue."""
        if existing_list is None:
            result_list = []
        else:
            result_list = existing_list.copy()
        result_list.append(item)
        return result_list

class ThreadSafeCounter:
    """Thread-safe counter with proper synchronization."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter atomically."""
        with self._lock:
            self._value += amount
            return self._value
    
    def get_value(self) -> int:
        """Get current counter value atomically."""
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0

class UserProcessor:
    """Main user processing class with dependency injection."""
    
    def __init__(self, 
                 config_manager: ConfigurationManager,
                 validator: InputValidator,
                 db_manager: DatabaseManager,
                 file_manager: FileOperationManager,
                 calc_engine: CalculationEngine):
        self.config = config_manager
        self.validator = validator
        self.db_manager = db_manager
        self.file_manager = file_manager
        self.calc_engine = calc_engine
        self.user_data: Dict[str, datetime] = {}
        self.temp_storage: List[str] = []
        self.error_count = 0
        self._lock = threading.Lock()
    
    def process_user_input(self, user_input: str) -> ProcessingResult:
        """Process user input with comprehensive validation and error handling."""
        # Validate input
        validation_result = self.validator.validate(user_input)
        if validation_result != ValidationResult.VALID:
            return ProcessingResult(
                success=False,
                message=f"Validation failed: {validation_result.value}",
                error_code=validation_result.value
            )
        
        try:
            # Sanitize input
            sanitized_input = SecurityUtils.sanitize_input(user_input)
            processed_input = sanitized_input.lower().strip().replace(" ", "_")
            
            # Thread-safe data storage
            with self._lock:
                self.user_data[processed_input] = datetime.now()
                self.temp_storage.append(processed_input)
            
            # Hash sensitive data
            data_hash = SecurityUtils.hash_sensitive_data(processed_input)
            
            # Save to database and file
            db_success = self.db_manager.save_user_data(processed_input, data_hash)
            file_success = self.file_manager.write_user_data("user_data.txt", processed_input)
            
            if not (db_success and file_success):
                self._increment_error_count()
                return ProcessingResult(
                    success=False,
                    message="Failed to save data",
                    error_code="save_failure"
                )
            
            logger.info(f"Successfully processed user: {processed_input}")
            return ProcessingResult(
                success=True,
                message="User input processed successfully",
                data={"processed_input": processed_input, "timestamp": datetime.now()}
            )
            
        except Exception as e:
            self._increment_error_count()
            logger.error(f"Processing error: {e}")
            return ProcessingResult(
                success=False,
                message=f"Processing failed: {str(e)}",
                error_code="processing_error"
            )
    
    def _increment_error_count(self) -> None:
        """Thread-safe error count increment."""
        with self._lock:
            self.error_count += 1
    
    def get_error_count(self) -> int:
        """Get current error count."""
        with self._lock:
            return self.error_count

def worker_increment_counter(counter: ThreadSafeCounter, iterations: int = 1000) -> None:
    """Worker function for counter increment operations."""
    for _ in range(iterations):
        counter.increment()

class ApplicationController:
    """Main application controller coordinating all components."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.validator = InputValidator(self.config.get("max_input_length", 500))
        self.db_manager = DatabaseManager(self.config.get("database_path", "app.db"))
        self.file_manager = FileOperationManager()
        self.calc_engine = CalculationEngine()
        self.user_processor = UserProcessor(
            self.config, self.validator, self.db_manager, 
            self.file_manager, self.calc_engine
        )
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive application test."""
        results = {}
        
        # Test string operations
        test_data = ["item1", "item2", "item3", "item4", "item5"]
        results["string_result"] = StringOperations.build_string_efficiently(test_data)
        
        # Test list operations (avoiding mutable default arguments)
        list1 = StringOperations.add_item_to_new_list("first")
        list2 = StringOperations.add_item_to_new_list("second")
        results["list1"] = list1
        results["list2"] = list2
        
        # Test mathematical operations
        math_result = self.calc_engine.safe_divide(10, 2)
        results["math_result"] = math_result
        
        # Test thread-safe operations
        counter = ThreadSafeCounter()
        threads = [
            threading.Thread(target=worker_increment_counter, args=(counter, 1000)),
            threading.Thread(target=worker_increment_counter, args=(counter, 1000))
        ]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        results["counter_value"] = counter.get_value()
        
        # Test user processing
        processing_result = self.user_processor.process_user_input("test_user")
        results["processing_success"] = processing_result.success
        results["processing_message"] = processing_result.message
        
        # Test calculation engine
        calc_params = CalculationParams(10, 2, 3, 4, 1, 2)
        calc_result = self.calc_engine.calculate_score(calc_params)
        results["calculation_result"] = calc_result
        
        results["error_count"] = self.user_processor.get_error_count()
        
        return results

def main() -> None:
    """Main execution function with proper error handling."""
    try:
        app = ApplicationController()
        results = app.run_comprehensive_test()
        
        logger.info("Application execution completed successfully")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
