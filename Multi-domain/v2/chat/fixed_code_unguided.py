"""
Improved Python Code - Following Best Practices
Fixed version of the problematic code example
"""
import sys
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

# Configure logging properly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserProcessor:
    """Handles user data processing operations."""
    
    def __init__(self):
        self.user_data: Dict[str, datetime] = {}
        self.temp_storage: List[str] = []
        self.error_count: int = 0
        self.config: Dict[str, Any] = {
            "debug": True,
            "version": "1.0",
            "max_input_length": 500
        }
    
    def format_input(self, user_input: str, suffix: Optional[str] = None) -> str:
        """Format user input with optional suffix."""
        if suffix is None:
            suffix = "default"
        return f"{user_input}{suffix}"
    
    def calculate_score(self, a: float, b: float, c: float, d: float, e: float, f: float) -> str:
        """Calculate a score based on multiple parameters."""
        try:
            result = a + b * c / d - e % f
            
            if result <= 0:
                return "negative"
            elif result == 50:
                return "fifty"
            elif result >= 100:
                return "very high"
            elif result % 2 != 0:
                return "odd"
            elif result <= 10:
                return "too low"
            elif result >= 90:
                return "too high"
            else:
                return "good"
                
        except (ZeroDivisionError, ValueError) as e:
            logger.error(f"Calculation error: {e}")
            return "error"
    
    def validate_input(self, user_input: str) -> bool:
        """Validate user input."""
        if not user_input or not isinstance(user_input, str):
            return False
        if len(user_input) > self.config["max_input_length"]:
            return False
        return True
    
    def save_user_data(self, processed_input: str) -> bool:
        """Save user data to file safely."""
        try:
            with open("user_data.txt", "a", encoding="utf-8") as f:
                f.write(f"{processed_input}\n")
            return True
        except IOError as e:
            logger.error(f"File save error: {e}")
            self.error_count += 1
            return False
    
    def process_user_input(self, user_input: str) -> bool:
        """Process user input with proper validation and error handling."""
        if not self.validate_input(user_input):
            logger.warning("Invalid user input")
            return False
        
        try:
            processed = user_input.lower().strip().replace(" ", "_")
            
            # Store data
            self.user_data[processed] = datetime.now()
            self.temp_storage.append(processed)
            
            # Save to file
            if not self.save_user_data(processed):
                return False
            
            logger.info(f"User processed: {processed}")
            return True
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.error_count += 1
            return False

class MathOperations:
    """Handle mathematical operations safely."""
    
    @staticmethod
    def safe_divide(a: float, b: float) -> Optional[float]:
        """Safely divide two numbers."""
        try:
            if b == 0:
                logger.warning("Division by zero attempted")
                return None
            return a / b
        except (TypeError, ValueError) as e:
            logger.error(f"Division error: {e}")
            return None

class StringUtils:
    """String utility functions."""
    
    @staticmethod
    def build_string(items: List[Any]) -> str:
        """Build string from list efficiently."""
        return ", ".join(str(item) for item in items)
    
    @staticmethod
    def add_item_to_list(item: Any, target_list: Optional[List[Any]] = None) -> List[Any]:
        """Add item to list safely."""
        if target_list is None:
            target_list = []
        target_list.append(item)
        return target_list

class ThreadSafeCounter:
    """Thread-safe counter implementation."""
    
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> None:
        """Increment counter safely."""
        with self._lock:
            self._counter += amount
    
    def get_value(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._counter

def increment_counter_worker(counter: ThreadSafeCounter, iterations: int = 1000) -> None:
    """Worker function for counter increment."""
    for _ in range(iterations):
        counter.increment()

def main():
    """Main execution function."""
    # Initialize processor
    processor = UserProcessor()
    
    # Test data processing
    test_data = ["item1", "item2", "item3", "item4", "item5"]
    result_string = StringUtils.build_string(test_data)
    
    # Test list operations
    list1 = StringUtils.add_item_to_list("first")
    list2 = StringUtils.add_item_to_list("second")
    
    # Test math operations
    math_result = MathOperations.safe_divide(10, 2)
    
    # Test thread-safe counter
    counter = ThreadSafeCounter()
    thread1 = threading.Thread(target=increment_counter_worker, args=(counter,))
    thread2 = threading.Thread(target=increment_counter_worker, args=(counter,))
    
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    
    # Process user input
    success = processor.process_user_input("test_user")
    
    # Output results
    logger.info("Program completed successfully")
    logger.info(f"Error count: {processor.error_count}")
    logger.info(f"Counter value: {counter.get_value()}")
    logger.info(f"List1: {list1}")
    logger.info(f"List2: {list2}")
    logger.info(f"Result string: {result_string}")
    logger.info(f"Math result: {math_result}")
    logger.info(f"Processing success: {success}")

if __name__ == "__main__":
    main()
