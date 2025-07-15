# Bad Practice Python Code - Educational Example
# WARNING: This code demonstrates what NOT to do

import *
from os import *
import sys, json, random, time, threading, socket, urllib, subprocess
from datetime import *

# Global variables everywhere
USER_DATA = {}
TEMP_STORAGE = []
CONFIG = None
ERROR_COUNT = 0
MAGIC_NUMBER = 42

# Terrible naming conventions
def a(b, c=None):
    global USER_DATA, TEMP_STORAGE, ERROR_COUNT
    x = b
    y = c if c else "default"
    z = x + str(y)
    return z

class data:
    def __init__(self):
        self.stuff = []
        self.things = {}
        self.numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    def do_something(self, param1, param2, param3, param4, param5, param6):
        try:
            result = param1 + param2 * param3 / param4 - param5 % param6
            if result > 0:
                if result < 100:
                    if result != 50:
                        if result % 2 == 0:
                            if result > 10:
                                if result < 90:
                                    return "good"
                                else:
                                    return "too high"
                            else:
                                return "too low"
                        else:
                            return "odd"
                    else:
                        return "fifty"
                else:
                    return "very high"
            else:
                return "negative"
        except:
            pass

# Function with way too many responsibilities
def process_user_input_and_save_to_file_and_send_email_and_log_and_validate(user_input):
    global USER_DATA, TEMP_STORAGE, ERROR_COUNT, CONFIG
    
    # No input validation
    data_obj = data()
    
    # Hard-coded values everywhere
    if len(user_input) > 500:
        return False
    
    # Deeply nested logic
    try:
        if user_input:
            if len(user_input) > 0:
                if user_input != "":
                    if user_input is not None:
                        processed = user_input.upper().lower().strip().replace(" ", "_")
                        
                        # Side effects everywhere
                        USER_DATA[processed] = datetime.now()
                        TEMP_STORAGE.append(processed)
                        
                        # File operations without proper handling
                        f = open("user_data.txt", "a")
                        f.write(processed + "\n")
                        f.close()
                        
                        # Database operations (simulated badly)
                        database_connection = "fake_connection"
                        query = "INSERT INTO users VALUES ('" + processed + "')"
                        
                        # Email sending (simulated)
                        email_body = "User " + processed + " has been processed on " + str(datetime.now())
                        
                        # Logging
                        print("User processed: " + processed)
                        
                        return True
    except Exception as e:
        ERROR_COUNT += 1
        print("Error occurred")
        return None

# Useless inheritance
class BadClass(object):
    def __init__(self):
        super(BadClass, self).__init__()
        self.value = 0
    
    def get_value(self):
        return self.value
    
    def set_value(self, val):
        self.value = val

class WorsClass(BadClass):
    def __init__(self):
        super(WorsClass, self).__init__()
    
    def get_value(self):
        return super(WorsClass, self).get_value()

# Terrible error handling
def divide_numbers(a, b):
    try:
        result = a / b
        return result
    except:
        try:
            result = float(a) / float(b)
            return result
        except:
            try:
                if b == 0:
                    return "Cannot divide by zero"
                else:
                    return a / b
            except:
                return "Error"

# Mutable default arguments
def add_item(item, target_list=[]):
    target_list.append(item)
    return target_list

# String concatenation in loops
def build_huge_string(items):
    result = ""
    for item in items:
        result = result + str(item) + ", "
    return result

# Circular imports (would cause issues in real modules)
# import bad_practice_python  # This would create circular import

# Memory leaks
class LeakyClass:
    instances = []
    
    def __init__(self, data):
        self.data = data
        self.instances.append(self)  # Never cleaned up
        
        # Circular reference
        self.self_ref = self

# Threading without locks
shared_counter = 0

def increment_counter():
    global shared_counter
    for i in range(1000):
        shared_counter += 1

# SQL injection vulnerability (simulated)
def get_user_data(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    # This would be vulnerable to SQL injection
    return f"Executing: {query}"

# Command injection vulnerability (simulated)
def run_user_command(command):
    full_command = "echo " + command
    # This would be vulnerable to command injection
    return f"Would execute: {full_command}"

# Hardcoded credentials
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef"
SECRET_TOKEN = "my_secret_token"

# Main execution with poor structure
if __name__ == "__main__":
    # Initialize global state
    CONFIG = {"debug": True, "version": "1.0", "author": "unknown"}
    
    # Create some objects
    obj1 = WorsClass()
    obj2 = LeakyClass("test data")
    
    # Process some data
    test_data = ["item1", "item2", "item3", "item4", "item5"]
    result_string = build_huge_string(test_data)
    
    # Use mutable defaults
    list1 = add_item("first")
    list2 = add_item("second")  # This will contain both items!
    
    # Terrible math
    math_result = divide_numbers(10, 0)
    
    # Threading issues
    thread1 = threading.Thread(target=increment_counter)
    thread2 = threading.Thread(target=increment_counter)
    thread1.start()
    thread2.start()
    
    # Process user input
    success = process_user_input_and_save_to_file_and_send_email_and_log_and_validate("test_user")
    
    # Print everything
    print("Program completed")
    print("Error count:", ERROR_COUNT)
    print("Shared counter:", shared_counter)
    print("List1:", list1)
    print("List2:", list2)
    print("Result string:", result_string)
    print("Math result:", math_result)
    
    # Exit without cleanup
    sys.exit(0)