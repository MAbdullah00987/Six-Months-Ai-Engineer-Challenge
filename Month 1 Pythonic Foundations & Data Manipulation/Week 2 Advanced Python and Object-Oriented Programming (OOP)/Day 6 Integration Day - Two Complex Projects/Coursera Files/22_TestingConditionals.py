
#Testing Conditionals in Python
#Definition of Testing Conditionals
#Testing conditionals refers to the process of verifying that conditional statements (if/elif/else) in your code work correctly under all possible scenarios. It involves checking that each branch of logic executes as expected with different inputs, edge cases, and boundary conditions.
#Simple explanation: Testing conditionals means making sure your "if-then-else" decisions produce the right outcomes for every possible situation.

#Why We Need to Test Conditionals
#1. Ensure Correctness
#Verify that each condition produces the expected result for all possible inputs.
#2. Catch Logic Errors
#Identify mistakes in conditional logic before they cause problems in production.
#3. Prevent Bugs
#Find and fix issues like off-by-one errors, incorrect operators, or missing conditions.
#4. Handle Edge Cases
#Test boundary values and unusual inputs that might break your logic.
#5. Increase Confidence
#Know your code works reliably before deploying to users.
#6. Documentation
#Tests serve as examples showing how code should behave.
#7. Refactoring Safety
#Safely modify code knowing tests will catch any regressions.
#8. Business Logic Validation
#Ensure business rules are implemented correctly (e.g., pricing, eligibility).

#How to Test Conditionals Professionally
#Testing Strategies:

#Test All Branches - Every if/elif/else path should be tested
#Boundary Testing - Test values at the edges of conditions
#Equivalence Partitioning - Test representative values from each range
#Negative Testing - Test with invalid/unexpected inputs
#Edge Cases - Test extreme values (0, negative, very large numbers)

#Professional Testing Approaches:
#1. Manual Testing (Print statements)
#2. Automated Unit Testing (unittest, pytest)
#3. Test-Driven Development (TDD) (Write tests first)
#4. Code Coverage Tools (Measure test completeness)
#5. Integration Testing (Test how conditionals work together)

# Examples of Testing Conditionals
#Example 1: Testing Grade Classification System
# Function to test
def calculate_grade(marks):
    """
    Calculate letter grade based on marks
    Returns: grade (str), feedback (str)
    """
    if marks < 0 or marks > 100:
        return "INVALID", "Marks must be between 0 and 100"
    elif marks >= 90:
        return "A+", "Outstanding"
    elif marks >= 80:
        return "A", "Excellent"
    elif marks >= 70:
        return "B", "Good"
    elif marks >= 60:
        return "C", "Average"
    elif marks >= 50:
        return "D", "Below Average"
    else:
        return "F", "Fail"


# PROFESSIONAL TESTING APPROACH 1: Manual Testing with Clear Output
print("="*70)
print("TESTING GRADE CLASSIFICATION SYSTEM")
print("="*70)

test_cases = [
    # (input, expected_grade, expected_feedback, description)
    (-5, "INVALID", "Marks must be between 0 and 100", "Negative marks"),
    (105, "INVALID", "Marks must be between 0 and 100", "Marks over 100"),
    (0, "F", "Fail", "Minimum boundary - Fail"),
    (49, "F", "Fail", "Just below pass"),
    (50, "D", "Below Average", "Exact pass boundary"),
    (59, "D", "Below Average", "Just below C"),
    (60, "C", "Average", "Exact C boundary"),
    (69, "C", "Average", "Just below B"),
    (70, "B", "Good", "Exact B boundary"),
    (79, "B", "Good", "Just below A"),
    (80, "A", "Excellent", "Exact A boundary"),
    (89, "A", "Excellent", "Just below A+"),
    (90, "A+", "Outstanding", "Exact A+ boundary"),
    (100, "A+", "Outstanding", "Maximum boundary"),
]

passed = 0
failed = 0

for marks, expected_grade, expected_feedback, description in test_cases:
    grade, feedback = calculate_grade(marks)
    
    if grade == expected_grade and feedback == expected_feedback:
        status = " PASS"
        passed += 1
    else:
        status = "❌ FAIL"
        failed += 1
    
    print(f"\nTest: {description}")
    print(f"  Input: {marks}")
    print(f"  Expected: {expected_grade} - {expected_feedback}")
    print(f"  Got: {grade} - {feedback}")
    print(f"  Status: {status}")

print(f"\n{'='*70}")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print(f"{'='*70}")


''''
#Example 2: Testing Age-Based Ticket Pricing with Unit Tests
import unittest

# Function to test
def calculate_ticket_price(age, is_student=False, is_senior=False):
    """
    Calculate ticket price based on age and status
    Returns: (price, discount_info)
    """
    base_price = 500
    
    if age < 0:
        return None, "Invalid age"
    elif age < 3:
        return 0, "Free for toddlers"
    elif age < 13:
        price = base_price * 0.5
        return price, "50% child discount"
    elif age < 18:
        if is_student:
            price = base_price * 0.6
            return price, "40% student discount"
        else:
            price = base_price * 0.7
            return price, "30% teen discount"
    elif age < 60:
        if is_student:
            price = base_price * 0.8
            return price, "20% student discount"
        else:
            return base_price, "Regular price"
    else:
        if is_senior or age >= 65:
            price = base_price * 0.6
            return price, "40% senior discount"
        else:
            return base_price, "Regular price"


# PROFESSIONAL TESTING APPROACH 2: Unit Testing with unittest
class TestTicketPricing(unittest.TestCase):
    """Comprehensive test suite for ticket pricing logic"""
    
    def test_invalid_age_negative(self):
        """Test negative age returns error"""
        price, info = calculate_ticket_price(-1)
        self.assertIsNone(price)
        self.assertEqual(info, "Invalid age")
    
    def test_toddler_free_entry(self):
        """Test free entry for children under 3"""
        price, info = calculate_ticket_price(2)
        self.assertEqual(price, 0)
        self.assertEqual(info, "Free for toddlers")
    
    def test_child_discount_boundary(self):
        """Test child discount at exact boundary"""
        # Age 12 should get child discount
        price, info = calculate_ticket_price(12)
        self.assertEqual(price, 250)
        self.assertEqual(info, "50% child discount")
        
        # Age 13 should NOT get child discount
        price, info = calculate_ticket_price(13)
        self.assertEqual(price, 350)
        self.assertEqual(info, "30% teen discount")
    
    def test_teen_student_discount(self):
        """Test student discount for teens"""
        price, info = calculate_ticket_price(16, is_student=True)
        self.assertEqual(price, 300)
        self.assertEqual(info, "40% student discount")
    
    def test_teen_regular_price(self):
        """Test regular teen price without student status"""
        price, info = calculate_ticket_price(16, is_student=False)
        self.assertEqual(price, 350)
        self.assertEqual(info, "30% teen discount")
    
    def test_adult_regular_price(self):
        """Test adult regular price"""
        price, info = calculate_ticket_price(30)
        self.assertEqual(price, 500)
        self.assertEqual(info, "Regular price")
    
    def test_adult_student_discount(self):
        """Test adult student discount"""
        price, info = calculate_ticket_price(25, is_student=True)
        self.assertEqual(price, 400)
        self.assertEqual(info, "20% student discount")
    
    def test_senior_discount_at_60(self):
        """Test senior discount starts at age 60"""
        price, info = calculate_ticket_price(60)
        self.assertEqual(price, 300)
        self.assertEqual(info, "40% senior discount")
    
    def test_senior_discount_at_65(self):
        """Test senior discount at 65"""
        price, info = calculate_ticket_price(65)
        self.assertEqual(price, 300)
        self.assertEqual(info, "40% senior discount")
    
    def test_very_old_age(self):
        """Test with very old age"""
        price, info = calculate_ticket_price(100)
        self.assertEqual(price, 300)
        self.assertEqual(info, "40% senior discount")
    
    def test_boundary_adult_to_senior(self):
        """Test boundary between adult and senior"""
        # 59 should be adult price
        price, info = calculate_ticket_price(59)
        self.assertEqual(price, 500)
        
        # 60 should be senior price
        price, info = calculate_ticket_price(60)
        self.assertEqual(price, 300)


# Run the tests
if __name__ == '__main__':
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS FOR TICKET PRICING")
    print("="*70 + "\n")
    
    # Run tests with verbose output
    unittest.main(argv=[''], verbosity=2, exit=False)


#Example 3: Testing Temperature Conversion with Edge Cases
# Function to test
def classify_temperature(celsius):
    """
    Classify temperature and provide recommendations
    Returns: (classification, recommendation)
    """
    if celsius < -273.15:  # Absolute zero
        return "INVALID", "Temperature below absolute zero is impossible"
    elif celsius < -40:
        return "Extreme Cold", "Life-threatening - Do not go outside"
    elif celsius < 0:
        return "Freezing", "Bundle up - Risk of frostbite"
    elif celsius == 0:
        return "Freezing Point", "Water freezes - Dress warmly"
    elif celsius < 15:
        return "Cold", "Wear warm clothes"
    elif celsius < 25:
        return "Comfortable", "Perfect weather"
    elif celsius < 35:
        return "Warm", "Stay hydrated"
    elif celsius < 45:
        return "Hot", "Avoid outdoor activities"
    else:
        return "Extreme Heat", "Dangerous - Stay indoors with AC"


# PROFESSIONAL TESTING APPROACH 3: Comprehensive Test Matrix
print("="*80)
print("TEMPERATURE CLASSIFICATION - COMPREHENSIVE TEST SUITE")
print("="*80)

# Test categories with multiple test cases each
test_matrix = {
    "Invalid Inputs": [
        (-300, "INVALID", "Below absolute zero"),
        (-273.16, "INVALID", "Below absolute zero"),
    ],
    
    "Extreme Cold": [
        (-273.15, "Extreme Cold", "Absolute zero boundary"),
        (-50, "Extreme Cold", "Extreme cold temperature"),
        (-40, "Extreme Cold", "Upper boundary of extreme cold"),
    ],
    
    "Freezing Range": [
        (-39.9, "Freezing", "Just above extreme cold"),
        (-20, "Freezing", "Mid freezing range"),
        (-1, "Freezing", "Just below freezing point"),
        (-0.1, "Freezing", "Very close to freezing point"),
    ],
    
    "Freezing Point": [
        (0, "Freezing Point", "Exact freezing point"),
    ],
    
    "Cold Range": [
        (0.1, "Cold", "Just above freezing"),
        (10, "Cold", "Mid cold range"),
        (14.9, "Cold", "Upper boundary of cold"),
    ],
    
    "Comfortable Range": [
        (15, "Comfortable", "Lower boundary comfortable"),
        (20, "Comfortable", "Perfect temperature"),
        (24.9, "Comfortable", "Upper boundary comfortable"),
    ],
    
    "Warm Range": [
        (25, "Warm", "Lower boundary warm"),
        (30, "Warm", "Mid warm range"),
        (34.9, "Warm", "Upper boundary warm"),
    ],
    
    "Hot Range": [
        (35, "Hot", "Lower boundary hot"),
        (40, "Hot", "Very hot"),
        (44.9, "Hot", "Upper boundary hot"),
    ],
    
    "Extreme Heat": [
        (45, "Extreme Heat", "Lower boundary extreme heat"),
        (50, "Extreme Heat", "Dangerous heat"),
        (60, "Extreme Heat", "Lethal temperature"),
    ],
}

total_tests = 0
total_passed = 0
total_failed = 0

for category, tests in test_matrix.items():
    print(f"\n{'─'*80}")
    print(f"Category: {category}")
    print(f"{'─'*80}")
    
    category_passed = 0
    category_failed = 0
    
    for temp, expected_class, description in tests:
        total_tests += 1
        classification, recommendation = classify_temperature(temp)
        
        if classification == expected_class:
            status = " PASS"
            category_passed += 1
            total_passed += 1
        else:
            status = " FAIL"
            category_failed += 1
            total_failed += 1
        
        print(f"  {description}")
        print(f"    Input: {temp}°C")
        print(f"    Expected: {expected_class}")
        print(f"    Got: {classification}")
        print(f"    {status}")
    
    print(f"\n  Category Results: {category_passed} passed, {category_failed} failed")

# Final summary
print(f"\n{'='*80}")
print(f"OVERALL TEST RESULTS")
print(f"{'='*80}")
print(f"Total Tests: {total_tests}")
print(f"Passed: {total_passed} ({(total_passed/total_tests)*100:.1f}%)")
print(f"Failed: {total_failed} ({(total_failed/total_tests)*100:.1f}%)")
print(f"{'='*80}")

# Coverage report
categories_tested = len(test_matrix)
print(f"\nCode Coverage:")
print(f"  - Tested {categories_tested} condition branches")
print(f"  - Tested boundary values: ")
print(f"  - Tested edge cases: ")
print(f"  - Tested invalid inputs: ")

#Example 4: Testing Login Authentication with Security Cases
# Function to test
def authenticate_user(username, password, captcha_passed=True, account_locked=False, attempts=0):
    """
    Authenticate user with multiple security checks
    Returns: (success, message, action_required)
    """
    # Input validation
    if not username or not password:
        return False, "Username and password are required", "PROMPT_CREDENTIALS"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters", "INVALID_INPUT"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters", "INVALID_INPUT"
    
    # Security checks
    if account_locked:
        return False, "Account is locked. Contact support", "ACCOUNT_LOCKED"
    
    if attempts >= 3 and not captcha_passed:
        return False, "Please complete CAPTCHA", "REQUIRE_CAPTCHA"
    
    # Simulate authentication
    valid_users = {
        "admin": "Admin@123",
        "user1": "User@Pass123",
        "testuser": "Test@Pass456"
    }
    
    if username not in valid_users:
        return False, "Invalid username", "LOGIN_FAILED"
    
    if valid_users[username] != password:
        return False, "Invalid password", "LOGIN_FAILED"
    
    return True, "Login successful", "GRANT_ACCESS"


# PROFESSIONAL TESTING APPROACH 4: Security-Focused Testing
print("="*80)
print("AUTHENTICATION SYSTEM - SECURITY TEST SUITE")
print("="*80)

# Organized test cases by security concern
security_tests = {
    "Input Validation Tests": [
        {
            "description": "Empty username",
            "inputs": {"username": "", "password": "Pass@123"},
            "expected": (False, "Username and password are required", "PROMPT_CREDENTIALS")
        },
        {
            "description": "Empty password",
            "inputs": {"username": "admin", "password": ""},
            "expected": (False, "Username and password are required", "PROMPT_CREDENTIALS")
        },
        {
            "description": "Short username",
            "inputs": {"username": "ab", "password": "Pass@123"},
            "expected": (False, "Username must be at least 3 characters", "INVALID_INPUT")
        },
        {
            "description": "Short password",
            "inputs": {"username": "admin", "password": "Pass@12"},
            "expected": (False, "Password must be at least 8 characters", "INVALID_INPUT")
        },
    ],
    
    "Account Security Tests": [
        {
            "description": "Locked account",
            "inputs": {"username": "admin", "password": "Admin@123", "account_locked": True},
            "expected": (False, "Account is locked. Contact support", "ACCOUNT_LOCKED")
        },
        {
            "description": "Multiple failed attempts without CAPTCHA",
            "inputs": {"username": "admin", "password": "Admin@123", "attempts": 3, "captcha_passed": False},
            "expected": (False, "Please complete CAPTCHA", "REQUIRE_CAPTCHA")
        },
        {
            "description": "Multiple attempts with CAPTCHA passed",
            "inputs": {"username": "admin", "password": "Admin@123", "attempts": 3, "captcha_passed": True},
            "expected": (True, "Login successful", "GRANT_ACCESS")
        },
    ],
    
    "Authentication Tests": [
        {
            "description": "Non-existent user",
            "inputs": {"username": "hacker", "password": "Pass@123"},
            "expected": (False, "Invalid username", "LOGIN_FAILED")
        },
        {
            "description": "Valid user wrong password",
            "inputs": {"username": "admin", "password": "WrongPass@123"},
            "expected": (False, "Invalid password", "LOGIN_FAILED")
        },
        {
            "description": "Valid credentials - admin",
            "inputs": {"username": "admin", "password": "Admin@123"},
            "expected": (True, "Login successful", "GRANT_ACCESS")
        },
        {
            "description": "Valid credentials - user1",
            "inputs": {"username": "user1", "password": "User@Pass123"},
            "expected": (True, "Login successful", "GRANT_ACCESS")
        },
    ],
    
    "Edge Cases": [
        {
            "description": "Username with spaces",
            "inputs": {"username": "ad min", "password": "Admin@123"},
            "expected": (False, "Invalid username", "LOGIN_FAILED")
        },
        {
            "description": "Case-sensitive username",
            "inputs": {"username": "Admin", "password": "Admin@123"},
            "expected": (False, "Invalid username", "LOGIN_FAILED")
        },
        {
            "description": "Minimum valid username length",
            "inputs": {"username": "abc", "password": "Pass@123"},
            "expected": (False, "Invalid username", "LOGIN_FAILED")
        },
    ],
}

total_tests = 0
passed_tests = 0
failed_tests = 0
security_issues = []

for test_category, tests in security_tests.items():
    print(f"\n{'─'*80}")
    print(f" {test_category}")
    print(f"{'─'*80}")
    
    for test_case in tests:
        total_tests += 1
        description = test_case["description"]
        inputs = test_case["inputs"]
        expected = test_case["expected"]
        
        # Run the test
        result = authenticate_user(**inputs)
        
        # Check if test passed
        if result == expected:
            status = " PASS"
            passed_tests += 1
        else:
            status = " FAIL"
            failed_tests += 1
            security_issues.append({
                "category": test_category,
                "test": description,
                "expected": expected,
                "got": result
            })
        
        print(f"\n  Test: {description}")
        print(f"    Inputs: {inputs}")
        print(f"    Expected: {expected}")
        print(f"    Got: {result}")
        print(f"    Status: {status}")

# Security report
print(f"\n{'='*80}")
print(f"SECURITY TEST REPORT")
print(f"{'='*80}")
print(f"Total Tests: {total_tests}")
print(f"Passed: {passed_tests} ({(passed_tests/total_tests)*100:.1f}%)")
print(f"Failed: {failed_tests}")

if security_issues:
    print(f"\n  SECURITY ISSUES FOUND:")
    for issue in security_issues:
        print(f"  - {issue['category']}: {issue['test']}")
else:
    print(f"\n All security tests passed!")

print(f"{'='*80}")
'''