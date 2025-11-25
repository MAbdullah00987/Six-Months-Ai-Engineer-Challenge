
#ested Conditionals in Python
#Definition of Nested Conditionals
#Nested conditionals are conditional statements (if, elif, else) placed inside other conditional statements. This creates multiple levels of decision-making, where an inner condition is only checked if the outer condition is True.
#Structure:
#if outer_condition:
#    if inner_condition:
#        # Code executes if both conditions are True
#    else:
        # Code executes if outer is True but inner is False
#else:
    # Code executes if outer condition is False

#Why We Need Nested Conditionals
#1. Multi-Level Decision Making
#Real-world problems often require checking multiple conditions in a hierarchical manner.
#2. Complex Logic Implementation
#When decisions depend on multiple factors that must be evaluated in a specific order.
#3. Dependent Conditions
#When the second condition only makes sense if the first condition is True.
#4. Validation and Security
#Layer-by-layer verification processes (e.g., username → password → 2FA).
#5. Better Code Organization
#Groups related conditions together, making code more readable and maintainable.
#6. Efficiency
#Avoids unnecessary checks by only evaluating inner conditions when outer conditions are met.
#7. Real-World Scenarios
#Many situations naturally have hierarchical decision structures (age → license → insurance).

#Examples of Nested Conditionals
#Example 1: Bank Loan Approval System
# Multi-factor loan approval
age = 30
income = 50000
credit_score = 720
has_collateral = True

if age >= 21:
    print("Age requirement met ✓")
    
    if income >= 30000:
        print("Income requirement met ✓")
        
        if credit_score >= 700:
            print("Credit score requirement met ✓")
            
            if has_collateral:
                print("\n🎉 LOAN APPROVED!")
                print("You qualify for premium rates")
            else:
                print("\n✓ LOAN APPROVED!")
                print("Standard rates apply")
        else:
            print(f"Credit score too low: {credit_score}")
            print("Minimum required: 700")
    else:
        print(f"Income too low: {income}")
        print("Minimum required: 30000")
else:
    print(f"Age requirement not met: {age}")
    print("Minimum age: 21")



#Example 2: Student Grade and Scholarship System
# Comprehensive student evaluation
marks = 88
attendance = 95
extracurricular = True
financial_need = True

if marks >= 60:
    print(f"Passed with {marks} marks")
    
    if marks >= 85:
        print("Excellence Award!")
        
        if attendance >= 90:
            print("Perfect Attendance Bonus")
            
            if extracurricular:
                if financial_need:
                    print("\nFULL SCHOLARSHIP AWARDED!")
                    scholarship = 100
                else:
                    print("\n 75% SCHOLARSHIP AWARDED!")
                    scholarship = 75
            else:
                print("\n50% Scholarship (Join activities for more)")
                scholarship = 50
        else:
            print(f"Attendance below 90%: {attendance}%")
            print("No scholarship due to attendance")
            scholarship = 0
    else:
        print("Good performance, keep improving!")
        scholarship = 0
else:
    print("Failed. Need to retake the course.")
    scholarship = 0

print(f"\nFinal Scholarship: {scholarship}%")


#Example 3: E-Commerce Order Processing
# Complete order validation system
is_logged_in = True
username = "john_doe"
cart_items = 3
payment_method = "credit_card"
card_valid = True
shipping_address = True
stock_available = True

if is_logged_in:
    print(f"Welcome back, {username}!")
    
    if cart_items > 0:
        print(f"You have {cart_items} items in cart")
        
        if stock_available:
            print("All items are in stock ✓")
            
            if shipping_address:
                print("Shipping address confirmed ✓")
                
                if payment_method == "credit_card":
                    if card_valid:
                        print("Payment method verified ✓")
                        print("\n ORDER PLACED SUCCESSFULLY!")
                        print("Estimated delivery: 3-5 days")
                    else:
                        print(" Card validation failed")
                        print("Please update your card details")
                elif payment_method == "cash_on_delivery":
                    print("COD selected ✓")
                    print("\n ORDER PLACED SUCCESSFULLY!")
                    print("Pay when you receive")
                else:
                    print("Invalid payment method")
            else:
                print("Please add shipping address")
        else:
            print("Some items are out of stock")
            print("Please remove them from cart")
    else:
        print("Your cart is empty")
        print("Add items to proceed")
else:
    print("Please login to place order")



#Example 4: Airport Security Clearance System
# Multi-layer security check
has_ticket = True
ticket_valid = True
has_id = True
id_verified = True
luggage_scanned = True
dangerous_items = False
boarding_time = 45  # minutes before flight

if has_ticket:
    if ticket_valid:
        print("✓ Ticket validated")
        
        if has_id:
            if id_verified:
                print("✓ Identity confirmed")
                
                if luggage_scanned:
                    if not dangerous_items:
                        print("✓ Security check passed")
                        
                        if boarding_time <= 60:
                            if boarding_time >= 30:
                                print("Boarding time appropriate")
                                print("\nCLEARED FOR BOARDING!")
                                print("Proceed to Gate 12")
                            else:
                                print("Boarding has started!")
                                print("Please hurry to Gate 12")
                        else:
                            print("Too early for boarding")
                            print(f"Please wait {boarding_time - 60} minutes")
                    else:
                        print("SECURITY ALERT!")
                        print("Dangerous items detected")
                        print("Please report to security desk")
                else:
                    print("Luggage not scanned")
                    print("Please go to scanning area")
            else:
                print("ID verification failed")
                print("Contact airline representative")
        else:
            print("ID required for boarding")
    else:
        print("Ticket expired or invalid")
else:
    print("No ticket found")



#Example 5: Smart Home Climate Control System
# Intelligent temperature and environment control
is_home = True
current_temp = 32  # Celsius
humidity = 70
air_quality = "good"
energy_saving_mode = False
time_of_day = 14  # 24-hour format

if is_home:
    print("👤 Presence detected - Activating systems")
    
    if current_temp > 28:
        print(f"High temperature: {current_temp}°C")
        
        if not energy_saving_mode:
            print("Turning AC ON")
            
            if humidity > 60:
                print("High humidity detected")
                
                if air_quality == "good":
                    print("Air quality is good")
                    print("Setting AC to dehumidify mode")
                    ac_mode = "dehumidify"
                else:
                    print("Poor air quality")
                    print("Activating air purifier")
                    print("Setting AC to cooling + purify")
                    ac_mode = "cooling_purify"
            else:
                print("Setting AC to normal cooling")
                ac_mode = "cooling"
                
            # Set temperature based on time
            if 6 <= time_of_day < 12:
                target_temp = 24
                print(f"Morning mode: Target {target_temp}°C")
            elif 12 <= time_of_day < 18:
                target_temp = 22
                print(f"Afternoon mode: Target {target_temp}°C")
            else:
                target_temp = 25
                print(f" Evening mode: Target {target_temp}°C")
        else:
            print("Energy saving mode ON")
            print("Setting AC to eco mode")
            target_temp = 26
            ac_mode = "eco"
    elif current_temp < 18:
        print(f"Low temperature: {current_temp}°C")
        print("Activating heater")
    else:
        print(f"Comfortable temperature: {current_temp}°C")
        print("Climate control on standby")
else:
    print("No one home")
    print("All systems in power-saving mode")
