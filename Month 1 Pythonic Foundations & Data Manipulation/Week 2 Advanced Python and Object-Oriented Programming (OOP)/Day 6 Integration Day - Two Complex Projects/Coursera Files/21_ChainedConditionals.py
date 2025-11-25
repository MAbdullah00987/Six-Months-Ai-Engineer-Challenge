

#Chained Conditionals in Python
#Definition of Chained Conditionals
#Chained conditionals are a series of conditional statements linked together using if, elif (else if), and else keywords. They allow checking multiple mutually exclusive conditions in sequence, where only ONE block of code will execute - the first condition that evaluates to True.
#Structure:
#if condition1:
#    # Executes if condition1 is True
#elif condition2:
#    # Executes if condition1 is False and condition2 is True
#elif condition3:
    # Executes if condition1 and condition2 are False and condition3 is True
#else:
    # Executes if all above conditions are False

#Why We Need Chained Conditionals
#1. Mutually Exclusive Choices
#When only one option should be selected from multiple possibilities (e.g., grade categories, day of week).
#2. Cleaner Code
#Avoids multiple independent if statements that would all be checked even after finding a match.
#3. Efficiency
#Stops checking conditions once a match is found, saving processing time.
#4. Better Readability
#Clearly shows that options are alternatives to each other, not independent checks.
#5. Range-Based Classification
#Perfect for categorizing values into different ranges (temperature zones, age groups, salary brackets).
#6. Priority-Based Logic
#Checks conditions in order of priority - first match wins.
#7. Prevents Logic Errors
#Ensures only one path executes, avoiding conflicts from multiple true conditions.

#Comparison: Why NOT Multiple IF Statements?
# Multiple IF Statements (INEFFICIENT):
score = 85

if score >= 90:
    print("Grade A")
if score >= 80:  # This still checks even though 85 < 90
    print("Grade B")
if score >= 70:  # This also checks unnecessarily
    print("Grade C")

score = 85

if score >= 90:
    print("Grade A")
elif score >= 80:  # Only checks if score < 90
    print("Grade B")
elif score >= 70:  # Never reached because score >= 80
    print("Grade C")



# Examples of Chained Conditionals

#Example 1: Comprehensive Grading System
# Advanced student grading with detailed feedback
marks = 78
attendance = 92

# Calculate final score with attendance bonus
if attendance >= 90:
    final_score = marks + 2  # Bonus for good attendance
    print(f"Attendance Bonus: +2 marks")
else:
    final_score = marks

print(f"Final Score: {final_score}")

# Grade assignment with detailed feedback
if final_score >= 90:
    grade = "A+"
    gpa = 4.0
    feedback = "Outstanding performance! Excellence achieved."
    scholarship = "Full scholarship eligible"
elif final_score >= 85:
    grade = "A"
    gpa = 3.7
    feedback = "Excellent work! Keep up the great effort."
    scholarship = "75% scholarship eligible"
elif final_score >= 80:
    grade = "A-"
    gpa = 3.3
    feedback = "Very good performance! You're doing great."
    scholarship = "50% scholarship eligible"
elif final_score >= 75:
    grade = "B+"
    gpa = 3.0
    feedback = "Good job! Little more effort needed."
    scholarship = "25% scholarship eligible"
elif final_score >= 70:
    grade = "B"
    gpa = 2.7
    feedback = "Satisfactory. Room for improvement."
    scholarship = "Not eligible"
elif final_score >= 65:
    grade = "B-"
    gpa = 2.3
    feedback = "Fair performance. Need to work harder."
    scholarship = "Not eligible"
elif final_score >= 60:
    grade = "C"
    gpa = 2.0
    feedback = "Passed but needs significant improvement."
    scholarship = "Not eligible"
elif final_score >= 50:
    grade = "D"
    gpa = 1.0
    feedback = "Barely passed. Serious effort required."
    scholarship = "Not eligible"
else:
    grade = "F"
    gpa = 0.0
    feedback = "Failed. Please retake the course."
    scholarship = "Not eligible"

# Display complete result
print(f"\n{'='*50}")
print(f"GRADE REPORT")
print(f"{'='*50}")
print(f"Grade: {grade}")
print(f"GPA: {gpa}")
print(f"Feedback: {feedback}")
print(f"Scholarship: {scholarship}")
print(f"{'='*50}")



#Example 2: Weather-Based Activity Planner
# Intelligent activity suggestion system
temperature = 28  # Celsius
humidity = 65
wind_speed = 15  # km/h
precipitation = 0  # mm

print(f"Temperature: {temperature}°C")
print(f"Humidity: {humidity}%")
print(f"Wind Speed: {wind_speed} km/h")
print(f"Precipitation: {precipitation}mm\n")

# Weather-based recommendations
if precipitation > 5:
    weather_type = "Heavy Rain"
    activity = "Stay indoors - Read a book or watch movies"
    clothing = "Raincoat and umbrella essential"
    caution = "Avoid outdoor activities"
elif precipitation > 0:
    weather_type = "Light Rain"
    activity = "Indoor activities recommended - Gym or shopping mall"
    clothing = "Light raincoat or umbrella"
    caution = "Be cautious if going out"
elif temperature > 35:
    weather_type = "Extremely Hot"
    activity = "Stay in AC - Swimming pool if available"
    clothing = "Light, breathable cotton clothes"
    caution = " Risk of heat stroke - Stay hydrated!"
elif temperature > 30:
    weather_type = "Very Hot"
    activity = "Indoor sports or early morning/evening walk"
    clothing = "Light colored, loose clothes"
    caution = "Drink plenty of water"
elif temperature > 25:
    weather_type = "Warm & Pleasant"
    activity = "Perfect for outdoor activities - Picnic, sports, cycling"
    clothing = "Comfortable casual wear"
    caution = "Great day to be outside!"
elif temperature > 20:
    weather_type = "Mild"
    activity = "Ideal for jogging, hiking, or sightseeing"
    clothing = "Light jacket recommended"
    caution = "Perfect weather - enjoy!"
elif temperature > 15:
    weather_type = "Cool"
    activity = "Outdoor activities with warm clothes - Walking, running"
    clothing = "Sweater or light jacket"
    caution = "Comfortable for most activities"
elif temperature > 10:
    weather_type = "Cold"
    activity = "Brisk walking or indoor activities"
    clothing = "Warm jacket, scarf recommended"
    caution = "Dress warmly"
elif temperature > 5:
    weather_type = "Very Cold"
    activity = "Limited outdoor time - Indoor activities better"
    clothing = "Heavy winter coat, gloves, hat"
    caution = " Bundle up well!"
else:
    weather_type = "Freezing"
    activity = "Stay indoors - Emergency outdoor only"
    clothing = "Full winter gear essential"
    caution = " Dangerous cold - Avoid prolonged exposure"

# Additional wind consideration
if wind_speed > 30:
    caution += " | 💨 Strong winds - Be extra careful!"

print(f"Weather: {weather_type}")
print(f"Suggested Activity: {activity}")
print(f"Clothing: {clothing}")
print(f"Advisory: {caution}")


#Example 3: Smart Traffic Management System
# Intelligent traffic signal and route recommendation
time_hour = 17  # 5 PM
day_of_week = "Friday"  # Monday-Sunday
weather = "clear"  # clear, rain, fog
accident_reported = False
construction_work = False
special_event = True  # Concert at stadium

print(f"Time: {time_hour}:00")
print(f"Day: {day_of_week}")
print(f"Weather: {weather}")
print(f"Special Event: {'Yes' if special_event else 'No'}\n")

# Traffic condition assessment
if accident_reported:
    traffic_status = "CRITICAL"
    signal_timing = "Extended green (90 seconds)"
    route = "Use alternate Route B via Ring Road"
    estimated_delay = "45-60 minutes"
    alert = " MAJOR ACCIDENT - Avoid main roads!"
elif construction_work:
    traffic_status = "HEAVY"
    signal_timing = "Extended green (75 seconds)"
    route = "Use Route C via Northern Bypass"
    estimated_delay = "30-40 minutes"
    alert = " Construction zone - Expect delays"
elif special_event and (17 <= time_hour <= 22):
    traffic_status = "VERY HEAVY"
    signal_timing = "Smart signals (60-90 seconds)"
    route = "Avoid stadium area - Use Eastern Highway"
    estimated_delay = "25-35 minutes"
    alert = " Event traffic - Stadium area congested"
elif day_of_week in ["Friday", "Saturday"] and (17 <= time_hour <= 20):
    traffic_status = "HEAVY"
    signal_timing = "Extended green (70 seconds)"
    route = "Multiple slow zones - Consider Route A"
    estimated_delay = "20-30 minutes"
    alert = " Weekend rush hour traffic"
elif (7 <= time_hour <= 9) or (17 <= time_hour <= 19):
    traffic_status = "MODERATE TO HEAVY"
    signal_timing = "Normal with adaptive timing (60 seconds)"
    route = "Main route okay but monitor updates"
    estimated_delay = "15-20 minutes"
    alert = " Peak hours - Regular congestion"
elif (12 <= time_hour <= 14):
    traffic_status = "MODERATE"
    signal_timing = "Normal (50 seconds)"
    route = "All routes clear"
    estimated_delay = "10-15 minutes"
    alert = "Lunch hour - Some slowdowns"
elif (22 <= time_hour <= 24) or (0 <= time_hour <= 5):
    traffic_status = "LIGHT"
    signal_timing = "Reduced timing (30 seconds)"
    route = "All routes clear - Fast travel"
    estimated_delay = "5-8 minutes"
    alert = " Night time - Smooth traffic"
else:
    traffic_status = "NORMAL"
    signal_timing = "Standard (45 seconds)"
    route = "All routes operating normally"
    estimated_delay = "8-12 minutes"
    alert = " Normal traffic flow"

# Weather impact
if weather == "rain":
    alert += " | 🌧️ Rainy - Drive carefully!"
    estimated_delay = estimated_delay.replace("minutes", "minutes (+5 for rain)")
elif weather == "fog":
    alert += " | 🌫️ Low visibility - Reduce speed!"
    estimated_delay = estimated_delay.replace("minutes", "minutes (+10 for fog)")

# Display traffic report
print(f"{'='*60}")
print(f"TRAFFIC MANAGEMENT SYSTEM")
print(f"{'='*60}")
print(f"Traffic Status: {traffic_status}")
print(f"Signal Timing: {signal_timing}")
print(f"Recommended Route: {route}")
print(f"Estimated Delay: {estimated_delay}")
print(f"Alert: {alert}")
print(f"{'='*60}")



#Example 4: BMI Health Assessment System
# Comprehensive BMI calculator with health recommendations
weight = 75  # kg
height = 1.75  # meters
age = 28
gender = "male"  # male or female
activity_level = "moderate"  # sedentary, light, moderate, active, very_active

# Calculate BMI
bmi = weight / (height ** 2)
print(f"Weight: {weight} kg")
print(f"Height: {height} m")
print(f"Age: {age} years")
print(f"Gender: {gender.capitalize()}")
print(f"Calculated BMI: {bmi:.2f}\n")

# BMI classification with detailed recommendations
if bmi < 16:
    category = "Severe Underweight"
    health_risk = "VERY HIGH RISK"
    recommendation = " URGENT: Consult a doctor immediately"
    diet = "High calorie, nutrient-dense meals - 6 small meals/day"
    exercise = "Light walking only - Avoid strenuous activity"
    target_weight = f"Gain {(16 * height**2) - weight:.1f} kg minimum"
    emoji = " "
elif bmi < 18.5:
    category = "Underweight"
    health_risk = "MODERATE RISK"
    recommendation = "Consult nutritionist for healthy weight gain"
    diet = "Increase calorie intake - Protein-rich foods, nuts, healthy fats"
    exercise = "Light strength training with proper nutrition"
    target_weight = f"Gain {(18.5 * height**2) - weight:.1f} kg"
    emoji = " "
elif bmi < 25:
    category = "Normal Weight"
    health_risk = "LOW RISK"
    recommendation = " Maintain current lifestyle"
    diet = "Balanced diet - Fruits, vegetables, lean protein, whole grains"
    exercise = "Regular exercise 150 min/week - Mix cardio and strength"
    target_weight = "Maintain current weight"
    emoji = " "
elif bmi < 30:
    category = "Overweight"
    health_risk = "INCREASED RISK"
    recommendation = "Weight loss recommended for better health"
    diet = "Reduce calories by 500/day - Low carb, high protein"
    exercise = "Cardio 30 min/day + strength training 3x/week"
    target_weight = f"Lose {weight - (25 * height**2):.1f} kg"
    emoji = "⚠️"
elif bmi < 35:
    category = "Obese Class I"
    health_risk = "HIGH RISK"
    recommendation = "Medical supervision recommended for weight loss"
    diet = "Structured meal plan - Consult dietitian"
    exercise = "Start with low-impact: Swimming, cycling - Build gradually"
    target_weight = f"Lose {weight - (25 * height**2):.1f} kg"
    emoji = "🚨"
elif bmi < 40:
    category = "Obese Class II"
    health_risk = "VERY HIGH RISK"
    recommendation = "⚠️ Seek medical help - Possible health complications"
    diet = "Medical nutrition therapy required"
    exercise = "Doctor-approved program only"
    target_weight = f"Lose {weight - (25 * height**2):.1f} kg"
    emoji = "🚨"
else:
    category = "Obese Class III (Morbid Obesity)"
    health_risk = "EXTREMELY HIGH RISK"
    recommendation = "🚨 URGENT: Immediate medical intervention required"
    diet = "Medical supervision mandatory - Possible bariatric surgery"
    exercise = "Physical therapy under medical supervision"
    target_weight = f"Lose {weight - (25 * height**2):.1f} kg"
    emoji = "🚨"

# Calculate ideal weight range
ideal_min = 18.5 * (height ** 2)
ideal_max = 25 * (height ** 2)

# Display comprehensive report
print(f"{'='*70}")
print(f"{emoji} BMI HEALTH ASSESSMENT REPORT {emoji}")
print(f"{'='*70}")
print(f"BMI: {bmi:.2f}")
print(f"Category: {category}")
print(f"Health Risk: {health_risk}")
print(f"\nRecommendation: {recommendation}")
print(f"\n DIET PLAN:")
print(f"   {diet}")
print(f"\nEXERCISE PLAN:")
print(f"   {exercise}")
print(f"\n TARGET:")
print(f"   {target_weight}")
print(f"   Ideal weight range: {ideal_min:.1f} - {ideal_max:.1f} kg")
print(f"{'='*70}")



#Example 5: Customer Service Priority System
# Advanced customer support ticket prioritization
issue_type = "payment_failed"
customer_tier = "premium"  # free, basic, premium, enterprise
account_age_days = 450
ticket_count_30days = 2
response_time_hours = 0  # Time since ticket created
business_hours = True
affected_users = 1

print(f" TICKET ANALYSIS")
print(f"Issue: {issue_type}")
print(f"Customer Tier: {customer_tier.upper()}")
print(f"Account Age: {account_age_days} days")
print(f"Recent Tickets: {ticket_count_30days}")
print(f"Business Hours: {'Yes' if business_hours else 'No'}\n")

# Priority classification with detailed routing
if issue_type == "security_breach" or issue_type == "data_loss":
    priority = "P0 - CRITICAL"
    sla = "15 minutes"
    team = "Security Team + Senior Engineer"
    escalation = "Immediate C-Level notification"
    compensation = "Full investigation + Credits"
    emoji = " "
    color = "RED"
elif affected_users > 100 or issue_type == "service_outage":
    priority = "P1 - URGENT"
    sla = "30 minutes"
    team = "Incident Response Team"
    escalation = "Management notification"
    compensation = "Service credits based on downtime"
    emoji = " "
    color = "ORANGE"
elif customer_tier == "enterprise" and issue_type in ["payment_failed", "integration_broken"]:
    priority = "P1 - URGENT"
    sla = "1 hour"
    team = "Enterprise Support + Technical Account Manager"
    escalation = "TAM immediate engagement"
    compensation = "Priority resolution + Consultation"
    emoji = " "
    color = "ORANGE"
elif customer_tier == "premium" and issue_type in ["payment_failed", "feature_not_working"]:
    priority = "P2 - HIGH"
    sla = "2 hours"
    team = "Premium Support Team"
    escalation = "Team lead review after 1 hour"
    compensation = "Possible credit based on impact"
    emoji = " "
    color = "YELLOW"
elif issue_type == "billing_dispute" or issue_type == "payment_failed":
    priority = "P2 - HIGH"
    sla = "4 hours"
    team = "Billing Support Specialist"
    escalation = "Supervisor review after 2 hours"
    compensation = "Case-by-case evaluation"
    emoji = " "
    color = "YELLOW"
elif ticket_count_30days >= 3 and account_age_days > 365:
    priority = "P3 - MEDIUM (Loyalty Escalation)"
    sla = "8 hours"
    team = "Senior Support Agent"
    escalation = "Customer success team notified"
    compensation = "Goodwill gesture considered"
    emoji = " "
    color = "BLUE"
elif customer_tier in ["premium", "enterprise"]:
    priority = "P3 - MEDIUM"
    sla = "12 hours"
    team = "Tier 2 Support"
    escalation = "Standard queue monitoring"
    compensation = "Standard resolution"
    emoji = " "
    color = "BLUE"
elif issue_type in ["feature_request", "how_to_question"]:
    priority = "P4 - LOW"
    sla = "24 hours"
    team = "Tier 1 Support / Knowledge Base"
    escalation = "Self-service suggested first"
    compensation = "Documentation provided"
    emoji = " "
    color = "GREEN"
else:
    priority = "P3 - MEDIUM"
    sla = "12 hours"
    team = "General Support Queue"
    escalation = "Standard workflow"
    compensation = "Standard resolution"
    emoji = " "
    color = "BLUE"

# Business hours adjustment
if not business_hours and priority in ["P0 - CRITICAL", "P1 - URGENT"]:
    team += " (ON-CALL ACTIVATED)"
    emoji += " "

# Display ticket routing information
print(f"{'='*70}")
print(f"{emoji} TICKET PRIORITIZATION SYSTEM")
print(f"{'='*70}")
print(f"Priority Level: {priority}")
print(f"Alert Color: {color}")
print(f"SLA Response Time: {sla}")
print(f"Assigned To: {team}")
print(f"Escalation Path: {escalation}")
print(f"Compensation Policy: {compensation}")
print(f"{'='*70}")
print(f"\nNEXT ACTIONS:")

if priority.startswith("P0") or priority.startswith("P1"):
    print(f"   1. Acknowledge ticket immediately")
    print(f"   2. Start investigation within SLA")
    print(f"   3. Provide hourly updates")
    print(f"   4. Document all actions")
elif priority.startswith("P2"):
    print(f"   1. Acknowledge within 30 minutes")
    print(f"   2. Investigate and respond per SLA")
    print(f"   3. Update customer every 2 hours")
else:
    print(f"   1. Acknowledge within 2 hours")
    print(f"   2. Investigate and provide solution")
    print(f"   3. Follow standard procedure")

print(f"{'='*70}")
