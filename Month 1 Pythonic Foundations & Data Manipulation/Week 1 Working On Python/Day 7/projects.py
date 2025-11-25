#Week One Projects:
#1: Simple Calculator: A command-line program that performs basic arithmetic.
#2: Number Guessing Game: A game where the computer picks a number and the user guesses it.
#3: Text-Based Adventure Game: A simple story with user choices that affect the outcome.
#4: Contact Book: A program to store and retrieve names and phone numbers in a dictionary.
#5: Password Generator: A script that creates random, strong passwords.
#6: Unit Converter: A tool to convert units (e.g., Celsius to Fahrenheit, kilometers to miles).
#7: FizzBuzz: Implement the classic FizzBuzz programming challenge.
#8: File Organizer: A script that sorts files in a directory into subfolders based on file type.
#9: Simple To-Do List: A command-line application to add, view, and delete tasks.
#10: Word Counter: A program that reads a text file and counts the frequency of each word.

#Solutions:

#1: Simple Calculator: A command-line program that performs basic arithmetic.
#Solution:
'''
print("--------------------Simple Calculator----------------------")

#Step one : Get User input
num1 = float(input("Enter first numbeer: "))
operator = input("Enter operator (+,-,*,/): ")
num2 = float(input("Enter second number: "))

#Step two : Perform Calculation based on operator

if operator == '+':
    result = num1 + num2
elif operator == '-':
    result = num1 - num2
elif operator == '*':
    result = num1 * num2
elif operator == '/':

  if num2 == 0:
      result = "Error! Cannot divide by zero."
  else:
      result = num1 / num2
else:
    result = "Invalid Operator!"
    
#Step three : Output the result
print("The result is: ", result)

#Step four: If User wants to perform more calculations

again = input("Do you want to perform another calculation? (yes/no): ").lower()
if again != 'yes':
    print("Calculator Ended. Goodbye!")
'''

#2: Number Guessing Game: A game where the computer picks a number and the user guesses it.
'''
import random
print("------------------Number Guessing Game---------------------")

import random

print("=== Number Guessing Game ===")

secret_number = random.randint(1, 50)
attempt_limit = 7   # user can try 7 times
attempts = 0

while attempts < attempt_limit:
    guess = int(input("Guess a number between 1 and 50: "))
    attempts += 1

    if guess < secret_number:
        print("Too low! Try again.\n")
    elif guess > secret_number:
        print("Too high! Try again.\n")
    else:
        print(f"Correct! The number was {secret_number}.")
        print(f"You guessed it in {attempts} attempts.\n")
        break

# If user runs out of attempts
if attempts == attempt_limit and guess != secret_number:
    print(" You have used all your attempts!")
    print(f"The correct number was: {secret_number}")
    print("Better luck next time!")

    '''
    

#3: Text-Based Adventure Game: A simple story with user choices that affect the outcome.

"""
Text-Based Adventure Game: The Mysterious Island
A story-driven game where player choices affect the outcome
"""
''''
import time
import sys

def print_slow(text, delay=0.03):
    """Print text with a typing effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def get_choice(options):
    """Get valid input from player"""
    while True:
        choice = input("\nYour choice: ").strip().lower()
        if choice in options:
            return choice
        print("Invalid choice. Please try again.")

def intro():
    """Game introduction"""
    print("\n" + "="*50)
    print("🏝️  THE MYSTERIOUS ISLAND  🏝️")
    print("="*50 + "\n")
    
    print_slow("You wake up on a sandy beach, waves crashing nearby.")
    time.sleep(1)
    print_slow("The last thing you remember is the ship sinking...")
    time.sleep(1)
    print_slow("You must find a way to survive and escape this island!\n")
    time.sleep(1)

def beach_scene():
    """First scene at the beach"""
    print("\n--- THE BEACH ---")
    print_slow("You stand up and look around.")
    print_slow("To your LEFT, you see a dense JUNGLE.")
    print_slow("To your RIGHT, you see CAVES along the rocky cliffs.")
    print_slow("Behind you is the endless OCEAN.\n")
    
    print("What do you want to do?")
    print("1. Explore the JUNGLE")
    print("2. Investigate the CAVES")
    print("3. Search the BEACH for supplies")
    
    choice = get_choice(['1', '2', '3'])
    
    if choice == '1':
        jungle_path()
    elif choice == '2':
        cave_path()
    else:
        beach_search()

def beach_search():
    """Search the beach"""
    print("\n--- SEARCHING THE BEACH ---")
    print_slow("You walk along the beach, searching for anything useful.")
    time.sleep(1)
    print_slow("You find a waterproof BACKPACK washed ashore!")
    time.sleep(1)
    print_slow("Inside you find: a KNIFE, a ROPE, and a LIGHTER.")
    time.sleep(1)
    print_slow("These will be useful!\n")
    time.sleep(1)
    
    print("Now what?")
    print("1. Head to the JUNGLE")
    print("2. Check out the CAVES")
    
    choice = get_choice(['1', '2'])
    
    if choice == '1':
        jungle_with_supplies()
    else:
        cave_with_supplies()

def jungle_path():
    """Jungle exploration without supplies"""
    print("\n--- THE JUNGLE ---")
    print_slow("You enter the thick jungle. It's dark and humid.")
    time.sleep(1)
    print_slow("Suddenly, you hear a GROWL behind you!")
    time.sleep(1)
    print_slow("A wild PANTHER appears, blocking your path!\n")
    
    print("What do you do?")
    print("1. Try to CLIMB a tree")
    print("2. RUN deeper into the jungle")
    print("3. Stand STILL and don't move")
    
    choice = get_choice(['1', '2', '3'])
    
    if choice == '1':
        print_slow("\nYou quickly climb the nearest tree.")
        print_slow("The panther circles below but eventually leaves.")
        print_slow("You wait until it's safe, then climb down.")
        jungle_village()
    elif choice == '2':
        print_slow("\nYou run as fast as you can!")
        print_slow("The panther chases you through the jungle.")
        print_slow("You trip on a root and fall...")
        game_over("The panther caught you. GAME OVER!")
    else:
        print_slow("\nYou freeze completely.")
        print_slow("The panther sniffs the air, watching you.")
        print_slow("After a tense moment, it loses interest and walks away.")
        jungle_village()

def jungle_with_supplies():
    """Jungle exploration with supplies"""
    print("\n--- THE JUNGLE (WITH SUPPLIES) ---")
    print_slow("You enter the jungle with your supplies.")
    time.sleep(1)
    print_slow("The knife makes cutting through vines easier.")
    time.sleep(1)
    print_slow("After some time, you hear voices ahead!")
    jungle_village()

def jungle_village():
    """Discovery of native village"""
    print("\n--- MYSTERIOUS VILLAGE ---")
    print_slow("You discover a small village in a clearing!")
    time.sleep(1)
    print_slow("Friendly natives greet you with smiles.")
    time.sleep(1)
    print_slow("They offer you food and tell you about a BOAT on the other side of the island.\n")
    
    print("What do you do?")
    print("1. Ask them to GUIDE you to the boat")
    print("2. THANK them and go find the boat yourself")
    
    choice = get_choice(['1', '2'])
    
    if choice == '1':
        print_slow("\nThe natives guide you safely through the island.")
        print_slow("You reach the boat and they help you repair it.")
        victory("With their help, you escape the island safely!")
    else:
        print_slow("\nYou thank them and head out alone.")
        print_slow("Without their guidance, you get lost in the jungle.")
        print_slow("Days pass, and you run out of food and water...")
        game_over("You couldn't find the boat. GAME OVER!")

def cave_path():
    """Cave exploration without supplies"""
    print("\n--- THE CAVES ---")
    print_slow("You enter the dark caves. It's cold and damp.")
    time.sleep(1)
    print_slow("You can barely see anything in the darkness.")
    time.sleep(1)
    print_slow("You hear water dripping somewhere ahead.\n")
    
    print("What do you do?")
    print("1. Continue DEEPER into the cave")
    print("2. Go BACK to the beach")
    
    choice = get_choice(['1', '2'])
    
    if choice == '1':
        print_slow("\nYou stumble forward in the darkness.")
        print_slow("Suddenly, the ground disappears beneath you!")
        print_slow("You fall into a deep pit...")
        game_over("You fell into a trap. GAME OVER!")
    else:
        print_slow("\nYou decide it's too dangerous without light.")
        beach_scene()

def cave_with_supplies():
    """Cave exploration with supplies"""
    print("\n--- THE CAVES (WITH SUPPLIES) ---")
    print_slow("You enter the caves with your lighter for light.")
    time.sleep(1)
    print_slow("The flickering flame reveals ancient paintings on the walls.")
    time.sleep(1)
    print_slow("You follow the tunnel and find a SECRET EXIT!")
    time.sleep(1)
    print_slow("It leads to the other side of the island.\n")
    time.sleep(1)
    print_slow("There, you discover an old but working SAILBOAT!")
    time.sleep(1)
    
    victory("You sail away from the island to safety!")

def game_over(message):
    """Bad ending"""
    print("\n" + "="*50)
    print("  " + message + "  ")
    print("="*50)
    print("\nTry again? Make different choices!\n")
    
    play_again()

def victory(message):
    """Good ending"""
    print("\n" + "="*50)
    print("  " + message + "  ")
    print("="*50)
    print("\n CONGRATULATIONS! YOU SURVIVED! \n")
    
    play_again()

def play_again():
    """Ask if player wants to play again"""
    print("Would you like to play again? (yes/no)")
    choice = get_choice(['yes', 'y', 'no', 'n'])
    
    if choice in ['yes', 'y']:
        main()
    else:
        print("\nThanks for playing! \n")

def main():
    """Main game function"""
    intro()
    beach_scene()

# Start the game
if __name__ == "__main__":
    main()

'''

#4: Contact Book: A program to store and retrieve names and phone numbers in a dictionary.
"""
Contact Book Program
A program to store and retrieve names and phone numbers
Uses dictionary data structure for efficient storage
"""
'''

# Global dictionary to store contacts
contacts = {}

def display_menu():
    """Display the main menu"""
    print("\n" + "="*50)
    print("  CONTACT BOOK MANAGER  ")
    print("="*50)
    print("1. Add New Contact")
    print("2. View All Contacts")
    print("3. Search Contact")
    print("4. Update Contact")
    print("5. Delete Contact")
    print("6. Exit")
    print("="*50)

def add_contact():
    """Add a new contact to the contact book"""
    print("\n--- ADD NEW CONTACT ---")
    
    # Get contact name
    name = input("Enter contact name: ").strip().title()
    
    if not name:
        print(" Name cannot be empty!")
        return
    
    # Check if contact already exists
    if name in contacts:
        print(f"  Contact '{name}' already exists!")
        choice = input("Do you want to update it? (yes/no): ").lower()
        if choice in ['yes', 'y']:
            update_contact(name)
        return
    
    # Get phone number
    phone = input("Enter phone number: ").strip()
    
    if not phone:
        print(" Phone number cannot be empty!")
        return
    
    # Optional: Add email
    email = input("Enter email (optional, press Enter to skip): ").strip()
    
    # Store contact with additional info
    contacts[name] = {
        'phone': phone,
        'email': email if email else 'N/A'
    }
    
    print(f" Contact '{name}' added successfully!")

def view_all_contacts():
    """Display all contacts in the contact book"""
    print("\n--- ALL CONTACTS ---")
    
    if not contacts:
        print(" No contacts found! Your contact book is empty.")
        return
    
    print(f"\nTotal Contacts: {len(contacts)}\n")
    print(f"{'Name':<20} {'Phone':<15} {'Email':<30}")
    print("-" * 65)
    
    # Sort contacts alphabetically by name
    for name in sorted(contacts.keys()):
        phone = contacts[name]['phone']
        email = contacts[name]['email']
        print(f"{name:<20} {phone:<15} {email:<30}")

def search_contact():
    """Search for a contact by name"""
    print("\n--- SEARCH CONTACT ---")
    
    if not contacts:
        print(" No contacts to search!")
        return
    
    search_name = input("Enter name to search: ").strip().title()
    
    if not search_name:
        print(" Search term cannot be empty!")
        return
    
    # Search for exact match
    if search_name in contacts:
        print(f"\n Contact Found!")
        print(f"Name:  {search_name}")
        print(f"Phone: {contacts[search_name]['phone']}")
        print(f"Email: {contacts[search_name]['email']}")
    else:
        # Search for partial matches
        matches = [name for name in contacts.keys() if search_name.lower() in name.lower()]
        
        if matches:
            print(f"\n🔍 Found {len(matches)} similar contact(s):")
            for name in matches:
                print(f"\nName:  {name}")
                print(f"Phone: {contacts[name]['phone']}")
                print(f"Email: {contacts[name]['email']}")
        else:
            print(f" No contact found with name '{search_name}'")

def update_contact(name=None):
    """Update an existing contact"""
    print("\n--- UPDATE CONTACT ---")
    
    if not contacts:
        print("📭 No contacts to update!")
        return
    
    # If name not provided, ask for it
    if not name:
        name = input("Enter contact name to update: ").strip().title()
    
    if not name:
        print(" Name cannot be empty!")
        return
    
    if name not in contacts:
        print(f" Contact '{name}' not found!")
        return
    
    # Display current information
    print(f"\nCurrent Information for '{name}':")
    print(f"Phone: {contacts[name]['phone']}")
    print(f"Email: {contacts[name]['email']}")
    
    print("\nWhat do you want to update?")
    print("1. Phone Number")
    print("2. Email")
    print("3. Both")
    print("4. Cancel")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        new_phone = input("Enter new phone number: ").strip()
        if new_phone:
            contacts[name]['phone'] = new_phone
            print(" Phone number updated!")
        else:
            print(" Phone number cannot be empty!")
    
    elif choice == '2':
        new_email = input("Enter new email: ").strip()
        contacts[name]['email'] = new_email if new_email else 'N/A'
        print(" Email updated!")
    
    elif choice == '3':
        new_phone = input("Enter new phone number: ").strip()
        new_email = input("Enter new email: ").strip()
        
        if new_phone:
            contacts[name]['phone'] = new_phone
            contacts[name]['email'] = new_email if new_email else 'N/A'
            print(" Contact updated!")
        else:
            print(" Phone number cannot be empty!")
    
    elif choice == '4':
        print("Update cancelled.")
    
    else:
        print(" Invalid choice!")

def delete_contact():
    """Delete a contact from the contact book"""
    print("\n--- DELETE CONTACT ---")
    
    if not contacts:
        print("📭 No contacts to delete!")
        return
    
    name = input("Enter contact name to delete: ").strip().title()
    
    if not name:
        print(" Name cannot be empty!")
        return
    
    if name not in contacts:
        print(f" Contact '{name}' not found!")
        return
    
    # Confirm deletion
    print(f"\nAre you sure you want to delete '{name}'?")
    print(f"Phone: {contacts[name]['phone']}")
    print(f"Email: {contacts[name]['email']}")
    
    confirm = input("\nType 'yes' to confirm deletion: ").lower()
    
    if confirm in ['yes', 'y']:
        del contacts[name]
        print(f" Contact '{name}' deleted successfully!")
    else:
        print("Deletion cancelled.")

def get_valid_choice():
    """Get valid menu choice from user"""
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print(" Invalid choice! Please enter a number between 1 and 6.")

def main():
    """Main program loop"""
    print("\n Welcome to Contact Book Manager! ")
    
    # Add some sample contacts for demonstration
    # Remove these lines if you want to start with an empty contact book
    contacts['John Doe'] = {'phone': '555-1234', 'email': 'john@example.com'}
    contacts['Jane Smith'] = {'phone': '555-5678', 'email': 'jane@example.com'}
    contacts['Bob Wilson'] = {'phone': '555-9012', 'email': 'bob@example.com'}
    
    while True:
        display_menu()
        choice = get_valid_choice()
        
        if choice == '1':
            add_contact()
        
        elif choice == '2':
            view_all_contacts()
        
        elif choice == '3':
            search_contact()
        
        elif choice == '4':
            update_contact()
        
        elif choice == '5':
            delete_contact()
        
        elif choice == '6':
            print("\n Thank you for using Contact Book Manager!")
            print("Goodbye! \n")
            break
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")

# Start the program
if __name__ == "__main__":
    main()
    '''
#5: Password Generator: A script that creates random, strong passwords.

"""
Password Generator
A script that creates random, strong passwords
Includes multiple options and password strength checker
"""

import random
import string

def display_menu():
    """Display the main menu"""
    print("\n" + "="*50)
    print("  PASSWORD GENERATOR  ")
    print("="*50)
    print("1. Generate Simple Password")
    print("2. Generate Strong Password")
    print("3. Generate Custom Password")
    print("4. Generate Multiple Passwords")
    print("5. Check Password Strength")
    print("6. Exit")
    print("="*50)

def generate_simple_password(length=8):
    """
    Generate a simple password with letters and numbers
    
    Args:
        length (int): Length of the password (default: 8)
    
    Returns:
        str: Generated password
    """
    # Characters to use: lowercase, uppercase, and digits
    characters = string.ascii_letters + string.digits
    
    # Generate random password
    password = ''.join(random.choice(characters) for _ in range(length))
    
    return password

def generate_strong_password(length=12):
    """
    Generate a strong password with letters, numbers, and symbols
    Ensures at least one of each character type
    
    Args:
        length (int): Length of the password (default: 12)
    
    Returns:
        str: Generated strong password
    """
    if length < 4:
        length = 4  # Minimum length to include all character types
    
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    symbols = string.punctuation
    
    # Ensure at least one character from each set
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(symbols)
    ]
    
    # Fill the rest with random characters from all sets
    all_characters = lowercase + uppercase + digits + symbols
    password += [random.choice(all_characters) for _ in range(length - 4)]
    
    # Shuffle to make it random (not predictable pattern)
    random.shuffle(password)
    
    return ''.join(password)

def generate_custom_password():
    """
    Generate a custom password based on user preferences
    
    Returns:
        str: Generated custom password
    """
    print("\n--- CUSTOM PASSWORD GENERATOR ---")
    
    # Get length
    while True:
        try:
            length = int(input("Enter password length (minimum 4): "))
            if length < 4:
                print(" Length must be at least 4!")
                continue
            break
        except ValueError:
            print(" Please enter a valid number!")
    
    # Ask for character types
    include_lowercase = input("Include lowercase letters? (yes/no): ").lower() in ['yes', 'y']
    include_uppercase = input("Include uppercase letters? (yes/no): ").lower() in ['yes', 'y']
    include_digits = input("Include numbers? (yes/no): ").lower() in ['yes', 'y']
    include_symbols = input("Include symbols? (yes/no): ").lower() in ['yes', 'y']
    
    # Check if at least one type is selected
    if not any([include_lowercase, include_uppercase, include_digits, include_symbols]):
        print(" You must select at least one character type!")
        return None
    
    # Build character set
    characters = ''
    guaranteed = []
    
    if include_lowercase:
        characters += string.ascii_lowercase
        guaranteed.append(random.choice(string.ascii_lowercase))
    
    if include_uppercase:
        characters += string.ascii_uppercase
        guaranteed.append(random.choice(string.ascii_uppercase))
    
    if include_digits:
        characters += string.digits
        guaranteed.append(random.choice(string.digits))
    
    if include_symbols:
        characters += string.punctuation
        guaranteed.append(random.choice(string.punctuation))
    
    # Generate password
    password = guaranteed + [random.choice(characters) for _ in range(length - len(guaranteed))]
    random.shuffle(password)
    
    return ''.join(password)

def generate_multiple_passwords():
    """Generate multiple passwords at once"""
    print("\n--- GENERATE MULTIPLE PASSWORDS ---")
    
    # Get number of passwords
    while True:
        try:
            count = int(input("How many passwords do you want? (1-20): "))
            if 1 <= count <= 20:
                break
            print(" Please enter a number between 1 and 20!")
        except ValueError:
            print(" Please enter a valid number!")
    
    # Get length
    while True:
        try:
            length = int(input("Enter password length (minimum 4): "))
            if length >= 4:
                break
            print(" Length must be at least 4!")
        except ValueError:
            print(" Please enter a valid number!")
    
    # Ask for password type
    print("\nPassword type:")
    print("1. Simple (letters + numbers)")
    print("2. Strong (letters + numbers + symbols)")
    choice = input("Enter choice (1-2): ").strip()
    
    print(f"\n Generated {count} passwords:\n")
    
    for i in range(1, count + 1):
        if choice == '1':
            password = generate_simple_password(length)
        else:
            password = generate_strong_password(length)
        
        print(f"{i}. {password}")

def check_password_strength(password=None):
    """
    Check the strength of a password
    
    Args:
        password (str): Password to check (optional)
    
    Returns:
        str: Strength rating
    """
    if not password:
        print("\n--- PASSWORD STRENGTH CHECKER ---")
        password = input("Enter password to check: ")
    
    if not password:
        print(" Password cannot be empty!")
        return
    
    # Initialize score
    score = 0
    feedback = []
    
    # Check length
    length = len(password)
    if length >= 12:
        score += 2
        feedback.append(" Good length (12+ characters)")
    elif length >= 8:
        score += 1
        feedback.append("  Decent length (8-11 characters)")
    else:
        feedback.append(" Too short (less than 8 characters)")
    
    # Check for lowercase
    if any(c.islower() for c in password):
        score += 1
        feedback.append(" Contains lowercase letters")
    else:
        feedback.append(" Missing lowercase letters")
    
    # Check for uppercase
    if any(c.isupper() for c in password):
        score += 1
        feedback.append(" Contains uppercase letters")
    else:
        feedback.append(" Missing uppercase letters")
    
    # Check for digits
    if any(c.isdigit() for c in password):
        score += 1
        feedback.append(" Contains numbers")
    else:
        feedback.append(" Missing numbers")
    
    # Check for symbols
    if any(c in string.punctuation for c in password):
        score += 2
        feedback.append(" Contains symbols")
    else:
        feedback.append(" Missing symbols")

    # Check for common patterns
    common_passwords = ['password', '123456', 'qwerty', 'abc123', '111111']
    if password.lower() in common_passwords:
        score = 0
        feedback.append(" This is a common password!")
    
    # Determine strength
    if score >= 7:
        strength = "🟢 STRONG"
        recommendation = "Excellent! This is a secure password."
    elif score >= 5:
        strength = "🟡 MODERATE"
        recommendation = "Good, but could be stronger. Consider adding more character types."
    elif score >= 3:
        strength = "🟠 WEAK"
        recommendation = "This password needs improvement. Add more variety."
    else:
        strength = "🔴 VERY WEAK"
        recommendation = "This password is not secure! Generate a new one."
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Password: {password}")
    print(f"Length: {length} characters")
    print(f"Strength: {strength}")
    print(f"Score: {score}/7")
    print(f"{'='*50}")
    
    print("\nFeedback:")
    for item in feedback:
        print(f"  {item}")
    
    print(f"\nRecommendation: {recommendation}")

def get_valid_choice():
    """Get valid menu choice from user"""
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print(" Invalid choice! Please enter a number between 1 and 6.")

def copy_to_clipboard_instruction(password):
    """Show instructions for copying password"""
    print("\n Tip: Select the password above and press Ctrl+C to copy it!")

def main():
    """Main program loop"""
    print("\n Welcome to Password Generator! ")
    print("Generate strong, random passwords easily!")
    
    while True:
        display_menu()
        choice = get_valid_choice()
        
        if choice == '1':
            # Simple password
            while True:
                try:
                    length = int(input("\nEnter password length (default 8): ") or "8")
                    if length < 1:
                        print(" Length must be at least 1!")
                        continue
                    break
                except ValueError:
                    print(" Please enter a valid number!")
            
            password = generate_simple_password(length)
            print(f"\n Generated Simple Password:")
            print(f"{'='*50}")
            print(f"{password}")
            print(f"{'='*50}")
            copy_to_clipboard_instruction(password)
            check_password_strength(password)
        
        elif choice == '2':
            # Strong password
            while True:
                try:
                    length = int(input("\nEnter password length (default 12): ") or "12")
                    if length < 4:
                        print(" Length must be at least 4!")
                        continue
                    break
                except ValueError:
                    print(" Please enter a valid number!")
            
            password = generate_strong_password(length)
            print(f"\n Generated Strong Password:")
            print(f"{'='*50}")
            print(f"{password}")
            print(f"{'='*50}")
            copy_to_clipboard_instruction(password)
            check_password_strength(password)
        
        elif choice == '3':
            # Custom password
            password = generate_custom_password()
            if password:
                print(f"\n Generated Custom Password:")
                print(f"{'='*50}")
                print(f"{password}")
                print(f"{'='*50}")
                copy_to_clipboard_instruction(password)
                check_password_strength(password)
        
        elif choice == '4':
            # Multiple passwords
            generate_multiple_passwords()
        
        elif choice == '5':
            # Check password strength
            check_password_strength()
        
        elif choice == '6':
            # Exit
            print("\n Thank you for using Password Generator!")
            print("Stay secure! \n")
            break
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")

# Start the program
if __name__ == "__main__":
    main()
