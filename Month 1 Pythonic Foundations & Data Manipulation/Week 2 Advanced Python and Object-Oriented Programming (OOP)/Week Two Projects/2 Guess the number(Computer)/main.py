

import random

def guess(x):
    # 1. Fixed 'radiant' to 'randint'
    random_number = random.randint(1, x)
    guess = 0
    
    while guess != random_number:
        guess = int(input(f'Guess a number between 1 and {x}: '))
        if guess < random_number:
            print("Sorry, guess again. Too low.")
        elif guess > random_number:
            # (I also made "High" lowercase for consistency)
            print("Sorry, guess again. Too high.")
            
    print(f"Yay, Congrats. You have guessed the number {random_number} correctly.")

# 2. Moved the call to 'guess(10)' outside the function.
# This makes it the main part of the script that runs.
if __name__ == "__main__":
    guess(10)
