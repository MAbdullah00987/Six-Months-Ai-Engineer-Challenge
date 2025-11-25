

class Question:
    """Represents a single quiz question with multiple choices"""
    
    def __init__(self, question_text, choices, correct_answer):
        """
        Initialize a Question object
        
        Args:
            question_text (str): The question to ask
            choices (list): List of possible answers
            correct_answer (str): The correct answer
        """
        self.question_text = question_text
        self.choices = choices
        self.correct_answer = correct_answer
    
    def display_question(self):
        """Display the question and its choices"""
        print(f"\n{self.question_text}")
        for i, choice in enumerate(self.choices, 1):
            print(f"{i}. {choice}")
    
    def check_answer(self, user_answer):
        """
        Check if the user's answer is correct
        
        Args:
            user_answer (str): The user's answer
            
        Returns:
            bool: True if correct, False otherwise
        """
        return user_answer.strip().lower() == self.correct_answer.strip().lower()


class Quiz:
    """Manages a collection of questions and quiz logic"""
    
    def __init__(self, title="Quiz"):
        """
        Initialize a Quiz object
        
        Args:
            title (str): The title of the quiz
        """
        self.title = title
        self.questions = []
        self.score = 0
        self.total_questions = 0
    
    def add_question(self, question):
        """
        Add a question to the quiz
        
        Args:
            question (Question): A Question object to add
        """
        self.questions.append(question)
        self.total_questions += 1
    
    def run_quiz(self):
        """Execute the quiz, ask all questions, and track score"""
        print(f"\n{'='*50}")
        print(f"Welcome to: {self.title}")
        print(f"{'='*50}")
        print(f"Total Questions: {self.total_questions}\n")
        
        for i, question in enumerate(self.questions, 1):
            print(f"\nQuestion {i}/{self.total_questions}")
            question.display_question()
            
            # Get user input
            while True:
                try:
                    user_input = input("\nYour answer (enter the number or text): ")
                    
                    # Check if user entered a number (choice index)
                    if user_input.isdigit():
                        choice_index = int(user_input) - 1
                        if 0 <= choice_index < len(question.choices):
                            user_answer = question.choices[choice_index]
                            break
                        else:
                            print("Invalid choice number. Try again.")
                    else:
                        user_answer = user_input
                        break
                except ValueError:
                    print("Invalid input. Please try again.")
            
            # Check answer
            if question.check_answer(user_answer):
                print("✓ Correct!")
                self.score += 1
            else:
                print(f"Wrong! The correct answer was: {question.correct_answer}")
        
        self.display_results()
    
    def display_results(self):
        """Display the final quiz results"""
        print(f"\n{'='*50}")
        print("QUIZ COMPLETED!")
        print(f"{'='*50}")
        print(f"Your Score: {self.score}/{self.total_questions}")
        
        percentage = (self.score / self.total_questions) * 100 if self.total_questions > 0 else 0
        print(f"Percentage: {percentage:.1f}%")
        
        # Grade feedback
        if percentage >= 90:
            print("Grade: A - Excellent! ")
        elif percentage >= 80:
            print("Grade: B - Great job! ")
        elif percentage >= 70:
            print("Grade: C - Good effort! ")
        elif percentage >= 60:
            print("Grade: D - Keep practicing! ")
        else:
            print("Grade: F - Study more! ")
        print(f"{'='*50}\n")


# Example Usage - Python Programming Quiz
def create_sample_quiz():
    """Create and return a sample quiz about Python"""
    
    # Create a new quiz
    quiz = Quiz("Python Programming Quiz")
    
    # Create questions
    q1 = Question(
        "What is Python?",
        ["A compiled language", "An interpreted language", "A markup language", "A database"],
        "An interpreted language"
    )
    
    q2 = Question(
        "Which keyword is used to define a function in Python?",
        ["function", "def", "func", "define"],
        "def"
    )
    
    q3 = Question(
        "What is the output of: print(type([]))?",
        ["<class 'list'>", "<class 'dict'>", "<class 'tuple'>", "<class 'set'>"],
        "<class 'list'>"
    )
    
    q4 = Question(
        "Which of the following is used to create a class in Python?",
        ["class", "Class", "def", "object"],
        "class"
    )
    
    q5 = Question(
        "What does OOP stand for?",
        ["Object Oriented Programming", "Optimal Operation Process", "Organized Object Protocol", "Operation Output Programming"],
        "Object Oriented Programming"
    )
    
    # Add questions to quiz
    quiz.add_question(q1)
    quiz.add_question(q2)
    quiz.add_question(q3)
    quiz.add_question(q4)
    quiz.add_question(q5)
    
    return quiz


# Main program execution
if __name__ == "__main__":
    # Create and run the sample quiz
    my_quiz = create_sample_quiz()
    my_quiz.run_quiz()
    
    # Option to create custom quiz
    print("\nWould you like to create your own quiz? (yes/no)")
    choice = input().strip().lower()
    
    if choice == 'yes':
        custom_quiz = Quiz("Custom Quiz")
        
        print("\nHow many questions would you like to add?")
        num_questions = int(input())
        
        for i in range(num_questions):
            print(f"\n--- Question {i+1} ---")
            q_text = input("Enter the question: ")
            
            print("Enter 4 choices:")
            choices = []
            for j in range(4):
                choice = input(f"Choice {j+1}: ")
                choices.append(choice)
            
            correct = input("Enter the correct answer: ")
            
            custom_quiz.add_question(Question(q_text, choices, correct))
        
        print("\nStarting your custom quiz...")
        custom_quiz.run_quiz()