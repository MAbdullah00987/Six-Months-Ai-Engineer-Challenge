
#Refactoring a Script to Use Classes and Objects
#Refactor a Previous Project: Take a script from Week 1 and refactor it to use classes and objects.


class Student:
    """Represents a single student with their information"""
    
    def __init__(self, name, age, grade):
        """Initialize a new student"""
        self.name = name
        self.age = age
        self.grade = grade
    
    def update_grade(self, new_grade):
        """Update this student's grade"""
        self.grade = new_grade
    
    def get_letter_grade(self):
        """Calculate letter grade from percentage"""
        if self.grade >= 90:
            return 'A'
        elif self.grade >= 80:
            return 'B'
        elif self.grade >= 70:
            return 'C'
        elif self.grade >= 60:
            return 'D'
        else:
            return 'F'
    
    def __str__(self):
        """String representation of student"""
        return f"{self.name} | Age: {self.age} | Grade: {self.grade}% ({self.get_letter_grade()})"


class GradeManager:
    """Manages a collection of students"""
    
    def __init__(self):
        """Initialize empty list of students"""
        self.students = []
    
    def add_student(self, name, age, grade):
        """Add a new student to the manager"""
        student = Student(name, age, grade)
        self.students.append(student)
        print(f"✓ Added student: {name}")
        return student
    
    def find_student(self, name):
        """Find a student by name"""
        for student in self.students:
            if student.name.lower() == name.lower():
                return student
        return None
    
    def update_grade(self, name, new_grade):
        """Update a student's grade"""
        student = self.find_student(name)
        if student:
            student.update_grade(new_grade)
            print(f"✓ Updated {name}'s grade to {new_grade}%")
        else:
            print(f"✗ Student '{name}' not found")
    
    def calculate_average(self):
        """Calculate average grade of all students"""
        if len(self.students) == 0:
            return 0
        total = sum(student.grade for student in self.students)
        return total / len(self.students)
    
    def get_top_student(self):
        """Get the student with the highest grade"""
        if len(self.students) == 0:
            return None
        return max(self.students, key=lambda s: s.grade)
    
    def display_all(self):
        """Display all students"""
        if len(self.students) == 0:
            print("No students found.")
            return
        
        print("\n" + "="*60)
        print("ALL STUDENTS")
        print("="*60)
        for i, student in enumerate(self.students, 1):
            print(f"{i}. {student}")
        print("="*60)
    
    def get_statistics(self):
        """Get statistics about the class"""
        if len(self.students) == 0:
            return "No students to analyze"
        
        average = self.calculate_average()
        top_student = self.get_top_student()
        total_students = len(self.students)
        
        stats = f"""
CLASS STATISTICS
{'='*60}
Total Students: {total_students}
Class Average: {average:.2f}%
Top Student: {top_student.name} with {top_student.grade}%
{'='*60}
"""
        return stats


# Main program
if __name__ == "__main__":
    print("🎓 STUDENT GRADE MANAGER (OOP Style)")
    print("-" * 60)
    
    # Create the grade manager
    manager = GradeManager()
    
    # Add some students
    manager.add_student("Alice Smith", 20, 85)
    manager.add_student("Bob Johnson", 22, 92)
    manager.add_student("Charlie Brown", 21, 78)
    
    # Display all students
    manager.display_all()
    
    # Show statistics
    print(manager.get_statistics())
    
    # Update a grade
    print("Updating Bob's grade...")
    manager.update_grade("Bob Johnson", 95)
    
    # Display updated list and statistics
    manager.display_all()
    print(manager.get_statistics())