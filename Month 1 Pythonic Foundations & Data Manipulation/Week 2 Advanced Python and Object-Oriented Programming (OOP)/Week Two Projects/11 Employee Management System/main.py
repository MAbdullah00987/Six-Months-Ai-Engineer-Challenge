
#Employee Management System 
#Employee Management System: Design classes for Employee and Department to manage company data.


class Employee:
    """Represents an individual employee in the company"""
    
    def __init__(self, emp_id, name, position, salary):
        """Initialize employee with basic information"""
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.salary = salary
    
    def display_info(self):
        """Display employee information"""
        print(f"\n{'='*50}")
        print(f"Employee ID: {self.emp_id}")
        print(f"Name: {self.name}")
        print(f"Position: {self.position}")
        print(f"Salary: ${self.salary:,.2f}")
        print(f"{'='*50}")
    
    def give_raise(self, amount):
        """Increase employee salary"""
        self.salary += amount
        print(f"✓ {self.name} received a raise of ${amount:,.2f}")
        print(f"  New salary: ${self.salary:,.2f}")
    
    def update_position(self, new_position):
        """Update employee's job position"""
        old_position = self.position
        self.position = new_position
        print(f"{self.name} promoted from {old_position} to {new_position}")
    
    def __str__(self):
        """String representation of employee"""
        return f"{self.name} ({self.position}) - ID: {self.emp_id}"


class Department:
    """Represents a department containing multiple employees"""
    
    def __init__(self, dept_name):
        """Initialize department with name and empty employee list"""
        self.dept_name = dept_name
        self.employees = []
    
    def add_employee(self, employee):
        """Add an employee to the department"""
        self.employees.append(employee)
        print(f"✓ {employee.name} added to {self.dept_name} department")
    
    def remove_employee(self, emp_id):
        """Remove an employee by their ID"""
        for employee in self.employees:
            if employee.emp_id == emp_id:
                self.employees.remove(employee)
                print(f"{employee.name} removed from {self.dept_name} department")
                return True
        print(f"Employee with ID {emp_id} not found")
        return False
    
    def list_employees(self):
        """Display all employees in the department"""
        print(f"\n{'='*60}")
        print(f"Department: {self.dept_name}")
        print(f"Total Employees: {len(self.employees)}")
        print(f"{'='*60}")
        
        if not self.employees:
            print("No employees in this department")
        else:
            for i, emp in enumerate(self.employees, 1):
                print(f"{i}. {emp}")
        print(f"{'='*60}")
    
    def calculate_total_payroll(self):
        """Calculate total salary expense for the department"""
        total = sum(emp.salary for emp in self.employees)
        print(f"\n{self.dept_name} Total Payroll: ${total:,.2f}")
        return total
    
    def find_employee(self, emp_id):
        """Find and return an employee by ID"""
        for employee in self.employees:
            if employee.emp_id == emp_id:
                return employee
        return None
    
    def get_average_salary(self):
        """Calculate average salary in the department"""
        if not self.employees:
            return 0
        avg = sum(emp.salary for emp in self.employees) / len(self.employees)
        print(f"{self.dept_name} Average Salary: ${avg:,.2f}")
        return avg


# ==================== DEMO USAGE ====================

def main():
    """Demonstration of the Employee Management System"""
    
    print("\nEMPLOYEE MANAGEMENT SYSTEM DEMO \n")
    
    # Create departments
    it_dept = Department("Information Technology")
    hr_dept = Department("Human Resources")
    sales_dept = Department("Sales")
    
    print("\n--- Creating Employees ---")
    # Create employees
    emp1 = Employee(101, "Alice Johnson", "Software Engineer", 85000)
    emp2 = Employee(102, "Bob Smith", "Senior Developer", 95000)
    emp3 = Employee(103, "Carol White", "HR Manager", 75000)
    emp4 = Employee(104, "David Brown", "Sales Executive", 65000)
    emp5 = Employee(105, "Eve Davis", "DevOps Engineer", 90000)
    
    print("\n--- Adding Employees to Departments ---")
    # Add employees to departments
    it_dept.add_employee(emp1)
    it_dept.add_employee(emp2)
    it_dept.add_employee(emp5)
    hr_dept.add_employee(emp3)
    sales_dept.add_employee(emp4)
    
    print("\n--- Listing All Departments ---")
    # List employees in each department
    it_dept.list_employees()
    hr_dept.list_employees()
    sales_dept.list_employees()
    
    print("\n--- Employee Operations ---")
    # Display specific employee info
    emp1.display_info()
    
    # Give raise
    emp1.give_raise(5000)
    
    # Promote employee
    emp4.update_position("Senior Sales Executive")
    
    print("\n--- Department Statistics ---")
    # Calculate payrolls
    it_dept.calculate_total_payroll()
    hr_dept.calculate_total_payroll()
    sales_dept.calculate_total_payroll()
    
    # Average salaries
    it_dept.get_average_salary()
    
    print("\n--- Finding an Employee ---")
    # Find employee
    found_emp = it_dept.find_employee(102)
    if found_emp:
        print(f"Found: {found_emp}")
        found_emp.display_info()
    
    print("\n--- Removing an Employee ---")
    # Remove employee
    it_dept.remove_employee(105)
    it_dept.list_employees()
    
    print("\n✓ Demo completed successfully!\n")


# Run the demo
if __name__ == "__main__":
    main()