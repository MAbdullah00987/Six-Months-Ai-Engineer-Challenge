
#Bank Account System - Complete Guide
#Heres how everything works out by step by step:

#Complete Step-by-Step Explanation
#This is a Bank Account Management System that simulates real banking operations using Object-Oriented Programming (OOP) in Python. It allows you to:
#Create bank accounts
#Deposit and withdraw money
#Check balances
#Transfer money between accounts
#Track transaction history
#Handle savings accounts with interest


class BankAccount:
    """A class to represent a bank account with basic banking operations."""
    
    # Class variable to track all accounts
    total_accounts = 0
    
    def __init__(self, account_holder, account_number, initial_balance=0):
        """
        Initialize a new bank account.
        
        Args:
            account_holder (str): Name of the account holder
            account_number (str): Unique account number
            initial_balance (float): Starting balance (default 0)
        """
        self.account_holder = account_holder
        self.account_number = account_number
        self.balance = initial_balance
        self.transaction_history = []
        
        # Increment total accounts counter
        BankAccount.total_accounts += 1
        
        # Record initial deposit if any
        if initial_balance > 0:
            self.transaction_history.append(f"Initial deposit: ${initial_balance:.2f}")
    
    def deposit(self, amount):
        """
        Deposit money into the account.
        
        Args:
            amount (float): Amount to deposit
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print("Error: Deposit amount must be positive!")
            return False
        
        self.balance += amount
        self.transaction_history.append(f"Deposit: +${amount:.2f}")
        print(f"Successfully deposited ${amount:.2f}")
        print(f"New balance: ${self.balance:.2f}")
        return True
    
    def withdraw(self, amount):
        """
        Withdraw money from the account.
        
        Args:
            amount (float): Amount to withdraw
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print("Error: Withdrawal amount must be positive!")
            return False
        
        if amount > self.balance:
            print(f"Error: Insufficient funds! Available balance: ${self.balance:.2f}")
            return False
        
        self.balance -= amount
        self.transaction_history.append(f"Withdrawal: -${amount:.2f}")
        print(f"Successfully withdrew ${amount:.2f}")
        print(f"New balance: ${self.balance:.2f}")
        return True
    
    def get_balance(self):
        """
        Display current account balance.
        
        Returns:
            float: Current balance
        """
        print(f"\n{'='*50}")
        print(f"Account Holder: {self.account_holder}")
        print(f"Account Number: {self.account_number}")
        print(f"Current Balance: ${self.balance:.2f}")
        print(f"{'='*50}\n")
        return self.balance
    
    def get_transaction_history(self):
        """Display all transactions for this account."""
        print(f"\n{'='*50}")
        print(f"Transaction History - {self.account_holder}")
        print(f"Account: {self.account_number}")
        print(f"{'='*50}")
        
        if not self.transaction_history:
            print("No transactions yet.")
        else:
            for i, transaction in enumerate(self.transaction_history, 1):
                print(f"{i}. {transaction}")
        
        print(f"{'='*50}\n")
    
    def transfer(self, recipient_account, amount):
        """
        Transfer money to another account.
        
        Args:
            recipient_account (BankAccount): The account to transfer to
            amount (float): Amount to transfer
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print("Error: Transfer amount must be positive!")
            return False
        
        if amount > self.balance:
            print(f"Error: Insufficient funds! Available balance: ${self.balance:.2f}")
            return False
        
        # Deduct from sender
        self.balance -= amount
        self.transaction_history.append(
            f"Transfer to {recipient_account.account_holder}: -${amount:.2f}"
        )
        
        # Add to recipient
        recipient_account.balance += amount
        recipient_account.transaction_history.append(
            f"Transfer from {self.account_holder}: +${amount:.2f}"
        )
        
        print(f"Successfully transferred ${amount:.2f} to {recipient_account.account_holder}")
        print(f"Your new balance: ${self.balance:.2f}")
        return True
    
    def __str__(self):
        """String representation of the account."""
        return f"BankAccount({self.account_holder}, {self.account_number}, Balance: ${self.balance:.2f})"


class SavingsAccount(BankAccount):
    """A savings account with interest rate."""
    
    def __init__(self, account_holder, account_number, initial_balance=0, interest_rate=0.02):
        """
        Initialize a savings account.
        
        Args:
            interest_rate (float): Annual interest rate (default 2%)
        """
        super().__init__(account_holder, account_number, initial_balance)
        self.interest_rate = interest_rate
    
    def apply_interest(self):
        """Apply interest to the account balance."""
        interest = self.balance * self.interest_rate
        self.balance += interest
        self.transaction_history.append(f"Interest earned: +${interest:.2f}")
        print(f"Interest applied: ${interest:.2f}")
        print(f"New balance: ${self.balance:.2f}")


# Demo: Using the Bank Account System
if __name__ == "__main__":
    print("\nWELCOME TO THE BANK ACCOUNT SYSTEM\n")
    
    # Create accounts
    print("Creating accounts...")
    account1 = BankAccount("Alice Johnson", "ACC001", 1000)
    account2 = BankAccount("Bob Smith", "ACC002", 500)
    savings = SavingsAccount("Charlie Brown", "SAV001", 2000, 0.05)
    
    print(f"\nTotal accounts created: {BankAccount.total_accounts}\n")
    
    # Demonstrate deposit
    print("\n--- DEPOSIT OPERATION ---")
    account1.deposit(500)
    
    # Demonstrate withdrawal
    print("\n--- WITHDRAWAL OPERATION ---")
    account2.withdraw(200)
    
    # Demonstrate balance inquiry
    print("\n--- BALANCE INQUIRY ---")
    account1.get_balance()
    
    # Demonstrate transfer
    print("\n--- TRANSFER OPERATION ---")
    account1.transfer(account2, 300)
    
    # Show transaction history
    print("\n--- TRANSACTION HISTORY ---")
    account1.get_transaction_history()
    account2.get_transaction_history()
    
    # Demonstrate savings account interest
    print("\n--- SAVINGS ACCOUNT INTEREST ---")
    savings.get_balance()
    savings.apply_interest()
    
    # Error handling demonstrations
    print("\n--- ERROR HANDLING DEMONSTRATIONS ---")
    account1.withdraw(10000)  # Insufficient funds
    account1.deposit(-50)      # Negative deposit
    
    print("\nCompleted successfully! \n")