
"""
Bank Account System
A complete banking system using Object-Oriented Programming
Features: Create accounts, deposit, withdraw, transfer, transaction history
"""

import random
from datetime import datetime

class BankAccount:
    """Base class for bank accounts"""
    
    # Class variable to track total accounts
    total_accounts = 0
    
    def __init__(self, account_holder, initial_deposit=0):
        """
        Initialize a bank account
        
        Args:
            account_holder (str): Name of account holder
            initial_deposit (float): Initial deposit amount
        """
        self.account_holder = account_holder
        self.account_number = self.generate_account_number()
        self.balance = initial_deposit
        self.transaction_history = []
        self.created_date = datetime.now()
        
        BankAccount.total_accounts += 1
        
        # Record initial deposit
        if initial_deposit > 0:
            self.add_transaction("Initial Deposit", initial_deposit)
    
    @staticmethod
    def generate_account_number():
        """Generate a random 10-digit account number"""
        return ''.join([str(random.randint(0, 9)) for _ in range(10)])
    
    def add_transaction(self, transaction_type, amount):
        """
        Add transaction to history
        
        Args:
            transaction_type (str): Type of transaction
            amount (float): Transaction amount
        """
        transaction = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': transaction_type,
            'amount': amount,
            'balance': self.balance
        }
        self.transaction_history.append(transaction)
    
    def deposit(self, amount):
        """
        Deposit money into account
        
        Args:
            amount (float): Amount to deposit
        
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print(" Deposit amount must be positive!")
            return False
        
        self.balance += amount
        self.add_transaction("Deposit", amount)
        print(f" Successfully deposited ${amount:.2f}")
        print(f" New balance: ${self.balance:.2f}")
        return True
    
    def withdraw(self, amount):
        """
        Withdraw money from account
        
        Args:
            amount (float): Amount to withdraw
        
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print(" Withdrawal amount must be positive!")
            return False
        
        if amount > self.balance:
            print(f" Insufficient funds! Available balance: ${self.balance:.2f}")
            return False
        
        self.balance -= amount
        self.add_transaction("Withdrawal", -amount)
        print(f" Successfully withdrew ${amount:.2f}")
        print(f" Remaining balance: ${self.balance:.2f}")
        return True
    
    def check_balance(self):
        """Display current balance"""
        print(f"\n{'='*50}")
        print(f"Account Holder: {self.account_holder}")
        print(f"Account Number: {self.account_number}")
        print(f"Current Balance: ${self.balance:.2f}")
        print(f"{'='*50}")
    
    def view_transaction_history(self):
        """Display all transactions"""
        print(f"\n{'='*70}")
        print(f"TRANSACTION HISTORY - {self.account_holder}")
        print(f"Account: {self.account_number}")
        print(f"{'='*70}")
        
        if not self.transaction_history:
            print("No transactions yet.")
            return
        
        print(f"{'Date':<20} {'Type':<20} {'Amount':<15} {'Balance':<15}")
        print("-" * 70)
        
        for transaction in self.transaction_history:
            amount = transaction['amount']
            amount_str = f"${amount:+.2f}"  # Show + or - sign
            balance_str = f"${transaction['balance']:.2f}"
            
            print(f"{transaction['date']:<20} {transaction['type']:<20} {amount_str:<15} {balance_str:<15}")
    
    def __str__(self):
        """String representation of account"""
        return f"Account({self.account_holder}, {self.account_number}, ${self.balance:.2f})"


class SavingsAccount(BankAccount):
    """Savings account with interest rate"""
    
    def __init__(self, account_holder, initial_deposit=0, interest_rate=0.03):
        """
        Initialize savings account
        
        Args:
            account_holder (str): Name of account holder
            initial_deposit (float): Initial deposit
            interest_rate (float): Annual interest rate (default 3%)
        """
        super().__init__(account_holder, initial_deposit)
        self.interest_rate = interest_rate
        self.account_type = "Savings"
    
    def apply_interest(self):
        """Apply interest to account balance"""
        interest = self.balance * self.interest_rate
        self.balance += interest
        self.add_transaction("Interest Credit", interest)
        print(f" Interest applied: ${interest:.2f}")
        print(f" New balance: ${self.balance:.2f}")


class CheckingAccount(BankAccount):
    """Checking account with overdraft protection"""
    
    def __init__(self, account_holder, initial_deposit=0, overdraft_limit=500):
        """
        Initialize checking account
        
        Args:
            account_holder (str): Name of account holder
            initial_deposit (float): Initial deposit
            overdraft_limit (float): Overdraft protection limit
        """
        super().__init__(account_holder, initial_deposit)
        self.overdraft_limit = overdraft_limit
        self.account_type = "Checking"
    
    def withdraw(self, amount):
        """
        Withdraw with overdraft protection
        
        Args:
            amount (float): Amount to withdraw
        
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print(" Withdrawal amount must be positive!")
            return False
        
        # Check if withdrawal exceeds balance + overdraft
        if amount > self.balance + self.overdraft_limit:
            print(f" Insufficient funds! Available: ${self.balance:.2f}")
            print(f"Overdraft limit: ${self.overdraft_limit:.2f}")
            return False
        
        self.balance -= amount
        self.add_transaction("Withdrawal", -amount)
        print(f" Successfully withdrew ${amount:.2f}")
        
        if self.balance < 0:
            print(f"  Account overdrawn by ${abs(self.balance):.2f}")
        
        print(f" Current balance: ${self.balance:.2f}")
        return True


class Bank:
    """Bank system to manage multiple accounts"""
    
    def __init__(self, bank_name):
        """
        Initialize bank
        
        Args:
            bank_name (str): Name of the bank
        """
        self.bank_name = bank_name
        self.accounts = {}  # Dictionary to store accounts
    
    def create_account(self, account_holder, account_type, initial_deposit=0):
        """
        Create a new bank account
        
        Args:
            account_holder (str): Name of account holder
            account_type (str): 'savings' or 'checking'
            initial_deposit (float): Initial deposit amount
        
        Returns:
            BankAccount: Created account object
        """
        if account_type.lower() == 'savings':
            account = SavingsAccount(account_holder, initial_deposit)
        elif account_type.lower() == 'checking':
            account = CheckingAccount(account_holder, initial_deposit)
        else:
            print(" Invalid account type! Choose 'savings' or 'checking'")
            return None
        
        self.accounts[account.account_number] = account
        
        print(f"\n Account created successfully!")
        print(f"Account Holder: {account_holder}")
        print(f"Account Type: {account_type.title()}")
        print(f"Account Number: {account.account_number}")
        print(f"Initial Balance: ${initial_deposit:.2f}")
        
        return account
    
    def get_account(self, account_number):
        """
        Retrieve account by account number
        
        Args:
            account_number (str): Account number
        
        Returns:
            BankAccount: Account object or None
        """
        return self.accounts.get(account_number)
    
    def transfer(self, from_account_number, to_account_number, amount):
        """
        Transfer money between accounts
        
        Args:
            from_account_number (str): Source account number
            to_account_number (str): Destination account number
            amount (float): Amount to transfer
        
        Returns:
            bool: True if successful, False otherwise
        """
        from_account = self.get_account(from_account_number)
        to_account = self.get_account(to_account_number)
        
        if not from_account:
            print(" Source account not found!")
            return False
        
        if not to_account:
            print(" Destination account not found!")
            return False
        
        if amount <= 0:
            print(" Transfer amount must be positive!")
            return False
        
        if from_account.balance < amount:
            print(f" Insufficient funds! Available: ${from_account.balance:.2f}")
            return False
        
        # Perform transfer
        from_account.balance -= amount
        to_account.balance += amount
        
        # Record transactions
        from_account.add_transaction(f"Transfer to {to_account.account_holder}", -amount)
        to_account.add_transaction(f"Transfer from {from_account.account_holder}", amount)
        
        print(f"\n Transfer successful!")
        print(f"From: {from_account.account_holder} ({from_account_number})")
        print(f"To: {to_account.account_holder} ({to_account_number})")
        print(f"Amount: ${amount:.2f}")
        
        return True
    
    def list_all_accounts(self):
        """Display all accounts in the bank"""
        print(f"\n{'='*70}")
        print(f"{self.bank_name.upper()} - ALL ACCOUNTS")
        print(f"{'='*70}")
        
        if not self.accounts:
            print("No accounts in the system.")
            return
        
        print(f"{'Account Number':<15} {'Holder Name':<25} {'Type':<12} {'Balance':<15}")
        print("-" * 70)
        
        for account_number, account in self.accounts.items():
            account_type = getattr(account, 'account_type', 'Standard')
            print(f"{account_number:<15} {account.account_holder:<25} {account_type:<12} ${account.balance:<14.2f}")
        
        print(f"\nTotal Accounts: {len(self.accounts)}")


def display_main_menu():
    """Display main menu"""
    print("\n" + "="*50)
    print("  BANK ACCOUNT SYSTEM  ")
    print("="*50)
    print("1. Create New Account")
    print("2. Deposit Money")
    print("3. Withdraw Money")
    print("4. Check Balance")
    print("5. Transfer Money")
    print("6. View Transaction History")
    print("7. List All Accounts")
    print("8. Apply Interest (Savings Only)")
    print("9. Exit")
    print("="*50)


def main():
    """Main program"""
    # Create bank instance
    bank = Bank("MyBank")
    
    print("\n Welcome to MyBank Account System! ")
    
    while True:
        display_main_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            # Create account
            print("\n--- CREATE NEW ACCOUNT ---")
            name = input("Enter account holder name: ").strip()
            
            if not name:
                print(" Name cannot be empty!")
                continue
            
            print("\nAccount Type:")
            print("1. Savings (3% interest)")
            print("2. Checking (Overdraft protection)")
            acc_type_choice = input("Choose account type (1-2): ").strip()
            
            acc_type = 'savings' if acc_type_choice == '1' else 'checking'
            
            try:
                initial = float(input("Enter initial deposit amount: $").strip() or "0")
                bank.create_account(name, acc_type, initial)
            except ValueError:
                print("Invalid amount!")
        
        elif choice == '2':
            # Deposit
            print("\n--- DEPOSIT MONEY ---")
            acc_num = input("Enter account number: ").strip()
            account = bank.get_account(acc_num)
            
            if not account:
                print("Account not found!")
                continue
            
            try:
                amount = float(input("Enter deposit amount: $").strip())
                account.deposit(amount)
            except ValueError:
                print("Invalid amount!")
        
        elif choice == '3':
            # Withdraw
            print("\n--- WITHDRAW MONEY ---")
            acc_num = input("Enter account number: ").strip()
            account = bank.get_account(acc_num)
            
            if not account:
                print("Account not found!")
                continue
            
            try:
                amount = float(input("Enter withdrawal amount: $").strip())
                account.withdraw(amount)
            except ValueError:
                print("Invalid amount!")
        
        elif choice == '4':
            # Check balance
            print("\n--- CHECK BALANCE ---")
            acc_num = input("Enter account number: ").strip()
            account = bank.get_account(acc_num)
            
            if not account:
                print("Account not found!")
                continue
            
            account.check_balance()
        
        elif choice == '5':
            # Transfer
            print("\n--- TRANSFER MONEY ---")
            from_acc = input("Enter source account number: ").strip()
            to_acc = input("Enter destination account number: ").strip()
            
            try:
                amount = float(input("Enter transfer amount: $").strip())
                bank.transfer(from_acc, to_acc, amount)
            except ValueError:
                print("Invalid amount!")
        
        elif choice == '6':
            # Transaction history
            print("\n--- VIEW TRANSACTION HISTORY ---")
            acc_num = input("Enter account number: ").strip()
            account = bank.get_account(acc_num)
            
            if not account:
                print("Account not found!")
                continue
            
            account.view_transaction_history()
        
        elif choice == '7':
            # List all accounts
            bank.list_all_accounts()
        
        elif choice == '8':
            # Apply interest
            print("\n--- APPLY INTEREST ---")
            acc_num = input("Enter savings account number: ").strip()
            account = bank.get_account(acc_num)
            
            if not account:
                print("Account not found!")
                continue
            
            if isinstance(account, SavingsAccount):
                account.apply_interest()
            else:
                print("Interest can only be applied to savings accounts!")
        
        elif choice == '9':
            # Exit
            print("\n Thank you for using MyBank!")
            print("Have a great day! \n")
            break
        
        else:
            print(" Invalid choice! Please enter 1-9.")
        
        # Pause
        input("\nPress Enter to continue...")


# Run the program
if __name__ == "__main__":
    main()