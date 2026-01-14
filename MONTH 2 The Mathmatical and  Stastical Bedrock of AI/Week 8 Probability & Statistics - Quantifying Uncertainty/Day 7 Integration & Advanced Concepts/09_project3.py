
#Project: Naive Bayes Classifier - Complete and test the spam detection 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class NaiveBayesSpamClassifier:
    """
    Multinomial Naive Bayes Classifier for Spam Detection
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the classifier
        
        Parameters:
        -----------
        alpha : float
            Laplace smoothing parameter (default=1.0)
        """
        self.alpha = alpha
        self.class_probs = {}
        self.word_probs = {}
        self.vocabulary = set()
        self.classes = []
        
    def preprocess_text(self, text):
        """
        Preprocess text: lowercase, remove special characters, tokenize
        
        Parameters:
        -----------
        text : str
            Input text to preprocess
            
        Returns:
        --------
        list : List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers, keep only letters
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = text.split()
        return tokens
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Parameters:
        -----------
        X : array-like
            Training messages
        y : array-like
            Training labels (spam/ham)
        """
        print("Training Naive Bayes Classifier...")
        
        # Get unique classes
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calculate class probabilities P(class)
        for cls in self.classes:
            self.class_probs[cls] = np.sum(y == cls) / n_samples
            
        # Build vocabulary and count words for each class
        word_counts = {cls: defaultdict(int) for cls in self.classes}
        class_word_totals = {cls: 0 for cls in self.classes}
        
        for message, label in zip(X, y):
            tokens = self.preprocess_text(message)
            for token in tokens:
                self.vocabulary.add(token)
                word_counts[label][token] += 1
                class_word_totals[label] += 1
        
        # Calculate word probabilities P(word|class) with Laplace smoothing
        vocab_size = len(self.vocabulary)
        
        for cls in self.classes:
            self.word_probs[cls] = {}
            for word in self.vocabulary:
                # Laplace smoothing
                count = word_counts[cls][word]
                self.word_probs[cls][word] = (count + self.alpha) / (class_word_totals[cls] + self.alpha * vocab_size)
        
        print(f"Training completed!")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Classes: {self.classes}")
        print(f"Class probabilities: {self.class_probs}")
        
    def predict_single(self, message):
        """
        Predict class for a single message
        
        Parameters:
        -----------
        message : str
            Message to classify
            
        Returns:
        --------
        str : Predicted class
        """
        tokens = self.preprocess_text(message)
        class_scores = {}
        
        for cls in self.classes:
            # Start with log probability of class
            score = np.log(self.class_probs[cls])
            
            # Add log probabilities of words
            for token in tokens:
                if token in self.vocabulary:
                    score += np.log(self.word_probs[cls][token])
            
            class_scores[cls] = score
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict(self, X):
        """
        Predict classes for multiple messages
        
        Parameters:
        -----------
        X : array-like
            Messages to classify
            
        Returns:
        --------
        array : Predicted classes
        """
        return np.array([self.predict_single(message) for message in X])
    
    def predict_proba(self, X):
        """
        Predict class probabilities for messages
        
        Parameters:
        -----------
        X : array-like
            Messages to classify
            
        Returns:
        --------
        array : Probability estimates
        """
        probas = []
        for message in X:
            tokens = self.preprocess_text(message)
            class_scores = {}
            
            for cls in self.classes:
                score = np.log(self.class_probs[cls])
                for token in tokens:
                    if token in self.vocabulary:
                        score += np.log(self.word_probs[cls][token])
                class_scores[cls] = score
            
            # Convert log probabilities to probabilities
            max_score = max(class_scores.values())
            exp_scores = {cls: np.exp(score - max_score) for cls, score in class_scores.items()}
            total = sum(exp_scores.values())
            probas.append([exp_scores[cls] / total for cls in self.classes])
        
        return np.array(probas)


def create_sample_dataset():
    """
    Create a sample spam/ham dataset for demonstration
    """
    spam_messages = [
        "free money now click here",
        "congratulations you won a prize claim now",
        "buy cheap pills online fast delivery",
        "earn money from home work online",
        "get rich quick click here now",
        "winner winner claim your prize today",
        "cheap viagra buy now limited offer",
        "work from home earn thousands weekly",
        "you have won lottery click to claim",
        "make money fast online casino bonus",
        "free gift card claim yours today",
        "lose weight fast miracle pill",
        "click here for free money prizes",
        "congratulations winner click now claim",
        "buy now limited time offer cheap",
        "earn cash online work home today",
        "free trial buy pills online cheap",
        "winner alert claim prize money now",
        "get paid online work from home",
        "miracle cure buy now limited stock"
    ]
    
    ham_messages = [
        "hey how are you doing today",
        "meeting scheduled for tomorrow at noon",
        "can you send me the report please",
        "thanks for your help yesterday",
        "lets catch up for coffee this weekend",
        "the project deadline is next week",
        "please review the attached document",
        "happy birthday hope you have great day",
        "see you at the office tomorrow",
        "reminder about the team meeting today",
        "can we reschedule our appointment",
        "thank you for the information",
        "looking forward to working with you",
        "please let me know your availability",
        "the conference call went well today",
        "have a great weekend see you monday",
        "congratulations on your promotion well deserved",
        "the weather is nice today perfect for walk",
        "thanks for taking the time to help",
        "please confirm receipt of this message"
    ]
    
    # Create DataFrame
    messages = spam_messages + ham_messages
    labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    
    df = pd.DataFrame({
        'message': messages,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(df):
    """
    Plot distribution of spam vs ham messages
    """
    plt.figure(figsize=(8, 6))
    counts = df['label'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    
    plt.bar(counts.index, counts.values, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Spam vs Ham Messages', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(counts.items()):
        plt.text(i, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(classifier, top_n=10):
    """
    Plot most important words for spam vs ham classification
    """
    spam_words = []
    ham_words = []
    
    for word in classifier.vocabulary:
        spam_prob = classifier.word_probs['spam'].get(word, 0)
        ham_prob = classifier.word_probs['ham'].get(word, 0)
        
        if spam_prob > 0 and ham_prob > 0:
            ratio = spam_prob / ham_prob
            if ratio > 1:
                spam_words.append((word, ratio))
            else:
                ham_words.append((word, 1/ratio))
    
    # Sort and get top words
    spam_words = sorted(spam_words, key=lambda x: x[1], reverse=True)[:top_n]
    ham_words = sorted(ham_words, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Spam words
    words_spam, scores_spam = zip(*spam_words) if spam_words else ([], [])
    ax1.barh(range(len(words_spam)), scores_spam, color='#FF6B6B', alpha=0.7)
    ax1.set_yticks(range(len(words_spam)))
    ax1.set_yticklabels(words_spam)
    ax1.set_xlabel('Spam/Ham Probability Ratio', fontsize=12)
    ax1.set_title('Top Spam Indicators', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Ham words
    words_ham, scores_ham = zip(*ham_words) if ham_words else ([], [])
    ax2.barh(range(len(words_ham)), scores_ham, color='#4ECDC4', alpha=0.7)
    ax2.set_yticks(range(len(words_ham)))
    ax2.set_yticklabels(words_ham)
    ax2.set_xlabel('Ham/Spam Probability Ratio', fontsize=12)
    ax2.set_title('Top Ham Indicators', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the spam classifier
    """
    print("=" * 60)
    print("NAIVE BAYES SPAM CLASSIFIER")
    print("=" * 60)
    print()
    
    # Create dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset size: {len(df)} messages")
    print(f"\nFirst few messages:")
    print(df.head(10))
    print()
    
    # Plot class distribution
    plot_class_distribution(df)
    
    # Split data
    X = df['message'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()
    
    # Train classifier
    classifier = NaiveBayesSpamClassifier(alpha=1.0)
    classifier.fit(X_train, y_train)
    print()
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    print()
    
    # Classification report
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, classifier.classes)
    
    # Plot feature importance
    plot_feature_importance(classifier, top_n=10)
    
    # Test with custom messages
    print("\n" + "=" * 60)
    print("TESTING WITH CUSTOM MESSAGES")
    print("=" * 60)
    
    test_messages = [
        "Congratulations! You won a free prize",
        "Hey, want to grab lunch tomorrow?",
        "Click here for cheap pills and miracle cure",
        "Meeting rescheduled to 3pm today",
        "Earn money fast working from home now"
    ]
    
    for msg in test_messages:
        prediction = classifier.predict_single(msg)
        probas = classifier.predict_proba([msg])[0]
        
        print(f"\nMessage: '{msg}'")
        print(f"Prediction: {prediction.upper()}")
        print(f"Probabilities: Spam={probas[1]:.3f}, Ham={probas[0]:.3f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()