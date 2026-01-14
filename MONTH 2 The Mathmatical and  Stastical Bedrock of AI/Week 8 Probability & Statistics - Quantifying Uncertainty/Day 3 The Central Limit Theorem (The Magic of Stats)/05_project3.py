
#Prject 3:
#Naive Bayes Classifier from Scratch - Begin implementation for spam detection (foundation work)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier implemented from scratch for spam detection.
    Uses Multinomial Naive Bayes with Laplace smoothing.
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        alpha : float
            Laplace smoothing parameter (default=1.0)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.word_probs = {}
        self.vocab = set()
        self.classes = []
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: lowercase, remove special characters, tokenize.
        
        Parameters:
        -----------
        text : str
            Input text to preprocess
            
        Returns:
        --------
        List[str] : List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = text.split()
        return tokens
    
    def fit(self, X: List[str], y: np.ndarray):
        """
        Train the Naive Bayes classifier.
        
        Parameters:
        -----------
        X : List[str]
            List of text documents
        y : np.ndarray
            Array of labels (0 for ham, 1 for spam)
        """
        self.classes = np.unique(y)
        n_samples = len(X)
        
        # Calculate class priors P(class)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
        
        # Build vocabulary and count words per class
        word_counts = {c: defaultdict(int) for c in self.classes}
        total_words = {c: 0 for c in self.classes}
        
        for text, label in zip(X, y):
            tokens = self._preprocess_text(text)
            for token in tokens:
                self.vocab.add(token)
                word_counts[label][token] += 1
                total_words[label] += 1
        
        # Calculate word probabilities P(word|class) with Laplace smoothing
        vocab_size = len(self.vocab)
        
        for c in self.classes:
            self.word_probs[c] = {}
            for word in self.vocab:
                # Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
                count = word_counts[c][word]
                self.word_probs[c][word] = (count + self.alpha) / (
                    total_words[c] + self.alpha * vocab_size
                )
        
        print(f"Training completed!")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Class priors: {self.class_priors}")
    
    def _predict_single(self, text: str) -> int:
        """
        Predict the class for a single text document.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        int : Predicted class (0 or 1)
        """
        tokens = self._preprocess_text(text)
        
        # Calculate log probabilities for numerical stability
        log_probs = {}
        
        for c in self.classes:
            # Start with log of class prior
            log_prob = np.log(self.class_priors[c])
            
            # Add log probabilities of words
            for token in tokens:
                if token in self.vocab:
                    log_prob += np.log(self.word_probs[c][token])
            
            log_probs[c] = log_prob
        
        # Return class with highest probability
        return max(log_probs, key=log_probs.get)
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict classes for multiple documents.
        
        Parameters:
        -----------
        X : List[str]
            List of text documents
            
        Returns:
        --------
        np.ndarray : Array of predictions
        """
        return np.array([self._predict_single(text) for text in X])
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict probability estimates for documents.
        
        Parameters:
        -----------
        X : List[str]
            List of text documents
            
        Returns:
        --------
        np.ndarray : Array of probability estimates
        """
        probas = []
        
        for text in X:
            tokens = self._preprocess_text(text)
            log_probs = {}
            
            for c in self.classes:
                log_prob = np.log(self.class_priors[c])
                for token in tokens:
                    if token in self.vocab:
                        log_prob += np.log(self.word_probs[c][token])
                log_probs[c] = log_prob
            
            # Convert log probabilities to probabilities using softmax
            max_log_prob = max(log_probs.values())
            exp_probs = {c: np.exp(log_probs[c] - max_log_prob) for c in self.classes}
            total = sum(exp_probs.values())
            probs = [exp_probs[c] / total for c in sorted(self.classes)]
            probas.append(probs)
        
        return np.array(probas)
    
    def get_top_words(self, class_label: int, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top n words for a given class.
        
        Parameters:
        -----------
        class_label : int
            Class label
        n : int
            Number of top words to return
            
        Returns:
        --------
        List[Tuple[str, float]] : List of (word, probability) tuples
        """
        word_prob_list = [(word, prob) for word, prob in self.word_probs[class_label].items()]
        word_prob_list.sort(key=lambda x: x[1], reverse=True)
        return word_prob_list[:n]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }


def plot_confusion_matrix(cm: np.ndarray, title: str = 'Confusion Matrix'):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_top_words(classifier: NaiveBayesClassifier, n_words: int = 15):
    """Plot top words for each class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    class_names = ['Ham', 'Spam']
    colors = ['#2ecc71', '#e74c3c']
    
    for idx, class_label in enumerate(classifier.classes):
        top_words = classifier.get_top_words(class_label, n_words)
        words, probs = zip(*top_words)
        
        axes[idx].barh(range(len(words)), probs, color=colors[idx], alpha=0.7)
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words)
        axes[idx].set_xlabel('Probability', fontsize=12)
        axes[idx].set_title(f'Top {n_words} Words in {class_names[idx]} Messages', 
                           fontsize=14, fontweight='bold')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(metrics: Dict[str, float]):
    """Plot metrics as a bar chart."""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6'], alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Classification Metrics', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# DEMO: Sample Dataset and Training
# ============================================================================

def create_sample_dataset():
    """Create a sample spam/ham dataset for demonstration."""
    
    # Sample spam messages
    spam_messages = [
        "Congratulations! You've won a free lottery prize of $1000000. Click here to claim now!",
        "URGENT: Your account will be closed. Verify your password immediately by clicking this link.",
        "Get rich quick! Make money fast from home. No experience needed. Act now!",
        "Free credit card offer! Apply now and get approved instantly with zero interest.",
        "You have been selected for a special offer. Buy cheap medications online now!",
        "Winner winner! Claim your prize money now. This is not a scam, click here.",
        "Work from home and earn $5000 per week. Limited time offer, register today!",
        "Your loan has been approved! Get cash now with no credit check required.",
        "Hot singles in your area want to meet you! Click to see profiles now.",
        "Increase your income overnight! Join our program and make easy money fast.",
        "Lowest prices on medications! Order now and save big on prescriptions.",
        "Free iPhone giveaway! You are the lucky winner. Claim your prize immediately.",
        "Lose weight fast with this one weird trick. Buy now and get discount.",
        "Your package is waiting. Pay the shipping fee to receive your prize.",
        "Exclusive deal just for you! Limited offer on luxury watches at 90% off.",
    ]
    
    # Sample ham (legitimate) messages
    ham_messages = [
        "Hey, are we still meeting for lunch at noon tomorrow? Let me know.",
        "Thanks for sending the project report. I'll review it by end of day.",
        "Can you pick up milk and eggs on your way home? We're running low.",
        "The team meeting has been rescheduled to Friday at 2 PM.",
        "Happy birthday! Hope you have a wonderful day with family and friends.",
        "I finished reading that book you recommended. It was really good!",
        "Don't forget we have a doctor's appointment on Tuesday at 3 PM.",
        "The weather forecast says it might rain this weekend. Bring an umbrella.",
        "Could you share the notes from yesterday's lecture? I missed the class.",
        "Dinner was great last night. Thanks for the invitation!",
        "I'm running a few minutes late. Will be there by 10:15.",
        "The quarterly report is due next Monday. Let's schedule a review meeting.",
        "My flight lands at 6 PM. Can you pick me up from the airport?",
        "The new policy goes into effect starting next month. Please review the document.",
        "Great presentation today! The client seemed very impressed with our proposal.",
    ]
    
    # Create labels (0 = ham, 1 = spam)
    X = ham_messages + spam_messages
    y = np.array([0] * len(ham_messages) + [1] * len(spam_messages))
    
    return X, y


def main():
    """Main function to demonstrate the Naive Bayes classifier."""
    
    print("=" * 70)
    print("NAIVE BAYES SPAM CLASSIFIER FROM SCRATCH")
    print("=" * 70)
    print()
    
    # Create sample dataset
    print("Creating sample dataset...")
    X, y = create_sample_dataset()
    print(f"Dataset size: {len(X)} messages")
    print(f"Ham messages: {np.sum(y == 0)}")
    print(f"Spam messages: {np.sum(y == 1)}")
    print()
    
    # Split data (simple 80-20 split)
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train = [X[i] for i in train_idx]
    y_train = y[train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = y[test_idx]
    
    print(f"Training set: {len(X_train)} messages")
    print(f"Test set: {len(X_test)} messages")
    print()
    
    # Train classifier
    print("Training Naive Bayes Classifier...")
    print("-" * 70)
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(X_train, y_train)
    print()
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    print()
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(y_test, y_pred)
    print("-" * 70)
    print("PERFORMANCE METRICS:")
    print("-" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print()
    
    # Display confusion matrix
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print()
    
    # Test on new messages
    print("=" * 70)
    print("TESTING ON NEW MESSAGES:")
    print("=" * 70)
    
    test_messages = [
        "Can we reschedule our meeting to next week?",
        "WIN FREE MONEY NOW! Click here for amazing prizes!",
        "Thanks for your help with the project yesterday.",
        "Get rich quick! Make thousands from home!"
    ]
    
    for msg in test_messages:
        pred = classifier.predict([msg])[0]
        proba = classifier.predict_proba([msg])[0]
        label = "SPAM" if pred == 1 else "HAM"
        confidence = proba[pred] * 100
        
        print(f"\nMessage: \"{msg}\"")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
    
    print()
    print("=" * 70)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Plot metrics
    plot_metrics_comparison(metrics)
    
    # Plot top words
    plot_top_words(classifier, n_words=15)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set style for plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Run the main demo
    main()

    