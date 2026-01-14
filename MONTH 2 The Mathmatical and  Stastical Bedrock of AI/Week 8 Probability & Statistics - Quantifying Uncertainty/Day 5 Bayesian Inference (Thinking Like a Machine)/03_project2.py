

#Task: Project - Naive Bayes Logic. Understand how a spam filter works using Bayes' theorem (Probability of "Spam" given the word "Buy").


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import re

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Sample email dataset
emails = [
    ("Buy cheap watches now!", "spam"),
    ("Meeting tomorrow at 3pm", "ham"),
    ("Buy viagra online cheap", "spam"),
    ("Can you send me the report?", "ham"),
    ("Congratulations! You won a prize", "spam"),
    ("Lunch with the team today", "ham"),
    ("Buy now and save money", "spam"),
    ("Please review the document", "ham"),
    ("Free money click here", "spam"),
    ("Conference call at 2pm", "ham"),
    ("Buy our products today", "spam"),
    ("Thanks for your help", "ham"),
    ("Cheap deals available now", "spam"),
    ("Project deadline next week", "ham"),
    ("Win a free vacation", "spam"),
    ("Meeting notes attached", "ham"),
    ("Buy discount medications", "spam"),
    ("See you tomorrow", "ham"),
    ("Free gift for you", "spam"),
    ("Budget review meeting", "ham"),
]

# Create DataFrame
df = pd.DataFrame(emails, columns=['message', 'label'])

print("=" * 80)
print("NAIVE BAYES SPAM FILTER PROJECT")
print("=" * 80)
print(f"\nDataset Size: {len(df)} emails")
print(f"Spam emails: {len(df[df['label'] == 'spam'])}")
print(f"Ham (legitimate) emails: {len(df[df['label'] == 'ham'])}")

# Text preprocessing function
def preprocess_text(text):
    """Convert text to lowercase and extract words"""
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words

# Extract words from all emails
df['words'] = df['message'].apply(preprocess_text)

print("\n" + "=" * 80)
print("SAMPLE EMAILS")
print("=" * 80)
for i in range(3):
    print(f"\n{i+1}. Message: {df.iloc[i]['message']}")
    print(f"   Label: {df.iloc[i]['label'].upper()}")
    print(f"   Words: {df.iloc[i]['words']}")

# Calculate prior probabilities
total_emails = len(df)
spam_count = len(df[df['label'] == 'spam'])
ham_count = len(df[df['label'] == 'ham'])

P_spam = spam_count / total_emails
P_ham = ham_count / total_emails

print("\n" + "=" * 80)
print("PRIOR PROBABILITIES (Before seeing any words)")
print("=" * 80)
print(f"P(Spam) = {spam_count}/{total_emails} = {P_spam:.4f} ({P_spam*100:.1f}%)")
print(f"P(Ham)  = {ham_count}/{total_emails} = {P_ham:.4f} ({P_ham*100:.1f}%)")

# Build vocabulary and word counts
spam_words = []
ham_words = []

for idx, row in df.iterrows():
    if row['label'] == 'spam':
        spam_words.extend(row['words'])
    else:
        ham_words.extend(row['words'])

spam_word_count = Counter(spam_words)
ham_word_count = Counter(ham_words)
vocab = set(spam_words + ham_words)

print(f"\nVocabulary size: {len(vocab)} unique words")
print(f"Total words in spam emails: {len(spam_words)}")
print(f"Total words in ham emails: {len(ham_words)}")

# Calculate word probabilities with Laplace smoothing
def calculate_word_probability(word, word_count, total_words, vocab_size):
    """Calculate P(word | class) with Laplace smoothing"""
    return (word_count.get(word, 0) + 1) / (total_words + vocab_size)

# Focus on the word "buy" as an example
target_word = "buy"

P_buy_given_spam = calculate_word_probability(
    target_word, spam_word_count, len(spam_words), len(vocab)
)
P_buy_given_ham = calculate_word_probability(
    target_word, ham_word_count, len(ham_words), len(vocab)
)

print("\n" + "=" * 80)
print(f"WORD ANALYSIS: '{target_word.upper()}'")
print("=" * 80)
print(f"Occurrences in spam emails: {spam_word_count.get(target_word, 0)}")
print(f"Occurrences in ham emails: {ham_word_count.get(target_word, 0)}")
print(f"\nP('{target_word}' | Spam) = {P_buy_given_spam:.6f}")
print(f"P('{target_word}' | Ham)  = {P_buy_given_ham:.6f}")

# Apply Bayes' Theorem
print("\n" + "=" * 80)
print(f"BAYES' THEOREM: P(Spam | '{target_word}')")
print("=" * 80)
print("\nFormula:")
print("P(Spam | 'buy') = [P('buy' | Spam) × P(Spam)] / P('buy')")
print("\nWhere:")
print("P('buy') = P('buy' | Spam) × P(Spam) + P('buy' | Ham) × P(Ham)")

# Calculate P(buy)
P_buy = (P_buy_given_spam * P_spam) + (P_buy_given_ham * P_ham)

# Calculate P(Spam | buy) using Bayes' Theorem
P_spam_given_buy = (P_buy_given_spam * P_spam) / P_buy
P_ham_given_buy = (P_buy_given_ham * P_ham) / P_buy

print(f"\nCalculation:")
print(f"P('buy') = ({P_buy_given_spam:.6f} × {P_spam:.4f}) + ({P_buy_given_ham:.6f} × {P_ham:.4f})")
print(f"P('buy') = {P_buy:.6f}")
print(f"\nP(Spam | 'buy') = ({P_buy_given_spam:.6f} × {P_spam:.4f}) / {P_buy:.6f}")
print(f"P(Spam | 'buy') = {P_spam_given_buy:.4f} ({P_spam_given_buy*100:.1f}%)")
print(f"P(Ham | 'buy')  = {P_ham_given_buy:.4f} ({P_ham_given_buy*100:.1f}%)")

# Naive Bayes Classifier
def classify_email(message, spam_word_count, ham_word_count, P_spam, P_ham, vocab):
    """Classify email as spam or ham using Naive Bayes"""
    words = preprocess_text(message)
    
    # Calculate log probabilities to avoid underflow
    log_prob_spam = np.log(P_spam)
    log_prob_ham = np.log(P_ham)
    
    for word in words:
        p_word_spam = calculate_word_probability(
            word, spam_word_count, len(spam_words), len(vocab)
        )
        p_word_ham = calculate_word_probability(
            word, ham_word_count, len(ham_words), len(vocab)
        )
        
        log_prob_spam += np.log(p_word_spam)
        log_prob_ham += np.log(p_word_ham)
    
    # Convert back from log space
    prob_spam = np.exp(log_prob_spam)
    prob_ham = np.exp(log_prob_ham)
    
    # Normalize
    total = prob_spam + prob_ham
    prob_spam_normalized = prob_spam / total
    prob_ham_normalized = prob_ham / total
    
    return 'spam' if prob_spam_normalized > prob_ham_normalized else 'ham', prob_spam_normalized

# Test the classifier
test_messages = [
    "Buy cheap products now",
    "Meeting scheduled for tomorrow",
    "Free money and prizes",
    "Can you review this document"
]

print("\n" + "=" * 80)
print("CLASSIFIER PREDICTIONS")
print("=" * 80)
for msg in test_messages:
    prediction, spam_prob = classify_email(
        msg, spam_word_count, ham_word_count, P_spam, P_ham, vocab
    )
    print(f"\nMessage: '{msg}'")
    print(f"Prediction: {prediction.upper()}")
    print(f"Spam probability: {spam_prob*100:.1f}%")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Prior Probabilities
ax1 = axes[0, 0]
categories = ['Spam', 'Ham']
probs = [P_spam, P_ham]
colors = ['#ff6b6b', '#51cf66']
ax1.bar(categories, probs, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Prior Probabilities P(Spam) and P(Ham)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
for i, v in enumerate(probs):
    ax1.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

# 2. Word "buy" likelihood
ax2 = axes[0, 1]
word_probs = [P_buy_given_spam, P_buy_given_ham]
ax2.bar(categories, word_probs, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title(f'P("{target_word}" | Class) - Word Likelihood', fontsize=14, fontweight='bold')
for i, v in enumerate(word_probs):
    ax2.text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

# 3. Posterior Probabilities
ax3 = axes[1, 0]
posterior_probs = [P_spam_given_buy, P_ham_given_buy]
ax3.bar(categories, posterior_probs, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Probability', fontsize=12)
ax3.set_title(f'P(Class | "{target_word}") - Posterior Probabilities', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 1)
for i, v in enumerate(posterior_probs):
    ax3.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

# 4. Top words in spam vs ham
ax4 = axes[1, 1]
top_spam = spam_word_count.most_common(8)
top_ham = ham_word_count.most_common(8)

words_spam = [w[0] for w in top_spam]
counts_spam = [w[1] for w in top_spam]

y_pos = np.arange(len(words_spam))
ax4.barh(y_pos, counts_spam, color='#ff6b6b', alpha=0.7, edgecolor='black')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(words_spam)
ax4.set_xlabel('Frequency', fontsize=12)
ax4.set_title('Most Common Words in Spam Emails', fontsize=14, fontweight='bold')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('naive_bayes_spam_filter.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("Visualizations saved as 'naive_bayes_spam_filter.png'")
print("=" * 80)
plt.show()

# Create a summary table
summary_data = {
    'Metric': [
        'Total Emails',
        'Spam Emails',
        'Ham Emails',
        'P(Spam)',
        'P(Ham)',
        f'P("{target_word}" | Spam)',
        f'P("{target_word}" | Ham)',
        f'P(Spam | "{target_word}")',
        f'P(Ham | "{target_word}")'
    ],
    'Value': [
        total_emails,
        spam_count,
        ham_count,
        f'{P_spam:.4f}',
        f'{P_ham:.4f}',
        f'{P_buy_given_spam:.6f}',
        f'{P_buy_given_ham:.6f}',
        f'{P_spam_given_buy:.4f} ({P_spam_given_buy*100:.1f}%)',
        f'{P_ham_given_buy:.4f} ({P_ham_given_buy*100:.1f}%)'
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("KEY INSIGHTS")
print(f"1. The word '{target_word}' appears {spam_word_count.get(target_word, 0)} times in spam")
print(f"   but only {ham_word_count.get(target_word, 0)} times in legitimate emails.")
print(f"\n2. When we see the word '{target_word}', there's a {P_spam_given_buy*100:.1f}% chance")
print("   the email is spam (using Bayes' Theorem).")
print("\n3. Naive Bayes assumes word independence (the 'naive' assumption)")
print("   and multiplies probabilities of individual words.")
print("\n4. Laplace smoothing (+1) prevents zero probabilities for unseen words.")
