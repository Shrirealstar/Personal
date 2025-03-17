import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def preprocess(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Script Validation (Remove non-alphabetic characters)
    tokens = word_tokenize(text.lower())  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Stop Word Removal
    stemmed = [PorterStemmer().stem(word) for word in tokens]  # Stemming
    return stemmed

text = "Hello! This is an example sentence for NLP preprocessing in Python."
print(preprocess(text))


import numpy as np

def min_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,  # Deletion
                dp[i][j-1] + 1,  # Insertion
                dp[i-1][j-1] + (s1[i-1] != s2[j-1])  # Substitution
            )
    return dp[m][n]

print(min_edit_distance("kitten",   "sitting"))

from collections import Counter
import numpy as np

docs = [
    ("fun couple love love", "comedy"),
    ("fast furious shoot", "action"),
    ("couple fly fast fun fun", "comedy"),
    ("furious shoot shoot fun", "action"),
    ("fly fast shoot love", "action")
]

vocab = set(word for doc, _ in docs for word in doc.split())
word_counts = {"comedy": Counter(), "action": Counter()}
class_counts = Counter()

for doc, label in docs:
    word_counts[label].update(doc.split())
    class_counts[label] += 1

def predict(doc):
    words = doc.split()
    total_docs = sum(class_counts.values())
    probs = {c: np.log(class_counts[c] / total_docs) for c in class_counts}
    for c in class_counts:
        for word in words:
            probs[c] += np.log((word_counts[c][word] + 1) / (sum(word_counts[c].values()) + len(vocab)))
    return max(probs, key=probs.get)

print(predict("fast couple shoot fly"))

from nltk.corpus import wordnet
import nltk

def get_synonyms_antonyms(word):
    synonyms, antonyms = set(), set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    return synonyms, antonyms

synonyms, antonyms = get_synonyms_antonyms("active")
print("Synonyms:", synonyms)
print("Antonyms:", antonyms)
