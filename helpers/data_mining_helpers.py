import nltk
import pandas as pd
import re
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)


def tokenize_text(text, remove_stopwords=True, min_length=2):
    """
    Tokenize text using NLTK, optionally removing stopwords and non-alphabetic tokens.
    
    Args:
        text (str): Input text to tokenize.
        remove_stopwords (bool): Whether to remove English stopwords.
        min_length (int): Minimum word length to keep.
    
    Returns:
        list: List of cleaned, lowercased tokens.
    """
    if not isinstance(text, str):
        return []
    
    # Tokenize into words
    tokens = nltk.word_tokenize(text.lower(), language='english')
    
    # Keep only alphabetic tokens of sufficient length
    tokens = [word for word in tokens if word.isalpha() and len(word) >= min_length]
    
    # Optionally remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    return tokens