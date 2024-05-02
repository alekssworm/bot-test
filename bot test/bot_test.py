import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Example training data
training_data = [
    ("This is a wonderful day.", 'statement'),
    ("What is the weather today?", 'question'),
    ("I am watching television.", 'statement'),
    ("What do you think about this book?", 'question'),
    ("I have a cat.", 'statement'),
    ("What is your name?", 'question')
]

# Text preprocessing function
def preprocess_text(text):
    return text.lower()

# Splitting data into training and testing sets
X_train, y_train = zip(*[(preprocess_text(text), label) for text, label in training_data])

# Creating a pipeline with the model
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Function to plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Visualization of learning curve
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
title = "Learning Curves (Naive Bayes)"
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=4)

plt.show()
