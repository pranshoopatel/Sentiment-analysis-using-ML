import ast
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from time import time
import json
import pickle
from matplotlib import pyplot as plt
from textblob import TextBlob

# Load review data from JSON file
reviewText = []
reviewRating = []

with open('Reviews_Digital_Music.json', 'r') as fileHandler:
    reviewDatas = fileHandler.read().split('\n')
    for review in reviewDatas:
        if review == "":
            continue
        r = json.loads(review)
        reviewText.append(r['reviewText'])
        reviewRating.append(r['overall'])

# Save to pickle files for later use (optional)
with open('review_text.pkl', 'wb') as saveReviewText, open('review_rating.pkl', 'wb') as saveReviewRating:
    pickle.dump(reviewText, saveReviewText)
    pickle.dump(reviewRating, saveReviewRating)

# Convert ratings to a numpy array for easier manipulation
ratings = np.array(reviewRating)

# Plot histogram of ratings
plt.hist(ratings, bins=np.arange(ratings.min(), ratings.max() + 2) - 0.5, rwidth=0.7)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Ratings', fontsize=18)
plt.show()


def getInitialData(data_file):
    print('Fetching initial data...')
    t = time()
    i = 0
    df = {}
    with open(data_file, 'r') as file_handler:
        for review in file_handler.readlines():
            df[i] = ast.literal_eval(review)
            i += 1
    reviews_df = pd.DataFrame.from_dict(df, orient='index')
    reviews_df.to_pickle('reviews_digital_music.pickle')
    print('Fetching data completed!')
    print('Fetching time: ', round(time()-t, 3), 's\n')

def prepareData(reviews_df):
    print('Preparing data...')
    t = time()
    reviews_df.rename(columns={"overall": "reviewRating"}, inplace=True)
    reviews_df.drop(columns=['reviewerID', 'asin', 'reviewerName', 'helpful', 'summary', 'unixReviewTime', 'reviewTime'], inplace=True)
    reviews_df = reviews_df[reviews_df.reviewRating != 3.0]  # Ignoring 3-star reviews -> neutral
    reviews_df = reviews_df.assign(sentiment=np.where(reviews_df['reviewRating'] >= 4.0, 1, 0))  # 1 -> Positive, 0 -> Negative
    stemmer = SnowballStemmer('english')
    stop_words = stopwords.words('english')
    reviews_df['cleaned'] = reviews_df['reviewText'].apply(
        lambda text: ' '.join([stemmer.stem(w) for w in re.sub('[^a-z]+|(quot)+', ' ', text.lower()).split() if w not in stop_words])
    )
    reviews_df.to_pickle('reviews_digital_music_preprocessed.pickle')
    print('Preparing data completed!')
    print('Preparing time: ', round(time()-t, 3), 's\n')

def preprocessData(reviews_df_preprocessed):
    print('Preprocessing data...')
    t = time()
    if 'cleaned' not in reviews_df_preprocessed.columns or 'sentiment' not in reviews_df_preprocessed.columns:
        raise KeyError("Expected columns 'cleaned' or 'sentiment' are missing in the DataFrame.")
    
    X = reviews_df_preprocessed['cleaned'].values
    y = reviews_df_preprocessed['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Preprocessing data completed!')
    print('Preprocessing time: ', round(time()-t, 3), 's\n')
    return X_train, X_test, y_train, y_test

def evaluate(y_test, prediction):
    print('Evaluating results...')
    t = time()
    scores = {
        'Accuracy': accuracy_score(y_test, prediction),
        'Precision': precision_score(y_test, prediction),
        'Recall': recall_score(y_test, prediction),
        'F1 Score': f1_score(y_test, prediction),
        'ROC AUC': roc_auc_score(y_test, prediction)
    }
    print(f'Accuracy: {scores["Accuracy"]:.2f}')
    print(f'Precision: {scores["Precision"]:.2f}')
    print(f'Recall: {scores["Recall"]:.2f}')
    print(f'F1 Score: {scores["F1 Score"]:.2f}')
    print(f'ROC AUC: {scores["ROC AUC"]:.2f}')
    print('Results evaluated!')
    print('Evaluation time: ', round(time()-t, 3), 's\n')
    return scores

# Load initial data
getInitialData('Reviews_Digital_Music.json')
reviews_df = pd.read_pickle('reviews_digital_music.pickle')

# Prepare data
prepareData(reviews_df)

# Load and prepare data
reviews_df_preprocessed = pd.read_pickle('reviews_digital_music_preprocessed.pickle')
X_train, X_test, y_train, y_test = preprocessData(reviews_df_preprocessed)

# Define models
models = {
    'MultinomialNB': Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
        ('chi', SelectKBest(score_func=chi2, k=50000)),
        ('clf', MultinomialNB())
    ]),
    'LogisticRegression': Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
        ('chi', SelectKBest(score_func=chi2, k=50000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ]),
    'LinearSVC': Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
        ('chi', SelectKBest(score_func=chi2, k=50000)),
        ('clf', LinearSVC(C=1.0, penalty='l2', max_iter=3000, class_weight='balanced'))
    ])
}

# Initialize a dictionary to store scores for plotting
all_scores = {model: {} for model in models.keys()}

# Train and evaluate models
for name, model in models.items():
    print(f'Training {name}...')
    t = time()
    model.fit(X_train, y_train)
    print(f'Training {name} completed!')
    print(f'Training time for {name}: ', round(time()-t, 3), 's\n')

    print(f'Predicting with {name}...')
    t = time()
    prediction = model.predict(X_test)
    print(f'Prediction with {name} completed!')
    print(f'Prediction time for {name}: ', round(time()-t, 3), 's\n')

    # Evaluate and store scores
    scores = evaluate(y_test, prediction)
    all_scores[name] = scores

    # Print confusion matrix and observation details
    print('Confusion matrix for {}: {}'.format(name, confusion_matrix(y_test, prediction)))
    l = len(y_test)
    s = y_test.sum()
    print(f'Total number of observations: {l}')
    print(f'Positives in observations: {s}')
    print(f'Negatives in observations: {l - s}')
    print(f'Majority class is: {s / l * 100:.2f}%')
    print('---'*20)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors
colors = {'MultinomialNB': 'red', 'LogisticRegression': 'blue', 'LinearSVC': 'yellow'}

# Define score labels and their positions
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
x = np.arange(len(labels))  # the label locations

# Get the list of model names to index correctly
model_names = list(colors.keys())

# Plot bars for each model
for i, model in enumerate(model_names):
    scores = [all_scores[model].get(label, 0) * 100 for label in labels]
    ax.bar(x + i * 0.2, scores, width=0.2, color=colors[model], label=model)

# Add some text for labels, title and custom x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores (%)')
ax.set_title('Performance of Different Models')
ax.set_xticks(x + 0.2)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
