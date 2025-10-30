# Step 1: Import libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np

# Step 2: Load a few categories from 20 Newsgroups dataset
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# Step 3: Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
X = vectorizer.fit_transform(data.data)
y_true = data.target  # Actual category labels

# Step 4: Apply K-Means Clustering
k = len(categories)  # Number of clusters = number of categories
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=42)
kmeans.fit(X)
y_pred = kmeans.labels_  # Predicted cluster labels

# Step 5: Define a function to calculate purity score
def purity_score(y_true, y_pred):
    table = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(table, axis=0)) / np.sum(table)

# Step 6: Evaluate clustering performance
print("Purity Score:", purity_score(y_true, y_pred))
