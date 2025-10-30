from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
doc1 = "Information retrieval is the process of obtaining relevant information from a collection of resources."
doc2 = "Text mining and information retrieval are important components of data science."

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# Compute cosine similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Display similarity
print("Cosine Similarity between doc1 and doc2:", similarity[0][0])
