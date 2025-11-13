# program 8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
# Sample Document Collection (you can load files instead)
documents = [
    "The cat sat on the mat.",
    "Dogs and cats are both pets.",
    "The dog chased the cat.",
    "Cats sleep on warm mats.",
    "Dogs bark and guard the house."
]
# Step 1: Convert to TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(documents)
# Step 2: Apply Truncated SVD (LSI)
num_topics = 2 # You can increase this number
svd_model = TruncatedSVD(n_components=num_topics, random_state=42)
X_lsi = svd_model.fit_transform(X_tfidf)
# Step 3: Display LSI Topics (term importance in each topic)
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd_model.components_):
    terms_in_topic = zip(terms, comp)
    sorted_terms = sorted(terms_in_topic, key=lambda x: x[1], reverse=True)
    print(f"\n Topic {i + 1}:")
    for term, weight in sorted_terms[:5]:
        print(f"  {term} ({weight:.4f})")
# Step 4: Show document-topic matrix
print("\n Document representation in reduced LSI space:")
for i, doc_vec in enumerate(X_lsi):
    print(f"Doc {i + 1}: {doc_vec}")
