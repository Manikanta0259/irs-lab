import spacy
from nltk.stem import PorterStemmer

# Load English model for text processing
nlp = spacy.load("en_core_web_sm")

# Step 1: Input text
text = "Text mining is the process of deriving meaningful information from natural language text."

# Step 2: Process the text using spaCy
doc = nlp(text)

# Step 3: Tokenize (split text into words) and remove stopwords like 'is', 'the', etc.
tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]

# Step 4: Apply stemming (convert words to their root form)
# Example: "running" → "run"
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Step 5: Display results
print("Original Text:\n", text)
print("\nTokens after Stop Word Removal:\n", tokens)
print("\nStemmed Tokens:\n", stemmed_tokens)
