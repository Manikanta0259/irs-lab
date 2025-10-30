# Step 1: Create two simple documents
doc1 = "sachin virat dhoni anu"
doc2 = "anu chinna purna ram"
documents = [doc1, doc2]

# Step 2: Find all unique words (vocabulary)
unique_words = sorted(set(" ".join(documents).lower().split()))
print("Unique Words:", unique_words)

# Step 3: Create a mapping for each unique word to an index
word_index = {word: i for i, word in enumerate(unique_words)}

# Function to create a signature (vector) for a document
def create_signature(words):
    signature = [0] * len(unique_words)
    for word in words:
        if word in word_index:
            signature[word_index[word]] = 1
    return signature

# Step 4: Create signature for each document
doc_signatures = [create_signature(doc.lower().split()) for doc in documents]

# Display document signatures
print("\nDocument Signatures:")
for i, sig in enumerate(doc_signatures):
    print(f"Document {i+1}: {sig}")

# Step 5: Search for a word
query = input("\nEnter a word to search: ").lower()

if query in word_index:
    print(f"\nDocuments containing '{query}':")
    found = False
    for i, sig in enumerate(doc_signatures):
        if sig[word_index[query]] == 1:
            print(f"- Document {i+1}: \"{documents[i]}\"")
            found = True
    if not found:
        print("No documents found.")
else:
    print(f"The word '{query}' is not in the list of unique words.")
