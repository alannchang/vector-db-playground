from vector_store import VectorStore
import numpy as np

# Create a VectorStore instance
vector_store = VectorStore()

# Define your sentences
sentences = [
    "I eat mango",
    "Mango is my favorite fruit",
    "Mango, apple, oranges are fruits",
    "Fruits are good for health",
    "My dog's name is Mango",
    "I am allergic to mango",
    "Do you like mangoes",
    "I do not like mangoes",
    "Why are we talking about mangoes",
    "I have a mango tree in my backyard"
]

# Tokenization and Vocabulary Creation
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocabulary
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    sentence_vectors[sentence] = vector

# Storing in VectorStore
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Searching for Similarity
query_sentence = input("What is your mango related query? ")
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()
for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=2)

# Print similar sentences
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")
