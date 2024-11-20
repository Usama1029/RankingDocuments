import os
import math
from collections import Counter

# Define document paths and titles
DOCUMENTS = {
    "file1.txt": "D:\\7th Semester\\IRASSIGNMENT\\documents\\climate.txt",
    "file2.txt": "D:\\7th Semester\\IRASSIGNMENT\\documents\\energy.txt",
    "file3.txt": "D:\\7th Semester\\IRASSIGNMENT\\documents\\data.txt",
    "file4.txt": "D:\\7th Semester\\IRASSIGNMENT\\documents\\blockchain.txt",
    "file5.txt": "D:\\7th Semester\\IRASSIGNMENT\\documents\\AI.txt"
}

TITLES = {
    "file1.txt": "Climate Change Overview",
    "file2.txt": "Energy Conservation Strategies",
    "file3.txt": "Data Science Introduction",
    "file4.txt": "Blockchain Basics",
    "file5.txt": "Artificial Intelligence Trends"
}

# Function to read document contents
def read_documents(documents):
    content = {}
    for filename, filepath in documents.items():
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().lower()  # Convert to lowercase for uniform matching
            content[filename] = text
    return content

# Tokenize the text and return a list of words
def tokenize(text):
    # Simple tokenizer that splits text by spaces and removes non-alphabetic characters
    tokens = [word.strip('.,!?()[]{}":;') for word in text.split() if word.isalpha()]
    return tokens

# Calculate keyword matching score based on frequency of query terms
def calculate_keyword_matching_score(query, document_tokens):
    query_words = query.lower().split()  # Tokenize the query
    score = 0
    
    for word in query_words:
        word_count = document_tokens.count(word)  # Count occurrences of the word
        score += word_count  # Add frequency to the score
    
    return score

# Calculate TF-IDF score for the query and document
def calculate_tfidf_score(query, document_tokens, total_documents, document_frequencies):
    query_words = query.lower().split()  # Tokenize the query
    score = 0
    total_words = len(document_tokens)
    
    for word in query_words:
        # Calculate term frequency (TF)
        term_count = document_tokens.count(word)
        tf = term_count / total_words if total_words else 0
        
        # Calculate inverse document frequency (IDF)
        doc_count = document_frequencies.get(word, 0)
        idf = math.log((total_documents / (1 + doc_count)), 10) if doc_count else 0
        
        # Add the TF-IDF score for this word
        score += tf * idf
    
    return score

# Build document frequencies for TF-IDF
def build_document_frequencies(documents):
    word_counts = Counter()
    total_documents = len(documents)
    
    for filename, content in documents.items():
        tokens = tokenize(content)
        unique_words = set(tokens)
        
        # Count unique words for this document
        for word in unique_words:
            word_counts[word] += 1
    
    return word_counts, total_documents

# Rank documents based on keyword matching
def rank_documents_by_keyword(query, documents):
    ranked_documents = []
    for filename, content in documents.items():
        document_tokens = tokenize(content)
        score = calculate_keyword_matching_score(query, document_tokens)
        ranked_documents.append((filename, TITLES.get(filename, "Untitled Document"), score))
    
    ranked_documents.sort(key=lambda x: x[2], reverse=True)  # Sort by score in descending order
    return ranked_documents

# Rank documents based on TF-IDF matching
def rank_documents_by_tfidf(query, documents, document_frequencies, total_documents):
    ranked_documents = []
    for filename, content in documents.items():
        document_tokens = tokenize(content)
        score = calculate_tfidf_score(query, document_tokens, total_documents, document_frequencies)
        ranked_documents.append((filename, TITLES.get(filename, "Untitled Document"), score))
    
    ranked_documents.sort(key=lambda x: x[2], reverse=True)  # Sort by score in descending order
    return ranked_documents

# Display the ranked documents
def display_ranked_documents(ranked_documents):
    if ranked_documents:
        print("\nDocuments ranked by relevance:")
        for filename, title, score in ranked_documents:
            print(f"- {title} ({filename}): Score = {score:.2f}")
    else:
        print("No documents found.")

# Main function with user interaction
def main():
    print("Reading and processing documents...")
    documents = read_documents(DOCUMENTS)
    
    # Build document frequencies for TF-IDF
    document_frequencies, total_documents = build_document_frequencies(documents)
    
    while True:
        print("\nSearch Menu:")
        print("1. Keyword Matching")
        print("2. TF-IDF Matching")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '1':
            query = input("Enter search query: ").strip()
            ranked_documents = rank_documents_by_keyword(query, documents)
            display_ranked_documents(ranked_documents)
            
        elif choice == '2':
            query = input("Enter search query: ").strip()
            ranked_documents = rank_documents_by_tfidf(query, documents, document_frequencies, total_documents)
            display_ranked_documents(ranked_documents)
            
        elif choice == '3':
            print("Exiting the document ranking system. Goodbye!")
            break
            
        else:
            print("Invalid choice, please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
