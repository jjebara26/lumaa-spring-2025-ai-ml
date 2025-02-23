#All the imports below should get the code to run on any machine with python
#Name: Joseph Jebara
from datasets import load_dataset                           #To use the huggingface dataset
from sklearn.feature_extraction.text import TfidfVectorizer #For TF-IDF calculations
from sklearn.metrics.pairwise import cosine_similarity      #For cosine similarity calculations
import numpy as np 

def load_and_prepare_data():
    return load_dataset("matoupines/books", split="train") #The dataset is located here if you want to have a look: https://huggingface.co/datasets/matoupines/books

def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def recommend_books(user_input, vectorizer, tfidf_matrix, books, top_n=5):
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()    #Calculating the cosine similarity of the user description and all book descriptions
    top_indices = [int(i) for i in np.argsort(similarities)[-top_n:][::-1]] #This line gets the top_n indices of all cosine similarities

    print("Top Recommendations:")                                           #Printing the title, author, category, avg rating, description, and similarity rating for each recommended book
    for i, idx in enumerate(top_indices):
        print(f"{i + 1}. {books[idx]['title']} by {books[idx]['authors']}\n   Category: {books[idx]['category']}\n   Avg Rating: {books[idx]['avg_rating']}\n   Description: {books[idx]['description']}\n   Similarity: {similarities[idx]:.4f}\n")


print("Loading dataset...")
books = load_and_prepare_data()

print("Building TF-IDF model...")
vectorizer, tfidf_matrix = build_vectorizer(books["description"])

while True:
    user_input = input("Enter a description (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    print("Generating recommendations...")
    recommend_books(user_input, vectorizer, tfidf_matrix, books)
