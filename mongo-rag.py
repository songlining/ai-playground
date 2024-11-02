from pymongo import MongoClient
#from transformers import BertTokenizer, BertModel
#import torch
import openai
import os
#import numpy as np
from sentence_transformers import SentenceTransformer
# Load the embedding model (https://huggingface.co/nomic-ai/nomic-embed-text-v1")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    

mongo_url = os.getenv("MONGO_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")

# connec to MongoDB
client = MongoClient(mongo_url)
db = client["sample_mflix"]
movies_collection = db["embedded_movies"]

# Define a function to generate embeddings
def get_embedding(data):
    """Generates vector embeddings for the given data."""
    embedding = openai.embeddings.create(input=data, model="text-embedding-ada-002").data[0].embedding
    #embedding = model.encode(data)
    #l = embedding.tolist()
    print("Len of embedding list >>>>>>>>>>>>>>>>>>>>>>>>> ", len(embedding))
    return embedding

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# def generate_and_store_embeddings():
#     count = 1
#     for movie in movies_collection.find({"embedding": {"$exists": False}}):
#         print(f"Processing movie #{count}: {movie.get("title")}")
#         plot_text = movie.get("fullplot") or movie.get("plot")
#         if plot_text:
#              embedding = openai.embeddings.create(input=plot_text, model="text-embedding-ada-002").data[0].embedding
#              movies_collection.update_one({"_id": movie["_id"]}, {"$set": {"embedding": embedding}})
#              print("Embedded.")
#         count = count + 1


# def find_similar_movies(query_text, top_n=3):
#     query_embedding = openai.embeddings.create(input=query_text, model="text-embedding-ada-002").data[0].embedding
#     all_movies = movies_collection.find({"plot_embedding": {"$exists": True}})

#     similarities = []
#     for movie in all_movies:
#         movie_embedding = np.array(movie["plot_embedding"])
#         similarity = np.dot(query_embedding, movie_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(movie_embedding))
#         similarities.append((movie, similarity))
    
#     # Sort by similarity score and return the top N movies
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return [movie for movie, _ in similarities[:top_n]]

# Define a function to run vector search queries
def get_query_results(query):
  """Gets results from a vector search query."""

  query_embedding = get_embedding(query)
  pipeline = [
      {
            "$vectorSearch": {
              "index": "vector",
              "queryVector": query_embedding,
              "path": "plot_embedding",
              "exact": True,
              "limit": 5
            }
      }, {
            "$project": {
              "_id": 0,
              "text": 1
         }
      }
  ]

  results = movies_collection.aggregate(pipeline)

  array_of_results = []
  for doc in results:
      array_of_results.append(doc)
  return array_of_results


def generate_response_with_rag(query_text):
    similar_movies = get_query_results(query_text)
    context = " ".join([f"{movie.get('title')}: {movie.get('fullplot') or movie.get('plot')}" for movie in similar_movies])
    augmented_query = f"Context: {context}\n\nQuestion: {query_text}\n\nAnswer:"

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=augmented_query,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()


#main

# generate_and_store_embeddings()

query = "Can you recommend movies like The Great Train Robbery"
response = generate_response_with_rag(query)
print("Generated Response", response)