from pymongo import MongoClient
import openai
import os

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
    return embedding


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
query = "Can you recommend movies like The Great Train Robbery"
response = generate_response_with_rag(query)
print("Generated Response", response)