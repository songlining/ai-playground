from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import os


mongo_url = os.getenv("MONGO_URL")
client = MongoClient(mongo_url)
db = client["sample_mflix"]
movies_collection = db["movies"]

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype=torch.float16  # Use float16 for efficiency
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def generate_and_store_embeddings():
    """
    generate embedding vectors with 384 dimensions
    """
    count = 1
    for movie in movies_collection.find():
        print(f"Processing movie #{count}: {movie.get("title")}")
        plot_text = movie.get("fullplot") or movie.get("plot")
        if plot_text:
            embeddings = embed_model.encode(plot_text)
            movies_collection.update_one({"_id": movie["_id"]}, {"$set": {"embedding": embeddings.tolist()}})
            print("embedded")
        count = count + 1

generate_and_store_embeddings()