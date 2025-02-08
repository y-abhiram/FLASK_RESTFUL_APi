import os
import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

# Disable GPU for this example (if desired)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask App
app = Flask(__name__)

# Load Sentence Transformer Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# Data Loading and Splitting Functions
# -------------------------------

def load_and_split_courses(url):
    """
    Loads content from the URL using LangChain's UnstructuredURLLoader and splits it into chunks.
    """
    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()
    texts = [doc.page_content for doc in docs]

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

# -------------------------------
# Create FAISS Vector Store
# -------------------------------

def create_vector_store(texts):
    if not texts:
        raise ValueError("No data to store in FAISS!")
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS L2 index
    index.add(embeddings)
    return index, texts

# -------------------------------
# Load and Index Course Data
# -------------------------------

COURSE_URL = "https://brainlox.com/courses/category/technical"
try:
    course_chunks = load_and_split_courses(COURSE_URL)
    if course_chunks:
        vector_store, stored_texts = create_vector_store(course_chunks)
        print(" Data loaded and vector store created successfully!")
    else:
        raise Exception("No course chunks found at the URL!")
except Exception as e:
    print(f" Error loading data: {e}")
    vector_store = None
    stored_texts = []

# -------------------------------
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Course Query API. Use the /query endpoint to search for courses."

# Flask API Endpoint for Querying
# -------------------------------

@app.route('/query', methods=['POST'])
def query():
    try:
        if not vector_store or not stored_texts:
            return jsonify({"error": "Vector store is not initialized."}), 500

        user_data = request.get_json()
        if not user_data or "query" not in user_data:
            return jsonify({"error": "Query parameter is missing!"}), 400

        user_query = user_data["query"].strip()
        if not user_query:
            return jsonify({"error": "Query cannot be empty!"}), 400

        # Convert query to embedding
        query_embedding = model.encode([user_query], convert_to_numpy=True)

        # FAISS search for the 5 nearest neighbors
        distances, indices = vector_store.search(query_embedding, k=5)
        results = [stored_texts[i] for i in indices[0] if i < len(stored_texts)]

        if not results:
            return jsonify({"error": "No matching results found."}), 404

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# Run Flask Server
# -------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

