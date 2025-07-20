import chromadb
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch # We need torch for the T5 model

# --- Configuration ---
DB_PATH = "vector_db"
COLLECTION_NAME = "historical_docs"
# We'll use a small, manageable text-to-text model from Google called T5
MODEL_NAME = 't5-small' 

# --- Initialization ---
print("Initializing models and database connection...")
# Initialize the embedding model (for searching)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the generative language model and its tokenizer (for answering)
# This will download the model the first time you run it.
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
llm_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Connect to the existing vector database
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print("Initialization complete.")

# --- Main Query Logic ---
def ask_question(question):
    """
    Takes a user's question, finds relevant context, and generates an answer.
    """
    print(f"\nProcessing question: '{question}'")
    
    # 1. Generate an embedding for the user's question
    question_embedding = embedding_model.encode([question])
    
    # 2. Query the database to find the most relevant text chunks
    # We'll ask for the top 3 most relevant chunks.
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    
    context_chunks = results['documents'][0]
    context = " ".join(context_chunks)
    
    print(f"\nFound relevant context: \n...{context[:500]}...") # Print a snippet of the context
    
    # 3. Construct the prompt for the language model
    # T5 models have a specific prompt format. We prepend "question: " and "context: "
    prompt = f"question: {question} context: {context}"
    
    # 4. Use the language model to generate an answer
    print("\nGenerating answer with the language model...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generate the answer
    outputs = llm_model.generate(input_ids, max_length=100) # You can adjust max_length
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# --- Execution ---
if __name__ == "__main__":
    # Let's ask a question based on the sample document text we saw earlier.
    # The OCR text was: "WATIONAL ARCHNES AMD RECORDS ADMESTAKTION..."
    # So, a good question would be about the "National Archives".
    
    test_question = "What is this document about?"
    
    final_answer = ask_question(test_question)
    
    print("\n" + "="*30)
    print(f"Question: {test_question}")
    print(f"Final Answer: {final_answer}")
    print("="*30)