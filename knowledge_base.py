import pytesseract
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import os

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
IMAGE_PATH = 'sample_doc.png'
DB_PATH = "vector_db" # Path to store the database
COLLECTION_NAME = "historical_docs"

# --- 1. OCR Function (from our previous script) ---
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'")
        return None
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        return text
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return None

# --- 2. Text Chunking Function ---
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks.
    A simple strategy is to split by paragraphs (double newlines) first.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    for p in paragraphs:
        if len(p.strip()) > 0: # Ignore empty paragraphs
            chunks.append(p.strip().replace("\n", " ")) # Replace single newlines with spaces
    return chunks

# --- 3. Main Knowledge Base Creation Logic ---
def create_knowledge_base():
    print("Starting knowledge base creation...")

    # Step 1: Extract text from the document
    print(f"Extracting text from {IMAGE_PATH}...")
    document_text = extract_text_from_image(IMAGE_PATH)
    if not document_text:
        print("Text extraction failed. Aborting.")
        return

    # Step 2: Chunk the text
    print("Chunking extracted text...")
    text_chunks = chunk_text(document_text)
    if not text_chunks:
        print("No text chunks were created. Aborting.")
        return
    print(f"Created {len(text_chunks)} text chunks.")

    # Step 3: Initialize the embedding model
    print("Initializing sentence-transformer model...")
    # This will download the model the first time you run it
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    # Step 4: Generate embeddings for each chunk
    print("Generating embeddings for each chunk...")
    embeddings = model.encode(text_chunks)
    
    # Step 5: Initialize and populate the vector database
    print("Initializing vector database (ChromaDB)...")
    # This creates a persistent database in the 'vector_db' folder
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Get or create the collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Generate IDs for each chunk
    chunk_ids = [str(i) for i in range(len(text_chunks))]
    
    # Step 6: Add the data to the collection
    print(f"Adding {len(text_chunks)} chunks to the '{COLLECTION_NAME}' collection...")
    collection.add(
        embeddings=embeddings,
        documents=text_chunks,
        ids=chunk_ids
    )
    
    print("-" * 30)
    print("Knowledge base created successfully!")
    print(f"Data is stored in the '{DB_PATH}' directory.")
    print(f"Total documents in collection: {collection.count()}")
    print("-" * 30)

# --- Execution ---
if __name__ == "__main__":
    create_knowledge_base()