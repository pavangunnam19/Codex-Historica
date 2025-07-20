import pytesseract
from PIL import Image
import os # We'll use this to check if the file exists

# --- Configuration ---
# This is the same line from our check script. It's good practice
# to have it in our main script as well.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
IMAGE_PATH = 'sample_doc.png'

# --- Main OCR Logic ---
def extract_text_from_image(image_path):
    """
    Opens an image file, extracts text using Tesseract OCR, and returns the text.
    """
    # Check if the file exists at the given path
    if not os.path.exists(image_path):
        return f"Error: File not found at '{image_path}'"

    try:
        # Use Pillow to open the image
        img = Image.open(image_path)
        
        # Use pytesseract to extract the text.
        # We specify 'eng' for English language.
        text = pytesseract.image_to_string(img, lang='eng')
        
        return text
    except Exception as e:
        return f"An error occurred: {e}"

# --- Execution ---
if __name__ == "__main__":
    print(f"Attempting to extract text from '{IMAGE_PATH}'...")
    print("-" * 30)
    
    extracted_text = extract_text_from_image(IMAGE_PATH)
    
    print("Extracted Text:")
    print(extracted_text)
    print("-" * 30)
    print("OCR process finished.")