import pytesseract
from PIL import Image

# If you're on Windows, you might need to tell pytesseract where to find the executable
# Uncomment the line below if you get an error, and make sure the path is correct
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version {version} is installed and accessible.")
    print("\nSetup is successful! We can now move to the next phase.")
except pytesseract.TesseractNotFoundError:
    print("--- TESSERACT NOT FOUND ---")
    print("Pytesseract cannot find the Tesseract-OCR installation.")
    print("Please make sure you checked 'Add Tesseract to system PATH' during installation.")
    print("If you missed it, you can either reinstall Tesseract or uncomment the line in the script with the correct path to tesseract.exe.")