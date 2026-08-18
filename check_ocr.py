import pytesseract


import os

image_path = "fullpage.jpg"

# Check if the file exists
if os.path.exists(image_path):
    print("Image found at:", image_path)
else:
    print("Image not found.")

# ... rest of code ...
