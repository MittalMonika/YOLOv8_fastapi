import os
from pdf2image import convert_from_path
from fastapi import UploadFile

def save_pdf_as_images(uploaded_file: UploadFile, base_path: str):
    # Create directories
    document_name = uploaded_file.filename
    output_dir = os.path.join(base_path, "Pdf_To_Images", document_name.replace(".pdf", ""))
    os.makedirs(output_dir, exist_ok=True)

    # Save the PDF file temporarily
    temp_pdf = os.path.join(base_path, document_name)
    with open(temp_pdf, "wb") as temp:
        temp.write(uploaded_file.file.read())  # Assuming the file is not too large to fit in memory

    # Convert PDF to a list of PNG images
    images = convert_from_path(temp_pdf)

    # Save PNG images to files
    for i, image in enumerate(images):
        image_filename = os.path.join(output_dir, f"output_page_{i + 1}.png")
        image.save(image_filename, "PNG")

    # Optional: Remove the temporary PDF file if it's no longer needed
    os.remove(temp_pdf)

    print(f"The images are saved in {output_dir}")
    return output_dir  # This is the path where the images are stored
