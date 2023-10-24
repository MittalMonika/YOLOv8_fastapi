import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
from PIL import Image  # required for loading the images

from pdfutils import save_pdf_as_images
from yolo_model import process_with_yolo
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up a specific folder to serve static files. This folder should exist in your project directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/process_pdf/")
async def process_pdf(request: Request, uploaded_file: UploadFile = File(...)):
    base_path = os.getcwd()
    output_dir = save_pdf_as_images(uploaded_file, base_path)

    # We should save processed images to the 'static' folder so they can be served as static files.
    static_images_dir = os.path.join("static", "processed_images")
#    static_images_dir = os.path.join(base_path, "static", "processed_images")
    print(static_images_dir,"static_images_dir")
    #static_images_dir = os.path.join(base_path, "static")
    os.makedirs(static_images_dir, exist_ok=True)  # Ensure the directory exists.

    processed_image_urls = []
    ocr_results_all_images = []
    for image_file in os.listdir(output_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_dir, image_file)

            # Process the image and save the new image to the 'static' folder.
            processed_image_name, ocr_results = process_with_yolo(image_path, static_images_dir)
            processed_image_path = os.path.join(static_images_dir, processed_image_name)

            # Generate URL for the processed image.
            #image_url = request.url_for('static', path=processed_image_name)
            # Generate URL for the processed image.
            image_url = request.url_for('static', path=f"processed_images/{processed_image_name}")


            processed_image_urls.append(image_url)

            ocr_results_all_images.extend(ocr_results)

    # Now, you return URLs instead of file paths.
    return {
        "processed_images": processed_image_urls,
        "ocr_results": ocr_results_all_images
    }
