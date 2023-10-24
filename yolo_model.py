# yolo_modelOOAB.pyA
from PIL import Image
from PIL import ImageDraw
import io
import os
import pandas as pd
import numpy as np
import cv2
from typing import Optional

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
import torch

import pytesseract

MODEL_PATH = "./models/sample_model/yolo8_best.pt"  # Define your model path here
_loaded_model = None  # This holds the model once it's loaded


def get_yolo_model() -> YOLO:
    """
    This function ensures that the model is loaded only once and reused for subsequent predictions.
    It's a form of lazy loading.

    Returns:
        YOLO: The YOLO model object ready for predictions.
    """
    global _loaded_model

    if _loaded_model is None:
        # Load the model if it hasn't been loaded yet
        _loaded_model = YOLO(MODEL_PATH)
    
    return _loaded_model

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, save_txt: bool = False ,image_size: int = 1248, conf: float = 0.5, augment: bool = False):
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save,
                        save_txt=save_txt ,
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        )
    
    return predictions

def draw_boxes_and_save(predictions, input_image: Image, output_path: str):
    """
    Draw bounding boxes on the input image based on the predictions and save the new image.

    Args:
        predictions: The predictions obtained from the YOLO model.
        input_image (Image): The original input image.
        output_path (str): Path to save the image with bounding boxes.
    """
    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(input_image)

    for pred in predictions:
        for *box, conf, cls in pred:  # x1, y1, x2, y2, confidence, class
            left, top, right, bottom = box
            draw.rectangle(((left, top), (right, bottom)), outline="red")  # You can customize the color

    # Save the image with bounding boxes
    input_image.save(output_path)

   
        
def detect_sample_model(input_image: Image,model_sample_model) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        save=False,
        save_txt=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    labels_dict = {
        0 : "bordered",
        1 : "borderless"
    }

    predict = extract_prediction_data(predict, labels_dict)
    return predict



def extract_prediction_data(results: list, labels_dict: dict):
    """
    Extract prediction data from the results provided by YOLO.

    Args:
        results (list): A list containing the predict output from YOLO.
        labels_dict (dict): A dictionary containing the labels names.

    Returns:
        list: A list of tuples, each containing:
              - (xmin, ymin, xmax, ymax): coordinates of the bounding box
              - confidence: confidence score of the prediction
              - class_id: class ID of the prediction
              - class_name: human-readable class name
    """
    # Extract bounding box coordinates
    boxes = results[0].to("cpu").numpy().boxes.xyxy  # Ensure this line is correct for your 'results' structure

    # Extract confidence scores
    confidences = results[0].to("cpu").numpy().boxes.conf

    # Extract class IDs and convert to integers
    class_ids = results[0].to("cpu").numpy().boxes.cls.astype(int)

    # Prepare a list to hold the processed predictions
    processed_predictions = []

    # Iterate over all extracted information
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        class_name = labels_dict.get(class_id, "Unknown")  # Get the human-readable name from the labels dictionary
        processed_predictions.append((box, confidence, class_id, class_name))

    return processed_predictions



def add_margin_to_bbox(xmin, ymin, xmax, ymax, margin, image_width, image_height):
    """
    Expands the bounding box by the specified margin, without exceeding the image boundaries.

    Parameters:
    xmin, ymin, xmax, ymax (int): Coordinates of the bounding box.
    margin (int): Margin to add around the bounding box.
    image_width, image_height (int): Dimensions of the image.

    Returns:
    tuple: New coordinates with the margin.
    """
    xmin = max(0, xmin - margin)  # Decrease xmin, but not less than 0
    ymin = max(0, ymin - margin)  # Decrease ymin, but not less than 0
    xmax = min(image_width, xmax + margin)  # Increase xmax, but not more than image width
    ymax = min(image_height, ymax + margin)  # Increase ymax, but not more than image height

    return xmin, ymin, xmax, ymax


def draw_bounding_boxes(image, predictions, output_path):
    """
    Draw bounding boxes on the image.

    Args:
    image_path (str): The file path of the image.
    predictions (list): A list of predictions, where each prediction
                        is a tuple containing:
                        - box: the coordinates of the bounding box (xmin, ymin, xmax, ymax).
                        - confidence: the confidence score.
                        - class_id: the class id.
                        - class_name: the class name.
     """     



    for coordinates, confidence, class_id, class_name in predictions:
        # Extract the bounding box coordinates
        coordinates = coordinates.astype(int)
        xmin, ymin, xmax, ymax = coordinates


        # When you have a bounding box and want to add a margin, call the function like this:

        # Define a margin
        margin = 15  # for example, 10 pixels

        # Get the dimensions of the image
        image_height, image_width = image.shape[:2]  # Assuming 'image' is your image object

        # Get new coordinates with margin
        xmin, ymin, xmax, ymax = add_margin_to_bbox(xmin, ymin, xmax, ymax, margin, image_width, image_height)
        # Now, you can use these new coordinates in your cv2.rectangle or any other function.
            
        # Draw a rectangle around the object
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Construct the label text
        label = f"{class_name}: {confidence:.2f}"

        # Choose a font for the label text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Put the label text above the rectangle
        cv2.putText(image, label, (xmin, ymin - 10), font, 0.5, (0, 255, 0), 2)

    # Save the image with bounding boxes
    #output_path = 'output_image.jpg'  # Specify the output path here
    cv2.imwrite(output_path, image)

    return (output_path)  # Optionally return the output file path



def perform_ocr(image, predictions):
    """
    Perform OCR on detected bounding boxes within the image.

    Args:
        image: The loaded image on which OCR should be performed.
        predictions: List of bounding box predictions with their coordinates.
    
    Returns:
        List of dictionaries containing OCR results for each bounding box.
    """
    ocr_results = []


    for coordinates, confidence, class_id, class_name in predictions:
        # Extract the bounding box coordinates
        coordinates = coordinates.astype(int)
        xmin, ymin, xmax, ymax = coordinates


        # When you have a bounding box and want to add a margin, call the function like this:

        # Define a margin
        margin = 15  # for example, 10 pixels

        # Get the dimensions of the image
        image_height, image_width = image.shape[:2]  # Assuming 'image' is your image object

        # Get new coordinates with margin
        xmin, ymin, xmax, ymax = add_margin_to_bbox(xmin, ymin, xmax, ymax, margin, image_width, image_height)
        # Now, you can use these new coordinates in your cv2.rectangle or any other function.
       
        roi = image[ymin:ymax, xmin:xmax]  # region of interest
        text = pytesseract.image_to_string(roi)

        ocr_result = {
            #'coordinates': coordinates,
            'text': text.strip()  # Remove whitespace
        }
        ocr_results.append(ocr_result)

    return ocr_results




from fastapi import HTTPException

def process_with_yolo(image_path: str, base_output_dir: str):
    """
    Process the image with the YOLO model and save the processed images in the same base directory.

    Args:
        image_path (str): The path of the image to process.
        base_output_dir (str): The base directory where processed images will be saved.

    Returns:
        str: Path to the saved processed image.
    """

    # Load model and image
    sample_model = get_yolo_model()
    try:
        input_image = Image.open(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading image: {e}")

    # Get predictions
    predictions = detect_sample_model(input_image,sample_model)

    print("predictions", predictions)
    # Draw boxes and save image
    processed_images_dir = os.path.join(base_output_dir, "processed_images")
    #os.makedirs(processed_images_dir, exist_ok=True) # i remove this now 

    processed_image_name = os.path.basename(image_path)  # keep original image name
    #processed_image_path = os.path.join(processed_images_dir, processed_image_name) # i remove this now 
    processed_image_path = os.path.join(base_output_dir, processed_image_name)
    #processed_image_path  = processed_image_name
    print("dir_name",base_output_dir,processed_image_name,processed_image_path)
    image = cv2.imread(image_path)
    draw_bounding_boxes(image,predictions, processed_image_path)
    ocr_results = perform_ocr(image, predictions)
    return (processed_image_name,ocr_results)  # or return some info based on your need





