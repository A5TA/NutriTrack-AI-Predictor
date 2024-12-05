import io
from flask import Flask, request, jsonify
import sys
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
import os
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

# The class labels as an array
LABELS = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 
    'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 
    'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 
    'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 
    'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 
    'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 
    'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 
    'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 
    'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 
    'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

# Load the pre-trained model
MODEL_PATH = "food101_mobilenet_v3_2.pth" 
# device = torch.device("cpu")  # Use CPU for inference

# Initialize and load the model
model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LABELS))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define image transformation
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    """
    Predict the class of an image from raw bytes.
    """
    try:
        # Open the image from the uploaded file content
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Error loading image: {str(e)}"}

    try:
        # Apply the necessary transformations
        transformed_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Ensure the tensor is on the same device as the model
        device = torch.device("cpu")  # Adjust if using GPU
        transformed_image = transformed_image.to(device)
        # print(f"Tensor shape: {transformed_image.shape}")
        # Perform inference
        with torch.no_grad():
            outputs = model(transformed_image)

        # Get the predicted class
        _, predicted_class = torch.max(outputs, 1)
        predicted_label = LABELS[predicted_class.item()]
        return {"predicted_class": predicted_label}
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

# POST endpoint to upload an image and get the predicted class.
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Load the image file for prediction
        image = file.read()  # Read the file content
        predicted_label = predict_image(image)
        return jsonify({"predicted_class": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# GET endpoint to test api
@app.route('/hello', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3050)