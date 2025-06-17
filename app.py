from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import numpy as np
import uvicorn
import tempfile
import os
from io import BytesIO
from PIL import Image
import logging
import cv2
import matplotlib.pyplot as plt
import base64
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cancer Prediction API",
    description="API for predicting whether an image shows cancerous or non-cancerous tissue",
    version="1.0.0"
)

# Global model variable
model = None
last_conv_layer = None

# Model configuration
MODEL_FILE_ID = "https://drive.google.com/file/d/1GhBRt-ZuvH6zjKzWQ1oX2jGrr-9GnqA-/view?usp=sharing"  # Replace with your actual file ID
MODEL_LOCAL_PATH = "best_model.keras"

def download_model_from_gdrive(file_id, destination):
    """
    Download model from Google Drive using the file ID
    
    Parameters:
    -----------
    file_id : str
        Google Drive file ID
    destination : str
        Local path where the model should be saved
    
    Returns:
    --------
    bool
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading model from Google Drive (ID: {file_id})")
        
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # First request to get the download confirmation
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle the download confirmation for large files
        if "download_warning" in response.text or "virus scan warning" in response.text:
            # Extract the confirmation token
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            else:
                # Try to find token in the response text
                import re
                token_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', response.text)
                if token_match:
                    token = token_match.group(1)
                else:
                    logger.error("Could not find confirmation token")
                    return False
            
            # Make the actual download request with confirmation
            url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
            response = session.get(url, stream=True)
        
        # Check if the response is successful
        if response.status_code == 200:
            # Save the file
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Model downloaded successfully to {destination}")
            return True
        else:
            logger.error(f"Failed to download model. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def ensure_model_exists():
    """
    Ensure the model file exists locally. Download from Google Drive if not found.
    
    Returns:
    --------
    bool
        True if model file exists or was downloaded successfully
    """
    if os.path.exists(MODEL_LOCAL_PATH):
        logger.info(f"Model file found locally at {MODEL_LOCAL_PATH}")
        return True
    
    logger.info("Model file not found locally. Attempting to download from Google Drive...")
    
    if MODEL_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        logger.error("Google Drive file ID not configured. Please set MODEL_FILE_ID.")
        return False
    
    return download_model_from_gdrive(MODEL_FILE_ID, MODEL_LOCAL_PATH)

def normalize_for_network(image):
    """
    Normalize image for neural network input
    """
    # Scale to [0,1]
    image = image.astype(np.float32) / 255.0
    
    # Normalize using ImageNet mean and std for transfer learning
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    return image

def predict_single_image(image_path, model, img_size=(224, 224)):
    """
    Predict whether a single image is benign or adenocarcinoma
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : keras.Model
        Trained model for prediction
    img_size : tuple
        Size to which the image will be resized (default: (224, 224))
    
    Returns:
    --------
    dict
        Prediction results including class label and probability
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalize image
    img_array = normalize_for_network(img_array)
    
    # Expand dimensions to create batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Define class label based on prediction
    class_label = "Benign" if prediction >= 0.5 else "Adenocarcinoma"
    probability = float(1 - prediction) if class_label == "Adenocarcinoma" else float(prediction)
    
    return {
        "class": class_label,
        "probability": probability,
        "raw_score": float(prediction)
    }

def process_image_file(file_content, img_size=(224, 224)):
    """Process image from file content and prepare for prediction."""
    try:
        # Open image from BytesIO
        img = Image.open(BytesIO(file_content))
        
        # Convert to RGB if needed (in case of RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(img_size)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Save original image array for GradCAM visualization
        orig_img_array = img_array.copy()
        
        # Normalize
        norm_img_array = normalize_for_network(img_array)
        
        # Expand dimensions for model input
        input_img_array = np.expand_dims(norm_img_array, axis=0)
        
        return input_img_array, orig_img_array
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Cannot process image: {str(e)}")

def gradcam(model, img_array, layer_name='conv5_block3_out', pred_index=None):
    """
    Generate Grad-CAM heatmap for model's decision with improved visualization

    Parameters:
    -----------
    model : keras.Model
        Trained model for prediction
    img_array : numpy.ndarray
        Preprocessed image as numpy array
    layer_name : str
        Name of the last convolutional layer in the model
    pred_index : int or None
        Index of the class to visualize. If None, uses the highest scoring class.

    Returns:
    --------
    numpy.ndarray
        Heatmap as a numpy array
    """
    # First, create a model that maps the input image to the activations
    # of the last conv layer and output of the prediction layer
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            # For binary classification, we want to use the actual class index
            # For a sigmoid output, we use 0 since there's only one output neuron
            pred_index = 0

        # For binary classification with sigmoid:
        if predictions.shape[1] == 1:
            # For "Benign" prediction (high score)
            if predictions[0][0] >= 0.5:
                class_channel = predictions[:, pred_index]
            # For "Adenocarcinoma" prediction (low score)
            else:
                # Invert the gradient direction for the low-score class
                class_channel = 1.0 - predictions[:, pred_index]
        else:
            class_channel = predictions[:, pred_index]

    # Extract gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Average gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by how important it is
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    # Apply a minimum threshold to make weak activations more visible
    # Only apply significant thresholding if the max value is above a certain level
    if np.max(heatmap) > 0.2:
        heatmap = np.maximum(heatmap, 0.1 * np.max(heatmap))

    return heatmap

def generate_gradcam_visualization(img_array, orig_img_array, model, layer_name, alpha=0.6):
    """
    Generate Grad-CAM visualization for a given image
    
    Parameters:
    -----------
    img_array : numpy.ndarray
        Preprocessed image array (normalized, expanded dimensions)
    orig_img_array : numpy.ndarray
        Original image array
    model : keras.Model
        The trained model
    layer_name : str
        Name of the layer to use for Grad-CAM
    alpha : float
        Transparency of the heatmap overlay
        
    Returns:
    --------
    tuple
        (visualization image as numpy array, prediction result dict)
    """
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Define class label based on prediction
    class_label = "Benign" if prediction >= 0.5 else "Adenocarcinoma"
    probability = float(prediction) if class_label == "Benign" else float(1 - prediction)
    
    # Generate heatmap
    heatmap = gradcam(model, img_array, layer_name)
    
    # Load the original image and normalize to [0,1]
    img = orig_img_array / 255.0
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply Gaussian blur to smooth the heatmap (optional)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert from BGR to RGB (since OpenCV uses BGR)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img * 255 * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display original image
    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')
    
    # Display image with Grad-CAM overlay
    ax2.imshow(superimposed_img)
    ax2.set_title("Grad-CAM Visualization", fontsize=14)
    ax2.axis('off')
    
    # Add a colored border based on the prediction
    color = 'red' if class_label == 'Adenocarcinoma' else 'green'
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # Return the figure and prediction result
    return buf, {
        "class": class_label,
        "probability": probability,
        "raw_score": float(prediction)
    }

@app.on_event("startup")
async def load_model():
    """Load the model on startup and find the last conv layer."""
    global model, last_conv_layer
    try:
        # Ensure model file exists (download if necessary)
        if not ensure_model_exists():
            logger.error("Failed to ensure model file exists")
            return
        
        logger.info(f"Loading model from {MODEL_LOCAL_PATH}")
        model = tf.keras.models.load_model(MODEL_LOCAL_PATH)
        
        # Find the last convolutional layer in the model
        for layer in reversed(model.layers):
            if 'conv' in layer.name and 'output' not in layer.name:
                last_conv_layer = layer.name
                logger.info(f"Using layer for GradCAM: {last_conv_layer}")
                break
        
        if not last_conv_layer:
            # Fallback to a common layer name in ResNet50
            last_conv_layer = 'conv5_block3_out'
            logger.info(f"No conv layer found, using default: {last_conv_layer}")
            
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # We'll continue without the model, but predictions will fail
        model = None
        last_conv_layer = None

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "Cancer Prediction API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not loaded"}
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image shows cancerous or non-cancerous tissue
    
    Parameters:
    -----------
    file : UploadFile
        The image file to analyze
    
    Returns:
    --------
    dict
        Prediction results including class label and probability
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file contents
        contents = await file.read()
        
        # For small images, we can process directly in memory
        if len(contents) < 10 * 1024 * 1024:  # Less than 10MB
            img_array, _ = process_image_file(contents)
            prediction = model.predict(img_array)[0][0]
            
            # Define class label based on prediction
            class_label = "Benign" if prediction >= 0.5 else "Adenocarcinoma"
            probability = float(1 - prediction) if class_label == "Adenocarcinoma" else float(prediction)
            
            return {
                "filename": file.filename,
                "class": class_label,
                "probability": probability,
                "raw_score": float(prediction)
            }
        
        # For larger images, use temporary file
        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(contents)
                temp_path = temp.name
            
            try:
                result = predict_single_image(temp_path, model)
                return {
                    "filename": file.filename,
                    **result
                }
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-with-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Predict whether an image shows cancerous or non-cancerous tissue and return GradCAM visualization
    
    Parameters:
    -----------
    file : UploadFile
        The image file to analyze
    
    Returns:
    --------
    FileResponse
        GradCAM visualization image
    """
    # Check if model is loaded
    if model is None or last_conv_layer is None:
        raise HTTPException(status_code=503, detail="Model or layer not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Process directly in memory for small images
        if len(contents) < 10 * 1024 * 1024:  # Less than 10MB
            img_array, orig_img_array = process_image_file(contents)
            
            # Generate GradCAM visualization
            viz_image_buf, result = generate_gradcam_visualization(
                img_array, orig_img_array, model, last_conv_layer
            )
            
            # Save visualization to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
                temp.write(viz_image_buf.getvalue())
                temp_path = temp.name
            
            # Return the file with visualization
            return FileResponse(
                temp_path,
                media_type="image/png",
                headers={"X-Prediction": result["class"], 
                         "X-Probability": str(result["probability"]),
                         "X-Filename": file.filename}
            )
        
        # For larger images, use temporary file approach
        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(contents)
                temp_path = temp.name
            
            try:
                # Load and preprocess the image
                img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                orig_img_array = img_array.copy()
                
                # Normalize and expand dimensions
                norm_img_array = normalize_for_network(img_array)
                input_img_array = np.expand_dims(norm_img_array, axis=0)
                
                # Generate GradCAM visualization
                viz_image_buf, result = generate_gradcam_visualization(
                    input_img_array, orig_img_array, model, last_conv_layer
                )
                
                # Save visualization to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as viz_temp:
                    viz_temp.write(viz_image_buf.getvalue())
                    viz_temp_path = viz_temp.name
                
                # Return the file with visualization
                return FileResponse(
                    viz_temp_path,
                    media_type="image/png",
                    headers={"X-Prediction": result["class"], 
                             "X-Probability": str(result["probability"]),
                             "X-Filename": file.filename}
                )
            finally:
                # Clean up temp files
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"GradCAM error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GradCAM error: {str(e)}")

@app.post("/gradcam-base64")
async def gradcam_base64(file: UploadFile = File(...)):
    """
    Generate GradCAM visualization and return as base64 encoded image
    
    Parameters:
    -----------
    file : UploadFile
        The image file to analyze
    
    Returns:
    --------
    dict
        Prediction results and base64 encoded GradCAM visualization
    """
    # Check if model is loaded
    if model is None or last_conv_layer is None:
        raise HTTPException(status_code=503, detail="Model or layer not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Process the image
        img_array, orig_img_array = process_image_file(contents)
        
        # Generate GradCAM visualization
        viz_image_buf, result = generate_gradcam_visualization(
            img_array, orig_img_array, model, last_conv_layer
        )
        
        # Convert to base64
        encoded_image = base64.b64encode(viz_image_buf.getvalue()).decode('utf-8')
        
        # Return the result with base64 encoded image
        return {
            "filename": file.filename,
            "class": result["class"],
            "probability": result["probability"],
            "raw_score": result["raw_score"],
            "gradcam_image": encoded_image
        }
    
    except Exception as e:
        logger.error(f"GradCAM base64 error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GradCAM base64 error: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
