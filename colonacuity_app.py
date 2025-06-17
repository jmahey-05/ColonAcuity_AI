import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
import time
from typing import Tuple, Optional, Dict, Any

# Configure the page
st.set_page_config(
    page_title="ColonAcuity AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve layout and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    .subheader {
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        color: #e0e0e0;
        font-weight: 400;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .benign {
        background-color: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .adenocarcinoma {
        background-color: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        background-color: #3498db;
        color: white;
        height: 3rem;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .info-box {
        background-color: rgba(52, 152, 219, 0.1);
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .result-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .progress-container {
        width: 100%;
        height: 2rem;
        background-color: #ecf0f1;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    .image-label {
        font-size: 1rem;
        text-align: center;
        color: #95a5a6;
        margin-top: 0.5rem;
    }
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    .gradcam-container {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .gradcam-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #ffffff;
        text-align: center;
    }
    .browse-files-button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stExpander {
        border: 1px solid #2c3e50;
        border-radius: 5px;
    }
    /* Remove the dashed border upload box */
    .st-emotion-cache-7ym5gk {
        border: none;  /* Remove the file uploader's default border */
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Define API endpoints
API_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
GRADCAM_ENDPOINT = f"{API_BASE_URL}/gradcam-base64"

# Helper functions
def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

def validate_image_file(file) -> Tuple[bool, str]:
    """Validate the uploaded image file."""
    if file is None:
        return False, "No file uploaded."
    
    try:
        img = Image.open(file)
        img.verify()  # Verify it's an image
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def analyze_image(file, endpoint: str = GRADCAM_ENDPOINT) -> Tuple[bool, Dict[str, Any]]:
    """Send the image to the API for analysis and return the results."""
    # Reset file pointer to beginning
    file.seek(0)
    
    try:
        # Get file name and determine content type
        file_name = file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Set appropriate content type based on extension
        content_type = 'image/jpeg'  # Default
        if file_ext == '.png':
            content_type = 'image/png'
        elif file_ext in ['.tiff', '.tif']:
            content_type = 'image/tiff'
        
        # Prepare files with explicit content type
        files = {"file": (file_name, file.read(), content_type)}
        
        # Reset file pointer after reading
        file.seek(0)
        
        # Send request to API
        response = requests.post(endpoint, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            return False, {"error": f"API error: {response.status_code} - {response.text}"}
    except Exception as e:
        return False, {"error": f"Request error: {str(e)}"}

def display_progress_bar(progress_message: str):
    """Display a progress bar with the given message."""
    with st.spinner(progress_message):
        # Create a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            # Update the progress bar
            progress_bar.progress(i + 1)
            # Small delay to simulate processing
            time.sleep(0.01)
        # Remove the progress bar
        progress_bar.empty()

def display_results(result: Dict[str, Any]):
    """Display the analysis results."""
    class_label = result.get("class", "Unknown")
    probability = result.get("probability", 0.0)
    
    # Create a card based on the prediction class
    card_class = "benign" if class_label == "Benign" else "adenocarcinoma"
    
    with st.container():
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-header">Analysis Results:</div>
            <div class="metric-container">
                <strong>Prediction:</strong> {class_label}
            </div>
            <div class="metric-container">
                <strong>Confidence:</strong> {probability * 100:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display additional interpretation
        if class_label == "Adenocarcinoma":
            st.error("⚠️ This image likely shows CANCEROUS tissue.")
        else:
            st.success("✅ This image likely shows NON-CANCEROUS tissue.")

# Main app layout
def main():
    # Check API health before proceeding
    api_healthy = check_api_health()
    
    # App header
    st.markdown('<h1 class="main-header">ColonAcuity AI - Colon Cancer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Detection using AI</p>', unsafe_allow_html=True)
    
    # Information about the app
    st.markdown("""
    This application uses a deep learning model to detect colon cancer from histopathological 
    images. Upload an image to get started with the analysis.
    """)
    
    # Display API status
    if not api_healthy:
        st.error("⚠️ Backend API is not available. Please make sure the FastAPI server is running.")
    
    # Initialize session state for keeping track of analysis state
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None
    
    # File uploader with simplified styling - no dotted border
    st.markdown('<p>Choose an image...</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"], 
                                    label_visibility="collapsed")
    
    # Reset session if no file is uploaded
    if uploaded_file is None:
        st.session_state.original_image = None
        st.session_state.current_file_id = None
    
    # Display uploaded image if available
    if uploaded_file is not None:
        # Generate a unique identifier for the current file
        current_file_id = f"{uploaded_file.name}_{os.path.getsize(uploaded_file.name) if os.path.exists(uploaded_file.name) else hash(uploaded_file.name)}"
        
        # Check if this is a new file compared to what we had before
        if st.session_state.current_file_id != current_file_id:
            # New file detected - reset session state
            st.session_state.analyzed = False
            st.session_state.result = None
            st.session_state.original_image = None
            st.session_state.current_file_id = current_file_id
        
        # Validate image
        is_valid, error_msg = validate_image_file(uploaded_file)
        
        if not is_valid:
            st.error(error_msg)
        else:
            # Reset file pointer to beginning after validation
            uploaded_file.seek(0)
            
            # Store the image in session state if it's not already there
            if st.session_state.original_image is None:
                image = Image.open(uploaded_file)
                st.session_state.original_image = image
            
            # Always display the original image at the top
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(st.session_state.original_image, use_container_width=True)
            
            # Results display
            if st.session_state.analyzed and st.session_state.result:
                # Display analysis results first
                display_results(st.session_state.result)
                
                # Display ONLY the GradCAM visualization below results
                st.markdown('<div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 20px;">', unsafe_allow_html=True)
                                # GradCAM title at bottom
                st.markdown('<div style="text-align: center; font-size: 18px; color: #7f8c8d; margin-top: 10px;">GradCAM Visualization</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display prediction at the top of GradCAM
                class_label = st.session_state.result.get("class", "Unknown")
                probability = st.session_state.result.get("probability", 0.0)
                st.markdown(f'<div style="text-align: center; font-size: 14px; color: #cccccc; margin-bottom: 10px;">Prediction: {class_label} (Probability: {probability*100:.2f}%)</div>', unsafe_allow_html=True)
                
                # Only display GradCAM image
                if "gradcam_image" in st.session_state.result:
                    image_data = base64.b64decode(st.session_state.result["gradcam_image"])
                    img = Image.open(io.BytesIO(image_data))
                    st.image(img, use_container_width=True)
                    st.markdown('<div style="text-align: center; font-size: 12px; color: #cccccc;"></div>', unsafe_allow_html=True)
                

                
                # Reset button
                if st.button("Analyze Another Image"):
                    st.session_state.analyzed = False
                    st.session_state.result = None
                    st.session_state.original_image = None
                    st.session_state.current_file_id = None
                    st.rerun()
            
            # Display analysis button if not yet analyzed and API is healthy
            elif api_healthy:
                if st.button("Analyze with AI"):
                    with st.container():
                        st.markdown('<div class="loading-container">', unsafe_allow_html=True)
                        display_progress_bar("Processing image...")
                        
                        # Analyze the image
                        success, result = analyze_image(uploaded_file)
                        
                        if success:
                            # Store the result in session state
                            st.session_state.result = result
                            st.session_state.analyzed = True
                            st.rerun()  # Rerun to update UI with results
                        else:
                            st.error(f"Error analyzing image: {result.get('error', 'Unknown error')}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # About the app expander - moved to the bottom
    with st.expander("ℹ️ About this App", expanded=False):
        st.markdown("""
        **ColonAcuity AI** is an advanced tool for analyzing colon tissue images to detect potential cancerous cells.
        
        **How to use:**
        1. Upload a high-quality image of colon tissue
        2. Click "Analyze with AI" to process the image
        3. Review the results and GradCAM visualization
        
        **What is GradCAM?**
        Gradient-weighted Class Activation Mapping (GradCAM) highlights the regions of the image that most influenced the model's prediction, helping medical professionals understand what features the AI is focusing on.
        """)

# Run the app
if __name__ == "__main__":
    main() 
