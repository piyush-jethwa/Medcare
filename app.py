import streamlit as st
import requests
import base64
from groq import Groq
from io import BytesIO
from PIL import Image

# --------------------------------------------------
# API Configuration
# --------------------------------------------------
if 'HF_TOKEN' not in st.secrets:
    st.error("Hugging Face Token not found. Please set it in Streamlit secrets.")
    st.stop()
    
HF_TOKEN = st.secrets['HF_TOKEN']
HF_API_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

# Check for Groq API Key
if 'GROQ_API_KEY' not in st.secrets:
    st.error("Groq API Key not found. Please set it in Streamlit secrets.")
    st.stop()

groq_client = Groq(api_key=st.secrets['GROQ_API_KEY'])

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Medical Image Diagnosis AI Agent")
st.write("Upload a medical image and get detailed diagnostic insights using an advanced AI Agent.")

uploaded_image = st.file_uploader("Upload an X-ray / MRI / CT image", type=["png", "jpg", "jpeg"])
symptoms = st.text_area("Describe symptoms (optional)", placeholder="Fever, chest pain, coughing...")

AGENT_SYSTEM_PROMPT = """You are an Advanced Medical Image Diagnosis AI Agent. Your task is to analyze medical images and provide:
- Possible conditions
- Medical reasoning
- Recommended next steps
- Risk level (Low/Medium/High)

Important:
- You are NOT a doctor.
- Provide medically accurate, but safe and general explanations."""

def analyze_image(image_bytes):
    """Analyze image using LLaVA model"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Convert image to base64
    image = Image.open(BytesIO(image_bytes))
    
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Prepare the prompt
    prompt = "Analyze this medical image and provide a detailed description of any visible abnormalities, conditions, or notable features."
    
    payload = {
        "inputs": {
            "text": prompt,
            "image": img_str
        }
    }
    
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        error_message = response.json().get('error', 'Unknown error')
        raise Exception(f"Hugging Face API error: {error_message} (Status code: {response.status_code})")
    
    return response.json()[0]['generated_text']

# --------------------------------------------------
# Run diagnosis
# --------------------------------------------------
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Diagnosis"):
        with st.spinner("Analyzing the medical image..."):
            try:
                image_bytes = uploaded_image.read()
                
                # Step 1: Analyze image with LLaVA
                with st.spinner("Analyzing image..."):
                    image_analysis = analyze_image(image_bytes)
                
                # Step 2: Generate diagnosis with Groq
                with st.spinner("Generating diagnosis..."):
                    result = generate_diagnosis(image_analysis, symptoms)
                
                st.subheader("🩺 Diagnosis Result")
                st.write(result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your API keys and try again. If the issue persists, the model might be temporarily unavailable.")
