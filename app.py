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
HF_API_URL = "https://router.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

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
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
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
    
    # Prepare the payload
    payload = {
        "inputs": {
            "prompt": "Analyze this medical image and provide a detailed description of any visible abnormalities, conditions, or notable features.",
            "image": f"data:image/jpeg;base64,{img_str}",
            "max_new_tokens": 500
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text']
    except requests.exceptions.HTTPError as http_err:
        error_msg = response.json().get('error', str(http_err))
        raise Exception(f"Hugging Face API error: {error_msg} (Status code: {response.status_code})")
    except Exception as e:
        raise Exception(f"Error processing the image: {str(e)}")

def generate_diagnosis(image_analysis, symptoms):
    """Generate diagnosis using Groq's API"""
    prompt = f"""{AGENT_SYSTEM_PROMPT}
    
Image Analysis:
{image_analysis}

Symptoms reported: {symptoms if symptoms else 'None provided'}

Please provide a detailed diagnosis based on the image analysis and symptoms above. Include:
1. Possible conditions
2. Medical reasoning
3. Recommended next steps
4. Risk level (Low/Medium/High)"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024,
    )
    
    return chat_completion.choices[0].message.content

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
