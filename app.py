import streamlit as st
import requests
import base64
from groq import Groq
from io import BytesIO
from PIL import Image
import json

# --------------------------------------------------
# API Configuration
# --------------------------------------------------
if 'HF_TOKEN' not in st.secrets:
    st.error("Hugging Face Token not found. Please set it in Streamlit secrets.")
    st.stop()
    
HF_TOKEN = st.secrets['HF_TOKEN']
# Using a more reliable model endpoint
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
    """Analyze image using LLaVA model with detailed error handling"""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        # Convert image to base64
        image = Image.open(BytesIO(image_bytes))
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)  # Reduce quality to decrease size
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare the payload
        payload = {
            "inputs": {
                "prompt": "Analyze this medical image and provide a detailed description of any visible abnormalities, conditions, or notable features.",
                "image": f"data:image/jpeg;base64,{img_str}",
                "max_new_tokens": 200
            }
        }
        
        # Make the API request with timeout
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Try to parse JSON
        try:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No generated text in response')
            return str(result)  # Return string representation if not in expected format
        except ValueError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}. Response: {response.text[:200]}")
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"\nStatus code: {e.response.status_code}"
            try:
                error_msg += f"\nResponse: {e.response.json()}"
            except:
                error_msg += f"\nResponse text: {e.response.text[:500]}"
        raise Exception(error_msg)
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def generate_diagnosis(image_analysis, symptoms):
    """Generate diagnosis using Groq's API"""
    try:
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
        
        if hasattr(chat_completion.choices[0].message, 'content'):
            return chat_completion.choices[0].message.content
        else:
            return "Error: Unexpected response format from Groq API"
            
    except Exception as e:
        raise Exception(f"Error generating diagnosis: {str(e)}")

# --------------------------------------------------
# Run diagnosis
# --------------------------------------------------
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Diagnosis"):
        with st.spinner("Analyzing the medical image..."):
            try:
                image_bytes = uploaded_image.read()
                
                # Show progress
                with st.spinner("Analyzing image..."):
                    image_analysis = analyze_image(image_bytes)
                    st.session_state['image_analysis'] = image_analysis
                
                # Generate diagnosis
                with st.spinner("Generating diagnosis..."):
                    result = generate_diagnosis(
                        image_analysis,
                        symptoms
                    )
                    st.session_state['diagnosis_result'] = result
                
                # Display results
                st.subheader("🩺 Diagnosis Result")
                st.write(result)
                
                # Show raw analysis in expander
                with st.expander("View Detailed Analysis"):
                    st.text_area("Image Analysis", image_analysis, height=200)
                
            except Exception as e:
                st.error("An error occurred during processing. Please try again.")
                st.error(f"Error details: {str(e)}")
                st.info("Tips: Try with a smaller image or check your API keys.")
