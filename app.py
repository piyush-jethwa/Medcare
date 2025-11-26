import streamlit as st
import requests
import base64
from groq import Groq

# --------------------------------------------------
# API Configuration
# --------------------------------------------------
# To deploy on Streamlit Cloud, set the HF_TOKEN and GROQ_API_KEY in the secrets.

# Check for Hugging Face Token
if 'HF_TOKEN' not in st.secrets:
    st.error("Hugging Face Token not found. Please set it in Streamlit secrets.")
    st.stop()
HF_TOKEN = st.secrets['HF_TOKEN']
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

# Check for Groq API Key
if 'GROQ_API_KEY' not in st.secrets:
    st.error("Groq API Key not found. Please set it in Streamlit secrets.")
    st.stop()

groq_client = Groq(api_key=st.secrets['GROQ_API_KEY'])

# --------------------------------------------------
# Title + UI
# --------------------------------------------------
st.title("Medical Image diagnosis AI Agent")
st.write("Upload a medical image and get detailed diagnostic insights using an advanced AI Agent.")

uploaded_image = st.file_uploader("Upload an X-ray / MRI / CT image", type=["png", "jpg", "jpeg"])

symptoms = st.text_area("Describe symptoms (optional)", placeholder="Fever, chest pain, coughing...")

# --------------------------------------------------
# Agent Prompt
# --------------------------------------------------
AGENT_SYSTEM_PROMPT = '''
You are an Advanced Medical Image diagnosis AI Agent.

Your role:
- Analyze medical images
- Give detailed diagnosis
- Mention possible conditions
- Provide medical reasoning step-by-step
- Identify abnormalities
- Suggest further tests
- Provide risk level (low / Medium / High)

Important:
- You are NOT a doctor.
- Provide medically accurate, but safe and general explanations.
'''

def get_image_description(image_bytes):
    """Get a detailed description of the image using Hugging Face's API."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, data=image_bytes)
    if response.status_code != 200:
        raise Exception(f"Hugging Face API error: {response.text}")
    return response.json()[0]['generated_text']

def generate_diagnosis(image_bytes, symptoms):
    # Get image description from Hugging Face
    image_description = get_image_description(image_bytes)
    
    # Prepare the prompt for Groq
    prompt = f"""You are an Advanced Medical Image Diagnosis AI Agent.
    
    Image Analysis:
    {image_description}
    
    Symptoms reported: {symptoms if symptoms else 'None provided'}
    
    Please provide a detailed diagnosis based on the image analysis and symptoms above. Include:
    1. Possible conditions
    2. Medical reasoning
    3. Recommended next steps
    4. Risk level (Low/Medium/High)
    
    Remember:
    - You are NOT a doctor.
    - Provide medically accurate, but safe and general explanations.
    """
    
    # Get diagnosis from Groq
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
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
                result = generate_diagnosis(image_bytes, symptoms)
                st.subheader("🩺 Diagnosis Result")
                st.write(result)
            except Exception as e:
                if "ResourceExhausted" in str(e):
                    st.error("API quota exceeded. Please check your Google AI Platform billing or wait and try again later.")
                else:
                    st.error(f"An error occurred: {e}")
