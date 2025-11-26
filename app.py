import streamlit as st
from groq import Groq

# --------------------------------------------------
# API KEY Configuration
# --------------------------------------------------
# To deploy on Streamlit Cloud, set the GROQ_API_KEY in the secrets.

if 'GROQ_API_KEY' in st.secrets:
    GROQ_API_KEY = st.secrets['GROQ_API_KEY']
else:
    st.error("Groq API Key not found. Please set it in Streamlit secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# List available models to find the correct one for vision
st.write("Checking for available Groq models...")
try:
    models_response = client.models.list()
    # The response object is not a simple list, we need to access the data attribute
    available_models = [model.id for model in models_response.data]
    st.write("Available models:")
    st.write(available_models)
except Exception as e:
    st.error(f"An error occurred while fetching models: {e}")
st.stop()

# --------------------------------------------------
# Title + UI
# --------------------------------------------------
st.title("🧠 Medical Image Diagnosis AI Agent")
st.write("Upload a medical image and get detailed diagnostic insights using an advanced AI Agent.")

uploaded_image = st.file_uploader("Upload an X-ray / MRI / CT image", type=["png", "jpg", "jpeg"])

symptoms = st.text_area("Describe symptoms (optional)", placeholder="Fever, chest pain, coughing...")

# --------------------------------------------------
# Agent Prompt
# --------------------------------------------------
AGENT_SYSTEM_PROMPT = '''
You are an Advanced Medical Image Diagnosis AI Agent.

Your role:
- Analyze medical images
- Give detailed diagnosis
- Mention possible conditions
- Provide medical reasoning step-by-step
- Identify abnormalities
- Suggest further tests
- Provide risk level (Low / Medium / High)

Important:
- You are NOT a doctor.
- Provide medically accurate, but safe and general explanations.
'''

def generate_diagnosis(image_bytes, symptoms):
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}
    prompt = f"{AGENT_SYSTEM_PROMPT}\n\nSymptoms reported: {symptoms if symptoms else 'Not provided'}"

    response = model.generate_content([prompt, image_part])
    return response.text

# --------------------------------------------------
# Run diagnosis
# --------------------------------------------------
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

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
