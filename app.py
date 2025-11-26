import streamlit as st
import google.generativeai as genai

# --------------------------------------------------
# API KEY Configuration
# --------------------------------------------------
# To deploy on Streamlit Cloud, set the GOOGLE_API_KEY in the secrets.
# For local development, create a .env file with GOOGLE_API_KEY="your_key".

if 'GOOGLE_API_KEY' in st.secrets:
    GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
else:
    st.error("Google API Key not found. Please set it in Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Use the vision model for image analysis
model = genai.GenerativeModel("gemini-2.5-flash-image")

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

def generate_diagnosis(image_bytes, symptoms):
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}
    prompt = f"{AGENT_SYSTEM_PROMPT}\n\nSymptoms reported: {symptoms if symptoms else 'Not provided'}"

    generation_config = genai.types.GenerationConfig(max_output_tokens=2048)
    response = model.generate_content([prompt, image_part], generation_config=generation_config)
    return response.text

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
