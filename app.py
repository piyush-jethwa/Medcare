import streamlit as st
import google.generativeai as genai

# --------------------------------------------------
# HARD-CODED API KEY (user should replace)
# --------------------------------------------------
GOOGLE_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-pro")

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
    st.image(uploaded_image, caption="Uploaded Image", width='stretch')

    if st.button("Generate Diagnosis"):
        with st.spinner("Analyzing the medical image..."):
            result = generate_diagnosis(uploaded_image.read(), symptoms)
        st.subheader("🩺 Diagnosis Result")
        st.write(result)
