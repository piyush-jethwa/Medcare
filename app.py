import streamlit as st
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from io import BytesIO
from groq import Groq

# --------------------------------------------------
# Model and API Configuration
# --------------------------------------------------
@st.cache_resource
def load_model():
    """Load the LLaVA model and processor"""
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    if torch.cuda.is_available():
        model = model.to('cuda')
    return processor, model

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

def analyze_image(image, processor, model):
    """Analyze image using LLaVA model locally"""
    try:
        # Prepare the prompt
        prompt = "USER: <image>\nAnalyze this medical image and provide a detailed description of any visible abnormalities, conditions, or notable features. ASSISTANT:"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode and clean up the response
        response = processor.decode(output[0][2:], skip_special_tokens=True)
        return response.split("ASSISTANT:")[-1].strip()
        
    except Exception as e:
        raise Exception(f"Error analyzing image: {str(e)}")

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
        return "Error: Unexpected response format from Groq API"
            
    except Exception as e:
        raise Exception(f"Error generating diagnosis: {str(e)}")

# --------------------------------------------------
# Run diagnosis
# --------------------------------------------------
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Diagnosis"):
        with st.spinner("Initializing model (this may take a moment)..."):
            try:
                # Load model and processor
                processor, model = load_model()
                
                # Load and preprocess image
                image = Image.open(BytesIO(uploaded_image.getvalue())).convert('RGB')
                
                # Analyze image
                with st.spinner("Analyzing image..."):
                    image_analysis = analyze_image(image, processor, model)
                    st.session_state['image_analysis'] = image_analysis
                
                # Generate diagnosis
                with st.spinner("Generating diagnosis..."):
                    result = generate_diagnosis(image_analysis, symptoms)
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
                st.info("If the issue persists, try with a different image or check your internet connection.")
