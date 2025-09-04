!pip install transformers torch streamlit PyPDF2 -q

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to generate response
def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Concept explanation (Smart City)
def concept_explanation(concept):
    prompt = f"Explain the smart city concept of {concept} in detail with real-life sustainable examples."
    return generate_response(prompt, max_length=400)

# Quiz generator (Smart City Awareness)
def quiz_generator(topic):
    prompt = f"Generate 5 quiz questions about {topic} in the context of smart cities and sustainability. Include multiple choice, true/false, and short answer types. At the end, provide all answers in a separate list."
    return generate_response(prompt, max_length=600)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏙 Citizen AI - Smart City Assistant")

tab1, tab2 = st.tabs(["City Concept Explanation", "Smart City Quiz Generator"])

with tab1:
    concept_input = st.text_input("Enter a smart city concept:", placeholder="e.g., Renewable Energy, Smart Transport")
    if st.button("Explain Concept"):
        if concept_input.strip():
            with st.spinner("Generating explanation..."):
                explanation = concept_explanation(concept_input)
            st.text_area("📖 Explanation", explanation, height=300)
        else:
            st.warning("Please enter a concept.")

with tab2:
    quiz_input = st.text_input("Enter a smart city topic:", placeholder="e.g., Waste Management, Green Energy")
    if st.button("Generate Quiz"):
        if quiz_input.strip():
            with st.spinner("Generating quiz..."):
                quiz = quiz_generator(quiz_input)
            st.text_area("📝 Quiz Questions", quiz, height=400)
        else:
            st.warning("Please enter a topic.")
