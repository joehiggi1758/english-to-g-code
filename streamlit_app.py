import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./gcode_translation_model")
tokenizer = AutoTokenizer.from_pretrained("./gcode_translation_model")

# Define translation function
def translate_to_gcode(instruction):
    inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs.input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("English to G-Code Translator")
st.write("Enter an English instruction, and this model will translate it to G-code.")

instruction = st.text_input("Enter instruction", "Move X=10")

if instruction:
    gcode = translate_to_gcode(instruction)
    st.write(f"Generated G-Code: {gcode}")
