import streamlit as st
import requests
import base64

st.title("Food Classification API")
st.write("This is a simple API that classifies food images into 20 classes")
st.write("The food can be classified on the following classes:")
st.write(
    """
    - Burger
    - Butter Naan
    - Chai
    - Chapati
    - Chole Bature
    - Dal Maakhani
    - Dhokla
    - Fried Rice
    - Idli
    - Jalebi
    - Kaathi Rolls
    - Kadai Paneer
    - Kulfi
    - Masala Dosa
    - Momos
    - Paani Puri
    - Pakode
    - Pav Bhaji
    - Pizza
    - Samosa
    """
)
st.write("Upload an image of food to classify")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    with open(uploaded_file.name, "rb") as f:
        image = f.read()
        image_base64 = base64.b64encode(image).decode("utf-8")
    response = requests.post(r'https://calorie-click-api.onrender.com/predict', json=image_base64)
    st.json(response.json())
