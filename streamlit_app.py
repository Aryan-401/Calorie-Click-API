import streamlit as st
import requests
import io
from PIL import Image
import base64

st.title("Food Classification API")
st.write("This is a simple API that classifies food images into 20 classes")
st.write("The food can be classified on the following classes:")
st.table(
    {
        " ": [
            "Burger",
            "Butter Naan",
            "Chai",
            "Chapati",
            "Chole Bature",
            ],
        "  ": [
            "Dal Maakhani",
            "Dhokla",
            "Fried Rice",
            "Idli",
            "Jalebi",
            ],
        "   ": [
            "Kaathi Rolls",
            "Kadai Paneer",
            "Kulfi",
            "Masala Dosa",
            "Momos",
            ],
        "    ": [
            "Paani Puri",
            "Pakode",
            "Pav Bhaji",
            "Pizza",
            "Samosa",
        ]
    }
)
st.write("Upload an image of food to classify")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    input_image = Image.open(uploaded_file)
    file_extension = uploaded_file.name.split(".")[-1].upper()
    if file_extension == "JPG":
        file_extension = "JPEG"
    buffered = io.BytesIO()
    input_image.save(buffered, format=file_extension)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    print(img_str)
    with st.spinner("This may take a while depending if the model is loaded or not. Please be patient."):
        response = requests.get(
            url="https://calorie-click-api.onrender.com")
        if response.status_code == 200:
            st.success("Model is loaded")

        response = requests.post(
            url="https://calorie-click-api.onrender.com/predict",
            json={"base64_image": img_str},
        )
        if response.status_code == 200:
            st.success("Image Classified")
            top_classes = response.json().get("top_classes")
            st.table(
                {
                    "Class": top_classes,
                }
            )
        else:
            st.error("Error occurred while classifying the image")