import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
import os

from recommendation import cnv, dme, drusen, normal

st.set_page_config(page_title="OCT Retinal Analysis", page_icon="👁️", layout="wide")


page = option_menu(
    menu_title="OCT Retinal Analysis Platform",
    options=["Home", "About", "Disease Identification"],
    menu_icon="#",
    icons=[None, None, None],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0px 25px",
            "background-color": "#0e1117",
            "justify-content": "center"
        },
        "icon": {"display": "none"},
        "nav-link": {
            "font-size": "20px",
            "color": "#6ca8ff",
            "margin": "0px 10px",
            "text-align": "center",
            "transition": "0.3s",
        },
        "nav-link-selected": {
            "background-color": "#1f4eff",
            "color": "white",
            "font-weight": "bold",
            "border-radius": "5px"
        },
        "menu-title": {
            "font-size": "30px",
            "font-weight": "bold",
            "text-align": "center",
            "margin": "auto auto",
            "flex": "1",
        },
    }
)


def model_prediction(test_image_path, threshold=0.7):
    model = tf.keras.models.load_model("Trained_Model.keras")
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x).flatten()
    pred_index = np.argmax(predictions)
    confidence = predictions[pred_index]

    if confidence < threshold:
        return None, confidence, predictions
    return pred_index, confidence, predictions


if page == "Home":
    st.markdown("""
    #### **Welcome to the Retinal OCT Analysis Platform**
    Optical Coherence Tomography (OCT) provides high-resolution cross-sectional images of the retina,
    allowing for early detection and monitoring of retinal diseases such as **CNV, DME, Drusen,** and **Normal Retina**.

    ---
    ##### **Key Features:**
    - Automated disease classification (CNV, DME, Drusen, Normal)
    - High-resolution OCT image analysis
    - Invalid image detection for non-retinal scans
    ---
    Upload an OCT scan to start disease identification!
    """)

elif page == "About":
    st.markdown("""
    #### **OCT (Optical Coherence Tomography)** captures cross-sectional images of the retina.
    This project classifies OCT images into 4 categories:
    - CNV (Choroidal Neovascularization)
    - DME (Diabetic Macular Edema)
    - Drusen (Early AMD)
    - Normal Retina

    Dataset: 84,495 high-resolution OCT scans split into train, validation, and test sets.
    Images were verified by multiple ophthalmologists to ensure accuracy.
    """)

elif page == "Disease Identification":
    st.markdown("""
    #### **OCT Retinal Disease Identification**
    """)
    test_image = st.file_uploader("Upload your OCT Image:")

    temp_file_path = None
    if test_image is not None:
        ext = os.path.splitext(test_image.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name

    if test_image is not None and st.button("Predict"):
        with st.spinner("Analyzing the image..."):
            result_index, confidence, predictions = model_prediction(temp_file_path)

        class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

        if result_index is None:
            st.error(f"Invalid Eye Scan Image! (Max confidence: {confidence*100:.2f}%)")
            st.image(test_image, caption="Uploaded Image")
        else:
            st.success(f"Model predicts: **{class_names[result_index]}** (Confidence: {confidence*100:.2f}%)")
            st.image(test_image, caption=f"Predicted: {class_names[result_index]}")

            with st.expander("Learn More"):
                if result_index == 0:
                    st.write("OCT scan showing **CNV with subretinal fluid.**")
                    st.markdown(cnv)
                elif result_index == 1:
                    st.write("OCT scan showing **DME with retinal thickening and intraretinal fluid.**")
                    st.markdown(dme)
                elif result_index == 2:
                    st.write("OCT scan showing **Drusen deposits in early AMD.**")
                    st.markdown(drusen)
                elif result_index == 3:
                    st.write("OCT scan showing **Normal retina with preserved foveal contour.**")
                    st.markdown(normal)
