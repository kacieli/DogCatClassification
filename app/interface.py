import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load your pre-trained model
model = tf.keras.models.load_model('dogcat_model1.h5')

# Load the breed-to-index mapping
with open("breed_to_index.json", "r") as json_file:
    breed_to_index = json.load(json_file)

# Reverse the mapping to get index-to-breed mapping
index_to_breed = {v: k for k, v in breed_to_index.items()}

# Define function to identify whether it's a cat or dog
def is_cat_or_dog(index):
    """
    Determine if the predicted class is a cat or a dog based on index ranges.
    """
    if index < 67:  # Assuming cat breeds are first
        return "Cat"
    else:
        return "Dog"

# Function to preprocess the image
def preprocess_image(image):
    """
    Prepares the image for prediction: converts to RGB, resizes, and normalizes.
    """
    image = image.convert("RGB")  # Convert to RGB for consistency
    image = image.resize((128, 128))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to create a table of top predictions
def create_predictions_table(predictions, labels, top_n=5):
    """
    Creates a table of the top N predictions with their confidence levels.
    """
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]  # Get top N class indices
    top_classes = [labels[i] for i in top_indices]
    top_confidences = predictions[0][top_indices]
    table_data = [{"Rank": i + 1, "Breed": top_classes[i], "Confidence": f"{top_confidences[i]:.2%}"} for i in range(top_n)]
    return table_data

# Streamlit Interface
st.title("Cat or Dog Breed Classifier")
st.write("Upload a picture to find out whether it's a cat or a dog, and discover its breed!")

uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, or PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Uploaded Image", use_column_width=True)
    st.write("Analyzing the image...")

    try:
        # Preprocess and predict
        processed_image = preprocess_image(image)
        predictions = tf.nn.softmax(model.predict(processed_image)).numpy()  # Convert logits to probabilities

        # Get the top prediction
        top_index = np.argmax(predictions[0])
        top_breed = index_to_breed[top_index]
        top_confidence = predictions[0][top_index]
        animal_type = is_cat_or_dog(top_index)

        # Display whether it's a cat or dog
        st.write(f"### It's a **{animal_type}**!")
        st.write(f"The model is **{top_confidence:.2%} sure** that this image is a **{animal_type}**.")

        # Explanation about the classification
        st.write(f"**Explanation:** The model first checks if the image resembles a cat or a dog based on the visual features.")
        st.write(f"Then, it identifies the specific breed from {len(index_to_breed)} possible classes.")
        st.write("Keep in mind that high confidence scores indicate stronger predictions.")

        # If cat or dog, show breed prediction
        st.write(f"### Predicted Breed: **{top_breed}**")

        # Display confidence table
        st.write("### Confidence Levels for Top Breeds")
        table_data = create_predictions_table(predictions, list(index_to_breed.values()))
        st.table(table_data)

        # Additional tips for better results
        st.write("### Tips:")
        st.write("- Ensure the image is clear and well-lit for better predictions.")
        st.write("- If the confidence is low, try another image for comparison.")

    except Exception as e:
        st.error("Something went wrong while processing your image. Please try again.")

# Instructions for running Streamlit
st.write("---")

# st.write("To start this application, run the following command in your terminal:")
# st.code("streamlit run interface.py", language="bash")




# Mac/Linux:
# source tf_env/bin/activate

# Windows
# tf_env\Scripts\activate

# streamlit run interface.py




# deactivate

# conda create --name tf_env python=3.9
# conda activate tf_env
# pip install tensorflow-macos tensorflow-metal
#
