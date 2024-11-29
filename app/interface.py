import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load your pre-trained model
model = tf.keras.models.load_model('dogcat_model1.h5')

# Define all class names if there are 187 classes
class_names = [f"Class {i}" for i in range(187)]  # Replace with actual class names if available

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

# Function to visualize prediction confidence
def visualize_prediction(predictions):
    """
    Displays a bar chart for top 5 predictions with their confidence levels.
    """
    top_indices = np.argsort(predictions[0])[-5:][::-1]  # Get top 5 class indices
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = predictions[0][top_indices]

    fig, ax = plt.subplots()
    ax.barh(top_classes, top_confidences, color="blue")
    ax.set_xlim([0, 1])
    ax.set_title("Top 5 Predictions")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Classes")
    ax.invert_yaxis()  # Invert y-axis for better visualization
    return fig, top_classes, top_confidences

# Dynamic explanation for confidence level
def confidence_explanation(confidence):
    """
    Provides a dynamic explanation based on the confidence score.
    """
    if confidence > 0.8:
        return "The model is highly confident about its prediction, meaning it strongly believes this image belongs to the predicted class."
    elif confidence > 0.5:
        return "The model is somewhat confident about its prediction, but there is some uncertainty. This might be because the image contains features shared by multiple classes."
    else:
        return "The model is not very confident about its prediction. This could be due to factors like low image quality, unusual angles, or the model lacking enough training data for this class."

# Dynamic explanation for the top 5 predictions
def plot_explanation(top_confidences):
    """
    Provides a dynamic explanation based on the spread of confidence levels.
    """
    if top_confidences[0] > 0.7 and top_confidences[0] - top_confidences[1] > 0.2:
        return "The model is quite confident about the top class, as it has a much higher confidence than the other classes."
    else:
        return "The model finds it challenging to decide between the top classes, as their confidence levels are close. This may indicate that the image shares features with multiple classes."

# Streamlit Interface
st.title("Dog & Cat Breed Classifier")
st.write("Upload a picture to find out what class it belongs to!")

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
        top_class = class_names[top_index]
        top_confidence = predictions[0][top_index]

        # Dynamic explanation for confidence score
        confidence_text = confidence_explanation(top_confidence)

        # Display the top prediction and its confidence
        st.write(f"### Prediction: **{top_class}**")
        st.write(f"The model is **{top_confidence:.2%} sure** that this image belongs to the class: **{top_class}**.")
        st.write(confidence_text)

        # Display confidence visualization
        st.write("### How Sure is the Model?")
        fig, top_classes, top_confidences = visualize_prediction(predictions)
        st.pyplot(fig)

        # Dynamic explanation for the plot
        plot_text = plot_explanation(top_confidences)
        st.write(plot_text)

        # Additional tips
        st.write("### Tips:")
        st.write("- Ensure the image is clear and well-lit for better predictions.")
        st.write("- If the confidence is low, consider using another image for comparison.")

    except Exception as e:
        st.error("Something went wrong while processing your image. Please try again.")

# Instructions for running Streamlit
st.write("---")
st.write("To start this application, run the following command in your terminal:")
st.code("streamlit run interface.py", language="bash")



# streamlit run interface.py
# source tf_env/bin/activate



