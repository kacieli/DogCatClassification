# Dog & Cat Breed Classifier

A **Streamlit application** to classify uploaded images into 187 categories using a pre-trained model. The app provides confidence explanations, visualizes top predictions, and offers tips for better predictions.



## How to Run

### Prerequisites
1. Install **Python 3.7+** and required libraries:
   ```bash
   pip install tensorflow streamlit pillow matplotlib
   ```
2. Place the pre-trained model file `dogcat_model1.h5` in the same directory as `interface.py`.


### Steps
1. Download or clone the repository:
  
2. Run the Streamlit app:
   ```bash
   streamlit run interface.py
   ```
3. Open the provided URL (e.g., `http://localhost:8501`) in your browser.



## Features
- Predicts the class of uploaded images.
- Dynamically explains confidence levels.
- Visualizes top 5 predictions in an easy-to-read chart.
- Provides tips for better prediction results.



## Example Outputs
- **Prediction**: `Class 38`
- **Confidence**: `The model is 72% sure this image belongs to Class 38.`
- **Visualization**: A bar chart showing top 5 predictions with confidence levels.

