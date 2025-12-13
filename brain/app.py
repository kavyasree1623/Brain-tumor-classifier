import streamlit as st 
from PIL import Image
import numpy as np
import pos  # make sure pos.py is in the same folder
import tensorflow as tf
import keras
import cv2

# Load model
model = pos.create_cct_model()
model.load_weights("checkpoint.weights.h5")  # Ensure this file is saved after training

# Compile the model for prediction (not training)
optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
)

# Streamlit UI
st.set_page_config(page_title="Tumor classifier project", layout="centered")
st.title("Brain tumor Classifier")
st.write("Upload a tumor classifier")

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    image_resized = cv2.resize(np.array(image), (32, 32))
    input_image = np.expand_dims(image_resized / 255.0, axis=0)  # normalize

    st.image(image, caption="Uploaded Image", use_column_width=False, width=250)

    # Make prediction
    prediction_logits = model.predict(input_image)
    predicted_class = np.argmax(prediction_logits, axis=1)[0]

    # Map class index to label
    labels = {
        0: "meningioma",
        1: "pituitary"
         }

    st.markdown("---")
    st.subheader("Prediction Result")
    st.success(f"**Predicted Tumor:** {labels[predicted_class]}")
