
# app.py
import streamlit as st
from PIL import Image
import numpy as np
import glob as gb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("Smile / Non-Smile Detection")
st.write("Upload an image and the model will predict whether it is a smile or non-smile.")

# --------------------------
# Load training data
# --------------------------
smile_data = gb.glob("Dataset/Smiled/*.jpg")
non_smile_data = gb.glob("Dataset/Non_smiled/*.jpg")

data = []
label = []

# Resize all images to 64x64
for image in smile_data:
    img = Image.open(image).convert('L').resize((64, 64))
    data.append(np.array(img).flatten() / 255.0)
    label.append(1)

for image in non_smile_data:
    img = Image.open(image).convert('L').resize((64, 64))
    data.append(np.array(img).flatten() / 255.0)
    label.append(0)

data = np.array(data)
label = np.array(label)

# --------------------------
# Scale training data and train model
# --------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(data)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, label)

# --------------------------
# Streamlit file uploader
# --------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load uploaded image and resize to same size as training
    image = Image.open(uploaded_file).convert('L').resize((64, 64))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess uploaded image
    image_array = np.array(image).flatten() / 255.0
    image_array = image_array.reshape(1, -1)
    
    # Scale uploaded image using the SAME scaler fitted on training data
    image_scaled = scaler.transform(image_array)
    
    # Predict
    prediction = model.predict(image_scaled)[0]
    probability = model.predict_proba(image_scaled)[0]
    
    if prediction == 1:
        st.success(f"Prediction: Smile 😄 (Confidence: {probability[1]*100:.2f}%)")
    else:
        st.error(f"Prediction: Non-Smile 😐 (Confidence: {probability[0]*100:.2f}%)")
