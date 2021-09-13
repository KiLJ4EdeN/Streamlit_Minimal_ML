# user interface
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
from preprocessing import preprocess_image


model = tf.keras.models.load_model('models/mnist.h5')

# UI
st.write("MNIST digit prediction")

# html input with an extra extension checker
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file:")
else:
    img = Image.open(file)
    st.image(img, use_column_width=True)
    img = preprocess_image(img)
    img = img.reshape(1, 28, 28, 1)
    pred = model.predict(img)
    label = np.argmax(pred)
    st.write(f'the prediction is: {label}')
    st.write('class probs:')
    st.write([f'{i}: {pred[0][i]}' for i in range(10)])
