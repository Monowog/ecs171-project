import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)


def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model 


model = load_model()

st.write("""
         # Trash Classification
         """)

file = st.file_uploader("Please upload an image of trash you would like to classify below:", type=["jpg","png"])

def import_and_predict(image, model):

    img_resized = image.resize((96, 128)) 
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names=['glass', 'paper', 'cardboard', 
           'metal', 'plastic', 'non-recyclable']
    string = "This image is most likely " + class_names[np.argmax(predictions)]
    st.success(string)

