import streamlit as st
import numpy as np
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource

def load_model():
    # model = tf.keras.models.load_model(INSERT FILE HERE)
    return None #model

model = load_model()

st.write("""
         # Trash Classification
         """)

file = st.file_uploader("Please upload an image of trash you would like to classify below:", type=["jpg","png"])

def import_and_predict(image_data, model):
    
    size = (312,584)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] #add dimension for model compatability
    #prediction = model.predict(img_reshape)

    return None #prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names=['glass', 'paper', 'cardboard', 
           'metal', 'plastic', 'trash']
    string = "This image is most likely " + class_names[np.argmax(predictions)]
    st.success(string)
