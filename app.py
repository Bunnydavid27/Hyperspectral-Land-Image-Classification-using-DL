import numpy as np
import tensorflow as tf
import pickle as pkl
import streamlit as st
from PIL import Image
from skimage import transform
import matplotlib.pyplot as plt
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('eurosat_overview.png') 

st.title('Hyperspectral Image Classification')
st.text('Upload Image')
class_name_list = np.array(['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
       'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River',
       'SeaLake'])
model_location = r'working\models\rgbmodel.h5'
model = tf.keras.models.load_model(model_location)
uploaded_image = st.file_uploader("Choose an Image",type=['jpg','png','jpeg'])
print(uploaded_image)
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image')
    if st.button('PREDICT'):
              #image = tf.keras.utils.load_img(r"Dataset/EuroSAT/Residential/Residential_504.jpg", target_size=(100,100,3))
              input_arr = tf.keras.utils.img_to_array(img)
              input_arr = transform.resize(input_arr,(100,100,3))
              # input_arr = np.array(img)
              plt.imshow(input_arr)
              input_arr = np.array([input_arr])  # Convert single image to a batch.
              predictions = model.predict(input_arr)
              output_class_id = np.argmax(predictions, axis=1)
              Land_covered= class_name_list[output_class_id]
              st.write(Land_covered)

     #python -m streamlit run app.py