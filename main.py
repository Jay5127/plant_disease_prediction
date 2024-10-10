import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st 
import keras

# working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = r"C:/Users/Jay/Documents/Deep Learning Projects/Plant disease detection/trained_model/plant_disease_detection.h5"

#load pretrained model
model = keras.models.load_model(model_path)

class_indices = json.load(open("app/class_indices.json"))

print(class_indices)

# Function to load and process the image using Pillow
def load_and_process(image_path , target_size = (224,224)):
  #load the image
  img = Image.open(image_path)

  #resize image
  img = img.resize(target_size)

  #Convert the image to numpy array
  img_array = np.array(img)

  #Add batch dimension
  img_array = np.expand_dims(img_array , axis=0)

  #Scale the image values 0 to 1
  img_array = img_array.astype('float32') / 255.
  return img_array

# Function to predict the class of an image
def predict_image_class(model , image_path , class_indices):
  processed_img = load_and_process(image_path)
  predict = model.predict(processed_img)
  predict_index = np.argmax(predict,axis =1 )[0]
  class_name = class_indices[str(predict_index)]
  return class_name


# Streamlit app

st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader('Upload an image...',type = ['jpg','jpeg','png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1 , col2 = st.columns(2)

    with col1:
       resized_image = image.resize((150,150))
       st.image(resized_image)
    
    with col2:
       if st.button('Classify'):
          prediction = predict_image_class(model , uploaded_image , class_indices)
          st.success(f'Prediction: {str(prediction)}')