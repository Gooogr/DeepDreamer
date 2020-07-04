import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from collections import defaultdict

DEMO_IMAGE_PATH = './test_images/flower_valley.jpg'

st.title("DeepDream interactive demo")

# Get content image from file_uploader
img_file_buffer = st.sidebar.file_uploader("Select image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
	image = np.array(Image.open(img_file_buffer))
else:
	image = np.array(Image.open(DEMO_IMAGE_PATH))
	
# Create mixed layers selector
layers_names = st.multiselect(
     'What InceptionV3 layers we will use?',
     options=['mixed{}'.format(i) for i in range(11)],
     default=['mixed2', 'mixed3', 'mixed4', 'mixed5'])

# Set default values of layers coefficients     
layers_coeff = defaultdict(lambda: 0.0)
layers_coeff['mixed2'] = 0.2
layers_coeff['mixed3'] = 3.
layers_coeff['mixed4'] = 2.
layers_coeff['mixed5'] = 1.5

for layer_name in layers_names:
	layers_coeff[layer_name] = st.slider(layer_name, 0.0, 5.0, layers_coeff[layer_name]) # dict update automatically
	

# Setting up sidebar hyperparameters
st.sidebar.markdown('Additional hyperparameters. Be careful.')
num_octave = st.sidebar.slider("Octave number", min_value=2, max_value=5, value=3)
step = st.sidebar.slider("Step", min_value=0.005, max_value=0.05, value=0.01, step = 0.005, format='%f')
octave_scale = st.sidebar.slider("Octave scale", min_value=1.1, max_value=2.0, value=1.4)
iterations = st.sidebar.slider("Iterations", min_value=10, max_value=30, value=20)
max_loss = st.sidebar.slider("Maximum loss", min_value=5, max_value=20, value=10)

if st.button('Start to dream'):
	st.write('ping')
	st.write(layers_coeff)
	




