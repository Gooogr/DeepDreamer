import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from collections import defaultdict
from main import predict
from keras import backend as K
from keras.applications import inception_v3
from utils import get_loss

DEMO_IMAGE_PATH = './imgs/flower_valley.jpg'

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

# Setting up model and loss
K.set_learning_phase(0) 
model = inception_v3.InceptionV3(weights='imagenet', 
								 include_top=False)	
loss = get_loss(layers_coeff, model)

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href

if st.button('Start to dream'):
	predicted_img = predict(img_file=image, 
							num_octave=num_octave, 
							octave_scale=octave_scale,
							iterations=iterations,
							step=step,
							max_loss=max_loss,
							model=model,
							loss=loss)
	st.image(predicted_img)
	result = Image.fromarray(predicted_img)
	st.markdown(get_image_download_link(result), 
				unsafe_allow_html=True)
	




