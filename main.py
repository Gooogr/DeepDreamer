import scipy
import numpy as np
import cv2
import argparse
from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image
from utils import *

##### Setting up constants, hyperparameters #####

# Constants
DEMO_IMAGE_PATH = './test_images/flower_valley.jpg'

# Loop's hyperparameters
STEP = 0.01          # Value of the gradient ascent step
NUM_OCTAVE = 3       # Octave amount
OCTAVE_SCALE = 1.4   # Scale of each octave (look at the picture at the end of notebook)
ITERATIONS = 20      # Iterations amount on the each octave
MAX_LOSS = 10.       # Max loss that prevent ugly artifacts

# Layers could be from mixed0 up to mixed10
# Check names with command model.summary()
layer_contribution = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5
}

##### Setting up model and loss #####

#Set learning phase = 0 (test mode). Prevent model learning for safety reasons.
K.set_learning_phase(0) 
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)		
loss = get_loss(layer_contribution, model)

#### Create K.function to fetch loss and gradients from model input

# Create tensor for result storing
dream = model.input 
# Fetch gradient and normalize it
grads = K.gradients(loss = loss, variables = dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  #1e-7 - safety measure to avoid division by 0
# And now we provide link between current result and his gradients with losses
fetch_loss_and_grads = K.function([dream], [loss, grads])
	
#### Predict	        
def predict(img_file, num_octave, octave_scale, iterations, step, max_loss):
	img = preprocess_img(img_file)

	# Create list of shapes correspond with octave scales 
	original_shape = img.shape[1:3]
	octave_shapes = [original_shape]
	for i in range(num_octave):
		scaled_shape = tuple([int(dim/(octave_scale ** i)) for dim in original_shape])
		octave_shapes.append(scaled_shape)
	octave_shapes = octave_shapes[::-1]

	orginal_img = np.copy(img)

	# Initialize shrunck image by the smallest image
	shrunck_original_img = resize_img(img, octave_shapes[0]) 

	for shape in octave_shapes:
		print('Processing image shape: ', shape)
		# Image gradient ascenting 
		img = resize_img(img, shape)
		img = gradient_ascent(img, iterations, step, fetch_loss_and_grads, max_loss)
		# Lost detail computation
		upscaled_shrunck_original_img = resize_img(shrunck_original_img, shape)
		same_original_size = resize_img(orginal_img, shape)
		lost_detail = same_original_size - upscaled_shrunck_original_img
		# Impute details
		img += lost_detail
		save_img(img, './notebook_images/scale_{}.png'.format(shape))
		
		shrunck_original_img = resize_img(orginal_img, shape)
		
	save_img(img, 'result.png') 
	print('Process finished, result was saved in the project root folder')
	
predict(img_file=DEMO_IMAGE_PATH, 
		num_octave=NUM_OCTAVE, 
		octave_scale=OCTAVE_SCALE,
		iterations=ITERATIONS,
		step=STEP,
		max_loss=MAX_LOSS)
