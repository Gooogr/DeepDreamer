import scipy
import numpy as np
import cv2
import argparse
import json
from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image
from utils import *

#Set learning phase = 0 (test mode). Prevent model learning for safety reasons.
K.set_learning_phase(0) 
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)	


##### Create parser #####
parser = argparse.ArgumentParser()
parser.add_argument('-i', action = 'store', type = str,
					dest = 'image_path', 
					help = 'Path to the input image',
					required = True)
parser.add_argument('-o', action = 'store', type = str,
					dest = 'json_path', 
					default = 'config.json',
					help = 'Path to the json configuration file')
					
def predict(img_file, num_octave, octave_scale, iterations, step, max_loss, model, loss,  write_result=False):
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
		img = gradient_ascent(img, iterations, step, fetch_loss_and_grads(model, loss), max_loss)
		# Lost detail computation
		upscaled_shrunck_original_img = resize_img(shrunck_original_img, shape)
		same_original_size = resize_img(orginal_img, shape)
		lost_detail = same_original_size - upscaled_shrunck_original_img
		# Impute details
		img += lost_detail
		save_img(img, './imgs/scale_{}.png'.format(shape))
		
		shrunck_original_img = resize_img(orginal_img, shape)
		
	if write_result:	
		save_img(img, 'result.png')
		print('Process finished, result was saved in the project root folder')
	else:
		pil_img = deprocess_img(np.copy(img))
		return pil_img
					
					
if __name__ == "__main__":
	args = parser.parse_args()
	img_path = args.image_path
	with open(args.json_path) as f:
		options = json.load(f)
		f.close()
		
	layer_contribution = options['layer_contribution']
	
	loss = get_loss(layer_contribution, model)
	        
	predict(img_file=img_path, 
			num_octave=options['num_octave'], 
			octave_scale=options['octave_scale'],
			iterations=options['iterations'],
			step=options['step'],
			max_loss=options['max_loss'],
			model = model,
			loss = loss,
			write_result=True)
